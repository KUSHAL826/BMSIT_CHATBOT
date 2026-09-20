import hashlib
import json
import logging
import re
import threading
import time
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup

from app.config import Config
from app.services.fsutil import atomic_write_json, read_json
from app.services.storage import StorageService

logger = logging.getLogger(__name__)

class ScraperStatus:
    def __init__(self):
        self.is_running = False
        self.phase = "idle"
        self.current_url = ""
        self.pages_scraped = 0
        self.total_target = 0
        self.status_message = "Idle"
        self.error = None
        self.last_result = None

    def to_dict(self):
        return {
            "is_running": self.is_running,
            "phase": self.phase,
            "current_url": self.current_url,
            "pages_scraped": self.pages_scraped,
            "total_target": self.total_target,
            "status_message": self.status_message,
            "error": self.error,
            "last_result": self.last_result
        }

scraper_status = ScraperStatus()

class WebsiteScraper:
    def __init__(self, base_url=None, max_pages=None, max_depth=None):
        settings = StorageService.load_settings()
        self.base_url = base_url or settings.get("bmsit_url", Config.BMSIT_DEFAULT_URL)
        self.max_pages = max_pages or settings.get("scrape_max_pages", Config.SCRAPE_MAX_PAGES)
        self.max_depth = max_depth or settings.get("scrape_depth", Config.SCRAPE_DEPTH)
        
        parsed = urlparse(self.base_url)
        self.domain = parsed.netloc
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 BMSIT-AI-Bot/1.0",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
        })

    def run_sync(self, keep_running_for_indexing=False):
        """
        Executes the scraping process synchronously and returns results.
        Updates scraper_status throughout execution.
        """
        global scraper_status
        scraper_status.is_running = True
        scraper_status.phase = "crawling"
        scraper_status.pages_scraped = 0
        scraper_status.total_target = self.max_pages
        scraper_status.status_message = f"Starting crawl of {self.base_url}"
        scraper_status.error = None

        # Targeted BMSIT URLs with real content
        priority_paths = [
            "",
            "/admissions.php",
            "/admission-query.php",
            "/fee-structure.php",
            "/hostel-facilities.php",
            "/placement-new.php",
            "/academics-overview.php",
            "/contact-details.php",
            "/dep-cse.php",
            "/dep-aiml.php",
            "/dep-ise.php",
            "/dep-ece-vlsi-mtech-system.php",
            "/dep-electrical.php",
            "/dep-mechanical.php",
            "/dep-civil.php",
            "/scholarships.php",
            "/library-facilities.php",
            "/autonomous.php",
            "/iqac.php",
            "/nirf-new.php",
            "/aicte-idea-lab.php",
            "/career.php"
        ]
        visited = set()
        queue = []
        for p in priority_paths:
            full_u = urljoin(self.base_url, p).rstrip("/")
            if (full_u, 0) not in queue:
                queue.append((full_u, 0))

        # Load previous state for change detection and seed existing pages
        previous_state = StorageService.load_scrape_state()
        old_pages = previous_state.get("pages", {})

        # Ensure previously scraped URLs stay in the queue in consistent order
        for old_u in old_pages:
            if old_u and not any(q[0] == old_u for q in queue):
                queue.append((old_u, 0))

        scraped_pages = {}
        detected_changes = []

        try:
            while queue and len(visited) < self.max_pages:
                url, depth = queue.pop(0)

                # Normalize URL
                url = url.split("#")[0].rstrip("/")
                if not url or url in visited:
                    continue
                
                # Check domain boundary
                u_domain = urlparse(url).netloc
                if u_domain and u_domain != self.domain and not u_domain.endswith("." + self.domain):
                    continue

                visited.add(url)
                scraper_status.current_url = url
                scraper_status.status_message = f"Scraping [{len(visited)}/{self.max_pages}]: {url}"

                page_data = self._fetch_and_clean_page(url)
                if not page_data:
                    continue

                scraped_pages[url] = page_data
                scraper_status.pages_scraped = len(scraped_pages)

                # Discover new links if depth allows
                if depth < self.max_depth and len(visited) + len(queue) < self.max_pages:
                    for link in page_data["links"]:
                        clean_link = link.split("#")[0].rstrip("/")
                        if clean_link and clean_link not in visited and clean_link not in [q[0] for q in queue]:
                            # Only crawl same domain
                            l_domain = urlparse(clean_link).netloc
                            if l_domain == self.domain or not l_domain:
                                queue.append((clean_link, depth + 1))

                # Polite delay
                time.sleep(Config.SCRAPE_DELAY)

            # Remove site chrome that repeats across pages (mega-menus, ticker
            # links, carousels). Without this, the vast majority of indexed text
            # is navigation noise and the assistant answers from menu labels
            # instead of real page content.
            scraper_status.phase = "cleaning"
            scraper_status.status_message = f"Removing repeated navigation text from {len(scraped_pages)} pages..."
            removed_lines = self._strip_cross_page_boilerplate(scraped_pages)
            logger.info("[Scraper] Filtered %s repeated boilerplate line(s) site-wide.", removed_lines)

            # Change detection runs on the CLEANED text, so a menu tweak on the
            # website no longer marks every page as modified.
            for url, page_data in scraped_pages.items():
                old_info = old_pages.get(url)
                if not old_info:
                    detected_changes.append({
                        "url": url,
                        "title": page_data["title"],
                        "change_type": "NEW",
                        "summary": f"New page discovered ({len(page_data['content'])} characters)"
                    })
                elif old_info.get("hash") != page_data["hash"]:
                    detected_changes.append({
                        "url": url,
                        "title": page_data["title"],
                        "change_type": "MODIFIED",
                        "summary": f"Content changed (diff size: {len(page_data['content']) - old_info.get('length', 0):+d} chars)"
                    })

            # Pages known from earlier crawls that this run did not reach.
            # They are reported but NOT treated as deletions: a crawl can stop
            # early on the page cap, and stored knowledge is never discarded
            # implicitly.
            for old_url, old_data in old_pages.items():
                if old_url not in scraped_pages:
                    detected_changes.append({
                        "url": old_url,
                        "title": old_data.get("title", "Unknown"),
                        "change_type": "MISSING",
                        "summary": "Not reached in this crawl - previously stored content retained"
                    })

            # Merge crawl state instead of overwriting it, so pages that were not
            # revisited are not rediscovered as "NEW" on every single run.
            merged_pages = dict(old_pages)
            for u, d in scraped_pages.items():
                merged_pages[u] = {
                    "hash": d["hash"],
                    "title": d["title"],
                    "length": len(d["content"]),
                    "last_seen": StorageService._now_ist_str()
                }

            changes_summary_text = self._format_changes_summary(detected_changes)
            StorageService.save_scrape_state({
                "pages": merged_pages,
                "changes": detected_changes,
                "last_scrape": StorageService._now_ist_str(),
                "summary": changes_summary_text
            })

            # Snapshot of raw crawled HTML text, merged with previous snapshots.
            scrape_batch_file = Config.SCRAPED_DIR / "bmsit_web_content.json"
            snapshot = read_json(scrape_batch_file, default={}) or {}
            if not isinstance(snapshot, dict):
                snapshot = {}
            snapshot.update(scraped_pages)
            atomic_write_json(scrape_batch_file, snapshot)

            result = {
                "status": "success",
                "total_pages": len(scraped_pages),
                "changes": detected_changes,
                "changes_summary": changes_summary_text,
                "scraped_pages": scraped_pages
            }
            scraper_status.last_result = result
            scraper_status.status_message = f"Crawled {len(scraped_pages)} pages successfully."
            return result

        except Exception as e:
            logger.exception("Scraping error: %s", e)
            scraper_status.error = str(e)
            scraper_status.status_message = f"Failed: {str(e)}"
            return {
                "status": "error",
                "message": str(e),
                "pages_scraped": len(scraped_pages)
            }
        finally:
            if not keep_running_for_indexing:
                scraper_status.is_running = False

    def _strip_cross_page_boilerplate(self, scraped_pages):
        """
        Drops lines that appear on a large share of the crawled pages.

        A college site repeats its mega-menu, news ticker and footer on every
        page. Those lines carry no page-specific meaning but dominate the index
        by volume, so retrieval keeps surfacing menu labels. Any line seen on
        more than ~22% of pages (and on at least 4 pages) is treated as chrome
        and removed. Content hashes are recomputed from the cleaned text.

        Returns the number of distinct lines filtered.
        """
        total_pages = len(scraped_pages)
        if total_pages < 4:
            return 0

        line_pages = {}
        for page in scraped_pages.values():
            for line in set(page["content"].splitlines()):
                normalized = line.strip().lower()
                if not normalized:
                    continue
                line_pages[normalized] = line_pages.get(normalized, 0) + 1

        threshold = max(6, int(total_pages * Config.SCRAPE_BOILERPLATE_RATIO))
        boilerplate = {
            line for line, count in line_pages.items()
            # Only short lines qualify. Long repeated paragraphs are usually
            # genuine content (an address block, a standard course description).
            if count >= threshold and len(line) <= 90
        }
        if not boilerplate:
            return 0

        gutted = 0
        for page in scraped_pages.values():
            original = page["content"]
            kept, previous = [], None
            for line in original.splitlines():
                normalized = line.strip().lower()
                if normalized in boilerplate:
                    continue
                # Collapse immediate repeats left behind by the removal.
                if normalized and normalized == previous:
                    continue
                previous = normalized
                kept.append(line)
            cleaned = "\n".join(kept).strip()

            # Guard: if stripping removed almost everything, this page is built
            # mostly from shared template text. Keep the original rather than
            # reducing the page to nothing - an empty page is dropped from the
            # knowledge base entirely, which silently loses real content.
            if len(cleaned) < max(200, int(len(original) * 0.15)):
                gutted += 1
                continue

            page["content"] = cleaned
            page["hash"] = hashlib.md5(cleaned.encode("utf-8")).hexdigest()

        if gutted:
            logger.info(
                "[Scraper] Kept %s page(s) unfiltered because boilerplate removal would have "
                "emptied them.", gutted,
            )
        return len(boilerplate)

    def _fetch_and_clean_page(self, url):
        """Fetches a page, strips navigation/footer boilerplate, extracts clean text and links."""
        try:
            resp = self.session.get(
                url, timeout=Config.SCRAPE_REQUEST_TIMEOUT, allow_redirects=True
            )
            if resp.status_code != 200:
                return None
            
            content_type = resp.headers.get("Content-Type", "")
            if "text/html" not in content_type:
                return None

            soup = BeautifulSoup(resp.text, "html.parser")

            # Extract links before removing navigation elements
            links = []
            for a in soup.find_all("a", href=True):
                href = a["href"].strip()
                if href and not href.startswith(("javascript:", "mailto:", "tel:", "#")):
                    full_link = urljoin(url, href)
                    # Ignore binary media and archives
                    if not re.search(r'\.(jpg|jpeg|png|gif|svg|pdf|zip|rar|tar|mp4|exe|css|js)$', full_link, re.I):
                        links.append(full_link)

            # Strip non-content and layout boilerplate tags
            for tag in soup(["script", "style", "noscript", "iframe", "svg", "header", "footer", "nav", "aside", "form"]):
                tag.decompose()

            # Strip BMSIT-specific navigation, sidebars, and footer sections
            boilerplate_selectors = [
                ".top-xtra-menu-bg", ".top-xtra-menu", ".top-menu", ".mainMenu", 
                ".sidebar", "#sidebar", ".footer", ".footer-bg", ".footer-area", 
                ".bottom-footer", "#footer", "#header", ".navbar", ".dropdown-menu", 
                ".modal", ".breadcrumbs", ".social-icons", ".quick-links", ".marquee",
                ".parentHorizontalTab", ".for-mobile-shift"
            ]
            for sel in boilerplate_selectors:
                for el in soup.select(sel):
                    el.decompose()

            # Determine specific, high-quality title
            title = ""
            h1 = soup.find("h1")
            if h1 and h1.get_text().strip():
                title = h1.get_text().strip()
            elif soup.find("h2"):
                h2 = soup.find("h2")
                if h2 and h2.get_text().strip():
                    title = h2.get_text().strip()

            if not title and soup.title and soup.title.string:
                title = soup.title.string.strip()

            # If title is generic college name, generate from URL path slug
            clean_title_lower = (title or "").lower()
            if not title or "bms institute of technology" in clean_title_lower or clean_title_lower in ["bmsit", "bmsit&m", "home"]:
                slug = urlparse(url).path.strip("/").replace(".php", "").replace("-", " ").title()
                if slug:
                    title = f"BMSIT {slug}"
                else:
                    title = "BMSIT Official Portal"

            # Clean main content from body
            body = soup.body
            if not body:
                return None

            text = body.get_text(separator="\n")
            lines = [line.strip() for line in text.splitlines() if line.strip() and len(line.strip()) > 1]
            cleaned_text = "\n".join(lines)

            if len(cleaned_text) < 60:
                # Too little content
                return None

            content_hash = hashlib.md5(cleaned_text.encode("utf-8")).hexdigest()

            return {
                "url": url,
                "title": title,
                "content": cleaned_text,
                "hash": content_hash,
                "links": list(set(links))
            }
        except Exception as e:
            logger.debug("Error fetching %s: %s", url, e)
            return None

    def _format_changes_summary(self, changes):
        if not changes:
            return "No changes detected. Website content is up to date."
        new_c = sum(1 for c in changes if c["change_type"] == "NEW")
        mod_c = sum(1 for c in changes if c["change_type"] == "MODIFIED")
        missing_c = sum(1 for c in changes if c["change_type"] in ("MISSING", "DELETED"))
        parts = []
        if new_c:
            parts.append(f"{new_c} new page{'s' if new_c > 1 else ''}")
        if mod_c:
            parts.append(f"{mod_c} updated page{'s' if mod_c > 1 else ''}")
        if missing_c:
            parts.append(f"{missing_c} page{'s' if missing_c > 1 else ''} not reached (content retained)")
        if not parts:
            return "No content changes detected. Knowledge base already up to date."
        return "Detected: " + ", ".join(parts)


def start_async_scrape(base_url=None, max_pages=None, max_depth=None, on_complete=None):
    """Launches the scraper in a background thread."""
    scraper = WebsiteScraper(base_url, max_pages, max_depth)

    def worker():
        result = scraper.run_sync()
        if on_complete:
            try:
                on_complete(result)
            except Exception as e:
                logger.error("Error in on_complete callback: %s", e)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    return thread
