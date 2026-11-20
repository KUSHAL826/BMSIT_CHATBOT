import hashlib
import io
import json
import logging
import os
import re
import tempfile
import threading
import time
from pathlib import Path
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
    # Extensions to skip: documents (PDF, Word, Excel, CSV), media, archives, and web assets.
    # Website scraper exclusively crawls HTML web pages. Documents are never loaded from website.
    SKIP_EXTENSIONS = (
        # Documents (never load or embed from website)
        ".pdf", ".docx", ".doc", ".xlsx", ".xls", ".csv", ".tsv",
        ".pptx", ".ppt", ".odt", ".ods", ".odp", ".rtf", ".txt",
        # Media & images
        ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".ico", ".bmp",
        ".mp4", ".mp3", ".wav", ".avi", ".mov", ".mkv",
        # Archives & executables
        ".zip", ".rar", ".tar", ".gz", ".7z",
        ".exe", ".dmg", ".apk", ".msi",
        # Web assets
        ".css", ".js", ".map", ".woff", ".woff2", ".ttf", ".eot"
    )

    def __init__(self, base_url=None, max_pages=None, max_depth=None):
        settings = StorageService.load_settings()
        self.base_url = (
            base_url
            or os.getenv("BMSIT_DEFAULT_URL")
            or settings.get("bmsit_url", Config.BMSIT_DEFAULT_URL)
        )
        # Environment variables take direct priority
        env_max_pages = os.getenv("SCRAPE_MAX_PAGES")
        env_depth = os.getenv("SCRAPE_DEPTH")
        self.max_pages = int(env_max_pages) if env_max_pages else (max_pages or Config.SCRAPE_MAX_PAGES)
        self.max_depth = int(env_depth) if env_depth else (max_depth or Config.SCRAPE_DEPTH)
        
        parsed = urlparse(self.base_url)
        self.domain = parsed.netloc.lower()
        self.apex_domain = self.domain.split(":")[0].removeprefix("www.")
        
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 BMSIT-AI-Bot/1.0",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
        })

    def _is_allowed_domain(self, target_url: str) -> bool:
        if not target_url:
            return False
        parsed = urlparse(target_url)
        netloc = parsed.netloc.lower().split(":")[0].removeprefix("www.")
        if not netloc:
            return True
        # Reject malformed URLs containing fake nested domain paths (e.g. /linkedin.com/, /bmsit.irins.org/)
        path_lower = parsed.path.lower()
        if any(ext_dom in path_lower for ext_dom in (".com/", ".org/", ".net/", ".edu/", ".gov/", ".io/")):
            return False
        return netloc == self.apex_domain or netloc.endswith("." + self.apex_domain)

    def run_sync(self, keep_running_for_indexing=False):
        """
        Executes the scraping process synchronously and returns results.
        Crawls HTML web pages across BMSIT recursively.
        Documents (PDF, Word, CSV) are skipped and never loaded.
        Updates scraper_status throughout execution.
        """
        global scraper_status
        scraper_status.is_running = True
        scraper_status.phase = "crawling"
        scraper_status.pages_scraped = 0
        scraper_status.total_target = self.max_pages
        scraper_status.status_message = f"Starting crawl of {self.base_url}"
        scraper_status.error = None

        # Comprehensive BMSIT seed paths across all institutional sections
        priority_paths = [
            "",
            "/about-us.php",
            "/governing-body.php",
            "/academic-council.php",
            "/administration.php",
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
            "/dep-ete.php",
            "/dep-mca.php",
            "/dep-mba.php",
            "/dep-physics.php",
            "/dep-chemistry.php",
            "/dep-maths.php",
            "/scholarships.php",
            "/library-facilities.php",
            "/autonomous.php",
            "/curriculum.php",
            "/academic-calendar.php",
            "/examinations.php",
            "/results.php",
            "/iqac.php",
            "/nirf-new.php",
            "/aicte-idea-lab.php",
            "/career.php",
            "/mandatory-disclosure.php"
        ]
        visited = set()
        queue = []
        anchor_map = {}

        for p in priority_paths:
            full_u = urljoin(self.base_url, p).split("#")[0].rstrip("/")
            if full_u and not any(q[0] == full_u for q in queue):
                queue.append((full_u, 0))

        # Load previous state for change detection and seed existing pages
        previous_state = StorageService.load_scrape_state()
        old_pages = previous_state.get("pages", {})

        # Ensure previously scraped URLs stay in the queue
        for old_u in old_pages:
            clean_old = old_u.split("#")[0].rstrip("/")
            if clean_old and not any(q[0] == clean_old for q in queue):
                queue.append((clean_old, 0))

        queued_urls = set(q[0] for q in queue)
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
                if not self._is_allowed_domain(url):
                    continue

                visited.add(url)
                scraper_status.current_url = url
                scraper_status.status_message = f"Scraping [{len(visited)}/{self.max_pages}]: {url}"

                anchor = anchor_map.get(url)
                page_data = self._fetch_and_clean_page(url, anchor_text=anchor)
                if not page_data:
                    continue

                scraped_pages[url] = page_data
                scraper_status.pages_scraped = len(scraped_pages)

                # Discover new links dynamically if depth allows
                if depth < self.max_depth and len(queue) < 10000:
                    for link_item in page_data.get("links", []):
                        if isinstance(link_item, tuple):
                            link_url, link_anchor = link_item
                        else:
                            link_url, link_anchor = link_item, ""

                        clean_link = link_url.split("#")[0].rstrip("/")
                        if clean_link and clean_link not in visited and clean_link not in queued_urls:
                            if self._is_allowed_domain(clean_link):
                                queue.append((clean_link, depth + 1))
                                queued_urls.add(clean_link)
                                if link_anchor and clean_link not in anchor_map:
                                    anchor_map[clean_link] = link_anchor

                # Polite delay
                time.sleep(Config.SCRAPE_DELAY)

            # Remove site chrome that repeats across HTML pages (mega-menus, footers)
            scraper_status.phase = "cleaning"
            scraper_status.status_message = f"Cleaning navigation boilerplate across {len(scraped_pages)} pages..."
            removed_lines = self._strip_cross_page_boilerplate(scraped_pages)
            logger.info("[Scraper] Filtered %s repeated boilerplate line(s) site-wide.", removed_lines)

            # Change detection runs on the CLEANED text
            for url, page_data in scraped_pages.items():
                old_info = old_pages.get(url)
                if not old_info:
                    detected_changes.append({
                        "url": url,
                        "title": page_data["title"],
                        "change_type": "NEW",
                        "summary": f"New page content ({len(page_data['content'])} characters)"
                    })
                elif old_info.get("hash") != page_data["hash"]:
                    detected_changes.append({
                        "url": url,
                        "title": page_data["title"],
                        "change_type": "MODIFIED",
                        "summary": f"Content updated (diff size: {len(page_data['content']) - old_info.get('length', 0):+d} chars)"
                    })

            # Pages known from earlier crawls that this run did not reach
            for old_url, old_data in old_pages.items():
                if old_url not in scraped_pages:
                    detected_changes.append({
                        "url": old_url,
                        "title": old_data.get("title", "Unknown"),
                        "change_type": "MISSING",
                        "summary": "Not reached in this crawl - previously stored content retained"
                    })

            # Merge crawl state instead of overwriting it
            merged_pages = dict(old_pages)
            for u, d in scraped_pages.items():
                merged_pages[u] = {
                    "hash": d["hash"],
                    "title": d["title"],
                    "length": len(d["content"]),
                    "is_document": d.get("is_document", False),
                    "file_type": d.get("file_type", "web"),
                    "last_seen": StorageService._now_ist_str()
                }

            changes_summary_text = self._format_changes_summary(detected_changes)
            state = StorageService.load_scrape_state()
            state["pages"] = merged_pages
            state["changes"] = detected_changes
            state["last_scrape"] = StorageService._now_ist_str()
            state["summary"] = changes_summary_text
            StorageService.save_scrape_state(state)

            # Snapshot of raw crawled content, merged with previous snapshots
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
            scraper_status.status_message = f"Crawled {len(scraped_pages)} pages and documents successfully."
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
        Drops lines that appear on a large share of the crawled HTML pages.
        Documents (PDF, DOCX) are preserved without stripping.
        """
        html_pages = {k: v for k, v in scraped_pages.items() if not v.get("is_document")}
        total_pages = len(html_pages)
        if total_pages < 4:
            return 0

        line_pages = {}
        for page in html_pages.values():
            for line in set(page["content"].splitlines()):
                normalized = line.strip().lower()
                if not normalized:
                    continue
                line_pages[normalized] = line_pages.get(normalized, 0) + 1

        threshold = max(6, int(total_pages * Config.SCRAPE_BOILERPLATE_RATIO))
        boilerplate = {
            line for line, count in line_pages.items()
            if count >= threshold and len(line) <= 90
        }
        if not boilerplate:
            return 0

        gutted = 0
        for page in html_pages.values():
            original = page["content"]
            kept, previous = [], None
            for line in original.splitlines():
                normalized = line.strip().lower()
                if normalized in boilerplate:
                    continue
                if normalized and normalized == previous:
                    continue
                previous = normalized
                kept.append(line)
            cleaned = "\n".join(kept).strip()

            if len(cleaned) < max(200, int(len(original) * 0.15)):
                gutted += 1
                continue

            page["content"] = cleaned
            page["hash"] = hashlib.md5(cleaned.encode("utf-8")).hexdigest()

        if gutted:
            logger.info(
                "[Scraper] Kept %s page(s) unfiltered because boilerplate removal would have emptied them.", gutted,
            )
        return len(boilerplate)

    def _fetch_and_clean_page(self, url, anchor_text=None):
        """
        Fetches an HTML web page, extracts clean text content, discovers internal
        links, and returns structured page data.
        Documents (PDF, Word, Excel, CSV) are never loaded or parsed from website.
        """
        try:
            path_lower = urlparse(url).path.lower()

            # Ignore non-HTML files (documents, images, videos, archives, executables)
            if any(path_lower.endswith(ext) for ext in self.SKIP_EXTENSIONS):
                return None

            resp = self.session.get(
                url, timeout=Config.SCRAPE_REQUEST_TIMEOUT, allow_redirects=True, stream=True
            )
            if resp.status_code != 200:
                return None
            
            content_type = (resp.headers.get("Content-Type") or "").lower()
            if "text/html" not in content_type:
                return None

            soup = BeautifulSoup(resp.text, "html.parser")

            # Extract links before removing navigation elements
            domain_without_scheme_re = re.compile(
                r'^(?:[a-zA-Z0-9-]+\.)+(?:com|org|net|edu|gov|in|io|co|ac\.in)(?:/|$)', re.I
            )
            discovered_links = []
            for a in soup.find_all("a", href=True):
                href = a["href"].strip()
                if not href or href.startswith(("javascript:", "mailto:", "tel:", "#")):
                    continue
                # Skip invalid developer placeholders (e.g. "NIL", "Scopus Author ID: ...")
                if re.search(r'\s', href) or href.upper() in ("NIL", "NULL", "NONE"):
                    continue
                # If href is an external domain without protocol (e.g. linkedin.com/...), prepend https://
                if domain_without_scheme_re.match(href):
                    href = "https://" + href

                full_link = urljoin(url, href).split("#")[0].rstrip("/")
                if not full_link or not full_link.startswith(("http://", "https://")):
                    continue
                
                # Check that full link does not contain a malformed nested domain in path
                p = urlparse(full_link)
                path_lower = p.path.lower()
                if any(ext_dom in path_lower for ext_dom in (".com/", ".org/", ".net/", ".edu/", ".gov/", ".io/")):
                    continue

                # Do NOT queue documents (PDF, Word, Excel, etc.) or media
                if any(path_lower.endswith(ext) for ext in self.SKIP_EXTENSIONS):
                    continue
                a_text = a.get_text().strip()
                a_text = re.sub(r'\s+', ' ', a_text)
                discovered_links.append((full_link, a_text))

            # Preserve essential footer contact & helpline information before removing footer
            footer_contact_text = ""
            footer_addr = soup.find(class_=re.compile(r'footer-address-box', re.I))
            if footer_addr:
                addr_lines = [l.strip() for l in footer_addr.get_text(separator="\n").splitlines() if l.strip()]
                if addr_lines:
                    footer_contact_text = "\n\n[BMSIT Institutional Address & Admissions Contacts]\n" + "\n".join(addr_lines)

            # Preserve semantic team & faculty cards with explicit labels
            for card in soup.find_all(class_=re.compile(r'(placement-member|faculty-member|faculty-card|team-member)', re.I)):
                name_el = card.find(class_=re.compile(r'(name|title)', re.I))
                role_el = card.find(class_=re.compile(r'(role|designation|post)', re.I))
                if name_el and role_el:
                    card_text = f"\nMember: {name_el.get_text().strip()} | Role: {role_el.get_text().strip()}\n"
                    card.replace_with(soup.new_string(card_text))

            # Strip non-content and layout boilerplate tags
            for tag in soup(["script", "style", "noscript", "iframe", "svg", "header", "footer", "nav", "aside", "form"]):
                tag.decompose()

            # Strip BMSIT-specific navigation, sidebars, submenus, and footer sections
            boilerplate_selectors = [
                ".top-xtra-menu-bg", ".top-xtra-menu", ".top-menu", ".mainMenu", 
                ".sidebar", "#sidebar", ".footer", ".footer-bg", ".footer-area", 
                ".bottom-footer", "#footer", "#header", ".navbar", ".dropdown-menu", 
                ".modal", ".breadcrumbs", ".social-icons", ".quick-links", ".marquee",
                ".for-mobile-shift", ".inner-common-menu-bg-1", ".resp-tabs-list",
                ".top-forOnline-btn-bg", ".top-admission-btn-bg", ".head-top-btn"
            ]
            for sel in boilerplate_selectors:
                for el in soup.select(sel):
                    el.decompose()

            # Determine specific, high-quality unique title
            title = ""
            h1 = soup.find("h1")
            slug = urlparse(url).path.strip("/").replace(".php", "").replace("-", " ").title()

            if "placement" in slug.lower() and "contact" in slug.lower():
                title = "BMSIT Placement Team Contact Details"
            elif slug.lower() == "contact details":
                title = "BMSIT College Main Office Contact Details"
            elif h1 and h1.get_text().strip():
                h1_txt = h1.get_text().strip()
                if slug and slug.lower() not in h1_txt.lower():
                    title = f"BMSIT {slug} - {h1_txt}"
                else:
                    title = f"BMSIT {h1_txt}"
            elif slug:
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

            # Append essential footer contact information if not already present in the page
            if footer_contact_text and "Contact for Admissions" not in cleaned_text:
                cleaned_text += footer_contact_text

            if len(cleaned_text) < 60:
                return None

            content_hash = hashlib.md5(cleaned_text.encode("utf-8")).hexdigest()

            return {
                "url": url,
                "title": title,
                "content": cleaned_text,
                "hash": content_hash,
                "links": discovered_links,
                "is_document": False,
                "file_type": "web"
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
