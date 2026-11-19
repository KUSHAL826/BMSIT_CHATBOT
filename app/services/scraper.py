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
    MEDIA_EXTENSIONS = (
        ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".ico", ".bmp",
        ".mp4", ".mp3", ".wav", ".avi", ".mov", ".mkv",
        ".zip", ".rar", ".tar", ".gz", ".7z",
        ".exe", ".dmg", ".apk", ".msi",
        ".css", ".js", ".map", ".woff", ".woff2", ".ttf", ".eot"
    )

    DOC_EXTENSIONS = {
        ".pdf": "pdf",
        ".docx": "docx",
        ".doc": "doc",
        ".csv": "csv",
        ".tsv": "csv",
        ".txt": "txt"
    }

    def __init__(self, base_url=None, max_pages=None, max_depth=None):
        settings = StorageService.load_settings()
        self.base_url = base_url or settings.get("bmsit_url", Config.BMSIT_DEFAULT_URL)
        self.max_pages = max_pages or settings.get("scrape_max_pages", Config.SCRAPE_MAX_PAGES)
        self.max_depth = max_depth or settings.get("scrape_depth", Config.SCRAPE_DEPTH)
        
        parsed = urlparse(self.base_url)
        self.domain = parsed.netloc.lower()
        self.apex_domain = self.domain.split(":")[0].removeprefix("www.")
        
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 BMSIT-AI-Bot/1.0",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,application/pdf,*/*;q=0.8"
        })

    def _is_allowed_domain(self, target_url: str) -> bool:
        if not target_url:
            return False
        netloc = urlparse(target_url).netloc.lower().split(":")[0].removeprefix("www.")
        if not netloc:
            return True
        return netloc == self.apex_domain or netloc.endswith("." + self.apex_domain)

    # -------------------------------------------------------------
    # Document Parsers (PDF, DOCX, DOC, CSV)
    # -------------------------------------------------------------
    def _parse_pdf_bytes(self, content_bytes, filename="document.pdf"):
        try:
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(content_bytes))
            pages_text = []
            for idx, p in enumerate(reader.pages):
                txt = (p.extract_text() or "").strip()
                if txt:
                    txt = re.sub(r'[ \t]+', ' ', txt)
                    pages_text.append(f"[Page {idx + 1}]\n{txt}")
            return "\n\n".join(pages_text).strip()
        except Exception as e:
            logger.warning("[Scraper] PDF parsing failed for %s: %s", filename, e)
            return ""

    def _parse_docx_bytes(self, content_bytes, filename="document.docx"):
        try:
            import docx
            doc = docx.Document(io.BytesIO(content_bytes))
            lines = []
            for p in doc.paragraphs:
                txt = p.text.strip()
                if txt:
                    lines.append(txt)
            for table in doc.tables:
                for row in table.rows:
                    row_txt = " | ".join(c.text.strip() for c in row.cells if c.text.strip())
                    if row_txt:
                        lines.append(row_txt)
            return "\n\n".join(lines).strip()
        except Exception as e:
            logger.warning("[Scraper] DOCX parsing failed for %s: %s", filename, e)
            return ""

    def _parse_doc_bytes(self, content_bytes, filename="document.doc"):
        try:
            from app.services.document_parser import DocumentParser
            with tempfile.NamedTemporaryFile(suffix=".doc", delete=False) as tmp:
                tmp.write(content_bytes)
                tmp_path = Path(tmp.name)
            try:
                sections = DocumentParser._parse_doc(tmp_path)
                return "\n\n".join(s["content"] for s in sections if s.get("content")).strip()
            finally:
                if tmp_path.exists():
                    tmp_path.unlink()
        except Exception as e:
            logger.warning("[Scraper] Legacy DOC parsing failed for %s: %s", filename, e)
            return ""

    def _parse_csv_bytes(self, content_bytes, filename="data.csv"):
        try:
            import csv
            text = content_bytes.decode("utf-8", errors="replace")
            reader = csv.reader(io.StringIO(text))
            lines = []
            for row in reader:
                row_str = " | ".join(c.strip() for c in row if c.strip())
                if row_str:
                    lines.append(row_str)
            return "\n".join(lines).strip()
        except Exception as e:
            logger.warning("[Scraper] CSV parsing failed for %s: %s", filename, e)
            return ""

    def run_sync(self, keep_running_for_indexing=False):
        """
        Executes the scraping process synchronously and returns results.
        Crawls web pages and linked documents (PDF, DOCX, DOC, CSV) recursively.
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
            # Documents (PDF, DOCX) are exempt from boilerplate removal
            scraper_status.phase = "cleaning"
            scraper_status.status_message = f"Removing repeated navigation text from {len(scraped_pages)} pages..."
            removed_lines = self._strip_cross_page_boilerplate(scraped_pages)
            logger.info("[Scraper] Filtered %s repeated boilerplate line(s) site-wide.", removed_lines)

            # Change detection runs on the CLEANED text
            for url, page_data in scraped_pages.items():
                old_info = old_pages.get(url)
                doc_suffix = f" ({page_data.get('file_type', 'doc').upper()} Document)" if page_data.get("is_document") else ""
                if not old_info:
                    detected_changes.append({
                        "url": url,
                        "title": page_data["title"],
                        "change_type": "NEW",
                        "summary": f"New content discovered{doc_suffix} ({len(page_data['content'])} characters)"
                    })
                elif old_info.get("hash") != page_data["hash"]:
                    detected_changes.append({
                        "url": url,
                        "title": page_data["title"],
                        "change_type": "MODIFIED",
                        "summary": f"Content changed{doc_suffix} (diff size: {len(page_data['content']) - old_info.get('length', 0):+d} chars)"
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
        Fetches an HTML web page or document (PDF, Word, CSV), extracts clean
        text content, discovers internal links, and returns structured page data.
        """
        try:
            resp = self.session.get(
                url, timeout=Config.SCRAPE_REQUEST_TIMEOUT, allow_redirects=True, stream=True
            )
            if resp.status_code != 200:
                return None
            
            content_type = (resp.headers.get("Content-Type") or "").lower()
            path_lower = urlparse(url).path.lower()

            # Ignore non-text media (images, videos, archives, executables)
            if any(path_lower.endswith(ext) for ext in self.MEDIA_EXTENSIONS):
                return None

            # Skip oversized files (> 20MB)
            content_length = resp.headers.get("Content-Length")
            if content_length and content_length.isdigit() and int(content_length) > 20 * 1024 * 1024:
                return None

            # Check if document format
            is_pdf = path_lower.endswith(".pdf") or "application/pdf" in content_type
            is_docx = path_lower.endswith(".docx") or "application/vnd.openxmlformats-officedocument.wordprocessingml" in content_type
            is_doc = path_lower.endswith(".doc") or "application/msword" in content_type
            is_csv = path_lower.endswith((".csv", ".tsv")) or "text/csv" in content_type
            is_txt = path_lower.endswith(".txt") and "text/plain" in content_type

            if is_pdf or is_docx or is_doc or is_csv or is_txt:
                content_bytes = resp.content
                if len(content_bytes) > 20 * 1024 * 1024:
                    return None

                if is_pdf:
                    cleaned_text = self._parse_pdf_bytes(content_bytes, filename=url)
                    doc_type = "pdf"
                elif is_docx:
                    cleaned_text = self._parse_docx_bytes(content_bytes, filename=url)
                    doc_type = "docx"
                elif is_doc:
                    cleaned_text = self._parse_doc_bytes(content_bytes, filename=url)
                    doc_type = "doc"
                elif is_csv:
                    cleaned_text = self._parse_csv_bytes(content_bytes, filename=url)
                    doc_type = "csv"
                else:
                    cleaned_text = content_bytes.decode("utf-8", errors="replace").strip()
                    doc_type = "txt"

                if len(cleaned_text) < 50:
                    return None

                filename = urlparse(url).path.split("/")[-1].split("?")[0]
                base_name = re.sub(r'\.(pdf|docx|doc|csv|tsv|txt)$', '', filename, flags=re.I)
                clean_name = re.sub(r'[-_+]', ' ', base_name).strip()

                if anchor_text and len(anchor_text) >= 4 and not re.match(r'^(click|here|download|view|pdf|link|read|more)$', anchor_text, re.I):
                    title = f"{anchor_text} ({doc_type.upper()} Document)"
                elif clean_name and len(clean_name) >= 3:
                    title = f"BMSIT {clean_name.title()} ({doc_type.upper()} Document)"
                else:
                    title = f"BMSIT Document {doc_type.upper()}"

                content_hash = hashlib.md5(cleaned_text.encode("utf-8")).hexdigest()

                return {
                    "url": url,
                    "title": title,
                    "content": cleaned_text,
                    "hash": content_hash,
                    "links": [],
                    "is_document": True,
                    "file_type": doc_type
                }

            # ---------------- Parse HTML Web Page ----------------
            if "text/html" not in content_type:
                return None

            soup = BeautifulSoup(resp.text, "html.parser")

            # Extract links before removing navigation elements
            discovered_links = []
            for a in soup.find_all("a", href=True):
                href = a["href"].strip()
                if not href or href.startswith(("javascript:", "mailto:", "tel:", "#")):
                    continue
                full_link = urljoin(url, href).split("#")[0].rstrip("/")
                if not full_link or not full_link.startswith(("http://", "https://")):
                    continue
                p_lower = urlparse(full_link).path.lower()
                if any(p_lower.endswith(ext) for ext in self.MEDIA_EXTENSIONS):
                    continue
                a_text = a.get_text().strip()
                a_text = re.sub(r'\s+', ' ', a_text)
                discovered_links.append((full_link, a_text))

            # Also discover embedded documents in iframes or embeds
            for el in soup.find_all(["iframe", "embed", "object"], src=True):
                src = el.get("src", "").strip()
                if src and not src.startswith(("javascript:", "#")):
                    full_link = urljoin(url, src).split("#")[0].rstrip("/")
                    p_lower = urlparse(full_link).path.lower()
                    if any(p_lower.endswith(ext) for ext in self.DOC_EXTENSIONS):
                        discovered_links.append((full_link, "Embedded Document"))

            # Strip non-content and layout boilerplate tags
            for tag in soup(["script", "style", "noscript", "iframe", "svg", "header", "footer", "nav", "aside", "form"]):
                tag.decompose()

            # Strip BMSIT-specific navigation, sidebars, and footer sections
            boilerplate_selectors = [
                ".top-xtra-menu-bg", ".top-xtra-menu", ".top-menu", ".mainMenu", 
                ".sidebar", "#sidebar", ".footer", ".footer-bg", ".footer-area", 
                ".bottom-footer", "#footer", "#header", ".navbar", ".dropdown-menu", 
                ".modal", ".breadcrumbs", ".social-icons", ".quick-links", ".marquee",
                ".for-mobile-shift"
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
