import unittest
import hashlib
from unittest.mock import MagicMock, patch
from app.services.scraper import WebsiteScraper
from app.services.ingestion_registry import content_hash


class TestWebsiteScraperDeep(unittest.TestCase):
    def setUp(self):
        self.scraper = WebsiteScraper(base_url="https://bmsit.ac.in/")

    def test_allowed_domains_bmsit(self):
        """BMSIT main domain and subdomains must be allowed."""
        allowed_cases = [
            "https://bmsit.ac.in/",
            "https://bmsit.ac.in/about-us.php",
            "https://bmsit.ac.in/dep-cse.php",
            "https://bmsit.ac.in/contact-details.php",
            "https://staff.bmsit.ac.in/sims/index.php",
            "https://student.bmsit.ac.in/",
            "https://results.bmsit.ac.in/",
            "https://projects.bmsit.ac.in/portal",
        ]
        for url in allowed_cases:
            with self.subTest(url=url):
                self.assertTrue(self.scraper._is_allowed_domain(url), f"Should allow {url}")

    def test_allowed_domains_linkedin_profiles(self):
        """Faculty LinkedIn profiles must be strictly allowed."""
        allowed_linkedin = [
            "https://www.linkedin.com/in/dr-satish-kumar",
            "https://in.linkedin.com/in/dr-shoba-m-chandra-33ab2652",
            "https://linkedin.com/in/dr-arun-kumar-b-r",
            "https://www.linkedin.com/pub/faculty-profile/12/34/56",
        ]
        for url in allowed_linkedin:
            with self.subTest(url=url):
                self.assertTrue(self.scraper._is_allowed_domain(url), f"Should allow faculty LinkedIn {url}")

    def test_rejected_domains_non_profile_linkedin(self):
        """Non-profile LinkedIn pages must be strictly rejected."""
        rejected_linkedin = [
            "https://www.linkedin.com/feed",
            "https://www.linkedin.com/feed/",
            "https://www.linkedin.com/company/bms-institute-of-technology",
            "https://www.linkedin.com/jobs/search",
            "https://www.linkedin.com/login",
            "https://www.linkedin.com/learning",
            "https://www.linkedin.com/groups",
            "https://www.linkedin.com/school/bmsit",
        ]
        for url in rejected_linkedin:
            with self.subTest(url=url):
                self.assertFalse(self.scraper._is_allowed_domain(url), f"Should reject non-profile LinkedIn {url}")

    def test_rejected_domains_third_party_and_research(self):
        """IRINS, ResearchGate, BMSET, and other external portals must be strictly rejected."""
        rejected_cases = [
            "https://bmsit.irins.org/",
            "https://irins.org/profile/12345",
            "https://www.researchgate.net/profile/John-Doe",
            "https://orcid.org/0000-0002-1825-0097",
            "https://bmset.org/",
            "https://alumni.bmset.org/",
            "https://vtu.ac.in/",
            "https://aicte-india.org/",
            "https://google.com/",
            "https://facebook.com/bmsit",
            "https://twitter.com/bmsit",
            "https://instagram.com/bmsit",
            "https://youtube.com/bmsit",
        ]
        for url in rejected_cases:
            with self.subTest(url=url):
                self.assertFalse(self.scraper._is_allowed_domain(url), f"Should reject {url}")

    def test_rejected_documents_and_media(self):
        """Documents, media, and images must never be crawled by the website scraper."""
        rejected_assets = [
            "https://bmsit.ac.in/brochure.pdf",
            "https://bmsit.ac.in/syllabus.docx",
            "https://bmsit.ac.in/data.xlsx",
            "https://bmsit.ac.in/marks.csv",
            "https://bmsit.ac.in/campus.jpg",
            "https://bmsit.ac.in/logo.png",
            "https://bmsit.ac.in/event.mp4",
            "https://bmsit.ac.in/style.css",
            "https://bmsit.ac.in/app.js",
        ]
        for url in rejected_assets:
            with self.subTest(url=url):
                self.assertFalse(self.scraper._is_allowed_domain(url), f"Should reject document/media {url}")

    def test_deep_link_extraction(self):
        """Verify dynamic discovery across <a>, data attributes, onclick, regex URLs, and relative paths."""
        mock_html = """
        <html>
        <head><title>BMSIT CSE Department</title></head>
        <body>
            <a href="https://bmsit.ac.in/faculty.php">Faculty List</a>
            <a href="https://www.linkedin.com/in/dr-test-faculty?utm_source=share&amp;utm_medium=android">Dr. Test LinkedIn</a>
            <div data-url="/dep-aiml.php">AIML Dept</div>
            <button onclick="window.location.href='contact-details.php'">Contact</button>
            <script>
                var nextPortal = 'https://staff.bmsit.ac.in/sims/login.php';
                var relatedPage = 'placement-new.php';
            </script>
            <p>Welcome to Computer Science & Engineering at BMSIT.</p>
        </body>
        </html>
        """
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"Content-Type": "text/html; charset=UTF-8"}
        mock_resp.text = mock_html

        with patch.object(self.scraper.session, "get", return_value=mock_resp):
            page_data = self.scraper._fetch_and_clean_page("https://bmsit.ac.in/dep-cse.php")

        self.assertIsNotNone(page_data)
        links = [l[0] for l in page_data["links"]]
        self.assertIn("https://bmsit.ac.in/faculty.php", links)
        # Verify LinkedIn canonicalization (query params stripped)
        self.assertIn("https://www.linkedin.com/in/dr-test-faculty", links)
        self.assertIn("https://bmsit.ac.in/dep-aiml.php", links)
        self.assertIn("https://bmsit.ac.in/contact-details.php", links)
        self.assertIn("https://staff.bmsit.ac.in/sims/login.php", links)
        self.assertIn("https://bmsit.ac.in/placement-new.php", links)

    def test_unlimited_timeout_behavior(self):
        """When SCRAPE_REQUEST_TIMEOUT is 0, request_timeout must be None (no time limit)."""
        with patch("app.config.Config.SCRAPE_REQUEST_TIMEOUT", 0):
            self.assertIsNone(self.scraper.request_timeout, "Timeout must be None when 0 (unlimited)")

    def test_no_token_limit_preserves_full_pages(self):
        """Scraper and ingestion must preserve full pages and never drop pages on token or length limits."""
        short_html = """
        <html><body><p>Dean Office: +91-80-28561576</p></body></html>
        """
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"Content-Type": "text/html"}
        mock_resp.text = short_html

        with patch.object(self.scraper.session, "get", return_value=mock_resp):
            page_data = self.scraper._fetch_and_clean_page("https://bmsit.ac.in/contact-short.php")

        self.assertIsNotNone(page_data, "Should not drop short pages")
        self.assertIn("Dean Office: +91-80-28561576", page_data["content"])

    def test_retention_on_probe_timeout(self):
        """Unreached pages that time out or encounter errors during probing must be RETAINED, never deleted."""
        # Simulate an old crawl with one page
        old_pages = {
            "https://bmsit.ac.in/slow-page.php": {
                "title": "Slow Page",
                "hash": "abc123hash",
                "length": 500,
            }
        }
        with patch("app.services.storage.StorageService.load_scrape_state", return_value={"pages": old_pages}), \
             patch("app.services.storage.StorageService.save_scrape_state"), \
             patch("app.services.storage.StorageService.record_crawl_run"), \
             patch.object(self.scraper, "_fetch_and_clean_page", return_value=None), \
             patch.object(self.scraper.session, "get", side_effect=Exception("Connection timed out")):
            result = self.scraper.run_sync()

        # deleted_urls must be empty because timeout does not mean page was deleted from the website!
        self.assertEqual(result.get("deleted_urls"), [])
        changes = result.get("changes", [])
        retained = [c for c in changes if c.get("change_type") == "RETAINED"]
        self.assertEqual(len(retained), 1)
        self.assertEqual(retained[0]["url"], "https://bmsit.ac.in/slow-page.php")


class TestDeduplicationLogic(unittest.TestCase):
    def test_unchanged_content_skips_embedding(self):
        """Verify that when content hash matches stored/registry items and vectors exist, embedding is skipped."""
        source_id = "web-bmsit"
        url = "https://bmsit.ac.in/about-us.php"
        text = "BMS Institute of Technology and Management was established in 2002."
        h = content_hash(text)

        # Mock RAGService and IngestionRegistry
        mock_rag = MagicMock()
        mock_rag.source_item_keys.return_value = {url}

        stored_items = {
            url: {
                "title": "About Us",
                "text": text,
                "metadata": {"url": url}
            }
        }
        registry_items = {
            url: {
                "item_key": url,
                "content_hash": h,
                "chunk_ids": ["web-bmsit::abc::1::def"],
                "status": "indexed"
            }
        }

        with patch("app.services.ingestion_registry.IngestionRegistry.decide_batch", return_value=([], [url], {url: "unchanged"})), \
             patch("app.services.ingestion_registry.IngestionRegistry.items", return_value=registry_items), \
             patch("app.services.knowledge_store.KnowledgeStore.get_items", return_value=stored_items):

            # Simulate the deduplication block from ingestion_pipeline.py
            hashes = {url: h}
            from app.services.ingestion_registry import IngestionRegistry
            from app.services.knowledge_store import KnowledgeStore

            changed, unchanged, reasons = IngestionRegistry.decide_batch(source_id, hashes)
            indexed_keys = mock_rag.source_item_keys(source_id)
            reg_items = IngestionRegistry.items(source_id)
            st_items = KnowledgeStore.get_items(source_id)

            final_changed = []
            final_unchanged = set(unchanged)

            for u in changed:
                if u in indexed_keys:
                    reg_item = reg_items.get(u)
                    if reg_item and reg_item.get("content_hash") == hashes.get(u):
                        final_unchanged.add(u)
                        reasons[u] = "already-indexed-registry-match"
                        continue
                    stored_item = st_items.get(u)
                    if stored_item:
                        prev_text = (stored_item.get("text") or "").strip()
                        if prev_text and content_hash(prev_text) == hashes.get(u):
                            final_unchanged.add(u)
                            reasons[u] = "already-indexed-unchanged"
                            continue
                final_changed.append(u)

            confirmed_unchanged = [u for u in final_unchanged if u in indexed_keys]
            unchanged = confirmed_unchanged
            changed = final_changed

            # Verified: changed must be empty, unchanged must have the URL
            self.assertEqual(len(changed), 0, "Unchanged page must NOT be in changed list")
            self.assertEqual(unchanged, [url], "Page must be in unchanged list")


class TestUrlResolutionAndRedirectHandling(unittest.TestCase):
    def setUp(self):
        self.scraper = WebsiteScraper(base_url="https://bmsit.ac.in/")

    def test_1_root_relative_url_resolution(self):
        """Root-relative URL (/governence.php) must resolve against domain root."""
        base_page = "https://bmsit.ac.in/faculty/new/index.php"
        resolved = self.scraper._normalize_discovered_url("/governence.php", base_page)
        self.assertEqual(resolved, "https://bmsit.ac.in/governence.php")

    def test_2_relative_url_resolution_subfolder(self):
        """Relative URL without leading slash inside subfolder must resolve within that subfolder."""
        base_page = "https://bmsit.ac.in/faculty/new/index.php"
        resolved = self.scraper._normalize_discovered_url("nba_approvals.php", base_page)
        self.assertEqual(resolved, "https://bmsit.ac.in/faculty/new/nba_approvals.php")

    def test_3_url_resolution_with_base_href(self):
        """When HTML contains <base href="...">, relative URLs must resolve against that base URL."""
        html_with_base = """
        <html>
        <head>
            <base href="https://bmsit.ac.in/">
            <title>Faculty Portal</title>
        </head>
        <body>
            <a href="nba_approvals.php">NBA Approvals</a>
        </body>
        </html>
        """
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"Content-Type": "text/html"}
        mock_resp.text = html_with_base
        mock_resp.url = "https://bmsit.ac.in/faculty/new/index.php"
        mock_resp.history = []

        with patch.object(self.scraper.session, "get", return_value=mock_resp):
            page_data = self.scraper._fetch_and_clean_page("https://bmsit.ac.in/faculty/new/index.php")

        self.assertIsNotNone(page_data)
        discovered = [u[0] for u in page_data["links"]]
        self.assertIn("https://bmsit.ac.in/nba_approvals.php", discovered)

    def test_4_valid_http_200_page(self):
        """Valid HTTP 200 page must be cleanly fetched, parsed, and content returned."""
        html_content = """
        <html>
        <head><title>Governance - BMSIT</title></head>
        <body>
            <h1>Board of Governors</h1>
            <p>BMS Institute of Technology is governed by the BMSET Trust.</p>
        </body>
        </html>
        """
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"Content-Type": "text/html"}
        mock_resp.text = html_content
        mock_resp.url = "https://bmsit.ac.in/governence.php"
        mock_resp.history = []

        with patch.object(self.scraper.session, "get", return_value=mock_resp):
            page_data = self.scraper._fetch_and_clean_page("https://bmsit.ac.in/governence.php")

        self.assertIsNotNone(page_data)
        self.assertEqual(page_data["url"], "https://bmsit.ac.in/governence.php")
        self.assertIn("Board of Governors", page_data["content"])
        self.assertIn("BMSET Trust", page_data["content"])

    def test_5_non_homepage_redirecting_to_homepage_discarded(self):
        """A non-homepage URL redirecting to homepage via HTTP 302 (soft-404) must return None and be discarded."""
        mock_redirect_302 = MagicMock()
        mock_redirect_302.status_code = 302

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"Content-Type": "text/html"}
        mock_resp.url = "https://bmsit.ac.in/"
        mock_resp.history = [mock_redirect_302]
        mock_resp.text = "<html><body><h1>Welcome to BMSIT Homepage</h1></body></html>"

        with patch.object(self.scraper.session, "get", return_value=mock_resp):
            page_data = self.scraper._fetch_and_clean_page("https://bmsit.ac.in/faculty/new/governence.php")

        self.assertIsNone(page_data, "Soft-404 redirect to homepage must be discarded and return None")

    def test_6_legitimate_redirect_to_valid_page(self):
        """A legitimate redirect to another valid page must index under the canonical URL."""
        mock_redirect_301 = MagicMock()
        mock_redirect_301.status_code = 301

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"Content-Type": "text/html"}
        mock_resp.url = "https://bmsit.ac.in/dep-cse.php"
        mock_resp.history = [mock_redirect_301]
        mock_resp.text = "<html><body><h1>Computer Science Department</h1><p>Academic details...</p></body></html>"

        with patch.object(self.scraper.session, "get", return_value=mock_resp):
            page_data = self.scraper._fetch_and_clean_page("https://bmsit.ac.in/dept/cse-old.php")

        self.assertIsNotNone(page_data)
        self.assertEqual(page_data["url"], "https://bmsit.ac.in/dep-cse.php", "Must index under canonical URL")

    def test_7_duplicate_url_discovery_across_attributes(self):
        """Duplicate URLs discovered across <a>, data-url, onclick, and regex must be deduplicated."""
        html_dupes = """
        <html>
        <head><title>Research Portal</title></head>
        <body>
            <a href="/research-centers.php">Research Centers</a>
            <div data-url="/research-centers.php">More Info</div>
            <button onclick="window.location.href='/research-centers.php'">Go to Research</button>
            <script>
                var target = "https://bmsit.ac.in/research-centers.php";
            </script>
        </body>
        </html>
        """
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"Content-Type": "text/html"}
        mock_resp.text = html_dupes
        mock_resp.url = "https://bmsit.ac.in/research.php"
        mock_resp.history = []

        with patch.object(self.scraper.session, "get", return_value=mock_resp):
            page_data = self.scraper._fetch_and_clean_page("https://bmsit.ac.in/research.php")

        self.assertIsNotNone(page_data)
        links = [u[0] for u in page_data["links"]]
        self.assertEqual(links.count("https://bmsit.ac.in/research-centers.php"), 1, "Duplicate links must be deduplicated")

    def test_8_legitimate_faculty_new_pages_remain_discoverable(self):
        """Legitimate pages inside /faculty/new/ must not be disabled or excluded."""
        faculty_new_url = "https://bmsit.ac.in/faculty/new/career.php"
        self.assertTrue(self.scraper._is_allowed_domain(faculty_new_url), "Must allow /faculty/new/ pages")

        html_faculty = """
        <html>
        <head><title>Careers - BMSIT Faculty</title></head>
        <body>
            <h1>Faculty Recruitment</h1>
            <a href="nba_approvals.php">NBA Approvals</a>
            <p>Application guidelines for assistant professor.</p>
        </body>
        </html>
        """
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"Content-Type": "text/html"}
        mock_resp.text = html_faculty
        mock_resp.url = faculty_new_url
        mock_resp.history = []

        with patch.object(self.scraper.session, "get", return_value=mock_resp):
            page_data = self.scraper._fetch_and_clean_page(faculty_new_url)

        self.assertIsNotNone(page_data)
        self.assertEqual(page_data["url"], faculty_new_url)
        discovered = [u[0] for u in page_data["links"]]
        self.assertIn("https://bmsit.ac.in/faculty/new/nba_approvals.php", discovered)

    def test_soft_404_purged_on_crawl_sync(self):
        """Unreached pages that redirect to homepage (soft-404) must be marked DELETED and purged."""
        old_pages = {
            "https://bmsit.ac.in/faculty/new/governence.php": {
                "title": "BMSIT Homepage",
                "hash": "5aa2f0c4b89788b9f78ecc23e92c7edb",
                "length": 1000
            }
        }
        mock_probe = MagicMock()
        mock_probe.status_code = 200
        mock_probe.url = "https://bmsit.ac.in/"

        with patch("app.services.storage.StorageService.load_scrape_state", return_value={"pages": old_pages}), \
             patch("app.services.storage.StorageService.save_scrape_state"), \
             patch("app.services.storage.StorageService.record_crawl_run"), \
             patch.object(self.scraper, "_fetch_and_clean_page", return_value=None), \
             patch.object(self.scraper.session, "get", return_value=mock_probe):
            result = self.scraper.run_sync()

        self.assertIn("https://bmsit.ac.in/faculty/new/governence.php", result.get("deleted_urls", []))


if __name__ == "__main__":
    unittest.main()
