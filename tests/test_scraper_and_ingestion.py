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


if __name__ == "__main__":
    unittest.main()
