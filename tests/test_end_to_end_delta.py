import unittest
import tempfile
import shutil
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch

from app.config import Config
from app.services.knowledge_store import KnowledgeStore
from app.services.ingestion_registry import IngestionRegistry, content_hash
from app.services.rag_service import RAGService
from app.services.ingestion_pipeline import run_website_delta_ingestion
from app.services.storage import StorageService


class TestDeltaIngestionEndToEnd(unittest.TestCase):
    def setUp(self):
        self.source_id = KnowledgeStore.WEBSITE_SOURCE_ID
        self.rag = RAGService.get_instance()

    def test_deduplication_lifecycle(self):
        """
        Verify:
        Run 1: New page -> Chunked & embedded, recorded in registry.
        Run 2: Identical page scraped -> Skipped, 0 chunks re-embedded.
        Run 3: Modified page content -> Re-chunked & re-embedded.
        """
        test_url = "https://bmsit.ac.in/test-unique-dept.php"
        original_text = "The Department of Artificial Intelligence and Machine Learning at BMSIT offers top tier education."
        modified_text = "The Department of Artificial Intelligence and Machine Learning at BMSIT updated its 2026 syllabus."

        # Setup mock scraper returning original page
        mock_crawl_1 = {
            "status": "success",
            "scraped_pages": {
                test_url: {
                    "url": test_url,
                    "title": "BMSIT AI&ML Department",
                    "content": original_text,
                    "hash": content_hash(original_text),
                    "file_type": "web",
                    "is_document": False,
                }
            },
            "changes": [{"url": test_url, "change_type": "NEW"}],
            "deleted_urls": []
        }

        dummy_embed_return = (np.ones((1, 768), dtype=np.float32), ["gemini:gemini-embedding-001"], 1)

        # Run 1: First ingestion (New page)
        with patch("app.services.scraper.WebsiteScraper.run_sync", return_value=mock_crawl_1), \
             patch.object(self.rag._embedder, "embed_documents", return_value=dummy_embed_return):
            res1 = run_website_delta_ingestion(trigger_label="Test Run 1")

        self.assertIn(res1["status"], ("success", "partial"))
        self.assertGreater(res1.get("pages_embedded", 0), 0, "First run must embed the new page")

        # Verify page is in knowledge store and registry
        items = KnowledgeStore.get_items(self.source_id)
        self.assertIn(test_url, items)
        reg_item = IngestionRegistry.get_item(self.source_id, test_url)
        self.assertIsNotNone(reg_item)
        self.assertEqual(reg_item["content_hash"], content_hash(original_text))

        # Run 2: Second ingestion with IDENTICAL text (Must skip embedding!)
        mock_crawl_2 = {
            "status": "success",
            "scraped_pages": {
                test_url: {
                    "url": test_url,
                    "title": "BMSIT AI&ML Department",
                    "content": original_text,
                    "hash": content_hash(original_text),
                    "file_type": "web",
                    "is_document": False,
                }
            },
            "changes": [],
            "deleted_urls": []
        }

        with patch("app.services.scraper.WebsiteScraper.run_sync", return_value=mock_crawl_2), \
             patch.object(self.rag._embedder, "embed_documents") as mock_embed:
            res2 = run_website_delta_ingestion(trigger_label="Test Run 2")

        # Embedder must NOT be called on unchanged content
        mock_embed.assert_not_called()
        self.assertEqual(res2.get("pages_embedded", 0), 0, "Second run must skip embedding for unchanged page")
        self.assertGreaterEqual(res2.get("pages_skipped", 0), 1, "Page must be counted as skipped")

        # Run 3: Third ingestion with MODIFIED content (Must re-embed!)
        mock_crawl_3 = {
            "status": "success",
            "scraped_pages": {
                test_url: {
                    "url": test_url,
                    "title": "BMSIT AI&ML Department",
                    "content": modified_text,
                    "hash": content_hash(modified_text),
                    "file_type": "web",
                    "is_document": False,
                }
            },
            "changes": [{"url": test_url, "change_type": "MODIFIED"}],
            "deleted_urls": []
        }

        with patch("app.services.scraper.WebsiteScraper.run_sync", return_value=mock_crawl_3), \
             patch.object(self.rag._embedder, "embed_documents", return_value=dummy_embed_return):
            res3 = run_website_delta_ingestion(trigger_label="Test Run 3")

        self.assertIn(res3["status"], ("success", "partial"))
        self.assertGreater(res3.get("pages_embedded", 0), 0, "Modified page must be embedded")
        reg_item_3 = IngestionRegistry.get_item(self.source_id, test_url)
        self.assertEqual(reg_item_3["content_hash"], content_hash(modified_text))

        # Cleanup test item
        KnowledgeStore.delete_items(self.source_id, [test_url])
        IngestionRegistry.forget_items(self.source_id, [test_url])
        self.rag.apply_source_update(self.source_id, [], replace_keys={test_url})


if __name__ == "__main__":
    unittest.main()
