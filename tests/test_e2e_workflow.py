import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import os
import unittest
from app.config import Config
from app.services.storage import StorageService
from app.services.rag_service import RAGService
from app.services.gemini_service import GeminiService
from app.services.scheduler_service import execute_bmsit_scrape_and_index, SchedulerService
from app.services.scraper import scraper_status

class TestE2EWorkflow(unittest.TestCase):
    def test_01_scheduler_configuration(self):
        """Test APScheduler configuration for 7:00 AM IST."""
        SchedulerService.start()
        status = SchedulerService.get_status()
        self.assertTrue(status["is_running"])
        self.assertIn("07:00", status["schedule"])
        self.assertIn("Asia/Kolkata", status["schedule"])
        print(f"\n[Test] Scheduler active: {status['schedule']}, Next run: {status['next_run_time']}")

    def test_02_delta_scrape_and_indexing_workflow(self):
        """
        Runs the delta pipeline end to end: crawl, hash, embed only what changed.

        'partial' is a valid success state: it means the embedding quota ran out
        part-way through, everything embedded so far was committed, and the
        remaining pages are deferred to the next run.
        """
        print("\n[Test] Running delta ingestion...")
        result = execute_bmsit_scrape_and_index(is_scheduled=False)

        self.assertIn(result["status"], ("success", "partial"),
                      f"Unexpected status: {result.get('status')} {result.get('message', '')}")
        self.assertGreater(result["indexed_chunks"], 0)
        self.assertFalse(scraper_status.is_running, "Scraper must not be left running")
        self.assertEqual(scraper_status.phase, "completed")

        # Nothing may be lost: every crawled page is either embedded, skipped as
        # unchanged, or explicitly deferred.
        accounted = (result["pages_embedded"] + result["pages_skipped"] + result["pages_deferred"])
        self.assertGreaterEqual(accounted, result["pages_changed"])

        history = StorageService.load_history()
        latest = history[0]
        self.assertIn(latest["status"], ("Indexed", "Partially Indexed"))
        self.assertGreater(latest["chunk_count"], 0)
        print(
            f"[Test] {result['pages_crawled']} crawled, {result['pages_embedded']} embedded, "
            f"{result['pages_skipped']} skipped unchanged, {result['pages_deferred']} deferred, "
            f"{latest['chunk_count']} chunks active."
        )

    def test_02b_second_run_skips_unchanged_pages(self):
        """A run immediately after the previous one must re-embed almost nothing."""
        print("\n[Test] Re-running ingestion to confirm delta skipping...")
        result = execute_bmsit_scrape_and_index(is_scheduled=True)
        self.assertIn(result["status"], ("success", "partial"))
        self.assertGreater(result["pages_skipped"], 0,
                           "Second run should skip pages whose content hash is unchanged")
        print(
            f"[Test] Second run skipped {result['pages_skipped']} unchanged page(s) and "
            f"embedded {result['pages_embedded']}."
        )

    def test_03_rag_retrieval_and_gemini_answers(self):
        """
        Tests that common questions on BMSIT are answered directly from the scraped website data
        and do NOT return the 'unknown / no verified information' message.
        """
        rag = RAGService.get_instance()
        stats = rag.get_stats()
        print(f"\n[Test] Current FAISS vectors: {stats['total_vectors']}, Chunks: {stats['total_chunks']}")
        self.assertGreater(stats["total_vectors"], 0)

        test_questions = [
            ("What courses or branches of engineering are offered at BMSIT?", ["computer", "engineering", "cse", "ise", "electronics"]),
            ("Tell me about BMSIT hostel facilities and contact", ["hostel", "room", "student", "bmsit"]),
            ("What are the placement statistics or placement cell at BMSIT?", ["placement", "training", "recruit", "cell"]),
            ("What is the address and location of BMSIT campus?", ["avalahalli", "yelahanka", "bengaluru", "doddaballapur"])
        ]

        for q, expected_keywords in test_questions:
            print(f"\n--- Testing Query: '{q}' ---")
            resp = GeminiService.generate_chat_response(q)
            reply = resp["reply"]
            sources = resp.get("sources", [])
            print(f"Sources cited ({len(sources)}): {[s.get('name') for s in sources[:3]]}")
            print(f"Reply preview: {reply[:250]}...")

            # Must not be the generic rejection or unknown template
            self.assertNotIn("As of now, I don't have verified information", reply, f"Failed for query '{q}': answered unknown despite data present!")
            self.assertFalse(resp.get("guardrail_triggered", False))

            # Must contain at least one relevant keyword in the answer
            reply_lower = reply.lower()
            matched = any(kw in reply_lower for kw in expected_keywords)
            self.assertTrue(matched, f"Reply did not contain expected keywords {expected_keywords}")

if __name__ == "__main__":
    unittest.main()
