"""
Rebuild the knowledge base from scratch, from the command line.

    .venv\\Scripts\\python.exe rebuild.py            full reset, then crawl and embed
    .venv\\Scripts\\python.exe rebuild.py --keep     keep existing data, embed only changes
    .venv\\Scripts\\python.exe rebuild.py --status   show current state and exit

Run it as many times as needed. If embedding quota runs out mid-way the run
reports "partial": everything embedded so far is kept and the remaining pages
resume on the next run.
"""
import logging
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from app.config import Config  # noqa: E402

logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)

from app.services.ingestion_pipeline import (  # noqa: E402
    reset_everything, run_website_delta_ingestion,
)
from app.services.ingestion_registry import IngestionRegistry  # noqa: E402
from app.services.knowledge_store import KnowledgeStore  # noqa: E402
from app.services.rag_service import RAGService  # noqa: E402


def show_status():
    rag = RAGService.get_instance()
    stats = rag.get_stats()
    embeddings = stats["embeddings"]

    print("\n--- knowledge base ---")
    print(f"  chunks indexed      : {stats['total_chunks']}")
    print(f"  vectors             : {stats['total_vectors']}")
    print(f"  tokens embedded     : {stats['total_tokens']}")
    print(f"  sources             : {stats['indexed_sources_count']}")
    print(f"  embedding spaces    : {stats['embedding_spaces']}")
    print(f"  awaiting re-embed   : {stats['needs_reembed']}")

    print("\n--- embedding backend ---")
    print(f"  active provider     : {embeddings['active_provider']}")
    print(f"  keys configured     : {embeddings['keys_configured']}")
    print(f"  keys usable now     : {embeddings['keys_usable']}")
    for key in embeddings["keys"]:
        state = "usable" if key["usable"] else f"cooling down {key['cooldown_seconds']}s"
        print(f"    {key['key']}: {state}, {key['requests']} request(s), "
              f"{key['quota_hits']} quota hit(s)")

    print("\n--- delta registry ---")
    registry = IngestionRegistry.stats()
    print(f"  tracked items       : {registry['tracked_items']}")
    print(f"  tracked chunks      : {registry['tracked_chunks']}")
    print(f"  failed items        : {registry['failed_items']}")
    print(f"  last embedded       : {registry['last_embedded']}")
    print(f"  stored pages        : "
          f"{len(KnowledgeStore.get_items(KnowledgeStore.WEBSITE_SOURCE_ID))}")
    print()


def main():
    args = set(sys.argv[1:])

    if "--status" in args:
        show_status()
        return 0

    if "--keep" not in args:
        print("Resetting all indexed data (uploaded files are kept)...")
        summary = reset_everything(remove_uploads=False)
        print(f"  removed {summary['removed_chunks']} vector(s) and "
              f"{summary['removed_sources']} stored source(s)\n")

    print(f"Crawling {Config.BMSIT_DEFAULT_URL} "
          f"(max {Config.SCRAPE_MAX_PAGES} pages, depth {Config.SCRAPE_DEPTH})")
    print(f"Chunking at {Config.CHUNK_TARGET_TOKENS}-{Config.CHUNK_MAX_TOKENS} tokens, "
          f"embedding with {Config.EMBED_MODEL}\n")

    result = run_website_delta_ingestion(is_scheduled=False)

    print("\n--- run result ---")
    for field in ("status", "pages_crawled", "pages_changed", "pages_embedded",
                  "pages_skipped", "pages_deferred", "pages_no_text",
                  "new_chunks", "replaced_chunks", "indexed_chunks"):
        if field in result:
            print(f"  {field:<18}: {result[field]}")
    if result.get("quota_note"):
        print(f"  note              : {result['quota_note']}")

    show_status()

    if result.get("status") == "partial":
        print("Quota ran out part-way. Everything embedded so far is saved.")
        print("Add more keys to GEMINI_API_KEYS, or run this again later:")
        print("  .venv\\Scripts\\python.exe rebuild.py --keep\n")
        return 2
    if result.get("status") != "success":
        print(f"Run failed: {result.get('message')}\n")
        return 1

    print("Done. Start the server with: .venv\\Scripts\\python.exe run.py\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
