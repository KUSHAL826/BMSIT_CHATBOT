"""
One-time, idempotent upgrade of existing data to the persistent knowledge model.

Runs on every startup and does nothing once the data is already in shape.

What it repairs:
  * Random per-run source ids become stable ids ("web-bmsit", "doc-<hash>"), so
    a re-scrape or re-upload updates the existing source instead of stacking a
    new history row whose chunks the next run would wipe.
  * Multiple website rows from earlier scrapes collapse into the single living
    website source; superseded duplicates are dropped.
  * Sources that have vectors but no entry in the knowledge store get one, so
    incremental updates and per-source deletes work on pre-existing data.
  * Rows left at "Processing" by an interrupted run are labelled honestly.
"""
import logging

from app.config import Config
from app.services.document_parser import DocumentParser
from app.services.fsutil import read_json
from app.services.ingestion_registry import IngestionRegistry, content_hash
from app.services.knowledge_store import KnowledgeStore
from app.services.rag_service import RAGService
from app.services.storage import StorageService

logger = logging.getLogger(__name__)


def _stable_id(entry):
    if entry.get("source_type") == "website":
        return KnowledgeStore.WEBSITE_SOURCE_ID
    return KnowledgeStore.document_source_id(entry.get("source", ""))


def _migrate_history(rag):
    history = StorageService.load_history()
    if not history:
        return

    remap = {}
    superseded = []
    kept = []
    seen_stable = set()
    changed = False

    for entry in history:  # newest first
        stable = _stable_id(entry)
        old_id = entry.get("id")

        if stable in seen_stable:
            # An older row for the same source: its content is superseded.
            if old_id and old_id != stable:
                superseded.append(old_id)
            changed = True
            continue

        seen_stable.add(stable)
        if old_id != stable:
            remap[old_id] = stable
            entry["id"] = stable
            changed = True

        if entry.get("source_type") == "website":
            entry["source"] = KnowledgeStore.WEBSITE_SOURCE_NAME
        kept.append(entry)

    if remap:
        moved = rag.remap_source_ids(remap)
        logger.info("[Migration] Re-pointed %s chunks onto stable source ids.", moved)
    if superseded:
        dropped = rag.drop_sources(superseded)
        logger.info("[Migration] Dropped %s chunks from %s superseded rows.", dropped, len(superseded))

    if changed:
        StorageService.save_history(kept)
        logger.info("[Migration] History normalised to %s stable source(s).", len(kept))


def _backfill_website_store(rag):
    source_id = KnowledgeStore.WEBSITE_SOURCE_ID
    if KnowledgeStore.load_source(source_id) is not None:
        return

    snapshot = read_json(Config.SCRAPED_DIR / "bmsit_web_content.json", default=None)
    items = {}

    if isinstance(snapshot, dict) and snapshot:
        for url, page in snapshot.items():
            text = (page or {}).get("content", "")
            if not text:
                continue
            title = (page or {}).get("title") or url
            items[url] = {
                "title": title,
                "text": text,
                "metadata": {"url": url, "title": title, "type": "web"},
            }
    else:
        items = rag.grouped_source_text(source_id)

    if not items:
        return

    KnowledgeStore.upsert_items(
        source_id, "website", KnowledgeStore.WEBSITE_SOURCE_NAME, items, replace_missing=False
    )
    logger.info("[Migration] Imported %s website page(s) into the knowledge store.", len(items))


def _backfill_document_store(rag, entry):
    source_id = entry.get("id")
    if not source_id or KnowledgeStore.load_source(source_id) is not None:
        return

    filename = entry.get("source", "")
    file_path = Config.UPLOADS_DIR / filename
    items = {}

    if file_path.exists():
        try:
            for idx, sec in enumerate(DocumentParser.parse_file(file_path), start=1):
                meta = sec.get("metadata", {}) or {}
                items[f"section-{idx}"] = {
                    "title": meta.get("section_title") or f"Section {idx}",
                    "text": sec.get("content", ""),
                    "metadata": meta,
                }
        except Exception as e:
            logger.warning("[Migration] Could not re-parse %s: %s", filename, e)

    if not items:
        items = rag.grouped_source_text(source_id)
    if not items:
        return

    KnowledgeStore.upsert_items(source_id, "document", filename, items, replace_missing=False)
    logger.info("[Migration] Imported document '%s' into the knowledge store.", filename)


def _backfill_registry(rag):
    """
    Rebuilds registry entries for content that was indexed before the registry
    existed, so the next daily run can do proper delta detection instead of
    re-embedding everything.
    """
    for record in KnowledgeStore.list_sources():
        source_id = record.get("source_id")
        source_type = record.get("source_type", "website")
        source_name = record.get("source_name", source_id)
        items = record.get("items", {})
        if not items:
            continue

        known = IngestionRegistry.known_keys(source_id)
        missing = [key for key in items if key not in known]
        if not missing:
            continue

        chunks_by_key = {}
        for chunk in rag.get_source_chunks(source_id):
            chunks_by_key.setdefault(chunk.get("item_key"), []).append(chunk)

        entries = []
        for key in missing:
            chunks = chunks_by_key.get(key, [])
            if not chunks:
                continue  # no vectors yet: the next run will embed it
            entries.append({
                "item_key": key,
                "content_hash": content_hash(items[key].get("text", "")),
                "chunk_ids": [c["chunk_id"] for c in chunks],
                "token_count": sum(int(c.get("tokens", 0)) for c in chunks),
                "embed_provider": chunks[0].get("embed_provider", "legacy:unknown"),
                "title": items[key].get("title"),
            })

        if entries:
            IngestionRegistry.record_many(source_id, entries, source_type, source_name)
            logger.info(
                "[Migration] Registered %s existing item(s) for %s.", len(entries), source_id
            )


def run_migrations():
    """Entry point called during app startup."""
    Config.ensure_directories()
    rag = RAGService.get_instance()

    try:
        _migrate_history(rag)
    except Exception as e:
        logger.exception("[Migration] History migration failed: %s", e)

    try:
        for entry in StorageService.load_history():
            if entry.get("source_type") == "website":
                _backfill_website_store(rag)
            else:
                _backfill_document_store(rag, entry)
    except Exception as e:
        logger.exception("[Migration] Knowledge store backfill failed: %s", e)

    try:
        _backfill_registry(rag)
    except Exception as e:
        logger.exception("[Migration] Registry backfill failed: %s", e)

    # Bring any locally-embedded vectors into the active embedding space.
    try:
        pending = rag.pending_reembed_count()
        if pending:
            logger.info("[Migration] %s vector(s) need re-embedding; upgrading in background.", pending)
            rag.reembed_pending_async()
    except Exception as e:
        logger.warning("[Migration] Could not start re-embedding: %s", e)
