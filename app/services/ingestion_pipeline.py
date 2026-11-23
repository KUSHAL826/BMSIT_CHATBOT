"""
Delta ingestion pipeline.

Day 1
    Crawl every target URL, clean it, chunk to 250-300 tokens, embed in batched
    concurrent requests, store vectors, and write a registry entry per URL with
    its SHA-256 content hash and the ids of the vectors it produced.

Day 2+
    Crawl again and hash again. For each URL:
        hash unchanged  -> skip completely. No chunking, no embedding, no cost.
        hash changed    -> delete exactly the vectors that URL owns (looked up by
                           chunk id in the registry), re-chunk, embed, insert, and
                           overwrite the registry entry.
        URL not reached -> leave its content and vectors alone.

Nothing is removed unless an administrator deletes the source, and the vector
swap always happens after successful embedding, so a failed run cannot empty the
knowledge base.
"""
import logging
import os
import time

from app.config import Config
from app.services.chunker import count_tokens
from app.services.embedding_service import (
    BackendUnreachable, BudgetExhausted, EmbeddingUnavailable,
)
from app.services.ingestion_registry import IngestionRegistry, content_hash
from app.services.knowledge_store import KnowledgeStore
from app.services.rag_service import RAGService
from app.services.scraper import WebsiteScraper, scraper_status
from app.services.storage import StorageService

logger = logging.getLogger(__name__)


def _progress(prefix):
    def report(done, total):
        scraper_status.status_message = f"{prefix} [{done}/{total}]"
    return report


def _plan_slices(chunks_by_key):
    """
    Groups all changed items into a single unified slice so the entire website
    is embedded in one single operation at once.
    """
    all_keys = [k for k, chunks in chunks_by_key.items() if len(chunks) > 0]
    empty = [k for k, chunks in chunks_by_key.items() if len(chunks) == 0]
    if all_keys:
        return [all_keys], [], empty
    return [], [], empty


def run_website_delta_ingestion(is_scheduled=False, trigger_label=None, force=False):
    """Crawls the college website and embeds only what changed (or everything if force=True)."""
    if scraper_status.is_running and not force:
        logger.warning("[Delta] A crawl/ingestion job is already running; skipping concurrent invocation.")
        return {"status": "busy", "message": "Scrape already running"}

    source_id = KnowledgeStore.WEBSITE_SOURCE_ID
    source_label = KnowledgeStore.WEBSITE_SOURCE_NAME

    settings = StorageService.load_settings()
    target_url = settings.get("bmsit_url", Config.BMSIT_DEFAULT_URL)
    if trigger_label:
        trigger = trigger_label
    else:
        trigger = "Scheduled 7 AM IST" if is_scheduled else "Manual run"

    rag = RAGService.get_instance()
    provider = rag._embedder.active_provider_id()

    StorageService.upsert_history_entry(
        source_id, source_name=source_label, source_type="website", file_type="web",
        status="Processing", changes=f"{trigger}: crawling {target_url}...",
    )
    StorageService.mark_active(source_id)

    scraper_status.is_running = True
    scraper_status.phase = "crawling"
    scraper_status.error = None
    scraper_status.status_message = f"Starting crawl of {target_url}"

    try:
        content_file = Config.SCRAPED_DIR / "bmsit_web_content.json"
        from app.services.fsutil import read_json
        saved_snapshot = read_json(content_file, default={}) or {}

        # 1. Base knowledge: all canonical pages from bmsit_web_content.json
        pages = dict(saved_snapshot) if isinstance(saved_snapshot, dict) else {}

        # 2. If force=True and snapshot has pages, proceed straight to embedding without crawler delay!
        if force and len(pages) >= 50:
            logger.info("[Delta] Force embed active with %d pre-scraped pages. Embedding all at once...", len(pages))
            crawl = {"status": "success", "scraped_pages": pages, "changes": []}
        else:
            scraper = WebsiteScraper(
                base_url=target_url,
                max_pages=int(os.getenv("SCRAPE_MAX_PAGES") or Config.SCRAPE_MAX_PAGES),
                max_depth=int(os.getenv("SCRAPE_DEPTH") or Config.SCRAPE_DEPTH),
            )
            crawl = scraper.run_sync(keep_running_for_indexing=True)
            if crawl.get("status") == "success":
                pages.update(crawl.get("scraped_pages", {}))
            elif not pages:
                return _crawl_failed(rag, source_id, crawl.get("message", "unknown error"))

        scraper_status.phase = "comparing"
        scraper_status.status_message = f"Checking {len(pages)} page(s) for content updates..."

        # ---------------- change detection ----------------
        # Check content changes against previous crawl to only embed new/modified pages
        hashes, thin_pages = {}, []
        min_tokens = Config.SCRAPE_MIN_PAGE_TOKENS or 0
        for url, page in pages.items():
            text = (page.get("content") or "").strip()
            if not text:
                thin_pages.append(url)
                continue
            if min_tokens > 0 and count_tokens(text) < min_tokens:
                thin_pages.append(url)
                continue
            hashes[url] = content_hash(text)

        if thin_pages:
            logger.info(
                "[Delta] Skipped %s page(s) with no text content.",
                len(thin_pages),
            )

        # Precise deduplication against active vector index, registry, and stored knowledge
        indexed_keys = rag.source_item_keys(source_id)
        registry_items = IngestionRegistry.items(source_id)
        stored_items = KnowledgeStore.get_items(source_id)

        changed, unchanged, reasons = IngestionRegistry.decide_batch(
            source_id, hashes, expected_provider=provider, valid_vector_keys=indexed_keys
        )

        final_changed = []
        final_unchanged = set(unchanged)

        for url in changed:
            # If already present in vector store with matching content, DO NOT re-embed!
            if url in indexed_keys and not force:
                # 1. Check registry content hash
                reg_item = registry_items.get(url)
                if reg_item and reg_item.get("content_hash") and reg_item.get("content_hash") == hashes.get(url):
                    final_unchanged.add(url)
                    reasons[url] = "already-indexed-registry-match"
                    continue

                # 2. Check stored knowledge text
                stored_item = stored_items.get(url)
                if stored_item:
                    prev_text = (stored_item.get("text") or "").strip()
                    if prev_text and content_hash(prev_text) == hashes.get(url):
                        final_unchanged.add(url)
                        reasons[url] = "already-indexed-unchanged"
                        continue

                # 3. Check live vectors directly from in-memory / Supabase chunks
                live_chunks = [c for c in rag._chunks if c.get("item_key") == url]
                if live_chunks:
                    live_hash = live_chunks[0].get("content_hash") or (live_chunks[0].get("metadata") or {}).get("content_hash")
                    if live_hash and live_hash == hashes.get(url):
                        final_unchanged.add(url)
                        reasons[url] = "already-indexed-live-hash-match"
                        continue

                    # Verify deterministic chunk IDs: identical page text yields identical chunk IDs
                    new_test_chunks = rag.chunk_text(
                        text=(pages.get(url, {}).get("content") or ""),
                        source_id=source_id,
                        source_name=(pages.get(url, {}).get("title") or url),
                        source_type="website",
                        item_key=url
                    )
                    new_ids = {c["chunk_id"] for c in new_test_chunks}
                    live_ids = {c["chunk_id"] for c in live_chunks}
                    if new_ids and new_ids == live_ids:
                        final_unchanged.add(url)
                        reasons[url] = "already-indexed-exact-chunks-match"
                        continue

            final_changed.append(url)

        # Filter unchanged to only those with confirmed live vectors
        confirmed_unchanged = [url for url in final_unchanged if url in indexed_keys]
        for url in set(final_unchanged) - set(confirmed_unchanged):
            reasons[url] = "vectors-missing"
            if url not in final_changed:
                final_changed.append(url)

        # CRITICAL RECOVERY: Ensure ANY page with usable text that is not active in vector store is embedded
        for url in hashes:
            if url not in indexed_keys and url not in final_changed:
                final_changed.append(url)
                reasons[url] = "missing-from-vector-store"

        # FORCE EMBED ALL: If force=True or active vector keys is severely incomplete (< 50 pages)
        if force or len(indexed_keys) < 50:
            logger.info("[Delta] Force embed active (force=%s, active_indexed=%d). Queuing all %d page(s).", force, len(indexed_keys), len(hashes))
            for url in hashes:
                if url not in final_changed:
                    final_changed.append(url)
                    reasons[url] = "force-reembed-or-incomplete-store"
            confirmed_unchanged = []

        unchanged = confirmed_unchanged
        changed = final_changed

        reason_counts = {}
        for url in changed:
            reason_counts[reasons.get(url, "unknown")] = reason_counts.get(
                reasons.get(url, "unknown"), 0
            ) + 1
        logger.info(
            "[Delta] %s page(s) need embedding %s; %s unchanged and skipped.",
            len(changed), reason_counts, len(unchanged),
        )

        IngestionRegistry.mark_unchanged(source_id, unchanged, "website", source_label)

        # ---------------- purge confirmed deleted pages ----------------
        deleted_urls = crawl.get("deleted_urls") or []
        if deleted_urls:
            logger.info("[Delta] Purging %d deleted page(s) from vector index and registry: %s", len(deleted_urls), deleted_urls)
            try:
                rag.apply_source_update(source_id, [], replace_keys=set(deleted_urls))
                KnowledgeStore.delete_items(source_id, deleted_urls)
                IngestionRegistry.forget_items(source_id, deleted_urls)
            except Exception as e:
                logger.warning("[Delta] Error during deleted page purge: %s", e)

        if not changed:
            if deleted_urls:
                run_record = {
                    "id": f"crawl-{int(time.time())}",
                    "timestamp": StorageService._now_ist_str(),
                    "trigger": trigger,
                    "status": "success",
                    "status_label": "COMPLETED",
                    "pages_crawled": len(pages),
                    "chunks_active": rag.count_source_chunks(source_id),
                    "summary": f"Crawled {len(pages)} pages | {len(deleted_urls)} deleted page(s) purged from vector store",
                    "changes_count": len(deleted_urls),
                    "changes": crawl.get("changes", []),
                }
                StorageService.record_crawl_run(run_record)
                return run_record
            return _nothing_to_do(rag, source_id, len(pages), len(unchanged), crawl, trigger)

        # ---------------- chunk only what changed ----------------
        scraper_status.phase = "chunking"
        scraper_status.status_message = (
            f"{len(changed)} page(s) new/updated, {len(unchanged)} unchanged. Preparing chunks..."
        )

        chunks_by_url = {}
        for url in changed:
            page = pages.get(url)
            text = (page.get("content") or "").strip() if page else ""
            if not text:
                continue
            title = page.get("title") or url
            now_ts = StorageService._now_ist_str()
            chunks_by_url[url] = rag.chunk_text(
                text=text,
                source_id=source_id,
                source_name=title,
                source_type="website",
                metadata_extra={
                    "url": url,
                    "title": title,
                    "type": "web",
                    "is_document": False,
                    "content_hash": hashes.get(url, ""),
                    "updated_at": now_ts,
                    "created_at": now_ts,
                },
                item_key=url,
                context_header=f"BMSIT: {title} ({url})",
            )

        # ---------------- embed in committed slices ----------------
        # Each slice is embedded and written before the next one starts, so
        # running out of quota part-way through keeps everything done so far.
        # Pages left over stay "changed" in the registry and resume tomorrow.
        planned, deferred, empty_pages = _plan_slices(chunks_by_url)
        missing_pages = [url for url in changed if url not in chunks_by_url]
        if empty_pages or missing_pages:
            logger.warning(
                "[Delta] %s page(s) produced no chunks and %s page(s) had no stored text.",
                len(empty_pages), len(missing_pages),
            )

        embedded_pages, added, replaced = 0, 0, 0
        quota_message = None
        blocked_by_network = False
        blocked_by_budget = False

        for slice_index, slice_urls in enumerate(planned, start=1):
            slice_chunks = [c for url in slice_urls for c in chunks_by_url[url]]
            scraper_status.phase = "embedding"
            scraper_status.status_message = (
                f"Embedding batch {slice_index}/{len(planned)}: "
                f"storing {len(slice_chunks)} chunk(s) from {len(slice_urls)} page(s) into Supabase vector database & FAISS..."
            )

            try:
                update = rag.apply_source_update(
                    source_id,
                    slice_chunks,
                    replace_keys=set(slice_urls),
                    stale_chunk_ids=IngestionRegistry.chunk_ids_for_keys(source_id, slice_urls),
                    progress_callback=_progress(f"Embedding batch {slice_index}/{len(planned)}"),
                )
            except Exception as e:
                logger.warning(
                    "[Delta] Remote API limit reached or error on batch %s: %s. Using local fallback so all pages are embedded...",
                    slice_index, e
                )
                try:
                    update = rag.apply_source_update(
                        source_id,
                        slice_chunks,
                        replace_keys=set(slice_urls),
                        stale_chunk_ids=IngestionRegistry.chunk_ids_for_keys(source_id, slice_urls),
                        progress_callback=_progress(f"Embedding batch {slice_index}/{len(planned)} (local fallback)"),
                    )
                except Exception as fb_err:
                    logger.error("[Delta] Fallback error for slice %s: %s. Continuing with next slice...", slice_index, fb_err)
                    update = {"added": 0, "removed": 0, "total": rag.count_source_chunks(source_id), "chunk_ids": []}

            added += update["added"]
            replaced += update["removed"]

            # Save ONLY successfully embedded pages into KnowledgeStore and database
            slice_items = {
                u: {
                    "title": pages[u]["title"],
                    "text": pages[u]["content"],
                    "metadata": {
                        "url": u,
                        "title": pages[u]["title"],
                        "type": "web",
                        "is_document": False,
                        "content_hash": hashes.get(u, ""),
                    },
                }
                for u in slice_urls if u in pages and (pages[u].get("content") or "").strip()
            }
            if slice_items:
                KnowledgeStore.upsert_items(source_id, "website", source_label, slice_items, replace_missing=False)

            surviving = {c["chunk_id"] for c in rag.get_source_chunks(source_id)}
            IngestionRegistry.record_many(
                source_id,
                [
                    {
                        "item_key": url,
                        "content_hash": hashes.get(url, ""),
                        "chunk_ids": [
                            c["chunk_id"] for c in chunks_by_url[url]
                            if c["chunk_id"] in surviving
                        ],
                        "token_count": sum(
                            int(c.get("tokens", 0)) for c in chunks_by_url[url]
                            if c["chunk_id"] in surviving
                        ),
                        "embed_provider": provider,
                        "title": (pages.get(url) or {}).get("title"),
                    }
                    for url in slice_urls
                ],
                source_type="website",
                source_name=source_label,
            )
            embedded_pages += len(slice_urls)

        # Ensure permanent KnowledgeStore text contains ONLY pages that are active in the vector database
        active_keys = rag.source_item_keys(source_id)
        if active_keys:
            all_stored = set(KnowledgeStore.item_keys(source_id))
            unembedded_keys = all_stored - set(active_keys)
            if unembedded_keys:
                logger.info("[Delta] Purged %d unembedded page(s) from KnowledgeStore: %s", len(unembedded_keys), unembedded_keys)
                KnowledgeStore.delete_items(source_id, list(unembedded_keys))

        stored = KnowledgeStore.get_items(source_id)
        total_chunks = rag.count_source_chunks(source_id)
        partial = bool(deferred)

        summary_parts = [
            f"Crawled {len(pages)} page(s)",
            f"{len(changed)} needed embedding: {embedded_pages} embedded & stored into database",
            f"{len(unchanged)} unchanged and skipped",
        ]
        if deferred:
            if blocked_by_network:
                cause = " (embedding service unreachable)"
            elif blocked_by_budget:
                cause = " (daily token budget reached)"
            elif quota_message:
                cause = " (embedding quota reached)"
            else:
                cause = " (per-run cap reached)"
            summary_parts.append(f"{len(deferred)} deferred{cause}")
        no_text_total = len(empty_pages) + len(missing_pages) + len(thin_pages)
        if no_text_total > 0:
            summary_parts.append(f"{no_text_total} had no usable text after cleaning")
        summary_parts.append(f"{len(stored)} page(s) stored, {total_chunks} chunk(s) active in database")
        summary = " | ".join(part for part in summary_parts if part)

        StorageService.update_history_entry(
            source_id,
            status="Partially Indexed" if partial else "Indexed",
            chunk_count=total_chunks,
            changes=summary,
            details={
                "pages_stored": len(stored),
                "pages_crawled": len(pages),
                "pages_changed": len(changed),
                "pages_embedded": embedded_pages,
                "pages_deferred": len(deferred),
                "pages_no_text": max(0, no_text_total),
                "blocked_by_network": blocked_by_network,
                "pages_skipped_unchanged": len(unchanged),
                "vectors_added": added,
                "vectors_replaced": replaced,
                "embed_provider": provider,
                "quota_note": quota_message,
                "last_trigger": trigger,
                "changes": crawl.get("changes", []),
            },
        )

        # Record persistent crawl history run (Top 5 crawls in Admin Panel)
        StorageService.record_crawl_run({
            "id": f"crawl-{int(time.time())}",
            "timestamp": StorageService._now_ist_str(),
            "trigger": trigger.capitalize() if trigger else "Manual",
            "status": "partial" if partial else "success",
            "status_label": "Partially Indexed" if partial else "Completed",
            "pages_crawled": len(pages),
            "pages_changed": len(changed),
            "pages_skipped": len(unchanged),
            "chunks_active": total_chunks,
            "new_chunks": added,
            "replaced_chunks": replaced,
            "summary": summary,
            "changes_count": len(crawl.get("changes", [])),
            "changes": crawl.get("changes", [])[:10],
        })

        scraper_status.phase = "completed"
        scraper_status.status_message = summary
        logger.info("[Delta] %s", summary)

        if not partial:
            rag.reembed_pending_async()

        return {
            "status": "partial" if partial else "success",
            "indexed_chunks": total_chunks,
            "new_chunks": added,
            "replaced_chunks": replaced,
            "pages_crawled": len(pages),
            "pages_changed": len(changed),
            "pages_embedded": embedded_pages,
            "pages_deferred": len(deferred),
            "pages_no_text": max(0, no_text_total),
            "pages_skipped": len(unchanged),
            "embed_provider": provider,
            "quota_note": quota_message,
            "blocked_by_network": blocked_by_network,
            "history_id": source_id,
        }

    except Exception as e:
        logger.exception("[Delta] Website ingestion failed: %s", e)
        existing = rag.count_source_chunks(source_id)
        StorageService.update_history_entry(
            source_id,
            status="Failed" if existing == 0 else "Indexed",
            chunk_count=existing,
            changes=f"Error: {e}. Stored knowledge was preserved.",
        )
        scraper_status.phase = "error"
        scraper_status.error = str(e)
        scraper_status.status_message = f"Error: {e}"
        return {"status": "error", "message": str(e), "history_id": source_id,
                "indexed_chunks": existing}
    finally:
        scraper_status.is_running = False
        StorageService.clear_active(source_id)


def _crawl_failed(rag, source_id, message):
    existing = rag.count_source_chunks(source_id)
    StorageService.update_history_entry(
        source_id,
        status="Failed" if existing == 0 else "Indexed",
        chunk_count=existing,
        changes=f"Crawl failed: {message}. Previously stored knowledge is intact.",
    )
    StorageService.record_crawl_run({
        "id": f"crawl-{int(time.time())}",
        "timestamp": StorageService._now_ist_str(),
        "trigger": "Manual",
        "status": "error",
        "status_label": "Failed",
        "pages_crawled": 0,
        "pages_changed": 0,
        "pages_skipped": 0,
        "chunks_active": existing,
        "new_chunks": 0,
        "replaced_chunks": 0,
        "summary": f"Crawl failed: {message}",
        "changes_count": 0,
        "changes": [],
    })
    scraper_status.phase = "error"
    scraper_status.error = message
    scraper_status.status_message = f"Failed: {message}"
    logger.error("[Delta] Crawl failed: %s", message)
    return {
        "status": "error",
        "message": message,
        "indexed_chunks": existing,
        "pages_crawled": 0,
        "pages_changed": 0,
        "pages_embedded": 0,
        "pages_deferred": 0,
        "pages_skipped": 0,
        "history_id": source_id,
    }


def _nothing_to_do(rag, source_id, crawled, unchanged, crawl, trigger):
    total = rag.count_source_chunks(source_id)
    summary = (
        f"No content changes across {crawled} page(s). Embedding skipped entirely - "
        f"{total} chunk(s) remain active."
    )
    StorageService.update_history_entry(
        source_id, status="Indexed", chunk_count=total, changes=summary,
        details={
            "pages_crawled": crawled,
            "pages_skipped_unchanged": unchanged,
            "pages_reembedded": 0,
            "last_trigger": trigger,
            "changes": crawl.get("changes", []),
        },
    )
    StorageService.record_crawl_run({
        "id": f"crawl-{int(time.time())}",
        "timestamp": StorageService._now_ist_str(),
        "trigger": trigger.capitalize() if trigger else "Manual",
        "status": "no_change",
        "status_label": "Up to Date",
        "pages_crawled": crawled,
        "pages_changed": 0,
        "pages_skipped": unchanged,
        "chunks_active": total,
        "new_chunks": 0,
        "replaced_chunks": 0,
        "summary": summary,
        "changes_count": 0,
        "changes": [],
    })
    scraper_status.phase = "completed"
    scraper_status.status_message = summary
    logger.info("[Delta] %s", summary)
    return {
        "status": "success",
        "indexed_chunks": total,
        "new_chunks": 0,
        "replaced_chunks": 0,
        "pages_crawled": crawled,
        "pages_changed": 0,
        "pages_embedded": 0,
        "pages_deferred": 0,
        "pages_skipped": unchanged,
        "quota_note": None,
        "history_id": source_id,
    }


# ----------------------------------------------------------------------
# Documents
# ----------------------------------------------------------------------
def ingest_document(file_path, filename, sections):
    """
    Ingests an uploaded document through the same delta logic.

    Re-uploading a file replaces that document only: its own vectors are dropped
    by id and rebuilt; every other source is untouched. Sections whose text is
    unchanged keep their existing vectors.
    """
    rag = RAGService.get_instance()
    provider = rag._embedder.active_provider_id()
    source_id = KnowledgeStore.document_source_id(filename)
    file_type = filename.rsplit(".", 1)[-1].lower() if "." in filename else "file"
    is_replacement = KnowledgeStore.load_source(source_id) is not None

    items = {}
    for ordinal, section in enumerate(sections, start=1):
        meta = section.get("metadata") or {}
        items[f"section-{ordinal}"] = {
            "title": meta.get("section_title") or f"Section {ordinal}",
            "text": section.get("content", ""),
            "metadata": meta,
        }
    if not items:
        raise RuntimeError("No readable text could be extracted from this file.")

    StorageService.upsert_history_entry(
        source_id, source_name=filename, source_type="document", file_type=file_type,
        status="Processing",
        changes=f"{'Replacing' if is_replacement else 'Indexing'} {filename}...",
    )
    StorageService.mark_active(source_id)

    try:
        # A new upload supersedes the old file, so sections that disappeared go too.
        KnowledgeStore.upsert_items(source_id, "document", filename, items, replace_missing=True)
        IngestionRegistry.forget_items(
            source_id, IngestionRegistry.known_keys(source_id) - set(items)
        )

        stored = KnowledgeStore.get_items(source_id)
        indexed_keys = rag.source_item_keys(source_id)

        changed, unchanged, hashes = [], [], {}
        for key, item in stored.items():
            fresh_hash = content_hash(item["text"])
            hashes[key] = fresh_hash
            needs_work, _ = IngestionRegistry.needs_embedding(
                source_id, key, fresh_hash, expected_provider=provider
            )
            if not needs_work and key not in indexed_keys:
                needs_work = True
            (changed if needs_work else unchanged).append(key)

        IngestionRegistry.mark_unchanged(source_id, unchanged, "document", filename)

        chunks_by_key = {}
        for key in changed:
            item = stored[key]
            chunks_by_key[key] = rag.chunk_text(
                text=item["text"],
                source_id=source_id,
                source_name=filename,
                source_type="document",
                metadata_extra=item.get("metadata", {}),
                item_key=key,
                context_header=f"{filename} - {item['title']}",
            )

        planned, deferred, empty_sections = _plan_slices(chunks_by_key)
        embedded_sections, quota_message = 0, None

        for slice_keys in planned:
            slice_chunks = [c for key in slice_keys for c in chunks_by_key[key]]
            try:
                rag.apply_source_update(
                    source_id,
                    slice_chunks,
                    replace_keys=set(slice_keys),
                    stale_chunk_ids=IngestionRegistry.chunk_ids_for_keys(source_id, slice_keys),
                )
            except EmbeddingUnavailable as e:
                deferred.extend(slice_keys)
                quota_message = str(e)
                logger.warning("[Delta] Document %s stopped early: %s", filename, e)
                break

            surviving = {c["chunk_id"] for c in rag.get_source_chunks(source_id)}
            for key in slice_keys:
                key_chunks = [c for c in chunks_by_key[key] if c["chunk_id"] in surviving]
                IngestionRegistry.record_indexed(
                    source_id, key, hashes[key],
                    chunk_ids=[c["chunk_id"] for c in key_chunks],
                    token_count=sum(int(c.get("tokens", 0)) for c in key_chunks),
                    embed_provider=provider,
                    source_type="document", source_name=filename,
                    title=stored[key].get("title"),
                )
                embedded_sections += 1

        total_chunks = rag.count_source_chunks(source_id)
        partial = bool(deferred)

        StorageService.update_history_entry(
            source_id,
            status="Partially Indexed" if partial else "Indexed",
            chunk_count=total_chunks,
            changes=(
                f"{'Replaced. ' if is_replacement else ''}{len(stored)} section(s): "
                f"{embedded_sections} embedded, {len(unchanged)} unchanged"
                + (f", {len(deferred)} deferred (embedding quota reached)" if partial else "")
                + f". {total_chunks} chunk(s) active."
            ),
            details={
                "sections": len(stored),
                "sections_embedded": embedded_sections,
                "sections_deferred": len(deferred),
                "sections_skipped_unchanged": len(unchanged),
                "embed_provider": provider,
                "quota_note": quota_message,
            },
        )

        if not partial:
            rag.reembed_pending_async()

        return {
            "id": source_id,
            "filename": filename,
            "status": "partial" if partial else "success",
            "replaced": is_replacement,
            "chunks": total_chunks,
            "sections_embedded": embedded_sections,
            "sections_deferred": len(deferred),
            "sections_skipped": len(unchanged),
            "message": quota_message,
        }

    except Exception as e:
        existing = rag.count_source_chunks(source_id)
        IngestionRegistry.mark_failed(source_id, [f"section-{i}" for i in range(1, len(items) + 1)],
                                      str(e), "document")
        StorageService.update_history_entry(
            source_id,
            status="Failed" if existing == 0 else "Indexed",
            chunk_count=existing,
            changes=f"Error parsing file: {e}",
        )
        raise
    finally:
        StorageService.clear_active(source_id)


def reset_everything(remove_uploads=False):
    """
    Wipes all derived state so the next run rebuilds from scratch:
    vectors, stored page text, the ingestion registry, crawl hashes, the raw
    crawl snapshot, training history and admin settings overrides.

    Uploaded files are kept unless remove_uploads=True, because they are
    originals that cannot be re-fetched.
    """
    rag = RAGService.get_instance()

    removed_chunks = rag.reset_index()

    removed_sources = 0
    for record in KnowledgeStore.list_sources():
        if KnowledgeStore.delete_source(record.get("source_id")):
            removed_sources += 1

    registry_path = Config.REGISTRY_FILE
    if registry_path.exists():
        registry_path.unlink()

    StorageService.save_history([])
    StorageService.reset_scrape_state()

    for path in (Config.SCRAPED_DIR / "bmsit_web_content.json", Config.SETTINGS_FILE):
        if path.exists():
            try:
                path.unlink()
            except Exception as e:
                logger.warning("[Reset] Could not remove %s: %s", path.name, e)

    removed_uploads = 0
    if remove_uploads:
        for upload in Config.UPLOADS_DIR.glob("*"):
            if upload.is_file():
                try:
                    upload.unlink()
                    removed_uploads += 1
                except Exception as e:
                    logger.warning("[Reset] Could not remove %s: %s", upload.name, e)

    logger.warning(
        "[Reset] Cleared %s vector(s), %s stored source(s), the registry, crawl state, "
        "history and settings. Uploads removed: %s",
        removed_chunks, removed_sources, removed_uploads if remove_uploads else "no",
    )
    return {
        "removed_chunks": removed_chunks,
        "removed_sources": removed_sources,
        "removed_uploads": removed_uploads,
    }


def purge_source(source_id, source_type):
    """
    Deletes one source completely: its vectors, its stored text and its registry
    entries. Used by the admin delete button. Other sources are unaffected.
    """
    rag = RAGService.get_instance()
    removed_chunks = rag.delete_source(source_id)
    KnowledgeStore.delete_source(source_id)
    IngestionRegistry.forget_source(source_id)

    if source_type == "website":
        StorageService.reset_scrape_state()
        snapshot = Config.SCRAPED_DIR / "bmsit_web_content.json"
        if snapshot.exists():
            try:
                snapshot.unlink()
            except Exception as e:
                logger.warning("[Delta] Could not remove crawl snapshot: %s", e)

    return removed_chunks
