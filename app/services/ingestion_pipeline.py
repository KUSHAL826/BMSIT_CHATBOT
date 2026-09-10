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
    Groups changed items into commit slices.

    Returns (slices, deferred, empty):
      slices   - lists of item keys embedded and committed together
      deferred - keys left out because the per-run chunk cap was reached
      empty    - keys that yielded no chunks at all (nothing to embed)

    `empty` exists so those items are reported instead of disappearing. They
    previously fell through every counter, which is why a run could claim 717
    changed pages, 0 embedded and only 85 deferred: the remainder produced no
    chunks and was silently dropped from the arithmetic.
    """
    slice_target = max(1, Config.EMBED_COMMIT_SLICE_CHUNKS)
    run_cap = Config.EMBED_MAX_CHUNKS_PER_RUN or 0

    # Biggest items first so one large page cannot be starved by the cap.
    ordered = sorted(chunks_by_key.items(), key=lambda kv: len(kv[1]), reverse=True)

    slices, deferred, empty = [], [], []
    current, current_count, budget_used = [], 0, 0

    for key, chunks in ordered:
        count = len(chunks)
        if not count:
            empty.append(key)
            continue
        if run_cap and budget_used + count > run_cap and budget_used > 0:
            deferred.append(key)
            continue

        current.append(key)
        current_count += count
        budget_used += count

        if current_count >= slice_target:
            slices.append(current)
            current, current_count = [], 0

    if current:
        slices.append(current)
    return slices, deferred, empty


def run_website_delta_ingestion(is_scheduled=False):
    """Crawls the college website and embeds only what changed."""
    source_id = KnowledgeStore.WEBSITE_SOURCE_ID
    source_label = KnowledgeStore.WEBSITE_SOURCE_NAME

    settings = StorageService.load_settings()
    target_url = settings.get("bmsit_url", Config.BMSIT_DEFAULT_URL)
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
        scraper = WebsiteScraper(
            base_url=target_url,
            max_pages=settings.get("scrape_max_pages", Config.SCRAPE_MAX_PAGES),
            max_depth=settings.get("scrape_depth", Config.SCRAPE_DEPTH),
        )
        crawl = scraper.run_sync(keep_running_for_indexing=True)

        if crawl.get("status") != "success":
            return _crawl_failed(rag, source_id, crawl.get("message", "unknown error"))

        pages = crawl.get("scraped_pages", {})
        scraper_status.phase = "comparing"
        scraper_status.status_message = f"Hashing {len(pages)} page(s) against the registry..."

        # ---------------- change detection ----------------
        # Pages too thin to answer anything are not worth embedding tokens on.
        hashes, thin_pages = {}, []
        for url, page in pages.items():
            text = (page.get("content") or "").strip()
            if not text:
                thin_pages.append(url)
                continue
            if count_tokens(text) < Config.SCRAPE_MIN_PAGE_TOKENS:
                thin_pages.append(url)
                continue
            hashes[url] = content_hash(text)

        if thin_pages:
            logger.info(
                "[Delta] Skipped %s page(s) with less than %s tokens of content.",
                len(thin_pages), Config.SCRAPE_MIN_PAGE_TOKENS,
            )

        changed, unchanged, reasons = IngestionRegistry.decide_batch(
            source_id, hashes, expected_provider=provider
        )

        # Safety net: trust the registry only if the vectors it claims really
        # exist in the index.
        indexed_keys = rag.source_item_keys(source_id)
        still_unchanged = []
        for url in unchanged:
            if url in indexed_keys:
                still_unchanged.append(url)
            else:
                reasons[url] = "vectors-missing"
                changed.append(url)
        unchanged = still_unchanged

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

        # Keep the permanent text store in sync for every crawled page, whether
        # or not it needs re-embedding.
        KnowledgeStore.upsert_items(
            source_id, "website", source_label,
            {
                url: {
                    "title": page["title"],
                    "text": page["content"],
                    "metadata": {"url": url, "title": page["title"], "type": "web"},
                }
                for url, page in pages.items()
            },
            replace_missing=False,
        )

        if not changed:
            return _nothing_to_do(rag, source_id, len(pages), len(unchanged), crawl, trigger)

        # ---------------- chunk only what changed ----------------
        scraper_status.phase = "chunking"
        scraper_status.status_message = (
            f"{len(changed)} page(s) changed, {len(unchanged)} unchanged. Chunking..."
        )

        stored = KnowledgeStore.get_items(source_id)
        chunks_by_url = {}
        for url in changed:
            item = stored.get(url)
            if not item:
                continue
            meta = item.get("metadata") or {}
            title = item.get("title") or url
            chunks_by_url[url] = rag.chunk_text(
                text=item["text"],
                source_id=source_id,
                source_name=title,
                source_type="website",
                metadata_extra={"url": meta.get("url", url), "title": title, "type": "web"},
                item_key=url,
                context_header=f"BMSIT page: {title} ({meta.get('url', url)})",
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
                f"{len(slice_chunks)} chunk(s) from {len(slice_urls)} page(s)..."
            )

            try:
                update = rag.apply_source_update(
                    source_id,
                    slice_chunks,
                    replace_keys=set(slice_urls),
                    stale_chunk_ids=IngestionRegistry.chunk_ids_for_keys(source_id, slice_urls),
                    progress_callback=_progress(f"Embedding batch {slice_index}/{len(planned)}"),
                )
            except EmbeddingUnavailable as e:
                # Registry is untouched for these pages, so they are retried later.
                deferred.extend(url for group in planned[slice_index - 1:] for url in group)
                quota_message = str(e)
                blocked_by_network = isinstance(e, BackendUnreachable)
                blocked_by_budget = isinstance(e, BudgetExhausted)
                logger.warning("[Delta] Stopping early: %s", e)
                break

            added += update["added"]
            replaced += update["removed"]

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
                        "title": (stored.get(url) or {}).get("title"),
                    }
                    for url in slice_urls
                ],
                source_type="website",
                source_name=source_label,
            )
            embedded_pages += len(slice_urls)

        total_chunks = rag.count_source_chunks(source_id)
        partial = bool(deferred)

        summary_parts = [
            f"Crawled {len(pages)} page(s)",
            f"{len(changed)} needed embedding: {embedded_pages} embedded",
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
        summary_parts.append(f"{len(stored)} page(s) stored, {total_chunks} chunk(s) active")
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
