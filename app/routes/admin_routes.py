import os
import re
import threading
from pathlib import Path
from flask import Blueprint, render_template, request, jsonify, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename

from app.config import Config
from app.services.storage import StorageService
from app.services.document_parser import DocumentParser
from app.services.ingestion_pipeline import ingest_document, purge_source, reset_everything
from app.services.ingestion_registry import IngestionRegistry
from app.services.knowledge_store import KnowledgeStore
from app.services.rag_service import RAGService
from app.services.scraper import WebsiteScraper, scraper_status
from app.services.scheduler_service import SchedulerService, execute_bmsit_scrape_and_index

admin_bp = Blueprint("admin", __name__, url_prefix="")

ALLOWED_EXTENSIONS = {"pdf", "docx", "doc", "csv"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _set_env_var(name, value):
    """
    Sets a variable for this process and persists it to .env.

    On a read-only filesystem the persist step is skipped with a warning; the
    value still applies until restart. Values are never logged.
    """
    os.environ[name] = value
    env_path = Config.BASE_DIR / ".env"
    try:
        lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
        replaced = False
        for index, line in enumerate(lines):
            if line.startswith(f"{name}="):
                lines[index] = f"{name}={value}"
                replaced = True
                break
        if not replaced:
            lines.append(f"{name}={value}")
        env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except Exception as e:
        print(f"[Admin] Could not persist {name} to .env ({e}). It applies until restart.")

@admin_bp.route("/")
def index():
    return redirect("/admin")

@admin_bp.route("/admin")
def admin_dashboard():
    history = StorageService.load_history()
    settings = StorageService.load_settings()
    scrape_state = StorageService.load_scrape_state()
    rag_stats = RAGService.get_instance().get_stats()
    sched_status = SchedulerService.get_status()

    return render_template(
        "admin.html",
        history=history,
        settings=settings,
        scrape_state=scrape_state,
        rag_stats=rag_stats,
        sched_status=sched_status
    )

@admin_bp.route("/api/admin/status", methods=["GET"])
def get_system_status():
    rag = RAGService.get_instance()
    stats = rag.get_stats()
    sched = SchedulerService.get_status()
    settings = StorageService.load_settings()
    scrape_state = StorageService.load_scrape_state()

    return jsonify({
        "status": "online",
        "rag": stats,
        "scheduler": sched,
        "settings": settings,
        "knowledge_store": KnowledgeStore.stats(),
        "registry": IngestionRegistry.stats(),
        "last_scrape": scrape_state.get("last_scrape"),
        "changes_count": len(scrape_state.get("changes", [])),
        "api_key_configured": bool(
            os.getenv("GEMINI_API_KEY") and os.getenv("GEMINI_API_KEY") != "your_api_key_here"
        ),
    })


@admin_bp.route("/api/admin/embeddings", methods=["GET"])
def get_embedding_status():
    """
    Embedding backend health: which keys are configured, which are usable right
    now, per-key request counts and quota hits, and the batch limits in force.
    Keys are shown masked.
    """
    rag = RAGService.get_instance()
    stats = rag.get_stats()
    return jsonify({
        "embeddings": stats["embeddings"],
        "spaces": stats["embedding_spaces"],
        "active_provider": stats["active_provider"],
        "pending_reembed": stats["needs_reembed"],
        "reembed": stats["reembed"],
        "total_tokens_indexed": stats["total_tokens"],
    })

@admin_bp.route("/api/admin/upload", methods=["POST"])
def upload_documents():
    """Handles PDF, DOC, DOCX, CSV uploads and triggers immediate RAG indexing."""
    if "files" not in request.files:
        return jsonify({"error": "No files provided in request"}), 400

    files = request.files.getlist("files")
    if not files or files[0].filename == "":
        return jsonify({"error": "No files selected"}), 400

    results = []

    for file in files:
        if not allowed_file(file.filename):
            results.append({
                "filename": file.filename,
                "status": "error",
                "message": f"Unsupported format. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
            })
            continue

        filename = secure_filename(file.filename)
        dest_path = Config.UPLOADS_DIR / filename
        file.save(dest_path)

        try:
            sections = DocumentParser.parse_file(dest_path)
            results.append(ingest_document(dest_path, filename, sections))
        except Exception as e:
            results.append({
                "filename": filename,
                "status": "error",
                "message": str(e)
            })

    return jsonify({"results": results})

@admin_bp.route("/api/admin/scrape", methods=["POST"])
def trigger_scrape_now():
    """Manual trigger: immediately scrapes the BMSIT website and updates FAISS."""
    if scraper_status.is_running:
        return jsonify({"status": "busy", "message": "A scrape job is already in progress"}), 409

    def background_task():
        execute_bmsit_scrape_and_index(is_scheduled=False)

    t = threading.Thread(target=background_task, daemon=True)
    t.start()

    return jsonify({
        "status": "started",
        "message": "Scraper started in background. Monitor progress in dashboard."
    })

@admin_bp.route("/api/admin/scrape/status", methods=["GET"])
def get_scrape_status():
    """Polls live status of the web scraper."""
    return jsonify(scraper_status.to_dict())

@admin_bp.route("/api/admin/history", methods=["GET"])
def get_training_history():
    """Fetches full training history."""
    history = StorageService.load_history()
    return jsonify({"history": history})

@admin_bp.route("/api/admin/source/<source_id>", methods=["DELETE"])
def delete_training_source(source_id):
    """
    Deletes any training folder or document.
    Immediately removes corresponding vectors from FAISS knowledge base and cleans disk.
    """
    deleted_entry = StorageService.delete_history_entry(source_id)
    if not deleted_entry:
        return jsonify({"error": "Source not found in history"}), 404

    # Vectors, stored text and registry entries for this source only.
    source_type = deleted_entry.get("source_type", "document")
    removed_chunks = purge_source(source_id, source_type)

    if source_type == "document":
        file_path = Config.UPLOADS_DIR / secure_filename(deleted_entry.get("source", ""))
        if file_path.exists():
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"[Admin] Could not delete physical file: {e}")

    rag = RAGService.get_instance()
    return jsonify({
        "status": "success",
        "deleted_source": deleted_entry.get("source"),
        "removed_chunks": removed_chunks,
        "remaining_chunks": rag.get_stats()["total_chunks"],
        "message": (
            f"Removed {deleted_entry.get('source')}: {removed_chunks} vectors, its stored "
            f"content and its registry entries. All other sources are untouched."
        )
    })


@admin_bp.route("/api/admin/reset", methods=["POST"])
def reset_knowledge_base():
    """
    Destructive: clears all vectors, stored page text, the ingestion registry,
    crawl hashes, history and settings overrides, so the next scrape rebuilds
    from zero. Requires {"confirm": "RESET"} in the body.

    Uploaded files are kept unless {"remove_uploads": true} is also sent, since
    originals cannot be re-fetched.
    """
    data = request.json or {}
    if data.get("confirm") != "RESET":
        return jsonify({
            "error": "Confirmation required.",
            "hint": 'Send {"confirm": "RESET"} to proceed. This deletes all indexed data.',
        }), 400
    if scraper_status.is_running:
        return jsonify({"error": "A scrape is in progress. Wait for it to finish."}), 409

    result = reset_everything(remove_uploads=bool(data.get("remove_uploads")))
    return jsonify({
        "status": "success",
        **result,
        "message": "Knowledge base cleared. Run a scrape to rebuild from scratch.",
    })


@admin_bp.route("/api/admin/knowledge", methods=["GET"])
def list_knowledge_sources():
    """Lists what the knowledge base currently holds, per source."""
    rag = RAGService.get_instance()
    sources = []
    for record in KnowledgeStore.list_sources():
        sid = record.get("source_id")
        items = record.get("items", {})
        sources.append({
            "source_id": sid,
            "source_name": record.get("source_name"),
            "source_type": record.get("source_type"),
            "items": len(items),
            "characters": sum(len(i.get("text", "")) for i in items.values()),
            "indexed_chunks": rag.count_source_chunks(sid),
            "registry": IngestionRegistry.stats(sid),
            "created": record.get("created"),
            "updated": record.get("updated"),
        })
    return jsonify({
        "sources": sources,
        "totals": KnowledgeStore.stats(),
        "registry": IngestionRegistry.stats(),
    })


@admin_bp.route("/api/admin/registry/<source_id>", methods=["GET"])
def get_registry_detail(source_id):
    """
    Per-item delta state: content hash, timestamps, vector ids and token counts.
    This is what the daily run compares against to decide what to re-embed.
    """
    items = IngestionRegistry.items(source_id)
    return jsonify({
        "source_id": source_id,
        "stats": IngestionRegistry.stats(source_id),
        "items": [
            {
                "item_key": key,
                "title": item.get("title"),
                "content_hash": item.get("content_hash"),
                "last_scraped": item.get("last_scraped"),
                "last_embedded": item.get("last_embedded"),
                "chunk_count": item.get("chunk_count", 0),
                "token_count": item.get("token_count", 0),
                "embed_provider": item.get("embed_provider"),
                "revision": item.get("revision", 1),
                "status": item.get("status"),
            }
            for key, item in sorted(items.items())
        ],
    })


@admin_bp.route("/api/admin/reembed", methods=["POST"])
def reembed_knowledge():
    """
    Re-embeds chunks that fell back to local vectors (or predate provider
    tagging) so they are searchable in the Gemini embedding space.
    """
    rag = RAGService.get_instance()
    pending = rag.pending_reembed_count()
    if pending == 0:
        return jsonify({"status": "ok", "message": "All vectors already use the active embedding model."})
    started = rag.reembed_pending_async()
    return jsonify({
        "status": "started" if started else "unavailable",
        "pending": pending,
        "message": (
            f"Upgrading {pending} vector(s) in the background."
            if started else
            "Cannot re-embed: no Gemini API key configured, or a job is already running."
        )
    })

@admin_bp.route("/api/admin/changes", methods=["GET"])
def get_website_changes():
    """Returns detected changes from the last website scrape."""
    state = StorageService.load_scrape_state()
    return jsonify({
        "changes": state.get("changes", []),
        "summary": state.get("summary", "No changes recorded yet."),
        "last_scrape": state.get("last_scrape")
    })

@admin_bp.route("/api/admin/settings", methods=["POST"])
def update_settings():
    """Updates BMSIT target URL, API Key, or scrape limits."""
    data = request.json or {}
    new_settings = {}
    
    if "bmsit_url" in data:
        new_settings["bmsit_url"] = data["bmsit_url"].strip()
    if "scrape_max_pages" in data:
        new_settings["scrape_max_pages"] = int(data["scrape_max_pages"])
    if "scrape_depth" in data:
        new_settings["scrape_depth"] = int(data["scrape_depth"])

    # API keys. Values are never echoed back in the response.
    if data.get("gemini_api_key", "").strip():
        _set_env_var("GEMINI_API_KEY", data["gemini_api_key"].strip())
    if data.get("gemini_api_keys", "").strip():
        pool = ",".join(
            part.strip() for part in re.split(r"[,\s;]+", data["gemini_api_keys"]) if part.strip()
        )
        _set_env_var("GEMINI_API_KEYS", pool)
    if data.get("gemini_api_key") or data.get("gemini_api_keys"):
        RAGService.get_instance()._embedder.pool.refresh()

    if "schedule_time" in data and str(data["schedule_time"]).strip():
        new_settings["schedule_time"] = str(data["schedule_time"]).strip()

    updated = StorageService.save_settings(new_settings)

    # Apply a changed cron time immediately instead of waiting for a restart.
    if "schedule_time" in new_settings:
        try:
            SchedulerService.reschedule()
        except Exception as e:
            print(f"[Admin] Could not reschedule daily job: {e}")

    return jsonify({"status": "success", "settings": updated})

@admin_bp.route("/api/admin/document/<filename>", methods=["GET"])
def view_document(filename):
    """Serves an uploaded document (PDF, CSV, etc.) for viewing/previewing."""
    clean_name = secure_filename(filename)
    file_path = Config.UPLOADS_DIR / clean_name
    if not file_path.exists():
        return jsonify({"error": "Document not found"}), 404

    # Determine mimetype for inline browser viewing
    ext = clean_name.rsplit(".", 1)[-1].lower() if "." in clean_name else ""
    mimetypes = {
        "pdf": "application/pdf",
        "csv": "text/plain",
        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "doc": "application/msword"
    }
    mimetype = mimetypes.get(ext, "application/octet-stream")

    response = send_from_directory(
        Config.UPLOADS_DIR,
        clean_name,
        mimetype=mimetype,
        as_attachment=False
    )
    response.headers["Content-Disposition"] = f'inline; filename="{clean_name}"'
    response.headers["X-Frame-Options"] = "SAMEORIGIN"
    return response

@admin_bp.route("/api/admin/document-content/<filename>", methods=["GET"])
def get_document_content(filename):
    """Extracts and returns parsed text sections from DOCX, DOC, CSV or PDF for clean modal preview."""
    clean_name = secure_filename(filename)
    file_path = Config.UPLOADS_DIR / clean_name
    if not file_path.exists():
        return jsonify({"error": "Document not found"}), 404

    ext = clean_name.rsplit(".", 1)[-1].lower() if "." in clean_name else ""
    try:
        sections = DocumentParser.parse_file(file_path)
        formatted = []
        for s in sections:
            meta = s.get("metadata", {})
            title = meta.get("section_title") or f"Section"
            if meta.get("page"):
                title = f"Page {meta.get('page')}"
            formatted.append({
                "title": title,
                "content": s.get("content", "")
            })
        return jsonify({
            "status": "success",
            "filename": clean_name,
            "type": ext,
            "sections": formatted
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Could not extract preview: {str(e)}"
        }), 500

@admin_bp.route("/api/admin/source-preview/<source_id>", methods=["GET"])
def get_source_preview(source_id):
    """Returns detailed preview content for any training source (document or website scrape)."""
    history = StorageService.load_history()
    target = next((h for h in history if h.get("id") == source_id), None)
    if not target:
        return jsonify({"error": "Training source not found"}), 404

    rag = RAGService.get_instance()
    matching_chunks = rag.get_source_chunks(source_id)
    stored_items = KnowledgeStore.get_items(source_id)

    if target.get("source_type") == "website":
        details = target.get("details", {})
        changes = details.get("changes", [])
        return jsonify({
            "status": "success",
            "source_type": "website",
            "title": target.get("source", "BMSIT Website"),
            "date": target.get("date"),
            "total_chunks": len(matching_chunks),
            "pages_count": details.get("pages_stored", details.get("pages_scraped", len(stored_items))),
            "pages_stored": len(stored_items),
            "changes": changes,
            "stored_pages": [
                {
                    "url": key,
                    "title": item.get("title"),
                    "characters": len(item.get("text", "")),
                    "updated": item.get("updated"),
                }
                for key, item in list(stored_items.items())[:60]
            ],
            "sample_chunks": [
                {
                    "id": c.get("chunk_id"),
                    "text": c.get("text", "")[:350] + ("..." if len(c.get("text", "")) > 350 else ""),
                    "metadata": c.get("metadata", {})
                }
                for c in matching_chunks[:12]
            ]
        })
    else:
        # Document
        filename = secure_filename(target.get("source", ""))
        file_path = Config.UPLOADS_DIR / filename
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        sections = []

        # Prefer the permanently stored text; it exists even if the original
        # file was removed from disk.
        for key, item in stored_items.items():
            sections.append({
                "title": item.get("title") or key,
                "content": item.get("text", "")
            })

        if not sections and file_path.exists():
            try:
                raw_sections = DocumentParser.parse_file(file_path)
                for s in raw_sections:
                    meta = s.get("metadata", {})
                    title = meta.get("section_title") or "Section"
                    if meta.get("page"):
                        title = f"Page {meta.get('page')}"
                    sections.append({
                        "title": title,
                        "content": s.get("content", "")
                    })
            except Exception as e:
                print(f"[Admin] Error parsing file for preview: {e}")

        # Fallback to indexed chunks if file was deleted or cannot be re-parsed
        if not sections and matching_chunks:
            for idx, c in enumerate(matching_chunks[:20]):
                meta = c.get("metadata", {})
                title = meta.get("section_title") or f"Chunk #{idx + 1}"
                sections.append({
                    "title": title,
                    "content": c.get("text", "")
                })

        return jsonify({
            "status": "success",
            "source_type": "document",
            "title": filename,
            "filename": filename,
            "type": ext,
            "sections": sections,
            "total_chunks": len(matching_chunks)
        })


