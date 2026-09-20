import os
import re
import threading
from pathlib import Path

import requests

from flask import (
    Blueprint,
    render_template,
    request,
    jsonify,
    redirect,
    url_for,
    send_from_directory,
    session
)
from werkzeug.utils import secure_filename

from app.config import Config
from app.services.storage import StorageService
from app.services.document_parser import DocumentParser
from app.services.ingestion_pipeline import (
    ingest_document,
    purge_source,
    reset_everything
)
from app.services.ingestion_registry import IngestionRegistry
from app.services.knowledge_store import KnowledgeStore
from app.services.rag_service import RAGService
from app.services.scraper import WebsiteScraper, scraper_status
from app.services.scheduler_service import (
    SchedulerService,
    execute_bmsit_scrape_and_index
)


admin_bp = Blueprint("admin", __name__, url_prefix="")


ALLOWED_EXTENSIONS = {
    "pdf",
    "docx",
    "doc",
    "csv"
}


def allowed_file(filename):
    return (
        "."
        in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


def _set_env_var(name, value):
    """
    Sets a variable for the current process.

    It is also written back to .env, but ONLY if that file already exists,
    which is the local-development case.

    In a hosted environment configuration comes from the platform,
    so creating a .env there would be misleading.

    Values are never logged.
    """

    os.environ[name] = value

    env_path = Config.ENV_FILE

    if not env_path.exists():
        print(
            f"[Admin] {name} updated for this process. "
            "No .env file present, so it was not persisted. "
            "Set it in your host's environment settings "
            "(Render: Dashboard -> Environment) to survive a restart."
        )
        return

    try:

        lines = env_path.read_text(
            encoding="utf-8"
        ).splitlines()

        replaced = False

        for index, line in enumerate(lines):

            if line.startswith(
                f"{name}="
            ):

                lines[index] = (
                    f"{name}={value}"
                )

                replaced = True
                break

        if not replaced:
            lines.append(
                f"{name}={value}"
            )

        env_path.write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8"
        )

    except Exception as e:

        print(
            f"[Admin] Could not persist {name} "
            f"to .env ({e}). "
            "It applies until restart."
        )


# ============================================================
# SUPABASE PERSISTENT ADMIN DATA
# ============================================================

def _supabase_headers():
    """
    Headers used for server-side access to Supabase.

    Uses the same SUPABASE_KEY already used by the application's
    Supabase vector storage service.
    """

    key = (
        os.getenv("SUPABASE_KEY")
        or getattr(
            Config,
            "SUPABASE_KEY",
            ""
        )
    )

    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def _load_supabase_vectors_for_admin():
    """
    Reads persistent vector metadata from Supabase.

    This is intentionally separate from the existing RAG/FAISS code.

    It is used by the Admin dashboard to restore knowledge information
    after a Render restart/redeploy.

    Returns:
        list[dict]
    """

    try:

        url = (
            os.getenv("SUPABASE_URL")
            or getattr(
                Config,
                "SUPABASE_URL",
                ""
            )
        ).rstrip("/")

        table = (
            os.getenv("SUPABASE_TABLE")
            or getattr(
                Config,
                "SUPABASE_TABLE",
                ""
            )
            or "vectors"
        )

        key = (
            os.getenv("SUPABASE_KEY")
            or getattr(
                Config,
                "SUPABASE_KEY",
                ""
            )
        )

        if not url or not key:

            print(
                "[Admin] Supabase credentials unavailable "
                "for persistent knowledge."
            )

            return []

        endpoint = (
            f"{url}/rest/v1/{table}"
        )

        params = {
            "select": (
                "id,"
                "source_id,"
                "source_name,"
                "source_type,"
                "item_key,"
                "text,"
                "metadata"
            ),
            "order": "id.desc",
            "limit": "10000",
        }

        response = requests.get(
            endpoint,
            headers=_supabase_headers(),
            params=params,
            timeout=30
        )

        if response.status_code >= 400:

            print(
                "[Admin] Supabase persistent "
                "knowledge read failed: "
                f"HTTP {response.status_code}: "
                f"{response.text[:500]}"
            )

            return []

        data = response.json()

        if not isinstance(
            data,
            list
        ):
            return []

        return data

    except Exception as e:

        print(
            "[Admin] Could not load persistent "
            f"Supabase vectors: {e}"
        )

        return []


# ============================================================
# BUILD TRAINING HISTORY FROM SUPABASE
# ============================================================

def _build_supabase_history():
    """
    Builds training-history entries from persistent Supabase vectors.

    One source can contain many chunks, therefore chunks are grouped
    using source_id.

    This allows the dashboard to continue showing previous training
    sources after a Render redeploy.
    """

    vectors = (
        _load_supabase_vectors_for_admin()
    )

    if not vectors:
        return []

    grouped = {}

    for row in vectors:

        source_id = (
            row.get("source_id")
            or row.get("source_name")
            or row.get("item_key")
        )

        if not source_id:
            continue

        source_id = str(
            source_id
        )

        source_name = (
            row.get("source_name")
            or source_id
            or "Unknown source"
        )

        source_type = (
            row.get("source_type")
            or "document"
        )

        source_type = str(
            source_type
        ).lower()

        if source_type in {
            "web",
            "website",
            "url",
            "scrape"
        }:

            display_type = "website"

        else:

            display_type = "document"

        if source_id not in grouped:

            grouped[source_id] = {

                "id": source_id,

                "source": source_name,

                "source_type": display_type,

                "file_type": (
                    "website"
                    if display_type == "website"
                    else (
                        str(
                            source_name
                        ).rsplit(
                            ".",
                            1
                        )[-1].lower()
                        if "."
                        in str(
                            source_name
                        )
                        else "document"
                    )
                ),

                "status": "Indexed",

                "chunk_count": 0,

                "changes": (
                    "Stored in Supabase"
                ),

                "date": "Persistent",

                "details": {
                    "persistent": True,
                    "chunks": [],
                },
            }

        grouped[source_id][
            "chunk_count"
        ] += 1

        if len(
            grouped[source_id][
                "details"
            ][
                "chunks"
            ]
        ) < 20:

            grouped[source_id][
                "details"
            ][
                "chunks"
            ].append({

                "id": row.get(
                    "id"
                ),

                "item_key": row.get(
                    "item_key"
                ),

                "text": (
                    row.get(
                        "text",
                        ""
                    )
                    or ""
                ),

                "metadata": (
                    row.get(
                        "metadata"
                    )
                    or {}
                ),
            })

    history = list(
        grouped.values()
    )

    history.sort(
        key=lambda item:
        item.get(
            "chunk_count",
            0
        ),
        reverse=True
    )

    return history


def _get_persistent_history():
    """
    Merges local training history with persistent Supabase sources,
    ensuring all active trained documents and web scrapes stay visible.
    """
    try:
        local_history = StorageService.load_history() or []
    except Exception as e:
        print(f"[Admin] Local history load error: {e}")
        local_history = []

    try:
        persistent_history = _build_supabase_history() or []
    except Exception as e:
        print(f"[Admin] Persistent Supabase history load error: {e}")
        persistent_history = []

    seen_keys = set()
    combined = []

    for item in local_history:
        key = str(item.get("source") or item.get("id") or "").strip().lower()
        if key and key not in seen_keys:
            seen_keys.add(key)
            combined.append(item)

    for item in persistent_history:
        key = str(item.get("source") or item.get("id") or "").strip().lower()
        if key and key not in seen_keys:
            seen_keys.add(key)
            combined.append(item)

    return combined


# ============================================================
# SUPABASE SOURCE CHUNKS
# ============================================================

def _get_supabase_source_chunks(source_id):
    """
    Retrieves chunks for one source directly from Supabase.

    Used when local FAISS/KnowledgeStore data disappeared
    after a Render redeployment.
    """

    vectors = (
        _load_supabase_vectors_for_admin()
    )

    matching = []

    for row in vectors:

        row_source_id = (
            row.get("source_id")
            or row.get("source_name")
            or row.get("item_key")
        )

        if str(
            row_source_id
        ) != str(
            source_id
        ):
            continue

        matching.append({

            "chunk_id": row.get(
                "id"
            ),

            "text": (
                row.get(
                    "text",
                    ""
                )
                or ""
            ),

            "metadata": (
                row.get(
                    "metadata"
                )
                or {}
            ),

            "item_key": row.get(
                "item_key"
            ),

            "source_id": row.get(
                "source_id"
            ),

            "source_name": row.get(
                "source_name"
            ),

            "source_type": row.get(
                "source_type"
            ),
        })

    return matching


# ============================================================
# SECURITY & AUTHENTICATION MIDDLEWARE
# ============================================================

@admin_bp.before_request
def require_admin_auth():
    """Enforces password authentication on all admin pages and API endpoints."""
    if request.path in ("/admin/login", "/admin/logout"):
        return None

    if Config.ADMIN_PASSWORD:
        if not session.get("admin_authenticated"):
            if request.path.startswith("/api/admin"):
                return jsonify({"error": "Unauthorized. Admin authentication required."}), 401
            return redirect(url_for("admin.login_page"))


@admin_bp.route("/admin/login", methods=["GET", "POST"])
def login_page():
    if request.method == "POST":
        password = (request.form.get("password") or "").strip()
        if password == Config.ADMIN_PASSWORD:
            session["admin_authenticated"] = True
            return redirect("/admin")
        return render_template("admin_login.html", error="Invalid admin security password.")

    if session.get("admin_authenticated"):
        return redirect("/admin")
    return render_template("admin_login.html")


@admin_bp.route("/admin/logout")
def logout():
    session.pop("admin_authenticated", None)
    return redirect(url_for("admin.login_page"))


# ============================================================
# ORIGINAL ADMIN ROUTES
# ============================================================

@admin_bp.route("/")
def index():
    return redirect("/admin")


@admin_bp.route("/admin")
def admin_dashboard():

    # Persistent history fallback.
    history = (
        _get_persistent_history()
    )

    settings = (
        StorageService.load_settings()
    )

    scrape_state = (
        StorageService.load_scrape_state()
    )

    rag_stats = (
        RAGService
        .get_instance()
        .get_stats()
    )

    sched_status = (
        SchedulerService.get_status()
    )

    return render_template(
        "admin.html",

        history=history,

        settings=settings,

        scrape_state=scrape_state,

        rag_stats=rag_stats,

        sched_status=sched_status
    )


# ============================================================
# SYSTEM STATUS
# ============================================================

@admin_bp.route(
    "/api/admin/status",
    methods=["GET"]
)
def get_system_status():

    rag = (
        RAGService.get_instance()
    )

    stats = (
        rag.get_stats()
    )

    sched = (
        SchedulerService.get_status()
    )

    settings = (
        StorageService.load_settings()
    )

    scrape_state = (
        StorageService.load_scrape_state()
    )

    return jsonify({

        "status": "online",

        "rag": stats,

        "scheduler": sched,

        "settings": settings,

        "knowledge_store":
            KnowledgeStore.stats(),

        "registry":
            IngestionRegistry.stats(),

        "last_scrape":
            scrape_state.get(
                "last_scrape"
            ),

        "changes_count":
            len(
                scrape_state.get(
                    "changes",
                    []
                )
            ),

        "api_key_configured": bool(
            os.getenv(
                "GEMINI_API_KEY"
            )
            and os.getenv(
                "GEMINI_API_KEY"
            ) != "your_api_key_here"
        ),
    })


# ============================================================
# EMBEDDING STATUS
# ============================================================

@admin_bp.route(
    "/api/admin/embeddings",
    methods=["GET"]
)
def get_embedding_status():

    rag = (
        RAGService.get_instance()
    )

    stats = (
        rag.get_stats()
    )

    return jsonify({

        "embeddings":
            stats["embeddings"],

        "spaces":
            stats["embedding_spaces"],

        "active_provider":
            stats["active_provider"],

        "pending_reembed":
            stats["needs_reembed"],

        "reembed":
            stats["reembed"],

        "total_tokens_indexed":
            stats["total_tokens"],
    })


# ============================================================
# DOCUMENT UPLOAD
# ============================================================

@admin_bp.route(
    "/api/admin/upload",
    methods=["POST"]
)
def upload_documents():

    """
    Handles PDF, DOC, DOCX, CSV uploads
    and triggers immediate RAG indexing.
    """

    if "files" not in request.files:

        return jsonify({
            "error":
                "No files provided in request"
        }), 400

    files = request.files.getlist(
        "files"
    )

    if (
        not files
        or files[0].filename == ""
    ):

        return jsonify({
            "error":
                "No files selected"
        }), 400

    results = []

    for file in files:

        if not allowed_file(
            file.filename
        ):

            results.append({

                "filename":
                    file.filename,

                "status":
                    "error",

                "message": (
                    "Unsupported format. "
                    "Allowed: "
                    + ", ".join(
                        ALLOWED_EXTENSIONS
                    )
                )
            })

            continue

        filename = secure_filename(
            file.filename
        )

        dest_path = (
            Config.UPLOADS_DIR
            / filename
        )

        file.save(
            dest_path
        )

        try:

            sections = (
                DocumentParser.parse_file(
                    dest_path
                )
            )

            results.append(
                ingest_document(
                    dest_path,
                    filename,
                    sections
                )
            )

        except Exception as e:

            results.append({

                "filename":
                    filename,

                "status":
                    "error",

                "message":
                    str(e)
            })

    return jsonify({
        "results":
            results
    })


# ============================================================
# MANUAL SCRAPE
# ============================================================

@admin_bp.route(
    "/api/admin/scrape",
    methods=["POST"]
)
def trigger_scrape_now():

    """
    Manual trigger: immediately scrapes
    the BMSIT website and updates FAISS.
    """

    if scraper_status.is_running:

        return jsonify({

            "status":
                "busy",

            "message":
                "A scrape job is already in progress"

        }), 409

    def background_task():

        execute_bmsit_scrape_and_index(
            is_scheduled=False
        )

    t = threading.Thread(
        target=background_task,
        daemon=True
    )

    t.start()

    return jsonify({

        "status":
            "started",

        "message": (
            "Scraper started in background. "
            "Monitor progress in dashboard."
        )
    })


@admin_bp.route(
    "/api/admin/scrape/status",
    methods=["GET"]
)
def get_scrape_status():

    """
    Polls live status of the web scraper.
    """

    return jsonify(
        scraper_status.to_dict()
    )


# ============================================================
# TRAINING HISTORY
# ============================================================

@admin_bp.route(
    "/api/admin/history",
    methods=["GET"]
)
def get_training_history():

    """
    Fetches full training history.

    Local StorageService history is preferred.

    Persistent Supabase vector metadata is
    used as fallback after redeployment.
    """

    history = (
        _get_persistent_history()
    )

    return jsonify({
        "history":
            history
    })


# ============================================================
# DELETE TRAINING SOURCE
# ============================================================

@admin_bp.route(
    "/api/admin/source/<source_id>",
    methods=["DELETE"]
)
def delete_training_source(
    source_id
):

    """
    Deletes any training folder or document.

    Removes corresponding vectors from the
    knowledge base and cleans disk.
    """

    deleted_entry = (
        StorageService.delete_history_entry(
            source_id
        )
    )

    if not deleted_entry:

        persistent_history = (
            _build_supabase_history()
        )

        deleted_entry = next(
            (
                item
                for item
                in persistent_history

                if str(
                    item.get("id")
                )
                == str(
                    source_id
                )
            ),
            None
        )

        if not deleted_entry:

            return jsonify({
                "error":
                    "Source not found in history"
            }), 404

    source_type = (
        deleted_entry.get(
            "source_type",
            "document"
        )
    )

    removed_chunks = purge_source(
        source_id,
        source_type
    )

    if source_type == "document":

        file_path = (
            Config.UPLOADS_DIR
            / secure_filename(
                deleted_entry.get(
                    "source",
                    ""
                )
            )
        )

        if file_path.exists():

            try:

                os.remove(
                    file_path
                )

            except Exception as e:

                print(
                    "[Admin] Could not "
                    f"delete physical file: {e}"
                )

    rag = (
        RAGService.get_instance()
    )

    return jsonify({

        "status":
            "success",

        "deleted_source":
            deleted_entry.get(
                "source"
            ),

        "removed_chunks":
            removed_chunks,

        "remaining_chunks":
            rag.get_stats()[
                "total_chunks"
            ],

        "message": (
            f"Removed "
            f"{deleted_entry.get('source')}: "
            f"{removed_chunks} vectors, "
            "its stored content and "
            "its registry entries. "
            "All other sources are untouched."
        )
    })


# ============================================================
# RESET KNOWLEDGE BASE
# ============================================================

@admin_bp.route(
    "/api/admin/reset",
    methods=["POST"]
)
def reset_knowledge_base():

    """
    Destructive operation.

    Clears all vectors, stored page text,
    ingestion registry, crawl hashes,
    history and settings overrides.

    Requires:
        {"confirm": "RESET"}
    """

    data = (
        request.json
        or {}
    )

    if data.get(
        "confirm"
    ) != "RESET":

        return jsonify({

            "error":
                "Confirmation required.",

            "hint": (
                'Send {"confirm": "RESET"} '
                "to proceed. "
                "This deletes all indexed data."
            ),

        }), 400

    if scraper_status.is_running:

        return jsonify({

            "error": (
                "A scrape is in progress. "
                "Wait for it to finish."
            )

        }), 409

    result = reset_everything(
        remove_uploads=bool(
            data.get(
                "remove_uploads"
            )
        )
    )

    return jsonify({

        "status":
            "success",

        **result,

        "message": (
            "Knowledge base cleared. "
            "Run a scrape to rebuild from scratch."
        ),
    })


# ============================================================
# KNOWLEDGE BASE
# ============================================================

@admin_bp.route(
    "/api/admin/knowledge",
    methods=["GET"]
)
def list_knowledge_sources():

    """
    Lists what the knowledge base currently holds,
    per source.

    IMPORTANT:

    Local KnowledgeStore is preferred.

    If Render has restarted/redeployed and the local
    KnowledgeStore is empty, persistent Supabase vectors
    are used automatically.

    Each source contains the first 5 stored chunks
    so the frontend can display them.
    """

    rag = (
        RAGService.get_instance()
    )

    sources = []

    # ========================================================
    # FIRST: TRY LOCAL KNOWLEDGE STORE
    # ========================================================

    local_records = []

    try:

        local_records = (
            KnowledgeStore.list_sources()
        )

    except Exception as e:

        print(
            "[Admin] Could not load local "
            f"knowledge sources: {e}"
        )

        local_records = []

    # ========================================================
    # LOCAL DATA EXISTS
    # ========================================================

    if local_records:

        for record in local_records:

            sid = record.get(
                "source_id"
            )

            items = record.get(
                "items",
                {}
            )

            sample_chunks = []

            for key, item in list(
                items.items()
            )[:5]:

                sample_chunks.append({

                    "chunk_id":
                        item.get(
                            "id"
                        ),

                    "item_key":
                        key,

                    "text":
                        item.get(
                            "text",
                            ""
                        ),

                    "metadata":
                        item.get(
                            "metadata",
                            {}
                        ),
                })

            sources.append({

                "source_id":
                    sid,

                "source_name":
                    record.get(
                        "source_name"
                    ),

                "source_type":
                    record.get(
                        "source_type"
                    ),

                "items":
                    len(
                        items
                    ),

                "characters":
                    sum(
                        len(
                            i.get(
                                "text",
                                ""
                            )
                        )
                        for i
                        in items.values()
                    ),

                "indexed_chunks":
                    rag.count_source_chunks(
                        sid
                    ),

                "registry":
                    IngestionRegistry.stats(
                        sid
                    ),

                "created":
                    record.get(
                        "created"
                    ),

                "updated":
                    record.get(
                        "updated"
                    ),

                "sample_chunks":
                    sample_chunks,
            })

        return jsonify({

            "sources":
                sources,

            "totals":
                KnowledgeStore.stats(),

            "registry":
                IngestionRegistry.stats(),

            "persistent":
                False,
        })

    # ========================================================
    # LOCAL DATA IS EMPTY
    # RESTORE FROM SUPABASE
    # ========================================================

    print(
        "[Admin] Local KnowledgeStore is empty. "
        "Restoring Knowledge Base from Supabase..."
    )

    vectors = (
        _load_supabase_vectors_for_admin()
    )

    if not vectors:

        print(
            "[Admin] No persistent vectors "
            "found in Supabase."
        )

        return jsonify({

            "sources": [],

            "totals": {

                "sources":
                    0,

                "items":
                    0,

                "chunks":
                    0,

                "characters":
                    0,
            },

            "registry":
                IngestionRegistry.stats(),

            "persistent":
                True,
        })

    # ========================================================
    # GROUP SUPABASE CHUNKS BY SOURCE
    # ========================================================

    grouped = {}

    for row in vectors:

        source_id = (
            row.get("source_id")
            or row.get("source_name")
            or row.get("item_key")
        )

        if not source_id:
            continue

        source_id = str(
            source_id
        )

        source_name = (
            row.get("source_name")
            or source_id
            or "Unknown source"
        )

        source_type = (
            row.get("source_type")
            or "document"
        )

        source_type = str(
            source_type
        ).lower()

        if source_type in {
            "web",
            "website",
            "url",
            "scrape"
        }:

            display_type = (
                "website"
            )

        else:

            display_type = (
                "document"
            )

        # ----------------------------------------------------
        # CREATE SOURCE
        # ----------------------------------------------------

        if source_id not in grouped:

            grouped[source_id] = {

                "source_id":
                    source_id,

                "source_name":
                    source_name,

                "source_type":
                    display_type,

                "items":
                    0,

                "characters":
                    0,

                "indexed_chunks":
                    0,

                "registry":
                    IngestionRegistry.stats(
                        source_id
                    ),

                "created":
                    None,

                "updated":
                    None,

                "sample_chunks":
                    [],
            }

        source = (
            grouped[source_id]
        )

        text = (
            row.get(
                "text",
                ""
            )
            or ""
        )

        # ----------------------------------------------------
        # COUNT CHUNK
        # ----------------------------------------------------

        source[
            "items"
        ] += 1

        source[
            "indexed_chunks"
        ] += 1

        source[
            "characters"
        ] += len(
            text
        )

        # ----------------------------------------------------
        # FIRST 5 CHUNKS
        # ----------------------------------------------------

        if len(
            source[
                "sample_chunks"
            ]
        ) < 5:

            source[
                "sample_chunks"
            ].append({

                "chunk_id":
                    row.get(
                        "id"
                    ),

                "item_key":
                    row.get(
                        "item_key"
                    ),

                "text":
                    text,

                "metadata":
                    row.get(
                        "metadata"
                    )
                    or {},

                "source_id":
                    row.get(
                        "source_id"
                    ),

                "source_name":
                    row.get(
                        "source_name"
                    ),

                "source_type":
                    row.get(
                        "source_type"
                    ),
            })

    # ========================================================
    # FINAL TOTALS
    # ========================================================

    sources = list(
        grouped.values()
    )

    total_chunks = sum(
        source.get(
            "indexed_chunks",
            0
        )
        for source
        in sources
    )

    total_characters = sum(
        source.get(
            "characters",
            0
        )
        for source
        in sources
    )

    persistent_totals = {

        "sources":
            len(
                sources
            ),

        "items":
            total_chunks,

        "chunks":
            total_chunks,

        "characters":
            total_characters,
    }

    print(
        f"[Admin] Restored "
        f"{len(sources)} source(s) and "
        f"{total_chunks} persistent chunk(s) "
        "from Supabase."
    )

    return jsonify({

        "sources":
            sources,

        "totals":
            persistent_totals,

        "registry":
            IngestionRegistry.stats(),

        "persistent":
            True,
    })


# ============================================================
# REGISTRY DETAILS
# ============================================================

@admin_bp.route(
    "/api/admin/registry/<source_id>",
    methods=["GET"]
)
def get_registry_detail(
    source_id
):

    """
    Per-item delta state.

    Content hash, timestamps, vector ids
    and token counts are returned here.
    """

    items = (
        IngestionRegistry.items(
            source_id
        )
    )

    return jsonify({

        "source_id":
            source_id,

        "stats":
            IngestionRegistry.stats(
                source_id
            ),

        "items": [

            {

                "item_key":
                    key,

                "title":
                    item.get(
                        "title"
                    ),

                "content_hash":
                    item.get(
                        "content_hash"
                    ),

                "last_scraped":
                    item.get(
                        "last_scraped"
                    ),

                "last_embedded":
                    item.get(
                        "last_embedded"
                    ),

                "chunk_count":
                    item.get(
                        "chunk_count",
                        0
                    ),

                "token_count":
                    item.get(
                        "token_count",
                        0
                    ),

                "embed_provider":
                    item.get(
                        "embed_provider"
                    ),

                "revision":
                    item.get(
                        "revision",
                        1
                    ),

                "status":
                    item.get(
                        "status"
                    ),
            }

            for key, item
            in sorted(
                items.items()
            )
        ],
    })


# ============================================================
# RE-EMBED
# ============================================================

@admin_bp.route(
    "/api/admin/reembed",
    methods=["POST"]
)
def reembed_knowledge():

    """
    Re-embeds chunks that fell back to local vectors
    or predate provider tagging.
    """

    rag = (
        RAGService.get_instance()
    )

    pending = (
        rag.pending_reembed_count()
    )

    if pending == 0:

        return jsonify({

            "status":
                "ok",

            "message": (
                "All vectors already use "
                "the active embedding model."
            )
        })

    started = (
        rag.reembed_pending_async()
    )

    return jsonify({

        "status":
            "started"
            if started
            else
            "unavailable",

        "pending":
            pending,

        "message": (

            f"Upgrading {pending} "
            "vector(s) in the background."

            if started

            else

            "Cannot re-embed: no Gemini API "
            "key configured, or a job is already running."
        )
    })


# ============================================================
# WEBSITE CHANGES
# ============================================================

@admin_bp.route(
    "/api/admin/changes",
    methods=["GET"]
)
def get_website_changes():

    state = (
        StorageService.load_scrape_state()
    )

    return jsonify({

        "changes":
            state.get(
                "changes",
                []
            ),

        "summary":
            state.get(
                "summary",
                "No changes recorded yet."
            ),

        "last_scrape":
            state.get(
                "last_scrape"
            )
    })


# ============================================================
# SETTINGS
# ============================================================

@admin_bp.route(
    "/api/admin/settings",
    methods=["POST"]
)
def update_settings():

    """
    Updates BMSIT target URL, API key,
    scrape limits and schedule.
    """

    data = (
        request.json
        or {}
    )

    new_settings = {}

    if "bmsit_url" in data:

        new_settings[
            "bmsit_url"
        ] = (
            data[
                "bmsit_url"
            ].strip()
        )

    if "scrape_max_pages" in data:

        new_settings[
            "scrape_max_pages"
        ] = int(
            data[
                "scrape_max_pages"
            ]
        )

    if "scrape_depth" in data:

        new_settings[
            "scrape_depth"
        ] = int(
            data[
                "scrape_depth"
            ]
        )

    # ========================================================
    # GEMINI API KEY
    # ========================================================

    if data.get(
        "gemini_api_key",
        ""
    ).strip():

        _set_env_var(
            "GEMINI_API_KEY",
            data[
                "gemini_api_key"
            ].strip()
        )

    if data.get(
        "gemini_api_keys",
        ""
    ).strip():

        pool = ",".join(

            part.strip()

            for part
            in re.split(
                r"[,\s;]+",
                data[
                    "gemini_api_keys"
                ]
            )

            if part.strip()
        )

        _set_env_var(
            "GEMINI_API_KEYS",
            pool
        )

    if (
        data.get(
            "gemini_api_key"
        )
        or data.get(
            "gemini_api_keys"
        )
    ):

        (
            RAGService
            .get_instance()
            ._embedder
            .pool
            .refresh()
        )

    # ========================================================
    # SCHEDULE
    # ========================================================

    if (
        "schedule_time"
        in data

        and str(
            data[
                "schedule_time"
            ]
        ).strip()
    ):

        new_settings[
            "schedule_time"
        ] = str(
            data[
                "schedule_time"
            ]
        ).strip()

    updated = (
        StorageService.save_settings(
            new_settings
        )
    )

    if (
        "schedule_time"
        in new_settings
    ):

        try:

            SchedulerService.reschedule()

        except Exception as e:

            print(
                "[Admin] Could not "
                f"reschedule daily job: {e}"
            )

    return jsonify({

        "status":
            "success",

        "settings":
            updated
    })


# ============================================================
# VIEW DOCUMENT
# ============================================================

@admin_bp.route(
    "/api/admin/document/<filename>",
    methods=["GET"]
)
def view_document(
    filename
):

    """
    Serves an uploaded document
    for browser viewing.
    """

    clean_name = secure_filename(
        filename
    )

    file_path = (
        Config.UPLOADS_DIR
        / clean_name
    )

    if not file_path.exists():

        return jsonify({
            "error":
                "Document not found"
        }), 404

    ext = (
        clean_name.rsplit(
            ".",
            1
        )[-1].lower()
        if "."
        in clean_name
        else ""
    )

    mimetypes = {

        "pdf":
            "application/pdf",

        "csv":
            "text/plain",

        "docx":
            (
                "application/vnd.openxmlformats-"
                "officedocument.wordprocessingml.document"
            ),

        "doc":
            "application/msword"
    }

    mimetype = mimetypes.get(
        ext,
        "application/octet-stream"
    )

    response = send_from_directory(

        Config.UPLOADS_DIR,

        clean_name,

        mimetype=mimetype,

        as_attachment=False
    )

    response.headers[
        "Content-Disposition"
    ] = (
        f'inline; filename="{clean_name}"'
    )

    response.headers[
        "X-Frame-Options"
    ] = "SAMEORIGIN"

    return response


# ============================================================
# DOCUMENT CONTENT
# ============================================================

@admin_bp.route(
    "/api/admin/document-content/<filename>",
    methods=["GET"]
)
def get_document_content(
    filename
):

    """
    Extracts and returns parsed text sections
    from DOCX, DOC, CSV or PDF.
    """

    clean_name = secure_filename(
        filename
    )

    file_path = (
        Config.UPLOADS_DIR
        / clean_name
    )

    if not file_path.exists():

        return jsonify({
            "error":
                "Document not found"
        }), 404

    ext = (
        clean_name.rsplit(
            ".",
            1
        )[-1].lower()
        if "."
        in clean_name
        else ""
    )

    try:

        sections = (
            DocumentParser.parse_file(
                file_path
            )
        )

        formatted = []

        for s in sections:

            meta = s.get(
                "metadata",
                {}
            )

            title = (
                meta.get(
                    "section_title"
                )
                or "Section"
            )

            if meta.get(
                "page"
            ):

                title = (
                    f"Page "
                    f"{meta.get('page')}"
                )

            formatted.append({

                "title":
                    title,

                "content":
                    s.get(
                        "content",
                        ""
                    )
            })

        return jsonify({

            "status":
                "success",

            "filename":
                clean_name,

            "type":
                ext,

            "sections":
                formatted
        })

    except Exception as e:

        return jsonify({

            "status":
                "error",

            "message": (
                "Could not extract preview: "
                f"{str(e)}"
            )

        }), 500


# ============================================================
# SOURCE PREVIEW
# ============================================================

@admin_bp.route(
    "/api/admin/source-preview/<source_id>",
    methods=["GET"]
)
def get_source_preview(
    source_id
):

    """
    Returns detailed preview content for any
    training source.

    Local data is preferred.

    Supabase persistent chunks are used
    as a fallback after Render redeployment.
    """

    history = (
        _get_persistent_history()
    )

    target = next(

        (
            h

            for h in history

            if str(
                h.get("id")
            )
            == str(
                source_id
            )
        ),

        None
    )

    if not target:

        return jsonify({
            "error":
                "Training source not found"
        }), 404

    rag = (
        RAGService.get_instance()
    )

    # ========================================================
    # LOCAL FAISS SOURCE CHUNKS
    # ========================================================

    matching_chunks = (
        rag.get_source_chunks(
            source_id
        )
    )

    # ========================================================
    # LOCAL KNOWLEDGE STORE
    # ========================================================

    stored_items = (
        KnowledgeStore.get_items(
            source_id
        )
    )

    # ========================================================
    # SUPABASE FALLBACK
    # ========================================================

    if not matching_chunks:

        matching_chunks = (
            _get_supabase_source_chunks(
                source_id
            )
        )

    # ========================================================
    # WEBSITE
    # ========================================================

    if target.get(
        "source_type"
    ) == "website":

        details = target.get(
            "details",
            {}
        )

        changes = details.get(
            "changes",
            []
        )

        return jsonify({

            "status":
                "success",

            "source_type":
                "website",

            "title":
                target.get(
                    "source",
                    "BMSIT Website"
                ),

            "date":
                target.get(
                    "date"
                ),

            "total_chunks":
                len(
                    matching_chunks
                ),

            "pages_count":
                details.get(
                    "pages_stored",
                    details.get(
                        "pages_scraped",
                        len(
                            stored_items
                        )
                    )
                ),

            "pages_stored":
                len(
                    stored_items
                ),

            "changes":
                changes,

            "stored_pages": [

                {

                    "url":
                        key,

                    "title":
                        item.get(
                            "title"
                        ),

                    "characters":
                        len(
                            item.get(
                                "text",
                                ""
                            )
                        ),

                    "updated":
                        item.get(
                            "updated"
                        ),
                }

                for key, item
                in list(
                    stored_items.items()
                )[:60]
            ],

            "sample_chunks": [

                {

                    "id":
                        c.get(
                            "chunk_id"
                        ),

                    "text": (

                        c.get(
                            "text",
                            ""
                        )[:350]

                        + (

                            "..."

                            if len(
                                c.get(
                                    "text",
                                    ""
                                )
                            ) > 350

                            else ""
                        )
                    ),

                    "metadata":
                        c.get(
                            "metadata",
                            {}
                        )
                }

                for c
                in matching_chunks[:12]
            ]
        })

    # ========================================================
    # DOCUMENT
    # ========================================================

    filename = secure_filename(

        target.get(
            "source",
            ""
        )
    )

    file_path = (
        Config.UPLOADS_DIR
        / filename
    )

    ext = (

        filename.rsplit(
            ".",
            1
        )[-1].lower()

        if "."
        in filename

        else ""
    )

    sections = []

    # ========================================================
    # FIRST: STORED TEXT
    # ========================================================

    for key, item in (
        stored_items.items()
    ):

        sections.append({

            "title": (
                item.get(
                    "title"
                )
                or key
            ),

            "content":
                item.get(
                    "text",
                    ""
                )
        })

    # ========================================================
    # SECOND: PHYSICAL FILE
    # ========================================================

    if (
        not sections
        and file_path.exists()
    ):

        try:

            raw_sections = (
                DocumentParser.parse_file(
                    file_path
                )
            )

            for s in raw_sections:

                meta = s.get(
                    "metadata",
                    {}
                )

                title = (
                    meta.get(
                        "section_title"
                    )
                    or "Section"
                )

                if meta.get(
                    "page"
                ):

                    title = (
                        f"Page "
                        f"{meta.get('page')}"
                    )

                sections.append({

                    "title":
                        title,

                    "content":
                        s.get(
                            "content",
                            ""
                        )
                })

        except Exception as e:

            print(
                "[Admin] Error parsing "
                f"file for preview: {e}"
            )

    # ========================================================
    # THIRD: SUPABASE / INDEXED CHUNKS
    # ========================================================

    if (
        not sections
        and matching_chunks
    ):

        for idx, c in enumerate(

            matching_chunks[:20]

        ):

            meta = c.get(
                "metadata",
                {}
            )

            title = (
                meta.get(
                    "section_title"
                )
                or f"Chunk #{idx + 1}"
            )

            sections.append({

                "title":
                    title,

                "content":
                    c.get(
                        "text",
                        ""
                    )
            })

    return jsonify({

        "status":
            "success",

        "source_type":
            "document",

        "title":
            filename,

        "filename":
            filename,

        "type":
            ext,

        "sections":
            sections,

        "total_chunks":
            len(
                matching_chunks
            )
    })