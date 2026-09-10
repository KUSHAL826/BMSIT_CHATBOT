"""
Configuration.

There are no default values here. Every setting must be present in .env.
If one is missing, startup fails immediately and names the missing variables,
instead of silently running with a value you did not choose. That silent
fallback is what previously made ingestion behaviour hard to explain: a stale
.env meant the code was using different limits than the ones written down.

Only filesystem paths and API keys are exempt: paths are structural, and keys
are optional because the app still runs (with reduced quality) without them.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

_MISSING = []


def _raw(name):
    value = os.getenv(name)
    if value is None or value.strip() == "":
        _MISSING.append(name)
        return None
    return value.strip()


def _req_str(name):
    return _raw(name)


def _req_int(name):
    value = _raw(name)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        _MISSING.append(f"{name} (must be an integer, got {value!r})")
        return None


def _req_float(name):
    value = _raw(name)
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        _MISSING.append(f"{name} (must be a number, got {value!r})")
        return None


def _req_bool(name):
    value = _raw(name)
    if value is None:
        return None
    if value.lower() in ("true", "1", "yes", "on"):
        return True
    if value.lower() in ("false", "0", "no", "off"):
        return False
    _MISSING.append(f"{name} (must be true or false, got {value!r})")
    return None


def _req_list(name):
    value = _raw(name)
    if value is None:
        return None
    items = [part.strip() for part in value.split(",") if part.strip()]
    if not items:
        _MISSING.append(f"{name} (must list at least one value)")
    return items


class Config:
    # ------------------------------------------------------------------
    # Paths (structural, not configurable)
    # ------------------------------------------------------------------
    BASE_DIR = BASE_DIR
    DATA_DIR = BASE_DIR / "data"
    UPLOADS_DIR = DATA_DIR / "uploads"
    SCRAPED_DIR = DATA_DIR / "scraped"
    INDEX_DIR = DATA_DIR / "index"
    KNOWLEDGE_DIR = DATA_DIR / "knowledge"
    HISTORY_FILE = DATA_DIR / "history.json"
    SCRAPE_STATE_FILE = DATA_DIR / "scrape_state.json"
    SETTINGS_FILE = DATA_DIR / "settings.json"
    REGISTRY_FILE = DATA_DIR / "ingestion_registry.json"

    # ------------------------------------------------------------------
    # Keys (optional: absence degrades quality, it is not a config error)
    # ------------------------------------------------------------------
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
    GEMINI_API_KEYS = os.getenv("GEMINI_API_KEYS", "").strip()

    # ------------------------------------------------------------------
    # Required settings
    # ------------------------------------------------------------------
    SECRET_KEY = _req_str("SECRET_KEY")

    # Crawler
    BMSIT_DEFAULT_URL = _req_str("BMSIT_DEFAULT_URL")
    SCRAPE_DEPTH = _req_int("SCRAPE_DEPTH")
    SCRAPE_MAX_PAGES = _req_int("SCRAPE_MAX_PAGES")
    SCRAPE_BOILERPLATE_RATIO = _req_float("SCRAPE_BOILERPLATE_RATIO")
    SCRAPE_REQUEST_TIMEOUT = _req_int("SCRAPE_REQUEST_TIMEOUT")
    SCRAPE_DELAY = _req_float("SCRAPE_DELAY")
    # Pages with less unique text than this are not worth embedding.
    SCRAPE_MIN_PAGE_TOKENS = _req_int("SCRAPE_MIN_PAGE_TOKENS")

    # Scheduler
    SCHEDULE_TIME = _req_str("SCHEDULE_TIME")
    SCHEDULE_TIMEZONE = _req_str("SCHEDULE_TIMEZONE")

    # Chunking
    CHUNK_TARGET_TOKENS = _req_int("CHUNK_TARGET_TOKENS")
    CHUNK_MAX_TOKENS = _req_int("CHUNK_MAX_TOKENS")
    CHUNK_MIN_TOKENS = _req_int("CHUNK_MIN_TOKENS")
    CHUNK_OVERLAP_TOKENS = _req_int("CHUNK_OVERLAP_TOKENS")

    # Embeddings
    EMBED_MODEL = _req_str("EMBED_MODEL")
    EMBEDDING_DIM = _req_int("EMBEDDING_DIM")
    EMBED_MAX_BATCH_TEXTS = _req_int("EMBED_MAX_BATCH_TEXTS")
    EMBED_MAX_BATCH_TOKENS = _req_int("EMBED_MAX_BATCH_TOKENS")
    EMBED_MAX_INPUT_TOKENS = _req_int("EMBED_MAX_INPUT_TOKENS")
    EMBED_MAX_CONCURRENCY = _req_int("EMBED_MAX_CONCURRENCY")
    EMBED_TIMEOUT = _req_int("EMBED_TIMEOUT")
    EMBED_RETRIES = _req_int("EMBED_RETRIES")
    EMBED_MIN_INTERVAL = _req_float("EMBED_MIN_INTERVAL")
    EMBED_KEY_COOLDOWN = _req_int("EMBED_KEY_COOLDOWN")
    EMBED_COMMIT_SLICE_CHUNKS = _req_int("EMBED_COMMIT_SLICE_CHUNKS")
    EMBED_MAX_CHUNKS_PER_RUN = _req_int("EMBED_MAX_CHUNKS_PER_RUN")
    # Reuse vectors for text already embedded before. This is the main reason a
    # large crawl fits inside a free quota.
    EMBED_CACHE_ENABLED = _req_bool("EMBED_CACHE_ENABLED")
    EMBED_CACHE_MAX_VECTORS = _req_int("EMBED_CACHE_MAX_VECTORS")
    # Self-imposed daily ceiling. 0 = unlimited (discover the real limit by
    # hitting HTTP 429, which is slower and less predictable).
    EMBED_DAILY_TOKEN_BUDGET = _req_int("EMBED_DAILY_TOKEN_BUDGET")

    # Retrieval / chat
    TOP_K = _req_int("TOP_K")
    CHAT_TIMEOUT = _req_int("CHAT_TIMEOUT")
    CHAT_MODELS = _req_list("CHAT_MODELS")

    # Server
    HOST = _req_str("HOST")
    PORT = _req_int("PORT")
    DEBUG = _req_bool("DEBUG")
    LOG_LEVEL = _req_str("LOG_LEVEL")

    @classmethod
    def validate(cls):
        """Raises if any required variable is missing or malformed."""
        if _MISSING:
            listing = "\n".join(f"  - {name}" for name in _MISSING)
            raise RuntimeError(
                "Configuration is incomplete. These variables must be set in "
                f"{BASE_DIR / '.env'}:\n{listing}\n\n"
                "Copy .env.example to .env and fill it in. There are no code "
                "defaults by design."
            )

        if cls.CHUNK_MAX_TOKENS < cls.CHUNK_TARGET_TOKENS:
            raise RuntimeError("CHUNK_MAX_TOKENS must be >= CHUNK_TARGET_TOKENS.")
        if cls.CHUNK_OVERLAP_TOKENS >= cls.CHUNK_TARGET_TOKENS:
            raise RuntimeError("CHUNK_OVERLAP_TOKENS must be < CHUNK_TARGET_TOKENS.")
        if cls.EMBED_MAX_INPUT_TOKENS < cls.CHUNK_MAX_TOKENS:
            raise RuntimeError("EMBED_MAX_INPUT_TOKENS must be >= CHUNK_MAX_TOKENS.")
        if cls.EMBED_MAX_BATCH_TOKENS < cls.EMBED_MAX_INPUT_TOKENS:
            raise RuntimeError("EMBED_MAX_BATCH_TOKENS must be >= EMBED_MAX_INPUT_TOKENS.")
        return True

    @classmethod
    def missing(cls):
        return list(_MISSING)

    @classmethod
    def ensure_directories(cls):
        for directory in (cls.DATA_DIR, cls.UPLOADS_DIR, cls.SCRAPED_DIR,
                          cls.INDEX_DIR, cls.KNOWLEDGE_DIR):
            directory.mkdir(exist_ok=True, parents=True)


Config.validate()
