"""
Configuration.

Resolution order for every setting:

    1. the process environment  (Render dashboard, render.yaml, shell export)
    2. a local .env file        (development convenience, optional)
    3. the documented default in _SPEC below

A physical .env file is NEVER required. `load_dotenv` is called with
override=False and only when the file exists, so platform-supplied variables
always win over a stale local file.

Settings marked REQUIRED in _SPEC have no default and startup fails if they are
absent from both sources. Every value, defaulted or not, is still type-checked
and cross-checked in validate(), so a malformed value fails loudly rather than
being silently ignored.
"""
import os
import secrets
from pathlib import Path

from dotenv import dotenv_values, load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = BASE_DIR / ".env"

# Load .env for local development only.
#   override=False -> a variable already present in the environment (which is
#   how Render supplies configuration) is never replaced by the file.
_ENV_FILE_KEYS = set()
if ENV_FILE.exists():
    load_dotenv(dotenv_path=ENV_FILE, override=False)
    _ENV_FILE_KEYS = {k for k, v in dotenv_values(ENV_FILE).items() if v is not None}

# Render sets RENDER=true in every service. Used only to choose a safe default
# bind address: containers must listen on all interfaces, local runs should not.
ON_PLATFORM = bool(os.getenv("RENDER") or os.getenv("PORT"))

REQUIRED = object()  # sentinel: no default, must be supplied

# name -> (kind, default)
# Defaults mirror .env.example exactly. Change them in one place only.
_SPEC = {
    # Server
    "HOST": ("str", "0.0.0.0" if ON_PLATFORM else "127.0.0.1"),
    "PORT": ("int", 5000),
    "DEBUG": ("bool", False),
    "LOG_LEVEL": ("str", "INFO"),
    # Empty means "generate one at startup". Flask uses it only to sign session
    # cookies and flash messages; this app has no login, so a per-process value
    # is harmless. A warning is emitted so it is never a silent surprise.
    "SECRET_KEY": ("str", ""),

    # Crawler
    "BMSIT_DEFAULT_URL": ("str", "https://bmsit.ac.in/"),
    "SCRAPE_DEPTH": ("int", 4),
    "SCRAPE_MAX_PAGES": ("int", 800),
    "SCRAPE_BOILERPLATE_RATIO": ("float", 0.30),
    "SCRAPE_REQUEST_TIMEOUT": ("int", 10),
    "SCRAPE_DELAY": ("float", 0.12),
    "SCRAPE_MIN_PAGE_TOKENS": ("int", 80),

    # Scheduler
    "SCHEDULE_TIME": ("str", "07:00"),
    "SCHEDULE_TIMEZONE": ("str", "Asia/Kolkata"),

    # Chunking
    "CHUNK_TARGET_TOKENS": ("int", 450),
    "CHUNK_MAX_TOKENS": ("int", 550),
    "CHUNK_MIN_TOKENS": ("int", 60),
    "CHUNK_OVERLAP_TOKENS": ("int", 30),

    # Embeddings
    "EMBED_MODEL": ("str", "gemini-embedding-001"),
    "EMBEDDING_DIM": ("int", 768),
    "EMBED_MAX_BATCH_TEXTS": ("int", 100),
    "EMBED_MAX_BATCH_TOKENS": ("int", 20000),
    "EMBED_MAX_INPUT_TOKENS": ("int", 2048),
    "EMBED_MAX_CONCURRENCY": ("int", 2),
    "EMBED_TIMEOUT": ("int", 60),
    "EMBED_RETRIES": ("int", 2),
    "EMBED_MIN_INTERVAL": ("float", 0.5),
    "EMBED_KEY_COOLDOWN": ("int", 3600),
    "EMBED_COMMIT_SLICE_CHUNKS": ("int", 150),
    "EMBED_MAX_CHUNKS_PER_RUN": ("int", 600),
    "EMBED_CACHE_ENABLED": ("bool", True),
    "EMBED_CACHE_MAX_VECTORS": ("int", 200000),
    "EMBED_DAILY_TOKEN_BUDGET": ("int", 400000),

    # Retrieval / chat
    "TOP_K": ("int", 8),
    "CHAT_TIMEOUT": ("int", 45),
    "CHAT_MODELS": ("list", "gemini-3.5-flash-lite,gemini-3.6-flash,gemini-3.8-flash"),
}

_MISSING = []    # required, absent from both sources
_INVALID = []    # present but unparseable
_WARNINGS = []   # non-fatal notes surfaced at startup
_SOURCES = {}    # name -> "environment" | ".env file" | "default"


def _coerce(name, kind, raw):
    if kind == "str":
        return raw
    if kind == "int":
        return int(raw)
    if kind == "float":
        return float(raw)
    if kind == "bool":
        lowered = raw.lower()
        if lowered in ("true", "1", "yes", "on"):
            return True
        if lowered in ("false", "0", "no", "off"):
            return False
        raise ValueError("expected true or false")
    if kind == "list":
        items = [part.strip() for part in raw.split(",") if part.strip()]
        if not items:
            raise ValueError("expected at least one comma-separated value")
        return items
    raise AssertionError(f"unknown kind {kind!r} for {name}")


def _get(name):
    """Resolves one setting from environment, then .env, then default."""
    kind, default = _SPEC[name]
    raw = os.getenv(name)

    if raw is None or raw.strip() == "":
        if default is REQUIRED:
            _MISSING.append(name)
            _SOURCES[name] = "MISSING"
            return None
        _SOURCES[name] = "default"
        # Defaults are given as native values, except list which reuses parsing.
        return _coerce(name, kind, default) if isinstance(default, str) and kind == "list" \
            else default

    _SOURCES[name] = ".env file" if name in _ENV_FILE_KEYS else "environment"
    try:
        return _coerce(name, kind, raw.strip())
    except ValueError as e:
        _INVALID.append(f"{name}={raw.strip()!r} ({e})")
        return None


def _generated_secret_key():
    """
    Fallback SECRET_KEY, regenerated each start.

    Deliberately not fatal: it lets `python run.py` work on a freshly created
    host with no variables configured at all. Set SECRET_KEY explicitly to keep
    signed cookies valid across restarts.
    """
    _WARNINGS.append(
        "SECRET_KEY was not set, so a random one was generated for this process. "
        "Set SECRET_KEY to keep session cookies valid across restarts."
    )
    _SOURCES["SECRET_KEY"] = "generated"
    return secrets.token_urlsafe(32)


class Config:
    # ------------------------------------------------------------------
    # Paths (structural, not configurable)
    # ------------------------------------------------------------------
    BASE_DIR = BASE_DIR
    ENV_FILE = ENV_FILE
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
    # API keys. Read live from the environment rather than frozen here, so the
    # admin panel can add a key at runtime. Absence degrades quality; it is not
    # a configuration error.
    # ------------------------------------------------------------------
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
    GEMINI_API_KEYS = os.getenv("GEMINI_API_KEYS", "").strip()

    # ------------------------------------------------------------------
    # Server
    # ------------------------------------------------------------------
    HOST = _get("HOST")
    PORT = _get("PORT")
    DEBUG = _get("DEBUG")
    LOG_LEVEL = _get("LOG_LEVEL")
    SECRET_KEY = _get("SECRET_KEY") or _generated_secret_key()

    # ------------------------------------------------------------------
    # Crawler
    # ------------------------------------------------------------------
    BMSIT_DEFAULT_URL = _get("BMSIT_DEFAULT_URL")
    SCRAPE_DEPTH = _get("SCRAPE_DEPTH")
    SCRAPE_MAX_PAGES = _get("SCRAPE_MAX_PAGES")
    SCRAPE_BOILERPLATE_RATIO = _get("SCRAPE_BOILERPLATE_RATIO")
    SCRAPE_REQUEST_TIMEOUT = _get("SCRAPE_REQUEST_TIMEOUT")
    SCRAPE_DELAY = _get("SCRAPE_DELAY")
    SCRAPE_MIN_PAGE_TOKENS = _get("SCRAPE_MIN_PAGE_TOKENS")

    # ------------------------------------------------------------------
    # Scheduler
    # ------------------------------------------------------------------
    SCHEDULE_TIME = _get("SCHEDULE_TIME")
    SCHEDULE_TIMEZONE = _get("SCHEDULE_TIMEZONE")

    # ------------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------------
    CHUNK_TARGET_TOKENS = _get("CHUNK_TARGET_TOKENS")
    CHUNK_MAX_TOKENS = _get("CHUNK_MAX_TOKENS")
    CHUNK_MIN_TOKENS = _get("CHUNK_MIN_TOKENS")
    CHUNK_OVERLAP_TOKENS = _get("CHUNK_OVERLAP_TOKENS")

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------
    EMBED_MODEL = _get("EMBED_MODEL")
    EMBEDDING_DIM = _get("EMBEDDING_DIM")
    EMBED_MAX_BATCH_TEXTS = _get("EMBED_MAX_BATCH_TEXTS")
    EMBED_MAX_BATCH_TOKENS = _get("EMBED_MAX_BATCH_TOKENS")
    EMBED_MAX_INPUT_TOKENS = _get("EMBED_MAX_INPUT_TOKENS")
    EMBED_MAX_CONCURRENCY = _get("EMBED_MAX_CONCURRENCY")
    EMBED_TIMEOUT = _get("EMBED_TIMEOUT")
    EMBED_RETRIES = _get("EMBED_RETRIES")
    EMBED_MIN_INTERVAL = _get("EMBED_MIN_INTERVAL")
    EMBED_KEY_COOLDOWN = _get("EMBED_KEY_COOLDOWN")
    EMBED_COMMIT_SLICE_CHUNKS = _get("EMBED_COMMIT_SLICE_CHUNKS")
    EMBED_MAX_CHUNKS_PER_RUN = _get("EMBED_MAX_CHUNKS_PER_RUN")
    EMBED_CACHE_ENABLED = _get("EMBED_CACHE_ENABLED")
    EMBED_CACHE_MAX_VECTORS = _get("EMBED_CACHE_MAX_VECTORS")
    EMBED_DAILY_TOKEN_BUDGET = _get("EMBED_DAILY_TOKEN_BUDGET")

    # ------------------------------------------------------------------
    # Retrieval / chat
    # ------------------------------------------------------------------
    TOP_K = _get("TOP_K")
    CHAT_TIMEOUT = _get("CHAT_TIMEOUT")
    CHAT_MODELS = _get("CHAT_MODELS")

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    @classmethod
    def validate(cls):
        """
        Raises if a required setting is absent, a supplied value is malformed, or
        two settings contradict each other.
        """
        problems = []

        if _MISSING:
            problems.append(
                "Missing required setting(s):\n"
                + "\n".join(f"    - {name}" for name in _MISSING)
            )
        if _INVALID:
            problems.append(
                "Malformed value(s):\n"
                + "\n".join(f"    - {item}" for item in _INVALID)
            )

        if problems:
            where = (
                "Set them as environment variables on your host "
                "(Render: Dashboard -> your service -> Environment)"
            )
            if not ON_PLATFORM:
                where = f"Set them in the environment, or in {ENV_FILE}"
            raise RuntimeError(
                "Configuration is incomplete.\n\n"
                + "\n\n".join(problems)
                + f"\n\n{where}.\n"
                "Resolution order is: process environment, then .env if present, "
                "then built-in defaults. A .env file is not required.\n"
                "See .env.example for the full list with documented defaults."
            )

        # Cross-field checks. These apply to defaults too.
        if cls.CHUNK_MAX_TOKENS < cls.CHUNK_TARGET_TOKENS:
            raise RuntimeError("CHUNK_MAX_TOKENS must be >= CHUNK_TARGET_TOKENS.")
        if cls.CHUNK_OVERLAP_TOKENS >= cls.CHUNK_TARGET_TOKENS:
            raise RuntimeError("CHUNK_OVERLAP_TOKENS must be < CHUNK_TARGET_TOKENS.")
        if cls.EMBED_MAX_INPUT_TOKENS < cls.CHUNK_MAX_TOKENS:
            raise RuntimeError("EMBED_MAX_INPUT_TOKENS must be >= CHUNK_MAX_TOKENS.")
        if cls.EMBED_MAX_BATCH_TOKENS < cls.EMBED_MAX_INPUT_TOKENS:
            raise RuntimeError("EMBED_MAX_BATCH_TOKENS must be >= EMBED_MAX_INPUT_TOKENS.")
        if cls.EMBEDDING_DIM <= 0:
            raise RuntimeError("EMBEDDING_DIM must be positive.")
        if cls.PORT <= 0 or cls.PORT > 65535:
            raise RuntimeError("PORT must be between 1 and 65535.")
        return True

    @classmethod
    def missing(cls):
        return list(_MISSING)

    @classmethod
    def invalid(cls):
        return list(_INVALID)

    @classmethod
    def warnings(cls):
        return list(_WARNINGS)

    @classmethod
    def sources(cls):
        """Where each setting came from. Useful for diagnosing a deploy."""
        return dict(_SOURCES)

    @classmethod
    def source_summary(cls):
        counts = {}
        for origin in _SOURCES.values():
            counts[origin] = counts.get(origin, 0) + 1
        parts = [f"{count} from {origin}" for origin, count in sorted(counts.items())]
        return ", ".join(parts) + (
            f" (.env {'found' if ENV_FILE.exists() else 'not present'})"
        )

    @classmethod
    def ensure_directories(cls):
        for directory in (cls.DATA_DIR, cls.UPLOADS_DIR, cls.SCRAPED_DIR,
                          cls.INDEX_DIR, cls.KNOWLEDGE_DIR):
            directory.mkdir(exist_ok=True, parents=True)


Config.validate()
