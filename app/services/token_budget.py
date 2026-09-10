"""
Daily embedding token ledger.

Free-tier embedding quota is a daily allowance. Discovering it by crashing into
HTTP 429 wastes the run's remaining time and leaves the backfill in an
unpredictable place. Instead this tracks what has been spent today and stops the
run deliberately once EMBED_DAILY_TOKEN_BUDGET is reached.

The result is a predictable multi-day backfill: each day embeds a known slice,
records it, and the delta registry means tomorrow simply continues. Set the
budget slightly below your real quota to leave room for query embeddings.

Stored in data/embed_ledger.json, keyed by date in the configured timezone.
"""
import logging
import threading
from datetime import datetime
from zoneinfo import ZoneInfo

from app.config import Config
from app.services.fsutil import atomic_write_json, read_json

logger = logging.getLogger(__name__)


class TokenBudget:
    _lock = threading.RLock()

    @staticmethod
    def _today():
        try:
            return datetime.now(ZoneInfo(Config.SCHEDULE_TIMEZONE)).strftime("%Y-%m-%d")
        except Exception:
            return datetime.now().strftime("%Y-%m-%d")

    @classmethod
    def _path(cls):
        return Config.DATA_DIR / "embed_ledger.json"

    @classmethod
    def _load(cls):
        data = read_json(cls._path(), default=None)
        if not isinstance(data, dict):
            data = {}
        data.setdefault("days", {})
        return data

    @classmethod
    def spent_today(cls):
        with cls._lock:
            day = cls._load()["days"].get(cls._today(), {})
            return int(day.get("tokens", 0)), int(day.get("requests", 0))

    @classmethod
    def remaining(cls):
        """Tokens left in today's budget, or None when unlimited."""
        budget = Config.EMBED_DAILY_TOKEN_BUDGET
        if not budget:
            return None
        tokens, _ = cls.spent_today()
        return max(0, budget - tokens)

    @classmethod
    def can_afford(cls, tokens):
        left = cls.remaining()
        return True if left is None else tokens <= left

    @classmethod
    def record(cls, tokens, requests=1, cached_tokens=0):
        """Adds usage to today's total. `cached_tokens` is what the cache saved."""
        with cls._lock:
            Config.ensure_directories()
            data = cls._load()
            today = cls._today()
            day = data["days"].setdefault(
                today, {"tokens": 0, "requests": 0, "cached_tokens": 0}
            )
            day["tokens"] = int(day.get("tokens", 0)) + int(tokens)
            day["requests"] = int(day.get("requests", 0)) + int(requests)
            day["cached_tokens"] = int(day.get("cached_tokens", 0)) + int(cached_tokens)

            # Keep the ledger small: 60 days of history is plenty.
            if len(data["days"]) > 60:
                for stale in sorted(data["days"])[:-60]:
                    data["days"].pop(stale, None)

            atomic_write_json(cls._path(), data)
            return day

    @classmethod
    def status(cls):
        tokens, requests = cls.spent_today()
        budget = Config.EMBED_DAILY_TOKEN_BUDGET
        data = cls._load()["days"].get(cls._today(), {})
        return {
            "date": cls._today(),
            "budget": budget or None,
            "tokens_spent": tokens,
            "tokens_remaining": cls.remaining(),
            "requests": requests,
            "tokens_saved_by_cache": int(data.get("cached_tokens", 0)),
            "exhausted": bool(budget) and tokens >= budget,
        }
