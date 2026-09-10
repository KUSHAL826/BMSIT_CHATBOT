import json
import os
import shutil
import threading
import uuid
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path

from app.config import Config
from app.services.fsutil import atomic_write_json, now_ist_str, read_json


class StorageService:
    """
    JSON persistence for training history, crawl state and admin settings.

    All writes are atomic and guarded by a lock, because the scrape runs on a
    background thread while Flask request threads read and write the same files.
    """

    _lock = threading.RLock()
    # Entries currently being ingested. Their "Processing" status must not be
    # rewritten by the stale-status repair below.
    _active_ingests = set()

    @staticmethod
    def _now_ist_str():
        return now_ist_str()

    # ------------------------------------------------------------------
    # Active ingest tracking
    # ------------------------------------------------------------------
    @classmethod
    def mark_active(cls, entry_id):
        with cls._lock:
            cls._active_ingests.add(entry_id)

    @classmethod
    def clear_active(cls, entry_id):
        with cls._lock:
            cls._active_ingests.discard(entry_id)

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------
    @classmethod
    def load_history(cls):
        """Loads training history, repairing any status left stale by a crash."""
        Config.ensure_directories()
        with cls._lock:
            history = read_json(Config.HISTORY_FILE, default=[])
            if not isinstance(history, list):
                return []

            from app.services.scraper import scraper_status

            updated = False
            for item in history:
                if item.get("status") != "Processing":
                    continue
                if item.get("id") in cls._active_ingests:
                    continue  # genuinely in progress right now
                if item.get("source_type") == "website" and scraper_status.is_running:
                    continue
                # Left over from an interrupted run.
                if item.get("chunk_count", 0) > 0:
                    item["status"] = "Indexed"
                else:
                    item["status"] = "Interrupted"
                    item["changes"] = (
                        "Previous run was interrupted before indexing finished. "
                        "Stored content is intact - run it again to re-index."
                    )
                updated = True

            if updated:
                cls._save_history_unlocked(history)
            return history

    @classmethod
    def _save_history_unlocked(cls, history):
        atomic_write_json(Config.HISTORY_FILE, history)

    @classmethod
    def save_history(cls, history):
        Config.ensure_directories()
        with cls._lock:
            cls._save_history_unlocked(history)

    @classmethod
    def get_history_entry(cls, entry_id):
        return next((h for h in cls.load_history() if h.get("id") == entry_id), None)

    @classmethod
    def add_history_entry(cls, source_name, source_type, file_type, status="Processing",
                          chunk_count=0, changes="Initial Ingestion", details=None, entry_id=None):
        """Adds a new training history record."""
        with cls._lock:
            history = cls.load_history()
            entry = {
                "id": entry_id or str(uuid.uuid4())[:8],
                "date": now_ist_str(),
                "source": source_name,
                "source_type": source_type,
                "file_type": file_type,
                "status": status,
                "chunk_count": chunk_count,
                "changes": changes,
                "details": details or {},
            }
            history.insert(0, entry)
            cls._save_history_unlocked(history)
            return entry

    @classmethod
    def upsert_history_entry(cls, entry_id, source_name, source_type, file_type,
                             status="Processing", chunk_count=None, changes=None, details=None):
        """
        Creates the entry if it does not exist, otherwise updates it in place and
        moves it to the top of the list.

        This is what stops every scrape and every re-upload from piling up a new
        row whose chunks then get wiped by the next run.
        """
        with cls._lock:
            history = cls.load_history()
            existing = next((h for h in history if h.get("id") == entry_id), None)

            if existing is None:
                entry = {
                    "id": entry_id,
                    "date": now_ist_str(),
                    "source": source_name,
                    "source_type": source_type,
                    "file_type": file_type,
                    "status": status,
                    "chunk_count": chunk_count or 0,
                    "changes": changes or "Initial ingestion",
                    "details": details or {},
                    "first_indexed": now_ist_str(),
                }
                history.insert(0, entry)
            else:
                entry = existing
                entry["date"] = now_ist_str()
                entry["source"] = source_name
                entry["source_type"] = source_type
                entry["file_type"] = file_type
                entry["status"] = status
                entry.setdefault("first_indexed", entry.get("date", now_ist_str()))
                if chunk_count is not None:
                    entry["chunk_count"] = chunk_count
                if changes is not None:
                    entry["changes"] = changes
                if details is not None:
                    merged = entry.get("details", {})
                    merged.update(details)
                    entry["details"] = merged
                history.remove(entry)
                history.insert(0, entry)

            cls._save_history_unlocked(history)
            return entry

    @classmethod
    def update_history_entry(cls, entry_id, **updates):
        """Updates an existing training history record."""
        with cls._lock:
            history = cls.load_history()
            for item in history:
                if item.get("id") == entry_id:
                    item.update(updates)
                    break
            cls._save_history_unlocked(history)

    @classmethod
    def delete_history_entry(cls, entry_id):
        """Deletes an entry from history and returns the deleted record."""
        with cls._lock:
            history = cls.load_history()
            deleted_entry = None
            new_history = []
            for item in history:
                if item.get("id") == entry_id and deleted_entry is None:
                    deleted_entry = item
                else:
                    new_history.append(item)
            cls._save_history_unlocked(new_history)
            return deleted_entry

    # ------------------------------------------------------------------
    # Scrape state
    # ------------------------------------------------------------------
    @classmethod
    def load_scrape_state(cls):
        """Loads previous website crawl state and content hashes."""
        Config.ensure_directories()
        with cls._lock:
            state = read_json(Config.SCRAPE_STATE_FILE, default=None)
            if not isinstance(state, dict):
                return {"pages": {}, "changes": [], "last_scrape": None}
            state.setdefault("pages", {})
            state.setdefault("changes", [])
            state.setdefault("last_scrape", None)
            return state

    @classmethod
    def save_scrape_state(cls, state):
        """Saves website crawl state and change logs."""
        Config.ensure_directories()
        with cls._lock:
            atomic_write_json(Config.SCRAPE_STATE_FILE, state)

    @classmethod
    def reset_scrape_state(cls):
        """Clears crawl hashes so the next scrape re-fetches and re-indexes everything."""
        cls.save_scrape_state({"pages": {}, "changes": [], "last_scrape": None,
                               "summary": "Crawl state reset after website knowledge was deleted."})

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------
    @classmethod
    def load_settings(cls):
        """Loads persistent admin runtime settings."""
        Config.ensure_directories()
        defaults = {
            "bmsit_url": Config.BMSIT_DEFAULT_URL,
            "scrape_depth": Config.SCRAPE_DEPTH,
            "scrape_max_pages": Config.SCRAPE_MAX_PAGES,
            "schedule_time": Config.SCHEDULE_TIME,
            "schedule_timezone": Config.SCHEDULE_TIMEZONE,
        }
        saved = read_json(Config.SETTINGS_FILE, default=None)
        if isinstance(saved, dict):
            defaults.update(saved)
        return defaults

    @classmethod
    def save_settings(cls, new_settings):
        """Saves persistent admin runtime settings."""
        with cls._lock:
            current = cls.load_settings()
            current.update(new_settings)
            atomic_write_json(Config.SETTINGS_FILE, current)
            return current
