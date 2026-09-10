"""
Permanent knowledge store.

Every piece of knowledge the bot can use lives here as raw text, one JSON
file per source, under data/knowledge/:

    data/knowledge/web-bmsit.json          -> every scraped page, keyed by URL
    data/knowledge/doc-<hash>.json         -> one uploaded document, keyed by section

Design rules (these are the bug fixes):

1. Source ids are DERIVED, not random. The same URL or the same filename always
   maps to the same id, so a re-scrape or a re-upload UPDATES the existing entry
   instead of creating a duplicate one.
2. Nothing is ever removed implicitly. A new scrape only touches the pages it
   actually re-crawled; pages it did not see keep their stored text and their
   vectors. Content disappears only when an admin deletes that source.
3. Deleting a source removes only that source's file, its uploaded document and
   its vectors. Other sources are untouched.
"""
import hashlib
import re
import threading

from app.config import Config
from app.services.fsutil import atomic_write_json, now_ist_str, read_json


class KnowledgeStore:
    # A single, stable source for all website knowledge.
    WEBSITE_SOURCE_ID = "web-bmsit"
    WEBSITE_SOURCE_NAME = "BMSIT Official Website"

    _lock = threading.RLock()

    # ------------------------------------------------------------------
    # Identity helpers
    # ------------------------------------------------------------------
    @staticmethod
    def document_source_id(filename):
        """Stable id for an uploaded document, derived from its filename."""
        key = (filename or "").strip().lower()
        return "doc-" + hashlib.md5(key.encode("utf-8")).hexdigest()[:10]

    @staticmethod
    def content_hash(text):
        return hashlib.md5((text or "").encode("utf-8")).hexdigest()

    @classmethod
    def is_website_source(cls, source_id):
        return source_id == cls.WEBSITE_SOURCE_ID

    @classmethod
    def _path(cls, source_id):
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", str(source_id))
        return Config.KNOWLEDGE_DIR / f"{safe}.json"

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------
    @classmethod
    def load_source(cls, source_id):
        """Returns the stored record for a source, or None."""
        Config.ensure_directories()
        return read_json(cls._path(source_id), default=None)

    @classmethod
    def get_items(cls, source_id):
        record = cls.load_source(source_id)
        return (record or {}).get("items", {})

    @classmethod
    def item_keys(cls, source_id):
        return set(cls.get_items(source_id).keys())

    @classmethod
    def list_sources(cls):
        Config.ensure_directories()
        records = []
        for path in sorted(Config.KNOWLEDGE_DIR.glob("*.json")):
            record = read_json(path, default=None)
            if record and record.get("source_id"):
                records.append(record)
        return records

    @classmethod
    def stats(cls):
        """Totals across the whole store, for the admin dashboard."""
        total_items = 0
        total_chars = 0
        sources = cls.list_sources()
        for record in sources:
            items = record.get("items", {})
            total_items += len(items)
            total_chars += sum(len(i.get("text", "")) for i in items.values())
        return {
            "stored_sources": len(sources),
            "stored_items": total_items,
            "stored_characters": total_chars,
        }

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------
    @classmethod
    def save_source(cls, record):
        with cls._lock:
            Config.ensure_directories()
            record["updated"] = now_ist_str()
            atomic_write_json(cls._path(record["source_id"]), record)
        return record

    @classmethod
    def upsert_items(cls, source_id, source_type, source_name, items, replace_missing=False):
        """
        Merges `items` into the stored source.

        items: {item_key: {"title": str, "text": str, "metadata": dict}}

        Returns a dict describing what actually changed:
            changed_keys  - new or modified items (these need re-embedding)
            new_keys      - items never seen before
            updated_keys  - items whose text changed
            removed_keys  - items dropped (only when replace_missing=True)
            unchanged     - count of items left exactly as they were
            total_items   - total items now stored for this source

        `replace_missing=False` is the important default: a partial crawl can
        never delete pages it simply did not visit this time.
        """
        with cls._lock:
            record = cls.load_source(source_id) or {
                "source_id": source_id,
                "source_type": source_type,
                "source_name": source_name,
                "created": now_ist_str(),
                "items": {},
            }
            record["source_type"] = source_type
            record["source_name"] = source_name
            stored = record.setdefault("items", {})

            new_keys, updated_keys, unchanged = [], [], 0

            for key, incoming in items.items():
                text = (incoming.get("text") or "").strip()
                if not text:
                    continue
                digest = cls.content_hash(text)
                existing = stored.get(key)

                if existing is None:
                    new_keys.append(key)
                elif existing.get("hash") != digest:
                    updated_keys.append(key)
                else:
                    # Same content: refresh the "last seen" stamp only.
                    existing["last_seen"] = now_ist_str()
                    unchanged += 1
                    continue

                stored[key] = {
                    "item_key": key,
                    "title": incoming.get("title") or key,
                    "text": text,
                    "hash": digest,
                    "metadata": incoming.get("metadata", {}),
                    "updated": now_ist_str(),
                    "last_seen": now_ist_str(),
                    "first_seen": (existing or {}).get("first_seen", now_ist_str()),
                }

            removed_keys = []
            if replace_missing:
                for key in list(stored.keys()):
                    if key not in items:
                        removed_keys.append(key)
                        stored.pop(key, None)

            cls.save_source(record)

            return {
                "changed_keys": new_keys + updated_keys,
                "new_keys": new_keys,
                "updated_keys": updated_keys,
                "removed_keys": removed_keys,
                "unchanged": unchanged,
                "total_items": len(stored),
                "record": record,
            }

    @classmethod
    def delete_items(cls, source_id, keys):
        """Removes specific items from a source. Used for granular deletes."""
        with cls._lock:
            record = cls.load_source(source_id)
            if not record:
                return 0
            removed = 0
            for key in keys:
                if record.get("items", {}).pop(key, None) is not None:
                    removed += 1
            if removed:
                cls.save_source(record)
            return removed

    @classmethod
    def delete_source(cls, source_id):
        """Deletes one source's stored content. Nothing else is affected."""
        with cls._lock:
            path = cls._path(source_id)
            if not path.exists():
                return False
            try:
                path.unlink()
                return True
            except Exception as e:
                print(f"[KnowledgeStore] Could not delete {path.name}: {e}")
                return False
