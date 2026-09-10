"""
Ingestion registry: the state that makes delta-scraping possible.

One JSON document, `data/ingestion_registry.json`, written atomically. For every
tracked item (a URL for the website source, a section for an uploaded document)
it records:

    content_hash   SHA-256 of the cleaned text - the change signal
    last_scraped   when it was last fetched
    last_embedded  when embeddings were last generated for it
    chunk_ids      the exact vector ids produced from it
    chunk_count    how many vectors it owns
    token_count    total tokens embedded, for cost/quota visibility
    embed_provider which embedding space those vectors live in
    status         indexed | unchanged | pending | failed

`chunk_ids` is the important part: on a change we delete precisely those vector
records and nothing else, instead of rebuilding a whole source.

A JSON file is deliberate here. The project has no database, the registry is
written by a single process, and it stays small (a few hundred KB for 1,000
URLs). Swapping in SQLite or Redis later only requires reimplementing this class.
"""
import hashlib
import threading

from app.config import Config
from app.services.fsutil import atomic_write_json, now_ist_str, read_json

REGISTRY_VERSION = 2

STATUS_INDEXED = "indexed"
STATUS_UNCHANGED = "unchanged"
STATUS_PENDING = "pending"
STATUS_FAILED = "failed"


def content_hash(text):
    """SHA-256 of the cleaned text. Prefixed so the algorithm is self-describing."""
    return "sha256:" + hashlib.sha256((text or "").encode("utf-8")).hexdigest()


class IngestionRegistry:
    _lock = threading.RLock()

    # ------------------------------------------------------------------
    # Load / save
    # ------------------------------------------------------------------
    @classmethod
    def _path(cls):
        return Config.REGISTRY_FILE

    @classmethod
    def load(cls):
        Config.ensure_directories()
        data = read_json(cls._path(), default=None)
        if not isinstance(data, dict) or "sources" not in data:
            return {"version": REGISTRY_VERSION, "updated": None, "sources": {}}
        data.setdefault("version", REGISTRY_VERSION)
        data.setdefault("sources", {})
        return data

    @classmethod
    def _save(cls, data):
        data["version"] = REGISTRY_VERSION
        data["updated"] = now_ist_str()
        atomic_write_json(cls._path(), data)

    @classmethod
    def _source(cls, data, source_id, source_type="website", source_name=None):
        sources = data.setdefault("sources", {})
        entry = sources.setdefault(source_id, {
            "source_id": source_id,
            "source_type": source_type,
            "source_name": source_name or source_id,
            "created": now_ist_str(),
            "items": {},
        })
        if source_name:
            entry["source_name"] = source_name
        entry["source_type"] = source_type
        entry.setdefault("items", {})
        return entry

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------
    @classmethod
    def get_item(cls, source_id, item_key):
        with cls._lock:
            data = cls.load()
            return data.get("sources", {}).get(source_id, {}).get("items", {}).get(item_key)

    @classmethod
    def items(cls, source_id):
        with cls._lock:
            return dict(cls.load().get("sources", {}).get(source_id, {}).get("items", {}))

    @classmethod
    def known_keys(cls, source_id):
        return set(cls.items(source_id).keys())

    @classmethod
    def chunk_ids(cls, source_id, item_key):
        item = cls.get_item(source_id, item_key) or {}
        return list(item.get("chunk_ids", []))

    @classmethod
    def chunk_ids_for_keys(cls, source_id, item_keys):
        """All vector ids owned by the given items, as a flat set."""
        items = cls.items(source_id)
        stale = set()
        for key in item_keys:
            stale.update(items.get(key, {}).get("chunk_ids", []))
        return stale

    @classmethod
    def decide_batch(cls, source_id, hashes, expected_provider=None):
        """
        Delta decision for many items using ONE registry read.

        `hashes` maps item_key -> fresh content hash.
        Returns (needs_work, skip, reasons) where reasons maps key -> why.

        Calling needs_embedding per item re-read the whole registry file each
        time, which is quadratic: a 3,000 page crawl meant 3,000 full reads of a
        multi-megabyte JSON file before any embedding even started.
        """
        snapshot = cls.items(source_id)
        needs_work, skip, reasons = [], [], {}

        for key, fresh_hash in hashes.items():
            item = snapshot.get(key)
            reason = cls._decide_one(item, fresh_hash, expected_provider)
            reasons[key] = reason
            if reason == "unchanged":
                skip.append(key)
            else:
                needs_work.append(key)

        return needs_work, skip, reasons

    @staticmethod
    def _decide_one(item, fresh_hash, expected_provider):
        if item is None:
            return "new"
        if item.get("content_hash") != fresh_hash:
            return "changed"
        if not item.get("chunk_ids"):
            return "no-vectors"
        if item.get("status") not in (STATUS_INDEXED, STATUS_UNCHANGED):
            return "previous-run-incomplete"
        if expected_provider and item.get("embed_provider") != expected_provider:
            return "embedding-model-changed"
        return "unchanged"

    @classmethod
    def needs_embedding(cls, source_id, item_key, fresh_hash, expected_provider=None):
        """
        The delta decision for one item.

        Returns (needs_work, reason). Skipping is only safe when the hash matches,
        vectors were actually produced, and they live in the embedding space we
        would use today - switching embedding model invalidates old vectors.
        """
        item = cls.get_item(source_id, item_key)
        if item is None:
            return True, "new"
        if item.get("content_hash") != fresh_hash:
            return True, "changed"
        if not item.get("chunk_ids"):
            return True, "no-vectors"
        if item.get("status") not in (STATUS_INDEXED, STATUS_UNCHANGED):
            return True, "previous-run-incomplete"
        if expected_provider and item.get("embed_provider") != expected_provider:
            return True, "embedding-model-changed"
        return False, "unchanged"

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------
    @classmethod
    def record_indexed(cls, source_id, item_key, fresh_hash, chunk_ids, token_count,
                       embed_provider, source_type="website", source_name=None, title=None):
        """Records a successful (re-)embedding of one item."""
        with cls._lock:
            data = cls.load()
            source = cls._source(data, source_id, source_type, source_name)
            existing = source["items"].get(item_key, {})
            source["items"][item_key] = {
                "item_key": item_key,
                "title": title or existing.get("title") or item_key,
                "content_hash": fresh_hash,
                "hash_algo": "sha256",
                "last_scraped": now_ist_str(),
                "last_embedded": now_ist_str(),
                "chunk_ids": list(chunk_ids),
                "chunk_count": len(chunk_ids),
                "token_count": int(token_count),
                "embed_provider": embed_provider,
                "status": STATUS_INDEXED,
                "first_seen": existing.get("first_seen", now_ist_str()),
                "revision": int(existing.get("revision", 0)) + 1,
            }
            cls._save(data)
            return source["items"][item_key]

    @classmethod
    def record_many(cls, source_id, entries, source_type="website", source_name=None):
        """
        Records many items in ONE load-modify-save cycle.

        `entries` is an iterable of dicts with keys: item_key, content_hash,
        chunk_ids, token_count, embed_provider, title.

        Calling record_indexed in a loop rewrites the whole registry per item,
        which is quadratic and unusably slow for hundreds of pages.
        """
        entries = list(entries)
        if not entries:
            return 0

        with cls._lock:
            data = cls.load()
            source = cls._source(data, source_id, source_type, source_name)
            stamp = now_ist_str()

            for entry in entries:
                key = entry["item_key"]
                existing = source["items"].get(key, {})
                chunk_ids = list(entry.get("chunk_ids", []))
                source["items"][key] = {
                    "item_key": key,
                    "title": entry.get("title") or existing.get("title") or key,
                    "content_hash": entry["content_hash"],
                    "hash_algo": "sha256",
                    "last_scraped": stamp,
                    "last_embedded": stamp,
                    "chunk_ids": chunk_ids,
                    "chunk_count": len(chunk_ids),
                    "token_count": int(entry.get("token_count", 0)),
                    "embed_provider": entry.get("embed_provider"),
                    "status": STATUS_INDEXED,
                    "first_seen": existing.get("first_seen", stamp),
                    "revision": int(existing.get("revision", 0)) + 1,
                }

            cls._save(data)
            return len(entries)

    @classmethod
    def mark_unchanged(cls, source_id, item_keys, source_type="website", source_name=None):
        """Refreshes the scrape timestamp for items that did not change."""
        if not item_keys:
            return 0
        with cls._lock:
            data = cls.load()
            source = cls._source(data, source_id, source_type, source_name)
            touched = 0
            for key in item_keys:
                item = source["items"].get(key)
                if not item:
                    continue
                item["last_scraped"] = now_ist_str()
                item["status"] = STATUS_INDEXED
                touched += 1
            cls._save(data)
            return touched

    @classmethod
    def mark_failed(cls, source_id, item_keys, reason, source_type="website"):
        if not item_keys:
            return 0
        with cls._lock:
            data = cls.load()
            source = cls._source(data, source_id, source_type)
            for key in item_keys:
                item = source["items"].setdefault(key, {"item_key": key, "chunk_ids": []})
                item["status"] = STATUS_FAILED
                item["failure_reason"] = str(reason)[:300]
                item["last_scraped"] = now_ist_str()
            cls._save(data)
            return len(item_keys)

    @classmethod
    def forget_items(cls, source_id, item_keys):
        with cls._lock:
            data = cls.load()
            source = data.get("sources", {}).get(source_id)
            if not source:
                return 0
            removed = 0
            for key in item_keys:
                if source.get("items", {}).pop(key, None) is not None:
                    removed += 1
            if removed:
                cls._save(data)
            return removed

    @classmethod
    def forget_source(cls, source_id):
        with cls._lock:
            data = cls.load()
            if data.get("sources", {}).pop(source_id, None) is None:
                return False
            cls._save(data)
            return True

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    @classmethod
    def stats(cls, source_id=None):
        data = cls.load()
        sources = data.get("sources", {})
        selected = {source_id: sources.get(source_id, {})} if source_id else sources

        tracked = chunks = tokens = failed = 0
        providers = {}
        last_embedded = None
        for source in selected.values():
            for item in (source.get("items") or {}).values():
                tracked += 1
                chunks += int(item.get("chunk_count", 0))
                tokens += int(item.get("token_count", 0))
                if item.get("status") == STATUS_FAILED:
                    failed += 1
                provider = item.get("embed_provider") or "unknown"
                providers[provider] = providers.get(provider, 0) + 1
                stamp = item.get("last_embedded")
                if stamp and (last_embedded is None or stamp > last_embedded):
                    last_embedded = stamp

        return {
            "tracked_items": tracked,
            "tracked_chunks": chunks,
            "embedded_tokens": tokens,
            "failed_items": failed,
            "providers": providers,
            "last_embedded": last_embedded,
            "registry_updated": data.get("updated"),
        }
