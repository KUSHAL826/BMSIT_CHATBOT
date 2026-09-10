"""
Content-addressed vector cache: the main defence against token exhaustion.

A college website reuses enormous amounts of text. The same course description,
the same fee table, the same department blurb appear on dozens of pages. Even
after boilerplate stripping, a 1,000 page crawl typically contains only a few
hundred pages' worth of *distinct* text.

This cache keys vectors by a hash of the exact text that was embedded. Before
calling the API, the pipeline asks the cache; anything seen before costs zero
tokens. Crucially this is done WITHOUT sharing chunk records between pages:
every page still owns its own chunk rows, so per-page update and delete stay
exact. Only the expensive part - the API call - is shared.

It also makes re-uploading a document nearly free, and makes a full rebuild
after an interrupted run cost only the tokens for text that is genuinely new.

Layout:
    data/index/vector_cache.npy    the vectors, one row each
    data/index/vector_cache.json   text hash + provider -> row number
"""
import hashlib
import logging
import threading

import numpy as np

from app.config import Config
from app.services.fsutil import atomic_write_bytes, atomic_write_json, read_json

logger = logging.getLogger(__name__)


def text_fingerprint(text):
    """Hash of the exact string that gets embedded."""
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()


class VectorCache:
    _instance = None
    _instance_lock = threading.Lock()

    def __init__(self):
        self._lock = threading.RLock()
        self._dim = Config.EMBEDDING_DIM
        self._rows = {}      # "provider|fingerprint" -> row index
        self._vectors = np.zeros((0, self._dim), dtype=np.float32)
        self._hits = 0
        self._misses = 0
        self._loaded = False

    @classmethod
    def get_instance(cls):
        with cls._instance_lock:
            if cls._instance is None:
                instance = cls()
                instance.load()
                cls._instance = instance
        return cls._instance

    @property
    def _map_file(self):
        return Config.INDEX_DIR / "vector_cache.json"

    @property
    def _vec_file(self):
        return Config.INDEX_DIR / "vector_cache.npy"

    @staticmethod
    def _key(fingerprint, provider):
        return f"{provider}|{fingerprint}"

    def load(self):
        with self._lock:
            Config.ensure_directories()
            rows = read_json(self._map_file, default={}) or {}
            vectors = np.zeros((0, self._dim), dtype=np.float32)

            if self._vec_file.exists():
                try:
                    loaded = np.load(str(self._vec_file))
                    if loaded.ndim == 2 and loaded.shape[1] == self._dim:
                        vectors = loaded.astype(np.float32)
                except Exception as e:
                    logger.warning("[Cache] Could not read vector cache: %s", e)

            # Drop any mapping that points past the vector array.
            rows = {k: v for k, v in rows.items() if isinstance(v, int) and 0 <= v < len(vectors)}

            self._rows = rows
            self._vectors = vectors
            self._loaded = True
            if rows:
                logger.info("[Cache] Loaded %s cached vector(s).", len(rows))

    def _persist(self):
        atomic_write_json(self._map_file, self._rows)
        vectors = self._vectors

        def write(tmp_path):
            with open(tmp_path, "wb") as handle:
                np.save(handle, vectors)

        atomic_write_bytes(self._vec_file, write)

    def lookup(self, texts, provider):
        """
        Returns (hits, misses) where hits maps position -> vector and misses is
        the list of positions that must be embedded.
        """
        if not Config.EMBED_CACHE_ENABLED:
            return {}, list(range(len(texts)))

        hits, misses = {}, []
        with self._lock:
            for position, text in enumerate(texts):
                row = self._rows.get(self._key(text_fingerprint(text), provider))
                if row is None:
                    misses.append(position)
                else:
                    hits[position] = self._vectors[row]

            self._hits += len(hits)
            self._misses += len(misses)

        return hits, misses

    def store(self, pairs, provider):
        """`pairs` is an iterable of (text, vector). Duplicates are ignored."""
        if not Config.EMBED_CACHE_ENABLED:
            return 0

        with self._lock:
            new_vectors, added = [], 0
            for text, vector in pairs:
                key = self._key(text_fingerprint(text), provider)
                if key in self._rows:
                    continue
                if len(self._rows) >= Config.EMBED_CACHE_MAX_VECTORS:
                    logger.warning(
                        "[Cache] Reached EMBED_CACHE_MAX_VECTORS (%s); not caching further.",
                        Config.EMBED_CACHE_MAX_VECTORS,
                    )
                    break
                self._rows[key] = len(self._vectors) + len(new_vectors)
                new_vectors.append(np.asarray(vector, dtype=np.float32))
                added += 1

            if added:
                block = np.vstack(new_vectors)
                self._vectors = (
                    block if len(self._vectors) == 0 else np.vstack([self._vectors, block])
                )
                self._persist()
            return added

    def stats(self):
        with self._lock:
            total = self._hits + self._misses
            return {
                "enabled": Config.EMBED_CACHE_ENABLED,
                "cached_vectors": len(self._rows),
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(self._hits / total, 3) if total else None,
            }

    def clear(self):
        with self._lock:
            self._rows = {}
            self._vectors = np.zeros((0, self._dim), dtype=np.float32)
            self._hits = self._misses = 0
            self._persist()
