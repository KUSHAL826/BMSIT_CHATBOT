"""
Embedding service: Gemini Developer API with a rotating pool of API keys.

Why a pool: a single key's daily embedding quota is the binding constraint when
back-filling thousands of pages. Configure several keys and the service spreads
requests across them, parks any key that reports a quota error, and keeps going
on the others. Day-to-day delta runs then need almost no quota at all.

Request shaping (both ceilings enforced, whichever binds first):
  * at most Config.EMBED_MAX_BATCH_TEXTS inputs per request  (default 250)
  * at most Config.EMBED_MAX_BATCH_TOKENS tokens per request (default 20,000)
  * a single input is truncated at Config.EMBED_MAX_INPUT_TOKENS

Batches are dispatched concurrently with asyncio under a bounded semaphore, with
a minimum spacing per key so a burst does not trip per-minute limits.

Vector spaces are identified by model, not by key: keys are interchangeable, so
`gemini:gemini-embedding-001` is one space no matter which key produced a vector.
Switching EMBED_MODEL creates a new space and triggers a re-embed instead of
silently mixing incompatible vectors.
"""
import asyncio
import hashlib
import logging
import math
import os
import re
import threading
import time

import numpy as np

from app.config import Config
from app.services.chunker import count_tokens
from app.services.token_budget import TokenBudget
from app.services.vector_cache import VectorCache

logger = logging.getLogger(__name__)

LOCAL_PROVIDER_ID = "local:hash-v1"


class EmbeddingUnavailable(RuntimeError):
    """Base: embeddings could not be produced right now."""


class QuotaExhausted(EmbeddingUnavailable):
    """Every configured API key reported a quota error."""


class BudgetExhausted(EmbeddingUnavailable):
    """Today's self-imposed token budget is used up. Deliberate, not an error."""


class BackendUnreachable(EmbeddingUnavailable):
    """The embedding endpoint could not be reached (DNS, offline, firewall)."""


def _normalize(matrix):
    """L2-normalises rows so inner product equals cosine similarity."""
    if not len(matrix):
        return matrix
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


def _mask(key):
    """A safe label for logs and the dashboard. Never exposes the key."""
    if not key:
        return "unset"
    return f"{key[:4]}...{key[-4:]}" if len(key) > 10 else "key"


# ----------------------------------------------------------------------
# Key pool
# ----------------------------------------------------------------------
class ApiKeyPool:
    """
    Tracks every configured key, whether it is usable right now, and when it
    was last called so requests stay spaced out.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._states = {}
        self._cursor = 0
        self.refresh()

    @staticmethod
    def _configured_keys():
        raw = []
        pooled = os.getenv("GEMINI_API_KEYS", "")
        if pooled:
            raw.extend(re.split(r"[,\s;]+", pooled))
        single = os.getenv("GEMINI_API_KEY", "")
        if single:
            raw.append(single)

        seen, keys = set(), []
        for candidate in raw:
            candidate = candidate.strip()
            if not candidate or candidate == "your_api_key_here" or candidate in seen:
                continue
            seen.add(candidate)
            keys.append(candidate)
        return keys

    def refresh(self):
        """Re-reads the environment. The admin panel can add a key at runtime."""
        with self._lock:
            configured = self._configured_keys()
            for key in configured:
                self._states.setdefault(key, {
                    "key": key,
                    "label": _mask(key),
                    "cooldown_until": 0.0,
                    "last_used": 0.0,
                    "requests": 0,
                    "quota_hits": 0,
                    "errors": 0,
                })
            for known in list(self._states):
                if known not in configured:
                    self._states.pop(known, None)
            return len(self._states)

    def configured_count(self):
        with self._lock:
            return len(self._states)

    def available_count(self):
        now = time.time()
        with self._lock:
            return sum(1 for s in self._states.values() if s["cooldown_until"] <= now)

    def acquire(self):
        """
        Returns the next usable key state, round-robin, or None if every key is
        parked. Also enforces the minimum spacing between calls on one key.
        """
        with self._lock:
            states = list(self._states.values())
            if not states:
                return None

            now = time.time()
            for offset in range(len(states)):
                state = states[(self._cursor + offset) % len(states)]
                if state["cooldown_until"] > now:
                    continue
                self._cursor = (self._cursor + offset + 1) % len(states)
                wait = Config.EMBED_MIN_INTERVAL - (now - state["last_used"])
                state["last_used"] = now + max(0.0, wait)
                state["requests"] += 1
                return state, max(0.0, wait)
            return None

    def park_for_quota(self, state):
        with self._lock:
            state["cooldown_until"] = time.time() + Config.EMBED_KEY_COOLDOWN
            state["quota_hits"] += 1
        logger.warning(
            "[Embeddings] Key %s is out of quota. Parked for %s minutes; "
            "%s of %s key(s) still usable.",
            state["label"], Config.EMBED_KEY_COOLDOWN // 60,
            self.available_count(), self.configured_count(),
        )

    def park_briefly(self, state, seconds=30, network=False):
        with self._lock:
            state["cooldown_until"] = max(state["cooldown_until"], time.time() + seconds)
            state["errors"] += 1
            if network:
                state["network_errors"] = state.get("network_errors", 0) + 1

    def network_error_count(self):
        with self._lock:
            return sum(s.get("network_errors", 0) for s in self._states.values())

    def quota_hit_count(self):
        with self._lock:
            return sum(s.get("quota_hits", 0) for s in self._states.values())

    def status(self):
        now = time.time()
        with self._lock:
            return [
                {
                    "key": state["label"],
                    "usable": state["cooldown_until"] <= now,
                    "cooldown_seconds": max(0, int(state["cooldown_until"] - now)),
                    "requests": state["requests"],
                    "quota_hits": state["quota_hits"],
                    "errors": state["errors"],
                }
                for state in self._states.values()
            ]


# ----------------------------------------------------------------------
# Backends
# ----------------------------------------------------------------------
class GeminiEmbeddingBackend:
    """Gemini Developer API `batchEmbedContents`, spread across the key pool."""

    remote = True

    def __init__(self, pool):
        self.pool = pool
        self.model = Config.EMBED_MODEL
        self.provider_id = f"gemini:{self.model}"

    def available(self):
        self.pool.refresh()
        return self.pool.available_count() > 0

    def _post(self, api_key, texts, task_type, with_task_type):
        import requests

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:batchEmbedContents?key={api_key}"
        )
        payload = []
        for text in texts:
            entry = {
                "model": f"models/{self.model}",
                "content": {"parts": [{"text": text}]},
                "outputDimensionality": Config.EMBEDDING_DIM,
            }
            if with_task_type:
                entry["taskType"] = task_type
            payload.append(entry)

        return requests.post(url, json={"requests": payload}, timeout=Config.EMBED_TIMEOUT)

    def _parse(self, response, expected):
        items = response.json().get("embeddings", [])
        if len(items) != expected:
            logger.warning("[Embeddings] Got %s vectors for %s inputs.", len(items), expected)
            return None
        vectors = []
        for item in items:
            values = list(item.get("values", []))
            if len(values) >= Config.EMBEDDING_DIM:
                vectors.append(values[: Config.EMBEDDING_DIM])
            else:
                vectors.append(values + [0.0] * (Config.EMBEDDING_DIM - len(values)))
        return _normalize(np.array(vectors, dtype=np.float32))

    async def embed_batch(self, texts, task_type):
        """
        Tries the batch on each usable key in turn. Returns None only when every
        key is parked or every attempt failed.
        """
        with_task_type = True
        attempts = max(1, self.pool.configured_count()) * Config.EMBED_RETRIES

        for _ in range(attempts):
            acquired = self.pool.acquire()
            if acquired is None:
                return None
            state, wait = acquired
            if wait > 0:
                await asyncio.sleep(wait)

            try:
                response = await asyncio.to_thread(
                    self._post, state["key"], texts, task_type, with_task_type
                )
            except Exception as e:
                message = str(e)
                offline = any(
                    marker in message.lower() for marker in
                    ("getaddrinfo", "nameresolution", "failed to resolve",
                     "connection aborted", "connection refused", "timed out",
                     "max retries exceeded", "ssl")
                )
                # Log the class of failure, never the URL: it carries the key.
                logger.warning(
                    "[Embeddings] Key %s: %s error (%s)",
                    state["label"], "network" if offline else "request",
                    type(e).__name__,
                )
                self.pool.park_briefly(state, 15, network=offline)
                continue

            if response.status_code == 200:
                parsed = self._parse(response, len(texts))
                if parsed is not None:
                    return parsed
                self.pool.park_briefly(state, 5)
                continue

            if response.status_code == 400 and with_task_type:
                # Older endpoints reject taskType; retry without it.
                with_task_type = False
                continue

            if response.status_code == 429:
                if "quota" in response.text.lower():
                    self.pool.park_for_quota(state)   # daily quota: move to next key
                else:
                    self.pool.park_briefly(state, 20)  # per-minute burst
                continue

            if response.status_code in (500, 502, 503, 504):
                self.pool.park_briefly(state, 10)
                continue

            if response.status_code in (401, 403):
                logger.error(
                    "[Embeddings] Key %s was rejected (HTTP %s). Check that it is valid "
                    "and that the Generative Language API is enabled.",
                    state["label"], response.status_code,
                )
                self.pool.park_for_quota(state)
                continue

            logger.warning(
                "[Embeddings] HTTP %s from key %s: %s",
                response.status_code, state["label"], response.text[:150],
            )
            self.pool.park_briefly(state, 10)

        return None


class LocalHashEmbeddingBackend:
    """
    Offline fallback: deterministic hashed bag-of-words vectors.

    Much weaker than the real model, but it keeps the assistant usable with no
    API access. These vectors live in their own space so they never dilute model
    vectors, and they are upgraded in the background once quota returns.
    """

    provider_id = LOCAL_PROVIDER_ID
    remote = False

    def available(self):
        return True

    def _vector(self, text):
        dim = Config.EMBEDDING_DIM
        vector = np.zeros(dim, dtype=np.float32)
        for position, token in enumerate(re.findall(r"\w+", (text or "").lower())):
            digest = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
            sign = 1.0 if ((digest >> 4) % 2 == 0) else -1.0
            vector[digest % dim] += sign / math.sqrt(position + 1)
        return vector

    async def embed_batch(self, texts, task_type):
        return _normalize(np.array([self._vector(t) for t in texts], dtype=np.float32))


# ----------------------------------------------------------------------
# Service
# ----------------------------------------------------------------------
class EmbeddingService:
    _instance = None
    _instance_lock = threading.Lock()

    def __init__(self):
        self.pool = ApiKeyPool()
        self.remote_backend = GeminiEmbeddingBackend(self.pool)
        self.local = LocalHashEmbeddingBackend()
        self._by_id = {
            self.remote_backend.provider_id: self.remote_backend,
            self.local.provider_id: self.local,
        }

    @classmethod
    def get_instance(cls):
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance

    # -- introspection -------------------------------------------------
    def remote_capacity_available(self):
        return self.remote_backend.available()

    def active_backend(self):
        return self.remote_backend if self.remote_capacity_available() else self.local

    def active_provider_id(self):
        """
        The space new content should land in.

        Note this is the remote model whenever any key is configured, even if all
        keys are momentarily parked, so a temporary quota block does not silently
        redefine the target space.
        """
        if self.pool.configured_count() > 0:
            return self.remote_backend.provider_id
        return self.local.provider_id

    def backend_for(self, provider_id):
        return self._by_id.get(provider_id)

    def status(self):
        return {
            "active_provider": self.active_provider_id(),
            "model": Config.EMBED_MODEL,
            "dimensions": Config.EMBEDDING_DIM,
            "keys_configured": self.pool.configured_count(),
            "keys_usable": self.pool.available_count(),
            "keys": self.pool.status(),
            "batch_limits": {
                "max_texts_per_request": Config.EMBED_MAX_BATCH_TEXTS,
                "max_tokens_per_request": Config.EMBED_MAX_BATCH_TOKENS,
                "max_concurrent_requests": Config.EMBED_MAX_CONCURRENCY,
                "max_chunks_per_run": Config.EMBED_MAX_CHUNKS_PER_RUN,
            },
            "token_budget": TokenBudget.status(),
            "vector_cache": VectorCache.get_instance().stats(),
        }

    # -- batching ------------------------------------------------------
    @staticmethod
    def build_batches(texts, token_counts=None):
        """Groups inputs into request-sized batches under both ceilings."""
        max_texts = Config.EMBED_MAX_BATCH_TEXTS
        max_tokens = Config.EMBED_MAX_BATCH_TOKENS

        batches, current, current_tokens = [], [], 0
        for index, text in enumerate(texts):
            tokens = token_counts[index] if token_counts else count_tokens(text)
            tokens = min(tokens, Config.EMBED_MAX_INPUT_TOKENS)
            if current and (len(current) >= max_texts or current_tokens + tokens > max_tokens):
                batches.append(current)
                current, current_tokens = [], 0
            current.append(index)
            current_tokens += tokens
        if current:
            batches.append(current)
        return batches

    @staticmethod
    def _truncate(text):
        limit_chars = Config.EMBED_MAX_INPUT_TOKENS * 4
        return text if len(text) <= limit_chars else text[:limit_chars]

    # -- embedding -----------------------------------------------------
    async def _embed_async(self, texts, token_counts, progress_callback, allow_local_fallback):
        prepared = [self._truncate(t) for t in texts]
        batches = self.build_batches(prepared, token_counts)
        vectors = [None] * len(prepared)
        providers = [None] * len(prepared)

        semaphore = asyncio.Semaphore(Config.EMBED_MAX_CONCURRENCY)
        done = {"count": 0}
        exhausted = {"hit": False}

        async def run_batch(indices):
            batch_texts = [prepared[i] for i in indices]
            async with semaphore:
                result, provider = None, None

                if self.remote_backend.available():
                    result = await self.remote_backend.embed_batch(batch_texts, "RETRIEVAL_DOCUMENT")
                    provider = self.remote_backend.provider_id

                if result is None:
                    if not allow_local_fallback:
                        exhausted["hit"] = True
                        return
                    result = await self.local.embed_batch(batch_texts, "RETRIEVAL_DOCUMENT")
                    provider = self.local.provider_id

                for row, index in enumerate(indices):
                    vectors[index] = result[row]
                    providers[index] = provider

            done["count"] += len(indices)
            if progress_callback:
                try:
                    progress_callback(done["count"], len(prepared))
                except Exception:
                    pass

        await asyncio.gather(*(run_batch(batch) for batch in batches))

        if exhausted["hit"] or any(v is None for v in vectors):
            keys = self.pool.configured_count()
            if self.pool.network_error_count() and not self.pool.quota_hit_count():
                raise BackendUnreachable(
                    "Could not reach generativelanguage.googleapis.com (DNS or network "
                    "failure), so no embeddings were generated. Check the internet "
                    "connection and run again. Nothing already indexed was lost."
                )
            raise QuotaExhausted(
                f"All {keys} configured API key(s) are out of embedding quota. Work already "
                "committed is kept; the rest resumes on the next run."
            )

        return np.array(vectors, dtype=np.float32), providers, len(batches)

    def embed_documents(self, texts, token_counts=None, progress_callback=None,
                        allow_local_fallback=None):
        """
        Embeds document chunks. Returns (vectors, providers, request_count).

        Order of operations, cheapest first:
          1. the vector cache answers anything whose exact text was embedded
             before, at zero token cost;
          2. what remains is checked against today's token budget;
          3. only genuinely new text reaches the API.

        With keys configured, `allow_local_fallback` defaults to False so a quota
        wall raises rather than quietly writing weak vectors that would then look
        "done" in the registry. With no keys at all it defaults to True so the
        app still works offline.
        """
        if not texts:
            return np.zeros((0, Config.EMBEDDING_DIM), dtype=np.float32), [], 0
        if allow_local_fallback is None:
            allow_local_fallback = self.pool.configured_count() == 0

        provider = self.active_provider_id()
        counts = token_counts or [count_tokens(t) for t in texts]

        cache = VectorCache.get_instance()
        hits, misses = cache.lookup(texts, provider)

        vectors = [None] * len(texts)
        providers = [None] * len(texts)
        for position, vector in hits.items():
            vectors[position] = vector
            providers[position] = provider

        cached_tokens = sum(counts[p] for p in hits)
        if hits:
            logger.info(
                "[Embeddings] Cache covered %s of %s chunk(s), saving about %s token(s).",
                len(hits), len(texts), cached_tokens,
            )

        requests_made = 0
        if misses:
            miss_tokens = sum(counts[p] for p in misses)

            if not TokenBudget.can_afford(miss_tokens) and not allow_local_fallback:
                remaining = TokenBudget.remaining()
                raise BudgetExhausted(
                    f"This slice needs about {miss_tokens} embedding token(s) but only "
                    f"{remaining} remain in today's budget of "
                    f"{Config.EMBED_DAILY_TOKEN_BUDGET}. Stopping deliberately; the "
                    "remaining pages resume on the next run."
                )

            miss_texts = [texts[p] for p in misses]
            miss_counts = [counts[p] for p in misses]
            fresh, fresh_providers, requests_made = run_async(
                self._embed_async(
                    miss_texts, miss_counts, progress_callback, allow_local_fallback
                )
            )

            for row, position in enumerate(misses):
                vectors[position] = fresh[row]
                providers[position] = fresh_providers[row]

            # Cache only real model vectors; local fallback vectors are transient.
            cacheable = [
                (miss_texts[row], fresh[row])
                for row in range(len(misses))
                if fresh_providers[row] == provider and provider != self.local.provider_id
            ]
            if cacheable:
                cache.store(cacheable, provider)

            spent = sum(
                miss_counts[row] for row in range(len(misses))
                if fresh_providers[row] != self.local.provider_id
            )
            TokenBudget.record(spent, requests_made, cached_tokens)
        elif cached_tokens:
            TokenBudget.record(0, 0, cached_tokens)

        return np.array(vectors, dtype=np.float32), providers, requests_made

    def embed_query(self, text, provider_id):
        """
        Embeds a query in one specific space, or returns None if that space is
        not reachable. Never substitutes another model: a query vector from a
        different space produces meaningless similarity scores.
        """
        backend = self.backend_for(provider_id)
        if backend is None or not backend.available():
            return None
        return run_async(backend.embed_batch([self._truncate(text)], "RETRIEVAL_QUERY"))


def run_async(coro):
    """Runs a coroutine from sync code (Flask thread, scheduler worker, CLI)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    box = {}

    def worker():
        try:
            box["value"] = asyncio.run(coro)
        except BaseException as e:  # re-raised on the calling thread
            box["error"] = e

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    thread.join()
    if "error" in box:
        raise box["error"]
    return box.get("value")
