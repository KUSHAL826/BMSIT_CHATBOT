"""
Vector store and retrieval.

Storage model: `chunks.json` (metadata) and `embeddings.npy` (vectors) are the
source of truth and must stay the same length. FAISS indexes are rebuilt from
them in memory, one index per embedding space, and written to disk only for
inspection. That removes the class of bug where a truncated index file silently
misaligned every search result.

Embedding spaces are keyed by provider id (for example `gemini:gemini-embedding-001`
or `local:hash-v1`). A query is only ever compared against vectors from its own
space, so switching model or falling back to local vectors cannot corrupt ranking.

Every mutation embeds first and swaps second, so an interrupted or failing run
can never leave the knowledge base empty.
"""
import difflib
import hashlib
import logging
import re
import threading

import faiss
import numpy as np

from app.config import Config
from app.services.chunker import count_tokens, split_into_token_chunks
from app.services.embedding_service import LOCAL_PROVIDER_ID, EmbeddingService
from app.services.fsutil import atomic_write_bytes, atomic_write_json, read_json
from app.services.supabase_service import SupabaseService

logger = logging.getLogger(__name__)

PROVIDER_LEGACY = "legacy:unknown"  # pre-registry vectors: lexical matching only

STOP_WORDS = {
    "what", "is", "the", "are", "of", "in", "for", "to", "at", "and", "a", "an", "on",
    "tell", "me", "about", "can", "you", "does", "when", "where", "how", "do", "did",
    "bmsit", "bms", "college", "institute", "there", "any", "please", "give", "list",
}


class RAGService:
    _instance = None
    _instance_lock = threading.Lock()

    def __init__(self):
        self._lock = threading.RLock()
        self._dim = Config.EMBEDDING_DIM
        self._chunks = []                 # aligned 1:1 with _embeddings rows
        self._embeddings = np.zeros((0, self._dim), dtype=np.float32)
        self._spaces = {}                 # provider id -> (faiss index, [positions])
        self._index = None                # combined index, for stats only
        self._embedder = EmbeddingService.get_instance()
        self._reembed_status = {"running": False, "done": 0, "total": 0, "message": "idle"}

    @classmethod
    def get_instance(cls):
        with cls._instance_lock:
            if cls._instance is None:
                instance = cls()
                instance.initialize()
                cls._instance = instance
        return cls._instance

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    @property
    def _index_file(self):
        return Config.INDEX_DIR / "faiss.index"

    @property
    def _chunks_file(self):
        return Config.INDEX_DIR / "chunks.json"

    @property
    def _emb_file(self):
        return Config.INDEX_DIR / "embeddings.npy"

    def initialize(self):
        Config.ensure_directories()
        supabase = SupabaseService.get_instance()
        with self._lock:
            chunks = []
            embeddings = np.zeros((0, self._dim), dtype=np.float32)
            loaded_from_supabase = False

            if supabase.is_enabled:
                logger.info("[RAG] Supabase is enabled. Loading existing vectors from Supabase PostgreSQL...")
                sp_chunks, sp_embeddings = supabase.fetch_all_chunks()
                if sp_chunks and len(sp_chunks) == len(sp_embeddings):
                    chunks = sp_chunks
                    embeddings = sp_embeddings
                    loaded_from_supabase = True
                    logger.info("[RAG] Successfully loaded %s chunk(s) from Supabase PostgreSQL.", len(chunks))

            # Fallback to local files if Supabase is disabled or empty
            if not loaded_from_supabase:
                chunks = read_json(self._chunks_file, default=[])
                if not isinstance(chunks, list):
                    chunks = []

                if self._emb_file.exists():
                    try:
                        loaded = np.load(str(self._emb_file))
                        if loaded.ndim == 2 and loaded.shape[1] == self._dim:
                            embeddings = loaded.astype(np.float32)
                        else:
                            logger.error(
                                "[RAG] embeddings.npy is %s but EMBEDDING_DIM is %s. Vectors "
                                "discarded; content will be re-embedded.", loaded.shape, self._dim,
                            )
                            chunks = []
                    except Exception as e:
                        logger.error("[RAG] Could not read embeddings.npy: %s", e)

                # If local data exists and Supabase is enabled but empty, backfill local data to Supabase
                if supabase.is_enabled and chunks and len(chunks) == len(embeddings):
                    logger.info("[RAG] Backfilling %s local chunk(s) to empty Supabase vector store...", len(chunks))
                    supabase.insert_chunks(chunks, embeddings)

            repaired = False
            if len(chunks) != len(embeddings):
                keep = min(len(chunks), len(embeddings))
                logger.error(
                    "[RAG] Integrity problem: %s chunks vs %s vectors. Truncating both to %s.",
                    len(chunks), len(embeddings), keep,
                )
                chunks, embeddings, repaired = chunks[:keep], embeddings[:keep], True

            for chunk in chunks:
                if not chunk.get("embed_provider"):
                    chunk["embed_provider"] = PROVIDER_LEGACY
                    repaired = True
                if not chunk.get("item_key"):
                    meta = chunk.get("metadata") or {}
                    chunk["item_key"] = meta.get("url") or meta.get("section_title") or "main"
                    repaired = True

            self._chunks = chunks
            self._embeddings = embeddings
            self._rebuild_spaces()
            if repaired or loaded_from_supabase:
                self._persist()

            logger.info(
                "[RAG] Loaded %s chunks (Supabase: %s). Spaces: %s",
                len(self._chunks), loaded_from_supabase, {p: len(v[1]) for p, v in self._spaces.items()},
            )

    def _rebuild_spaces(self):
        by_provider = {}
        for position, chunk in enumerate(self._chunks):
            by_provider.setdefault(chunk.get("embed_provider", PROVIDER_LEGACY), []).append(position)

        spaces = {}
        for provider, positions in by_provider.items():
            index = faiss.IndexFlatIP(self._dim)
            if positions:
                index.add(np.ascontiguousarray(self._embeddings[positions]))
            spaces[provider] = (index, positions)
        self._spaces = spaces

        combined = faiss.IndexFlatIP(self._dim)
        if len(self._embeddings):
            combined.add(np.ascontiguousarray(self._embeddings))
        self._index = combined

    def _persist(self):
        Config.ensure_directories()
        atomic_write_json(self._chunks_file, self._chunks)
        embeddings = self._embeddings

        def write_npy(tmp_path):
            with open(tmp_path, "wb") as handle:
                np.save(handle, embeddings)

        atomic_write_bytes(self._emb_file, write_npy)
        atomic_write_bytes(self._index_file, lambda tmp: faiss.write_index(self._index, tmp))

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def get_stats(self):
        with self._lock:
            source_ids = {c.get("source_id") for c in self._chunks if c.get("source_id")}
            provider_counts = {p: len(v[1]) for p, v in self._spaces.items()}
            active = self._embedder.active_provider_id()
            stale = sum(count for provider, count in provider_counts.items() if provider != active)
            return {
                "total_chunks": len(self._chunks),
                "total_vectors": self._index.ntotal if self._index is not None else 0,
                "indexed_sources_count": len(source_ids),
                "embedding_spaces": provider_counts,
                "active_provider": active,
                "needs_reembed": stale,
                "reembed": dict(self._reembed_status),
                "embeddings": self._embedder.status(),
                "total_tokens": sum(int(c.get("tokens", 0)) for c in self._chunks),
            }

    def get_source_chunks(self, source_id):
        with self._lock:
            return [dict(c) for c in self._chunks if c.get("source_id") == source_id]

    def count_source_chunks(self, source_id):
        with self._lock:
            return sum(1 for c in self._chunks if c.get("source_id") == source_id)

    def source_item_keys(self, source_id):
        with self._lock:
            return {
                c.get("item_key") for c in self._chunks
                if c.get("source_id") == source_id and c.get("item_key")
            }

    # ------------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------------
    def chunk_text(self, text, source_id, source_name, source_type, metadata_extra=None,
                   item_key=None, context_header=None):
        """
        Splits text into 250-300 token chunks with stable, unique ids.

        `context_header` (page title and URL, or file and section) is attached to
        the chunk rather than fed through the splitter, so it is available for
        embedding and citation without producing a junk header-only chunk.
        """
        pieces = split_into_token_chunks(text)
        if not pieces:
            return []

        key = item_key or "main"
        key_digest = hashlib.md5(str(key).encode("utf-8")).hexdigest()[:10]

        chunks = []
        for ordinal, piece in enumerate(pieces, start=1):
            # The text digest makes the id specific to this revision of the
            # content. Re-chunking identical text yields the same id (idempotent),
            # while edited text yields a new one, so the registry's recorded ids
            # always identify exactly the vectors that are live.
            text_digest = hashlib.md5(piece["text"].encode("utf-8")).hexdigest()[:8]
            chunks.append({
                "chunk_id": f"{source_id}::{key_digest}::{ordinal}::{text_digest}",
                "source_id": source_id,
                "source_name": source_name,
                "source_type": source_type,
                "item_key": key,
                "text": piece["text"],
                "tokens": piece["tokens"],
                "context_header": context_header or "",
                "metadata": metadata_extra or {},
            })
        return chunks

    @staticmethod
    def embedding_text(chunk):
        """
        What actually gets embedded: the chunk body ONLY.

        The page title and URL are deliberately excluded. They live in
        `context_header` and are used for citations and for the title-match boost
        during scoring, but keeping them out of the embedded string means the
        same paragraph appearing on twenty different pages produces one identical
        string - so the vector cache serves nineteen of them for free. Including
        the header made every copy unique and cost twenty times the tokens for no
        retrieval benefit.
        """
        return chunk.get("text", "")

    @staticmethod
    def _dedupe_key(chunk):
        normalized = re.sub(r"\s+", " ", chunk.get("text", "")).strip().lower()
        return hashlib.md5(normalized.encode("utf-8")).hexdigest()

    @staticmethod
    def dedupe_chunks(source_id, new_chunks, keys_being_replaced=None, replace_all=False):
        """
        Drops repeated chunks WITHIN each item (page or section).

        Deliberately scoped per item, not per source. Deduplicating across pages
        looks like a saving but breaks the delta model: a page's chunks would be
        discarded because a different page happened to contain the same text, so
        the registry would record few or zero vector ids for it, the page would
        have no retrievable content of its own, and every later run would try to
        re-embed it forever. Cross-page repetition is handled where it belongs -
        by stripping site boilerplate during crawling.
        """
        seen_per_item, unique, dropped = {}, [], 0
        for chunk in new_chunks:
            item_key = chunk.get("item_key", "main")
            seen = seen_per_item.setdefault(item_key, set())
            digest = RAGService._dedupe_key(chunk)
            if digest in seen:
                dropped += 1
                continue
            seen.add(digest)
            unique.append(chunk)

        if dropped:
            logger.info(
                "[RAG] Dropped %s chunk(s) repeated within their own item for %s.",
                dropped, source_id,
            )
        return unique, dropped

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------
    def _prepare(self, new_chunks, progress_callback=None):
        """Embeds outside the lock and tags each chunk with its embedding space."""
        texts = [self.embedding_text(c) for c in new_chunks]
        tokens = [int(c.get("tokens") or count_tokens(c.get("text", ""))) for c in new_chunks]
        vectors, providers, requests_made = self._embedder.embed_documents(
            texts, token_counts=tokens, progress_callback=progress_callback
        )
        if len(vectors) != len(new_chunks):
            raise RuntimeError(
                f"Embedding returned {len(vectors)} vectors for {len(new_chunks)} chunks."
            )

        prepared = []
        for chunk, provider in zip(new_chunks, providers):
            enriched = dict(chunk)
            enriched["embed_provider"] = provider or LOCAL_PROVIDER_ID
            enriched.setdefault("item_key", "main")
            prepared.append(enriched)

        logger.info(
            "[RAG] Embedded %s chunk(s) in %s request(s).", len(prepared), requests_made
        )
        return prepared, vectors

    def _append(self, prepared, vectors):
        self._chunks.extend(prepared)
        if len(self._embeddings) == 0:
            self._embeddings = vectors
        elif len(vectors):
            self._embeddings = np.vstack([self._embeddings, vectors])

    def _drop(self, predicate):
        """Removes chunks matching `predicate`. Caller must hold the lock."""
        keep_positions, keep_chunks, removed = [], [], 0
        for position, chunk in enumerate(self._chunks):
            if predicate(chunk):
                removed += 1
            else:
                keep_positions.append(position)
                keep_chunks.append(chunk)

        if removed:
            self._chunks = keep_chunks
            self._embeddings = (
                self._embeddings[keep_positions]
                if keep_positions and len(self._embeddings)
                else np.zeros((0, self._dim), dtype=np.float32)
            )
        return removed

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------
    def index_chunks(self, new_chunks, progress_callback=None):
        """Embeds and appends chunks to Supabase and memory."""
        if not new_chunks:
            return 0
        source_id = new_chunks[0].get("source_id")
        new_chunks, _ = self.dedupe_chunks(source_id, new_chunks)
        
        # Filter out chunks that already exist in memory (prevent duplicates)
        existing_ids = {c.get("chunk_id") for c in self._chunks if c.get("chunk_id")}
        new_chunks = [c for c in new_chunks if c.get("chunk_id") not in existing_ids]

        if not new_chunks:
            return 0

        prepared, vectors = self._prepare(new_chunks, progress_callback)
        supabase = SupabaseService.get_instance()
        if supabase.is_enabled:
            supabase.insert_chunks(prepared, vectors)

        with self._lock:
            self._append(prepared, vectors)
            self._rebuild_spaces()
            self._persist()
        return len(prepared)

    def apply_source_update(self, source_id, new_chunks, replace_keys=None, replace_all=False,
                            stale_chunk_ids=None, progress_callback=None):
        """
        Appends new embeddings/chunks for a source without deleting existing chunks.

        When new/changed pages are scraped, add their new embeddings/chunks to the
        existing Supabase vector store — never delete or replace previous chunks.
        Reuse existing chunks when the content hash is unchanged and prevent duplicates.
        """
        supabase = SupabaseService.get_instance()
        
        # Prevent duplicates: skip chunks whose chunk_id or exact text already exists
        existing_ids = {c.get("chunk_id") for c in self._chunks if c.get("chunk_id")}
        existing_digests = {self._dedupe_key(c) for c in self._chunks}

        filtered_chunks = []
        if new_chunks:
            deduped_chunks, _ = self.dedupe_chunks(source_id, new_chunks, replace_keys, replace_all)
            for chunk in deduped_chunks:
                cid = chunk.get("chunk_id")
                digest = self._dedupe_key(chunk)
                if cid in existing_ids or digest in existing_digests:
                    continue
                filtered_chunks.append(chunk)

        prepared, vectors = [], np.zeros((0, self._dim), dtype=np.float32)
        if filtered_chunks:
            prepared, vectors = self._prepare(filtered_chunks, progress_callback)

        removed = 0
        # Requirement: When new/changed pages are scraped, add their new embeddings/chunks
        # to the existing Supabase vector store — never delete or replace previous chunks.
        if not supabase.is_enabled and not replace_all:
            # Only in non-Supabase strict local mode with explicit replacement:
            pass  # Keep previous chunks preserved in all cases as required

        if supabase.is_enabled and prepared:
            supabase.insert_chunks(prepared, vectors)

        with self._lock:
            self._append(prepared, vectors)
            self._rebuild_spaces()
            self._persist()
            total = sum(1 for c in self._chunks if c.get("source_id") == source_id)

        logger.info(
            "[RAG] %s: -%s chunk(s), +%s chunk(s), %s active. (Supabase: %s)",
            source_id, removed, len(prepared), total, supabase.is_enabled,
        )
        return {
            "removed": removed,
            "added": len(prepared),
            "total": total,
            "chunk_ids": [c["chunk_id"] for c in prepared],
        }

    # ------------------------------------------------------------------
    # Deletion
    # ------------------------------------------------------------------
    def delete_source(self, source_id):
        """Removes every vector for one source. Other sources are untouched."""
        supabase = SupabaseService.get_instance()
        if supabase.is_enabled:
            supabase.delete_source(source_id)

        with self._lock:
            removed = self._drop(lambda c: c.get("source_id") == source_id)
            if removed:
                self._rebuild_spaces()
                self._persist()
        if removed:
            logger.info("[RAG] Deleted source %s (%s chunks).", source_id, removed)
        return removed

    def reset_index(self):
        """Empties the vector store completely. Used by the admin reset."""
        supabase = SupabaseService.get_instance()
        if supabase.is_enabled:
            supabase.reset_table()

        with self._lock:
            removed = len(self._chunks)
            self._chunks = []
            self._embeddings = np.zeros((0, self._dim), dtype=np.float32)
            self._vocabulary = None
            self._rebuild_spaces()
            self._persist()
        logger.warning("[RAG] Index reset: %s chunk(s) removed.", removed)
        return removed

    def delete_chunk_ids(self, chunk_ids):
        """Removes specific vector records by id, as recorded in the registry."""
        targets = set(chunk_ids or [])
        if not targets:
            return 0
        with self._lock:
            removed = self._drop(lambda c: c.get("chunk_id") in targets)
            if removed:
                self._rebuild_spaces()
                self._persist()
        return removed

    def delete_source_items(self, source_id, item_keys):
        return self.apply_source_update(source_id, [], replace_keys=item_keys)["removed"]

    def drop_sources(self, source_ids):
        targets = set(source_ids or [])
        if not targets:
            return 0
        with self._lock:
            removed = self._drop(lambda c: c.get("source_id") in targets)
            if removed:
                self._rebuild_spaces()
                self._persist()
        return removed

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------
    def remap_source_ids(self, mapping):
        """Rewrites chunk source ids, preserving vectors (no re-embedding)."""
        if not mapping:
            return 0
        changed = 0
        with self._lock:
            for chunk in self._chunks:
                old = chunk.get("source_id")
                new = mapping.get(old)
                if not new or new == old:
                    continue
                chunk["source_id"] = new
                chunk_id = chunk.get("chunk_id", "")
                chunk["chunk_id"] = (
                    new + chunk_id[len(old):] if chunk_id.startswith(f"{old}::")
                    else f"{new}::{chunk_id}"
                )
                changed += 1
            if changed:
                self._rebuild_spaces()
                self._persist()
        return changed

    def grouped_source_text(self, source_id):
        """Reassembles chunk text per item key, for backfilling the content store."""
        grouped = {}
        with self._lock:
            for chunk in self._chunks:
                if chunk.get("source_id") != source_id:
                    continue
                key = chunk.get("item_key") or "main"
                bucket = grouped.setdefault(key, {
                    "texts": [], "metadata": chunk.get("metadata", {}),
                    "title": chunk.get("source_name"),
                })
                bucket["texts"].append(chunk.get("text", ""))
        return {
            key: {"title": v["title"], "text": "\n\n".join(v["texts"]), "metadata": v["metadata"]}
            for key, v in grouped.items()
        }

    def pending_reembed_count(self):
        """Chunks whose vectors are not in the currently active embedding space."""
        active = self._embedder.active_provider_id()
        with self._lock:
            return sum(1 for c in self._chunks if c.get("embed_provider") != active)

    def reembed_pending(self, batch_limit=None):
        """
        Re-embeds chunks that live in an older or weaker space (local fallback,
        legacy vectors, or a previous embedding model) into the active space.
        Only vectors change; stored text is untouched.
        """
        backend = self._embedder.active_backend()
        if not backend.remote:
            return {"upgraded": 0, "message": "No remote embedding backend is available."}

        active = backend.provider_id
        with self._lock:
            targets = [
                (position, self.embedding_text(chunk), int(chunk.get("tokens") or 0))
                for position, chunk in enumerate(self._chunks)
                if chunk.get("embed_provider") != active
            ]
        if not targets:
            return {"upgraded": 0, "message": f"All vectors already use {active}."}
        if batch_limit:
            targets = targets[:batch_limit]

        self._reembed_status.update({
            "running": True, "done": 0, "total": len(targets),
            "message": f"Re-embedding {len(targets)} chunk(s) with {active}...",
        })

        upgraded = 0
        try:
            vectors, providers, _ = self._embedder.embed_documents(
                [text for _, text, _ in targets],
                token_counts=[tokens for _, _, tokens in targets],
                progress_callback=lambda done, total: self._reembed_status.update({"done": done}),
            )
            with self._lock:
                for row, (position, _, _) in enumerate(targets):
                    if providers[row] != active or position >= len(self._chunks):
                        continue
                    self._embeddings[position] = vectors[row]
                    self._chunks[position]["embed_provider"] = active
                    upgraded += 1
                if upgraded:
                    self._rebuild_spaces()
                    self._persist()
        finally:
            self._reembed_status.update({
                "running": False,
                "message": f"Upgraded {upgraded} vector(s) to {active}.",
            })

        return {"upgraded": upgraded, "message": self._reembed_status["message"]}

    def reembed_pending_async(self):
        if self._reembed_status.get("running"):
            return False
        if not self._embedder.active_backend().remote or self.pending_reembed_count() == 0:
            return False
        threading.Thread(target=self.reembed_pending, daemon=True).start()
        return True

    # ------------------------------------------------------------------
    # Retrieval
    ACRONYM_SYNONYMS = {
        "hod": ["head", "associate head", "department head", "chair", "incharge", "lead"],
        "cluster": ["division", "section", "department", "unit", "cluster", "batch", "group"],
        "division": ["cluster", "section", "department", "unit", "batch"],
        "section": ["cluster", "division", "batch"],
        "dept": ["department", "division", "branch"],
        "department": ["dept", "division", "branch"],
        "cse": ["computer science", "engineering"],
        "ise": ["information science", "engineering"],
        "ece": ["electronics", "communication", "engineering"],
        "eee": ["electrical", "electronics", "engineering"],
        "mech": ["mechanical", "engineering"],
        "cv": ["civil", "engineering"],
        "civil": ["civil engineering"],
        "ai": ["artificial intelligence"],
        "ml": ["machine learning"],
        "aiml": ["artificial intelligence", "machine learning"],
        "aids": ["artificial intelligence", "data science"],
        "principal": ["head of institution", "director"],
        "fee": ["fees", "tuition", "payment", "cost", "charges"],
        "fees": ["fee", "tuition", "payment", "cost", "charges"],
        "hostel": ["accommodation", "dormitory", "residence"],
        "placement": ["placements", "recruiters", "jobs", "hiring", "offers"],
        "placements": ["placement", "recruiters", "jobs", "hiring", "offers"],
        "prof": ["professor", "faculty", "doctor", "dr"],
        "dr": ["doctor", "professor", "faculty"],
    }

    @classmethod
    def _expand_synonyms(cls, query):
        """Expands college acronyms and common terms with semantic synonyms."""
        words = re.findall(r"\w+", (query or "").lower())
        expansions = []
        for word in words:
            if word in cls.ACRONYM_SYNONYMS:
                expansions.extend(cls.ACRONYM_SYNONYMS[word])
        return " ".join(set(expansions))

    @staticmethod
    def _query_stems(query):
        words = [
            w.lower() for w in re.findall(r"\w+", query or "")
            if w.lower() not in STOP_WORDS and (len(w) > 2 or w.isdigit())
        ]
        return [w[:-1] if (w.endswith("s") and not w.endswith("ss") and len(w) > 3) else w for w in words]

    def _index_vocabulary(self):
        """
        Set of words present in the indexed content, cached and rebuilt whenever
        the chunk count changes. Used to repair misspelled queries.
        """
        with self._lock:
            cached = getattr(self, "_vocabulary", None)
            if cached and cached[0] == len(self._chunks):
                return cached[1]

            vocabulary = set()
            for chunk in self._chunks:
                for word in re.findall(r"[a-z]{4,}", chunk.get("text", "").lower()):
                    vocabulary.add(word)
                for word in re.findall(r"[a-z]{4,}", str(chunk.get("source_name", "")).lower()):
                    vocabulary.add(word)

            self._vocabulary = (len(self._chunks), vocabulary)
            return vocabulary

    def correct_terms(self, stems):
        """
        Maps each query term to itself plus close spellings found in the index.

        "admissoin", "hostal" or "plcement" previously matched nothing: the
        lexical sweep needs exact word boundaries, and hashed fallback vectors
        have no notion of similar spellings either. The result looked identical
        to having no data on the topic.
        """
        if not stems:
            return [], {}

        vocabulary = self._index_vocabulary()
        if not vocabulary:
            return list(stems), {}

        expanded, corrections = [], {}
        candidates = None

        for stem in stems:
            expanded.append(stem)
            if len(stem) < 4 or stem in vocabulary:
                continue

            # Restrict the fuzzy search to words of a similar length and first
            # letter; over a large vocabulary this matters for speed.
            if candidates is None:
                candidates = sorted(vocabulary)
            pool = [
                word for word in candidates
                if abs(len(word) - len(stem)) <= 2 and word[0] == stem[0]
            ]
            matches = difflib.get_close_matches(stem, pool or candidates, n=3, cutoff=0.82)
            if matches:
                corrections[stem] = matches
                expanded.extend(matches)

        if corrections:
            logger.info("[RAG] Spelling repaired: %s", corrections)
        return expanded, corrections

    def retrieve(self, query, top_k=None):
        """
        FAISS Pure Dense Vector Semantic Search:
        Embeds the query into dense vector space and searches FAISS index using vector similarity.
        Works across all dimensions for any natural language input, including undefined words or phrasing.
        """
        with self._lock:
            if not self._chunks:
                return []
            chunks = self._chunks
            spaces = dict(self._spaces)

        k = top_k or Config.TOP_K
        expanded_synonyms = self._expand_synonyms(query)
        raw_stems = self._query_stems(f"{query} {expanded_synonyms}")
        stems, corrections = self.correct_terms(raw_stems)

        dense_scores = {}

        for provider, (index, positions) in spaces.items():
            if not positions or index.ntotal == 0 or provider == PROVIDER_LEGACY:
                continue

            # FAISS vector search with raw user query
            query_vector = self._embedder.embed_query(query, provider)
            if query_vector is not None and len(query_vector):
                search_k = min(len(positions), max(k * 5, 40))
                scores, ids = index.search(np.ascontiguousarray(query_vector), search_k)
                hits = [
                    (positions[i], float(s)) for s, i in zip(scores[0], ids[0])
                    if 0 <= i < len(positions)
                ]
                if hits:
                    values = [s for _, s in hits]
                    low, high = min(values), max(values)
                    spread = (high - low) or 1.0
                    for position, score in hits:
                        normalized = (score - low) / spread if spread > 0 else score
                        dense_scores[position] = max(dense_scores.get(position, 0.0), normalized)

            # Also embed synonym-enriched query to boost recall for domain terms
            if expanded_synonyms:
                enriched_text = f"{query} {expanded_synonyms}"
                query_vector_enriched = self._embedder.embed_query(enriched_text, provider)
                if query_vector_enriched is not None and len(query_vector_enriched):
                    search_k = min(len(positions), max(k * 5, 40))
                    scores, ids = index.search(np.ascontiguousarray(query_vector_enriched), search_k)
                    hits = [
                        (positions[i], float(s)) for s, i in zip(scores[0], ids[0])
                        if 0 <= i < len(positions)
                    ]
                    if hits:
                        values = [s for _, s in hits]
                        low, high = min(values), max(values)
                        spread = (high - low) or 1.0
                        for position, score in hits:
                            normalized = (score - low) / spread if spread > 0 else score
                            dense_scores[position] = max(dense_scores.get(position, 0.0), normalized)

        candidates = {}

        def consider(position, dense_score):
            chunk = chunks[position]
            text_lower = chunk.get("text", "").lower()
            title_lower = (
                f"{chunk.get('source_name', '')} {(chunk.get('metadata') or {}).get('title', '')}"
            ).lower()

            keyword_hits = sum(1 for stem in stems if re.search(r"\b" + re.escape(stem), text_lower))
            title_hits = sum(1 for stem in stems if stem in title_lower)

            entry = dict(chunk)
            # FAISS dense vector similarity is the primary score
            entry["score"] = round(dense_score + keyword_hits * 0.15 + title_hits * 0.10, 4)
            entry["kw_matches"] = keyword_hits
            entry["dense_score"] = round(dense_score, 4)
            current = candidates.get(position)
            if current is None or entry["score"] > current["score"]:
                candidates[position] = entry

        for position, score in dense_scores.items():
            consider(position, score)

        # Fallback lexical sweep if dense FAISS index is empty/sparse
        if stems and len(candidates) < k:
            for position, chunk in enumerate(chunks):
                if position in candidates:
                    continue
                text_lower = chunk.get("text", "").lower()
                keyword_hits = sum(
                    1 for stem in stems if re.search(r"\b" + re.escape(stem), text_lower)
                )
                if keyword_hits >= 2 or (len(stems) == 1 and keyword_hits >= 1):
                    consider(position, 0.10)

        ranked = sorted(candidates.values(), key=lambda c: c["score"], reverse=True)

        final, per_item = [], {}
        for chunk in ranked:
            key = (chunk.get("source_id"), chunk.get("item_key"))
            if per_item.get(key, 0) >= 3:
                continue
            per_item[key] = per_item.get(key, 0) + 1
            final.append(chunk)
            if len(final) >= k:
                break
        return final
