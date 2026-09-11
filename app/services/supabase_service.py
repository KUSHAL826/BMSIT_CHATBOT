"""
Supabase PostgreSQL + pgvector Storage Service.

Provides persistence for knowledge chunks and vector embeddings in a Supabase
PostgreSQL database equipped with the pgvector extension.

Uses standard HTTP PostgREST API (via requests) to query and upsert vector data
without needing platform-specific native binary dependencies.
"""
import hashlib
import json
import logging
import numpy as np
import requests

from app.config import Config

logger = logging.getLogger(__name__)


class SupabaseService:
    _instance = None

    def __init__(self):
        self.url = (Config.SUPABASE_URL or "").rstrip("/")
        self.key = Config.SUPABASE_KEY or ""
        self.table = Config.SUPABASE_TABLE or "vectors"
        self.db_url = Config.SUPABASE_DB_URL or ""

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def is_enabled(self):
        return bool((self.url and self.key) or self.db_url)

    def _headers(self):
        return {
            "apikey": self.key,
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def fetch_all_chunks(self):
        """
        Loads all chunks and vector embeddings from Supabase table with pagination.

        Returns (chunks_list, embeddings_matrix_np)
        """
        if not self.is_enabled:
            return [], np.zeros((0, Config.EMBEDDING_DIM), dtype=np.float32)

        endpoint = f"{self.url}/rest/v1/{self.table}"
        headers = self._headers()
        limit = 1000
        offset = 0

        raw_rows = []
        try:
            while True:
                req_url = f"{endpoint}?select=*&limit={limit}&offset={offset}"
                res = requests.get(req_url, headers=headers, timeout=30)
                if res.status_code != 200:
                    logger.error("[Supabase] Failed to fetch chunks: HTTP %s %s", res.status_code, res.text)
                    break
                batch = res.json()
                if not isinstance(batch, list) or not batch:
                    break
                raw_rows.extend(batch)
                if len(batch) < limit:
                    break
                offset += limit

            logger.info("[Supabase] Fetched %s row(s) from table '%s'.", len(raw_rows), self.table)
        except Exception as e:
            logger.exception("[Supabase] Error loading vectors from Supabase: %s", e)
            return [], np.zeros((0, Config.EMBEDDING_DIM), dtype=np.float32)

        chunks = []
        vec_list = []
        dim = Config.EMBEDDING_DIM

        for row in raw_rows:
            emb_raw = row.get("embedding")
            vec = None
            if isinstance(emb_raw, list):
                vec = emb_raw
            elif isinstance(emb_raw, str):
                try:
                    vec = json.loads(emb_raw)
                except Exception:
                    vec = None

            if vec is None or len(vec) == 0:
                vec = [0.0] * dim
            elif len(vec) < dim:
                vec = vec + [0.0] * (dim - len(vec))
            elif len(vec) > dim:
                vec = vec[:dim]

            chunk = {
                "chunk_id": row.get("id") or row.get("chunk_id"),
                "source_id": row.get("source_id", ""),
                "source_name": row.get("source_name", ""),
                "source_type": row.get("source_type", ""),
                "item_key": row.get("item_key", ""),
                "text": row.get("text", ""),
                "tokens": int(row.get("tokens") or 0),
                "context_header": row.get("context_header", ""),
                "metadata": row.get("metadata") if isinstance(row.get("metadata"), dict) else {},
                "embed_provider": row.get("embed_provider") or "gemini:gemini-embedding-001",
                "content_hash": row.get("content_hash", ""),
            }
            chunks.append(chunk)
            vec_list.append(vec)

        if not chunks:
            return [], np.zeros((0, dim), dtype=np.float32)

        embeddings = np.array(vec_list, dtype=np.float32)
        # L2-normalize vectors if flat inner product is used
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embeddings = (embeddings / norms).astype(np.float32)

        return chunks, embeddings

    def insert_chunks(self, chunks, vectors):
        """
        Inserts or merges chunks into the Supabase vectors table in batches.

        Never overwrites or deletes existing chunks unless an exact ID matches.
        """
        if not self.is_enabled or not chunks:
            return 0

        endpoint = f"{self.url}/rest/v1/{self.table}"
        headers = self._headers()
        headers["Prefer"] = "resolution=merge-duplicates"

        rows = []
        for idx, chunk in enumerate(chunks):
            vec = vectors[idx]
            emb_list = vec.tolist() if hasattr(vec, "tolist") else list(vec)

            text_content = chunk.get("text", "")
            c_hash = chunk.get("content_hash") or hashlib.md5(text_content.encode("utf-8")).hexdigest()

            rows.append({
                "id": chunk.get("chunk_id"),
                "source_id": chunk.get("source_id", ""),
                "source_name": chunk.get("source_name", ""),
                "source_type": chunk.get("source_type", ""),
                "item_key": chunk.get("item_key", ""),
                "text": text_content,
                "tokens": int(chunk.get("tokens", 0)),
                "context_header": chunk.get("context_header", ""),
                "metadata": chunk.get("metadata", {}),
                "embed_provider": chunk.get("embed_provider", "gemini:gemini-embedding-001"),
                "content_hash": c_hash,
                "embedding": emb_list,
            })

        batch_size = 50
        inserted_count = 0
        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            try:
                res = requests.post(endpoint, headers=headers, json=batch, timeout=30)
                if res.status_code in (200, 201, 204):
                    inserted_count += len(batch)
                else:
                    logger.error("[Supabase] Insert batch failed HTTP %s: %s", res.status_code, res.text)
            except Exception as e:
                logger.exception("[Supabase] Exception inserting vector batch: %s", e)

        logger.info("[Supabase] Successfully inserted %s vector chunk(s) into Supabase.", inserted_count)
        return inserted_count

    def delete_source(self, source_id):
        """Deletes chunks belonging to source_id from Supabase."""
        if not self.is_enabled or not source_id:
            return 0
        endpoint = f"{self.url}/rest/v1/{self.table}?source_id=eq.{source_id}"
        headers = self._headers()
        try:
            res = requests.delete(endpoint, headers=headers, timeout=30)
            if res.status_code in (200, 204):
                logger.info("[Supabase] Deleted source '%s' from Supabase.", source_id)
                return True
            else:
                logger.error("[Supabase] Delete source failed HTTP %s: %s", res.status_code, res.text)
        except Exception as e:
            logger.exception("[Supabase] Error deleting source '%s': %s", source_id, e)
        return False

    def reset_table(self):
        """Empties the vectors table in Supabase."""
        if not self.is_enabled:
            return False
        endpoint = f"{self.url}/rest/v1/{self.table}?id=neq.00000000-0000-0000-0000-000000000000"
        headers = self._headers()
        try:
            res = requests.delete(endpoint, headers=headers, timeout=30)
            if res.status_code in (200, 204):
                logger.warning("[Supabase] Table '%s' reset successfully.", self.table)
                return True
        except Exception as e:
            logger.exception("[Supabase] Error resetting table: %s", e)
        return False
