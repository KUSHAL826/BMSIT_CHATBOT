"""
Production Reranker Service for BMSIT AI Chatbot.

Solves the critical RAG problem:
"Sometimes even data is there in the backend, it's telling data not available."

How it works:
1. Candidate Pool Expansion: Retrieval fetches an expanded candidate pool (20-30 chunks)
   to ensure zero false negatives.
2. Cross-Scoring & Reranking: Reranker jointly evaluates (query, chunk_text) to measure
   direct answer relevance, exact question-entity alignment, and factual coverage.
3. Ranking Reordering: Promotes the most directly informative passages to the very top
   and filters out tangential noise chunks before context is fed to the LLM.
"""
import json
import logging
import math
import os
import re
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

STOP_WORDS = {
    "what", "is", "the", "are", "of", "in", "for", "to", "at", "and", "a", "an", "on",
    "tell", "me", "about", "can", "you", "does", "when", "where", "how", "do", "did",
    "bmsit", "bms", "college", "institute", "there", "any", "please", "give", "list"
}


class RerankerService:
    """
    Two-Tier Production Reranker:
    - Tier 1: Fast Cross-Attention LLM Pointwise/Listwise Scorer (when online).
    - Tier 2: High-Precision Cross-Token Coverage & Entity Density Scorer (local & deterministic).
    """
    _instance = None

    @classmethod
    def get_instance(cls) -> "RerankerService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: int = 10,
        api_key: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Reranks a list of candidate chunks based on direct relevance to the query.
        Returns the top_k reranked chunks with updated 'score' and 'rerank_score'.
        """
        if not chunks:
            return []

        if len(chunks) <= 1:
            return chunks

        effective_key = api_key or os.getenv("GEMINI_API_KEY", "")

        # Try Tier 1: LLM-based scoring if candidate pool is reasonable (e.g. up to 15 chunks)
        if effective_key and effective_key != "your_api_key_here" and len(chunks) >= 2:
            try:
                reranked = self._llm_rerank(query, chunks[:15], top_k=top_k, api_key=effective_key)
                if reranked:
                    # Append any remaining candidates scored by local tier
                    remaining = [c for c in chunks if c not in reranked]
                    if len(reranked) < top_k and remaining:
                        local_remaining = self._local_rerank(query, remaining)
                        reranked.extend(local_remaining[:(top_k - len(reranked))])
                    logger.info(f"[Reranker] LLM rerank successful for {len(reranked)} chunks.")
                    return reranked[:top_k]
            except Exception as e:
                logger.debug(f"[Reranker] LLM reranking skipped: {e}")

        # Tier 2: Local Cross-Token Semantic Reranker
        local_reranked = self._local_rerank(query, chunks)
        return local_reranked[:top_k]

    def _local_rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        High-precision deterministic cross-token reranking:
        - Exact query phrase matches (weight: 3.0)
        - Key entity overlap (weight: 2.0)
        - Source title & section relevance (weight: 1.5)
        - Initial vector similarity score (weight: 1.0)
        - Penalty for short, boilerplate fragments
        """
        query_lower = query.lower()
        raw_terms = [
            w for w in re.findall(r"\w+", query_lower)
            if w not in STOP_WORDS and (len(w) > 2 or w.isdigit())
        ]
        stems = [w[:-1] if (w.endswith("s") and not w.endswith("ss") and len(w) > 3) else w for w in raw_terms]

        scored_chunks = []
        for chunk in chunks:
            text = chunk.get("text", "")
            text_lower = text.lower()
            source_name = str(chunk.get("source_name", "")).lower()
            metadata = chunk.get("metadata") or {}
            sec = str(metadata.get("section_title") or "").lower()
            initial_score = float(chunk.get("score", 0.0))

            # 1. Exact phrase match bonus
            exact_phrase_bonus = 0.0
            if len(raw_terms) >= 2:
                phrase = " ".join(raw_terms[:4])
                if phrase in text_lower:
                    exact_phrase_bonus = 0.40

            # 2. Key stem density & coverage
            matched_stems = set()
            term_occurrences = 0
            for st in stems:
                matches = len(re.findall(r"\b" + re.escape(st), text_lower))
                if matches > 0:
                    matched_stems.add(st)
                    term_occurrences += matches

            stem_coverage = len(matched_stems) / max(len(stems), 1)
            coverage_score = stem_coverage * 0.45

            # 3. Title and section alignment
            title_match = sum(1 for st in stems if st in source_name or st in sec)
            title_score = min(title_match * 0.15, 0.30)

            # 4. Numerical/factual presence (if query asks for fees, cutoffs, dates, or intake)
            numeric_bonus = 0.0
            has_numeric_query = any(k in query_lower for k in ["fee", "fees", "cost", "cutoff", "intake", "phone", "package", "ctc"])
            if has_numeric_query and re.search(r"(₹|\b\d{2,6}\b|lpa|inr)", text_lower):
                numeric_bonus = 0.25

            # 5. Composite rerank score
            rerank_score = (
                (initial_score * 0.35) +
                coverage_score +
                exact_phrase_bonus +
                title_score +
                numeric_bonus
            )

            # Slight penalty for very short boilerplate chunks (< 80 chars)
            if len(text.strip()) < 80:
                rerank_score *= 0.70

            entry = dict(chunk)
            entry["rerank_score"] = round(rerank_score, 4)
            # Update primary score to reflect reranked ordering
            entry["score"] = round(rerank_score, 4)
            scored_chunks.append(entry)

        # Sort by rerank score descending
        scored_chunks.sort(key=lambda c: c["rerank_score"], reverse=True)
        return scored_chunks

    def _llm_rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int,
        api_key: str
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Fast cross-attention reranker via Gemini.
        Returns candidate indices sorted from most relevant to least relevant.
        """
        items_payload = []
        for idx, c in enumerate(candidates):
            src = c.get("source_name", "BMSIT")
            snippet = re.sub(r"\s+", " ", c.get("text", "")[:280]).strip()
            items_payload.append(f"[{idx}] Source: {src} | Text: {snippet}")

        items_text = "\n".join(items_payload)
        prompt = f"""You are a precise passage reranker for a college information search engine.
Question: "{query}"

Evaluate each candidate passage below and rank them by how directly and accurately they answer or provide facts for the question.
Passages:
{items_text}

Output JSON ONLY in this format:
{{"ranked_indices": [most_relevant_index, second_index, ...]}}"""

        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash-lite:generateContent?key={api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.0,
                "responseMimeType": "application/json",
                "maxOutputTokens": 100
            }
        }

        resp = requests.post(url, json=payload, timeout=8)
        if resp.status_code == 200:
            res_json = resp.json()
            candidates_out = res_json.get("candidates", [])
            if candidates_out:
                parts = candidates_out[0].get("content", {}).get("parts", [])
                if parts and parts[0].get("text"):
                    parsed = json.loads(parts[0]["text"])
                    ranked_indices = parsed.get("ranked_indices", [])
                    if isinstance(ranked_indices, list) and ranked_indices:
                        result = []
                        seen = set()
                        for idx in ranked_indices:
                            if isinstance(idx, int) and 0 <= idx < len(candidates) and idx not in seen:
                                seen.add(idx)
                                item = dict(candidates[idx])
                                # Higher rank gets boosted score
                                item["rerank_score"] = round(1.0 - (len(result) * 0.05), 3)
                                item["score"] = item["rerank_score"]
                                result.append(item)
                        return result

        return None
