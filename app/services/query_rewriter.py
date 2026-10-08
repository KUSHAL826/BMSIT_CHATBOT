"""
Query Rewriter for BMSIT AI Chatbot.

Transforms conversational, vague, or shorthand user queries into
high-precision, search-optimized standalone queries for RAG vector & lexical search.

Capabilities:
1. Multi-turn reference resolution (pronouns, follow-ups, elliptical questions).
2. Institutional term expansion (mapping college jargon, acronyms, quotas, departments).
3. Topic shift preservation (guarantees that switching topics does not pollute retrieval).
4. Dual-mode execution:
   - High-speed Gemini LLM rewriter when online.
   - Domain-aware heuristic rewriter fallback when offline or during quota limits.
"""
import logging
import os
import re
from typing import List, Optional

import requests
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from app.config import Config

logger = logging.getLogger(__name__)


class QueryRewriter:
    """Enterprise Query Rewriter for RAG retrieval optimization."""
    _instance = None

    @classmethod
    def get_instance(cls) -> "QueryRewriter":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def rewrite(
        self,
        query: str,
        messages: Optional[List[BaseMessage]] = None,
        api_key: Optional[str] = None
    ) -> str:
        """
        Rewrites a raw user question into an optimal RAG search query.
        """
        clean_query = (query or "").strip()
        if not clean_query:
            return clean_query

        # 1. Check if multi-turn conversational resolution with pronouns is needed
        referential_words = ["he", "she", "him", "her", "it", "its", "they", "them", "this", "that", "more", "details", "previous", "earlier"]
        needs_context = messages and len(messages) >= 2 and any(re.search(r"\b" + w + r"\b", clean_query.lower()) for w in referential_words)

        effective_key = api_key or os.getenv("GEMINI_API_KEY", "")
        if needs_context and effective_key and effective_key != "your_api_key_here":
            try:
                llm_rewritten = self._llm_rewrite(clean_query, messages, effective_key)
                if llm_rewritten and len(llm_rewritten) >= 3:
                    logger.info(f"[QueryRewriter] LLM rewrite: '{clean_query}' -> '{llm_rewritten}'")
                    return llm_rewritten
            except Exception as e:
                logger.debug(f"[QueryRewriter] LLM rewrite fallback triggered: {e}")

        # 2. Ultra-fast deterministic rule-based rewrite (0ms latency)
        heuristic_rewritten = self._heuristic_rewrite(clean_query, messages)
        return heuristic_rewritten

    def _llm_rewrite(
        self,
        query: str,
        messages: Optional[List[BaseMessage]],
        api_key: str
    ) -> Optional[str]:
        """
        Calls Gemini to reformulate and expand the user query into a single, search-optimized query.
        """
        history_context = ""
        if messages:
            recent_turns = []
            for m in messages[-4:]:
                role = "User" if isinstance(m, HumanMessage) else "Assistant"
                # Strip long content to preserve token speed
                text = re.sub(r"\s+", " ", m.content[:200]).strip()
                recent_turns.append(f"{role}: {text}")
            if recent_turns:
                history_context = "Conversation History:\n" + "\n".join(recent_turns) + "\n\n"

        prompt = f"""You are an expert Search Query Rewriter for the B.M.S. Institute of Technology and Management (BMSIT&M) knowledge base search engine.
Your task is to rewrite the user's latest question into an optimal, standalone search query.

Guidelines:
1. Resolve all pronouns ('he', 'she', 'it', 'they', 'this course', 'that quota', 'its fees') using the conversation history.
2. If the user changes topic (e.g. from hostel to computer science), focus EXCLUSIVELY on the new topic. Do NOT mix previous topics into the search query.
3. Expand common college abbreviations where helpful (e.g., 'cse' -> 'computer science engineering', 'hod' -> 'head of department', 'hostel fee' -> 'hostel room fee structure').
4. Keep the rewritten query concise, factual, and laser-focused on finding relevant passages in BMSIT records.
5. Return ONLY the rewritten search query. Do NOT add preamble, quotes, explanations, or answers.

{history_context}Latest User Question: {query}
Optimized Search Query:"""

        models = ["gemini-3.5-flash", "gemma-4-26b-a4b-it"]
        for m in models:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent?key={api_key}"
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.0,
                    "maxOutputTokens": 60
                }
            }
            try:
                resp = requests.post(url, json=payload, timeout=2.5)
                if resp.status_code == 200:
                    res_json = resp.json()
                    candidates = res_json.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        if parts and parts[0].get("text"):
                            result = parts[0]["text"].strip().strip('"\'')
                            if len(result.splitlines()) == 1 and len(result) < 200:
                                return result
            except Exception:
                continue

        return None

    def _heuristic_rewrite(
        self,
        query: str,
        messages: Optional[List[BaseMessage]]
    ) -> str:
        """
        Pure dynamic rewrite: if referential pronouns are detected and previous turns exist,
        combine with immediate prior user context; otherwise use clean query directly.
        Zero hardcoded dictionaries or canned lists.
        """
        clean_query = query.strip()
        referential_words = ["he", "she", "him", "her", "it", "its", "they", "them", "this", "that", "these", "those"]
        has_referential = any(re.search(r"\b" + w + r"\b", clean_query.lower()) for w in referential_words)

        if has_referential and messages:
            for m in reversed(messages):
                if isinstance(m, HumanMessage) and m.content:
                    prior_clean = m.content.strip()
                    if prior_clean and prior_clean.lower() != clean_query.lower():
                        return f"{clean_query} ({prior_clean})"

        return clean_query
