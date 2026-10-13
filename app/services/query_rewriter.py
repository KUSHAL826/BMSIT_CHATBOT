"""
Query Rewriter for BMSIT AI Chatbot.

Transforms conversational, vague, or shorthand user queries into
high-precision, search-optimized standalone queries for RAG vector & lexical search.

Zero hardcoded word lists or static dictionaries.
Uses Gemini 3.5 Flash Lite for multi-turn conversational contextualization.
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
        Rewrites conversational follow-up or shorthand queries into standalone search queries.
        Pure dynamic logic: zero predefined word lists or hardcoded dictionaries.
        """
        clean_query = (query or "").strip()
        if not clean_query:
            return clean_query

        # If there is no previous conversation history, the query is already standalone (0ms latency)
        if not messages or len(messages) < 2:
            return clean_query

        # Standalone detailed queries (6+ words) have full context on their own (0ms latency)
        if len(clean_query.split()) >= 6:
            return clean_query

        # Multi-turn reference resolution via Gemini 3.5 Flash Lite
        effective_key = api_key or os.getenv("GEMINI_API_KEY", "")
        if not effective_key:
            pooled = os.getenv("GEMINI_API_KEYS", "")
            if pooled:
                effective_key = pooled.split(",")[0].strip()

        if effective_key and effective_key != "your_api_key_here":
            try:
                llm_rewritten = self._llm_rewrite(clean_query, messages, effective_key)
                if llm_rewritten and len(llm_rewritten) >= 3:
                    logger.info(f"[QueryRewriter] LLM rewrite: '{clean_query}' -> '{llm_rewritten}'")
                    return llm_rewritten
            except Exception as e:
                logger.debug(f"[QueryRewriter] LLM rewrite fallback triggered: {e}")

        # Dynamic fallback rewrite (0ms latency, zero hardcoded dictionaries)
        return self._heuristic_rewrite(clean_query, messages)

    def _llm_rewrite(
        self,
        query: str,
        messages: Optional[List[BaseMessage]],
        api_key: str
    ) -> Optional[str]:
        """
        Fast LLM query reformulation using gemini-3.5-flash-lite only with pooled HTTP session.
        """
        history_context = ""
        if messages:
            recent_turns = []
            for m in messages[-4:]:
                role = "User" if isinstance(m, HumanMessage) else "Assistant"
                text = re.sub(r"\s+", " ", m.content[:200]).strip()
                recent_turns.append(f"{role}: {text}")
            if recent_turns:
                history_context = "Conversation History:\n" + "\n".join(recent_turns) + "\n\n"

        prompt = f"""You are an expert Search Query Rewriter for the B.M.S. Institute of Technology and Management (BMSIT&M) knowledge base search engine.
Your task is to rewrite the user's latest question into an optimal, standalone search query.

Guidelines:
1. Resolve all pronouns ('he', 'she', 'it', 'they', 'this course', 'that quota', 'its fees') using the conversation history.
2. If the user changes topic, focus EXCLUSIVELY on the new topic. Do NOT mix previous topics into the search query.
3. Keep the rewritten query concise, factual, and laser-focused on finding relevant passages in BMSIT records.
4. Return ONLY the rewritten search query. Do NOT add preamble, quotes, explanations, or answers.

{history_context}Latest User Question: {query}
Optimized Search Query:"""

        try:
            from app.services.gemini_service import GeminiService
            session = GeminiService._get_session()
        except Exception:
            session = requests.Session()

        models = [getattr(Config, "GEMINI_MODEL", "gemini-3.5-flash-lite") or "gemini-3.5-flash-lite", "gemini-3.5-flash-lite", "gemini-2.5-flash-lite", "gemini-flash-latest"]
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "maxOutputTokens": 60
            }
        }
        for m in models:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent?key={api_key}"
            try:
                resp = session.post(url, json=payload, timeout=2.5)
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
        Pure dynamic rewrite: combine short follow-up with prior user query context.
        Zero hardcoded dictionaries.
        """
        clean_query = query.strip()
        if messages:
            for m in reversed(messages):
                if isinstance(m, HumanMessage) and m.content:
                    prior_clean = m.content.strip()
                    if prior_clean and prior_clean.lower() != clean_query.lower():
                        return f"{clean_query} ({prior_clean})"

        return clean_query
