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

# Canonical BMSIT department & acronym mapping
BMSIT_CANONICAL_TERMS = {
    "cse": "Computer Science & Engineering",
    "ise": "Information Science & Engineering",
    "ece": "Electronics & Communication Engineering",
    "eee": "Electrical & Electronics Engineering",
    "mech": "Mechanical Engineering",
    "me": "Mechanical Engineering",
    "cv": "Civil Engineering",
    "civil": "Civil Engineering",
    "aiml": "Artificial Intelligence & Machine Learning",
    "ai&ml": "Artificial Intelligence & Machine Learning",
    "aids": "Artificial Intelligence & Data Science",
    "vlsi": "M.Tech in VLSI System Design",
    "mca": "Master of Computer Applications",
    "mba": "Master of Business Administration",
    "pmsss": "Prime Minister Special Scholarship Scheme (PMSSS / J&K Quota)",
    "comedk": "COMEDK entrance quota and counseling",
    "kcet": "Karnataka Common Entrance Test (KCET)",
    "hod": "Head of Department (HoD / Professor & Head)",
    "associate hod": "Associate Head / HoD of CSE Division or Cluster",
    "associate head": "Associate Head / HoD of CSE Division or Cluster",
    "cluster": "Cluster / Division in CSE Department",
    "clusters": "Clusters / Divisions in CSE Department",
    "cluster 1": "CSE Cluster 1 Division 1 Associate Head HoD",
    "cluster 2": "CSE Cluster 2 Division 2 Associate Head HoD",
    "cluster 3": "CSE Cluster 3 Division 3 Associate Head HoD",
    "cluster 4": "CSE Cluster 4 Division 4 Associate Head HoD",
    "cluster 5": "CSE Cluster 5 Division 5 Associate Head HoD",
    "principal": "Principal / Head of Institution",
    "hostel": "Hostel Facilities, accommodation, room types, and fee structure",
    "placement": "Placements, packages, recruiters, and statistics",
}

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

        models = ["gemini-3-flash-preview", "gemma-4-26b-a4b-it"]
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
        High-precision rule-based query expansion and reference resolution.
        """
        rewritten = query
        query_lower = query.lower()

        # Specific cluster handling: in BMSIT CSE department, clusters correspond to divisions
        for c_idx in range(1, 6):
            if f"cluster {c_idx}" in query_lower or f"cluster-{c_idx}" in query_lower or f"division {c_idx}" in query_lower:
                rewritten = f"{rewritten} CSE Division {c_idx} Cluster {c_idx} Associate Head HoD"
                break

        if ("associate hod" in query_lower or "associate head" in query_lower) and "cluster" not in query_lower and "division" not in query_lower:
            rewritten = f"{rewritten} Associate Head Division Cluster CSE"

        # Check for pronouns or short follow-ups needing context
        referential_words = ["he", "she", "him", "her", "it", "its", "they", "them", "this", "that", "more", "details"]
        has_referential = any(re.search(r"\b" + w + r"\b", query_lower) for w in referential_words)

        # Context resolution from previous user turn
        if has_referential and messages:
            last_user_topic = None
            for m in reversed(messages):
                if isinstance(m, HumanMessage):
                    for alias, full_name in BMSIT_CANONICAL_TERMS.items():
                        if re.search(r"\b" + re.escape(alias) + r"\b", m.content.lower()):
                            last_user_topic = full_name
                            break
                    if last_user_topic:
                        break

            if last_user_topic and last_user_topic.lower() not in query_lower:
                rewritten = f"{rewritten} for {last_user_topic}"

        # Expand key acronyms if present
        for alias, full_name in BMSIT_CANONICAL_TERMS.items():
            pattern = r"\b" + re.escape(alias) + r"\b"
            if re.search(pattern, query_lower) and alias in ["cse", "ise", "ece", "eee", "mech", "aiml", "pmsss", "hod"]:
                if full_name.lower() not in rewritten.lower():
                    rewritten = f"{rewritten} ({full_name})"

        # Ensure college scope if not present
        if not any(k in rewritten.lower() for k in ["bmsit", "bms institute", "college"]):
            rewritten = f"{rewritten} BMSIT"

        return rewritten.strip()
