"""
LangChain Chat History & Conversational Memory Manager for BMSIT AI Chatbot.

Provides:
- LangChain message abstraction (HumanMessage, AIMessage, SystemMessage).
- In-memory session chat history with automatic windowing/trimming.
- History-Aware Standalone Query Reformulation:
  Resolves ambiguous follow-up questions, pronouns, and topic shifts into
  clean, standalone queries for high-precision RAG vector retrieval without
  polluting search vectors with unrelated past context.
"""
import logging
import os
import re
import threading
import time
import warnings
from typing import Dict, List, Optional

import requests
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message=".*InMemoryChatMessageHistory.*")
    from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from app.config import Config

logger = logging.getLogger(__name__)

# Max conversational turns to retain in active history window (10 turns = 20 messages)
MAX_HISTORY_TURNS = 10
SESSION_TTL_SECONDS = 3600 * 6  # 6 hours TTL for inactive sessions


class LangChainHistoryManager:
    """
    Manages conversational memory across user chat sessions using LangChain.
    """
    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self._sessions: Dict[str, InMemoryChatMessageHistory] = {}
        self._last_active: Dict[str, float] = {}
        self._session_lock = threading.RLock()

    @classmethod
    def get_instance(cls) -> "LangChainHistoryManager":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance

    def _cleanup_stale_sessions(self):
        """Removes sessions that have been idle longer than SESSION_TTL_SECONDS."""
        now = time.time()
        stale_ids = [
            sid for sid, ts in self._last_active.items()
            if now - ts > SESSION_TTL_SECONDS
        ]
        for sid in stale_ids:
            self._sessions.pop(sid, None)
            self._last_active.pop(sid, None)

    def get_session_history(self, session_id: str) -> InMemoryChatMessageHistory:
        """Retrieves or creates a LangChain InMemoryChatMessageHistory for a session."""
        sid = str(session_id).strip()
        with self._session_lock:
            self._cleanup_stale_sessions()
            if sid not in self._sessions:
                self._sessions[sid] = InMemoryChatMessageHistory()
            self._last_active[sid] = time.time()
            return self._sessions[sid]

    def sync_history(
        self,
        session_id: Optional[str],
        client_history: Optional[List[dict]] = None
    ) -> List[BaseMessage]:
        """
        Synchronizes client-supplied history with backend LangChain memory.
        Trims history to the maximum allowed window size.
        """
        history_obj = None
        if session_id:
            history_obj = self.get_session_history(str(session_id).strip())

        messages: List[BaseMessage] = []

        if client_history and isinstance(client_history, list):
            # Reconstruct LangChain message objects from client payload
            for turn in client_history:
                if not isinstance(turn, dict):
                    continue
                role = str(turn.get("role", "")).lower()
                text = str(turn.get("text") or turn.get("content") or "").strip()
                if not text:
                    continue
                if role in ("user", "human"):
                    messages.append(HumanMessage(content=text))
                elif role in ("assistant", "ai", "bot"):
                    messages.append(AIMessage(content=text))
                elif role == "system":
                    messages.append(SystemMessage(content=text))
        elif history_obj:
            messages = list(history_obj.messages)


        # Windowing: Keep only the most recent turns
        max_messages = MAX_HISTORY_TURNS * 2
        if len(messages) > max_messages:
            messages = messages[-max_messages:]

        if history_obj:
            with self._session_lock:
                history_obj.clear()
                for msg in messages:
                    history_obj.add_message(msg)

        return messages

    def append_turn(
        self,
        session_id: Optional[str],
        user_message: str,
        ai_message: str
    ):
        """Appends a completed interaction turn to session memory."""
        if not session_id:
            return
        with self._session_lock:
            history_obj = self.get_session_history(session_id)
            history_obj.add_user_message(user_message.strip())
            history_obj.add_ai_message(ai_message.strip())
            # Enforce max window
            max_messages = MAX_HISTORY_TURNS * 2
            if len(history_obj.messages) > max_messages:
                trimmed = history_obj.messages[-max_messages:]
                history_obj.clear()
                for msg in trimmed:
                    history_obj.add_message(msg)

    def clear_session(self, session_id: str):
        """Clears memory for a given session."""
        with self._session_lock:
            self._sessions.pop(session_id, None)
            self._last_active.pop(session_id, None)

    @staticmethod
    def format_history_for_prompt(messages: List[BaseMessage]) -> str:
        """
        Renders LangChain messages into a structured conversation history block for the LLM.
        """
        if not messages:
            return ""

        formatted_lines = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                formatted_lines.append(f"User: {msg.content}")
            elif isinstance(msg, AIMessage):
                # Clean out excessive markdown or URLs in history to save context tokens
                clean_ai = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", str(msg.content))
                formatted_lines.append(f"Assistant: {clean_ai}")

        if not formatted_lines:
            return ""

        return "CONVERSATION HISTORY (Previous turns in this session):\n" + "\n".join(formatted_lines) + "\n\n"

    # Patterns indicating pronouns, references, or short follow-ups
    FOLLOW_UP_INDICATORS = [
        r"\b(he|him|his|she|her|hers|it|its|they|them|their|these|those|this|that)\b",
        r"\b(who is (he|she|that)|tell about (him|her|them)|what about (him|her|it|them))\b",
        r"\b(what are (the|their|its) fees|how much does it cost|what is the cutoff)\b",
        r"\b(how do i apply|what is the eligibility|who is the hod|who is the head)\b",
        r"\b(tell me more|more info|more details|explain more|what else|and for)\b",
        r"^(what about|and|how about|what is|tell me)\b"
    ]

    def reformulate_standalone_query(
        self,
        user_message: str,
        messages: List[BaseMessage],
        api_key: Optional[str] = None
    ) -> str:
        """
        Uses LangChain conversational context to reformulate elliptical/referential questions
        into self-contained standalone search queries for RAG retrieval.
        
        CRITICAL PRODUCTION SAFEGUARD:
        If the query is already standalone or shifts to a new topic (e.g. 'What about Computer Science?'),
        it ensures the new topic is queried directly without contaminating it with previous topics (like hostels).
        """
        msg_clean = user_message.strip()
        if not messages:
            return msg_clean

        msg_lower = msg_clean.lower()
        words = re.findall(r"\w+", msg_lower)

        # Check if question has referential signals or is a short follow-up (< 8 words)
        is_referential = any(re.search(pat, msg_lower) for pat in self.FOLLOW_UP_INDICATORS)
        is_short = len(words) <= 7

        if not (is_referential or is_short):
            # Already a detailed, independent query
            return msg_clean

        # Attempt fast LLM-based reformulation if API key is provided
        if api_key and api_key != "your_api_key_here":
            try:
                reformulated = self._llm_reformulate(msg_clean, messages, api_key)
                if reformulated and len(reformulated) >= 3:
                    logger.info(f"[HistoryManager] Reformulated: '{msg_clean}' -> '{reformulated}'")
                    return reformulated
            except Exception as e:
                logger.debug(f"[HistoryManager] Fast LLM reformulation skipped: {e}")

        # Intelligent heuristic reformulation fallback
        return self._heuristic_reformulate(msg_clean, messages)

    def _llm_reformulate(self, query: str, messages: List[BaseMessage], api_key: str) -> Optional[str]:
        """Calls Gemini with minimal tokens to synthesize a standalone query."""
        recent_dialogue = []
        for m in messages[-4:]:
            role = "User" if isinstance(m, HumanMessage) else "Assistant"
            text = m.content[:200]
            recent_dialogue.append(f"{role}: {text}")

        history_str = "\n".join(recent_dialogue)
        prompt = (
            "Given the conversation history about BMSIT (B.M.S. Institute of Technology and Management), "
            "reformulate the latest user question into a concise, standalone search query for the college knowledge base.\n"
            "- If the question references prior topics (using pronouns like 'he', 'she', 'it', 'its fees', 'the HOD'), resolve them.\n"
            "- If the question starts a new topic, do NOT mention the previous topic.\n"
            "- Output ONLY the standalone search query. Do NOT answer the question.\n\n"
            f"History:\n{history_str}\n\n"
            f"User Question: {query}\n"
            "Standalone Query:"
        )

        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash-lite:generateContent?key={api_key}"
        resp = requests.post(
            url,
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.0, "maxOutputTokens": 60}
            },
            timeout=8
        )
        if resp.status_code == 200:
            candidates = resp.json().get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                if parts and parts[0].get("text"):
                    result = parts[0]["text"].strip()
                    # Sanity check: Ensure it didn't return an answer or long sentence
                    if len(result.splitlines()) == 1 and len(result) < 150:
                        return result.strip('"\'')
        return None

    def _heuristic_reformulate(self, query: str, messages: List[BaseMessage]) -> str:
        """
        Heuristic entity resolution when offline or during API timeouts.
        Extracts key subject nouns from the most recent user turn.
        """
        msg_lower = query.lower()
        # Find key entity in the last user query
        last_user_msg = ""
        for m in reversed(messages):
            if isinstance(m, HumanMessage):
                last_user_msg = m.content
                break

        if not last_user_msg:
            return query

        # Detect specific entities in previous user question
        college_entities = [
            "computer science", "cse", "information science", "ise", "electronics", "ece",
            "electrical", "eee", "mechanical", "civil", "ai&ml", "aiml", "aids", "vlsi",
            "hostel", "placement", "placements", "admissions", "admission", "scholarship",
            "principal", "library", "canteen", "bus", "transport"
        ]
        found_entity = None
        last_lower = last_user_msg.lower()
        for ent in college_entities:
            if re.search(r"\b" + re.escape(ent) + r"\b", last_lower):
                found_entity = ent
                break

        if found_entity and found_entity not in msg_lower:
            # If current query is asking for details like "fees", "hod", "eligibility", "packages"
            details = ["fee", "fees", "hod", "head", "placement", "cutoff", "package", "eligibility", "rules", "rooms", "seats"]
            if any(d in msg_lower for d in details):
                return f"{query} for {found_entity} at BMSIT"

        return query
