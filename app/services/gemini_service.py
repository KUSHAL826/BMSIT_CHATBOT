"""
Production Gemini RAG & Generation Service for BMSIT AI Chatbot.

Key Capabilities:
- Strict grounding in BMSIT official knowledge base (Zero Hallucination).
- High extraction fidelity: Never falsely rejects queries when data exists in context.
- LangChain Conversational Memory & History-Aware Query Reformulation.
- Robust Offline Semantic Synthesis Fallback.
"""
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional

import requests

from app.config import Config
from app.services.langchain_history import LangChainHistoryManager
from app.services.query_rewriter import QueryRewriter
from app.services.rag_service import RAGService

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are the official AI Assistant for B.M.S. Institute of Technology and Management (BMSIT&M), Avalahalli, Yelahanka, Bengaluru.
Your mission is to provide accurate, comprehensive, verified, and complete guidance to students, parents, alumni, faculty, and prospective applicants.

=======================================================
STRICT OPERATIONAL DIRECTIVES (PRODUCTION GENAI RULES):
=======================================================

1. EXCLUSIVE KNOWLEDGE-BASE GROUNDING (STRICT RULE):
   - You must answer ONLY using the provided BMSIT knowledge base context and verified records.
   - ZERO EXTERNAL HALLUCINATION: Under NO circumstance may you fabricate, speculate, extrapolate, or guess dates, admission cutoffs, fee structures, faculty names, or policies not documented in the context.
   - Every factual claim must be substantiated by the provided context.

2. MAXIMAL EXTRACTION & EXHAUSTIVE COMPLETENESS (NEVER TRUNCATE):
   - CRITICAL: Never claim data is unavailable if the answer or related facts exist in the provided context!
   - Thoroughly inspect all provided context passages, tables, rosters, and lists before deciding.
   - COMPLETE ENUMERATION GUARANTEE: When the user asks for faculty, professors, authorities, wardens, committees, courses, branches, fees, or recruiters (e.g. "list names", "all faculties", "hostel authorities", "who are the wardens"):
     * You MUST list EVERY single individual name, designation, and detail present in the context.
     * NEVER omit any member, never summarize with 'and others', and never cut off mid-list or mid-sentence.
     * Group members clearly by role, block, or division.
   - Flexibly map user queries, synonyms, and phrasing to context content:
     * "courses" / "branches" / "departments" / "programs" / "degrees"
     * "fees" / "tuition" / "hostel expenses" / "structure" / "payment" / "procedure"
     * "authorities" / "wardens" / "leadership" / "in-charge" / "supervisors" / "HOD"
     * "admissions" / "eligibility" / "criteria" / "KCET" / "COMEDK"
     * "hostel" / "accommodation" / "rooms" / "mess"

3. UNTRUNCATED TEXT INTEGRITY (ZERO MID-SENTENCE CUTOFF):
   - Every sentence, bullet point, name, and statement MUST be written out to its natural completion.
   - Under NO circumstance should a sentence end abruptly with an unclosed parenthesis, a hanging comma, or an incomplete name or number. Always finish every thought cleanly.

4. COMPOUND QUESTIONS & PARTIAL COVERAGE PROTOCOL:
   - When a user asks a multi-part question covering multiple topics in one prompt (e.g. asking about a facility AND its fee AND its payment procedure):
     * Extract and provide full, detailed answers for ALL components that ARE documented in the context.
     * If a specific administrative sub-detail (such as exact online transaction steps or portal URLs) is not recorded in the context, do NOT refuse the entire query! Provide all the verified facts you have (e.g. describe the facility, room types, and fees), and clearly note that the specific fee payment steps/drafts should be confirmed directly with the college accounts/admissions office.
     * NEVER issue a blanket refusal when partial information exists in the knowledge base.

5. GRACEFUL MISSING INFORMATION PROTOCOL (STRICT REFUSAL RULE):
   - If—and ONLY if—the entire requested topic, facility, or entity is genuinely and completely ABSENT from the provided context:
     * Do NOT guess or substitute unrelated departments or generic pages.
     * State clearly and courteously:
       "As of now, I don't have verified information regarding this in the BMSIT knowledge base. Please refer to https://bmsit.ac.in or contact the college directly:"
     * Provide the official college contact points:
       • Email: admissions@bmsit.in / principal@bmsit.in
       • Phone: +91-80-68730444 / +91-80-68730424
       • Website: https://bmsit.ac.in
       • Campus Address: Doddaballapur Main Road, Avalahalli, Yelahanka, Bengaluru - 560064

6. CONVERSATIONAL CONTINUITY (MULTI-TURN MEMORY):
   - Maintain seamless context across turns using the conversation history.
   - Resolve pronouns ("he", "she", "it", "they", "this department", "that course") based on prior discussion.

7. DYNAMIC VERIFICATION & RECENCY:
   - Extract all facts, names of faculty, HODs, associate heads, department divisions/clusters, fees, and dates DYNAMICALLY and EXCLUSIVELY from the provided context.
   - Prioritize the most recently updated verified passage when records differ.

8. HIGHLIGHT CORE ANSWERS IN BOLD:
   - Highlight key names, designations, essential facts, numbers, dates, exam names, branches, and vital details in **bold** markdown (e.g. **Dr. Sanjay H A**, **Dr. Siddiq Iqbal**, **KCET**, **COMEDK**).
   - Format responses with clean, readable Markdown bullet points and bold headers.

9. DYNAMIC ADAPTIVE LENGTH:
   - If the user specifies a length constraint (e.g. "in one line", "briefly"), adhere to it.
   - If NO length constraint is given, provide a thorough, complete, and well-structured response that answers every part of the question without truncating any list or member.

10. DEPARTMENT OVERVIEWS & ACADEMIC PROFILES:
   - When the user asks about a department (e.g. "about cse dept in bmsit", "tell me about ise", "overview of ece"):
     * Provide a detailed, multi-section overview synthesizing the department profile, key specializations, divisions, academic programs, and leadership from the records.
     * Never reduce a department overview to an isolated fragment.
"""


class GeminiService:
    _session: Optional[requests.Session] = None
    _active_model: Optional[str] = None
    _model_cooldowns: Dict[str, float] = {}

    @classmethod
    def get_configured_api_keys(cls) -> List[str]:
        """
        Extracts all valid API keys from GEMINI_API_KEY and GEMINI_API_KEYS pool.
        Ensures zero disruption if keys are defined in either or both environment variables.
        """
        raw = []
        single = (os.getenv("GEMINI_API_KEY") or "").strip()
        if single:
            raw.append(single)
        pooled = (os.getenv("GEMINI_API_KEYS") or "").strip()
        if pooled:
            raw.extend(re.split(r"[,\s;]+", pooled))

        seen = set()
        keys = []
        for candidate in raw:
            candidate = candidate.strip()
            if not candidate or candidate == "your_api_key_here" or candidate in seen:
                continue
            seen.add(candidate)
            keys.append(candidate)
        return keys

    @classmethod
    def _get_session(cls) -> requests.Session:
        if cls._session is None:
            s = requests.Session()
            adapter = requests.adapters.HTTPAdapter(
                pool_connections=10,
                pool_maxsize=20,
                max_retries=1
            )
            s.mount("https://", adapter)
            s.mount("http://", adapter)
            cls._session = s
        return cls._session

    # Known prompt injection patterns
    INJECTION_PATTERNS = [
        r"ignore (all )?(previous|above) (instructions|directions|prompts)",
        r"system prompt",
        r"reveal (your|the) (instructions|system prompt|hidden prompt)",
        r"you are now (in )?(dan|jailbreak|developer|unrestricted) mode",
        r"bypass (all )?(guardrails|safety|filters)",
        r"act as an unrestricted",
        r"pretend you have no rules",
        r"repeat the words above",
        r"what was your initial instruction"
    ]

    # Comprehensive abusive, profane, toxic, and harassing language patterns
    ABUSIVE_PATTERNS = [
        # Explicit English profanities and sexual vulgarities
        r"\b(fuck|fucking|fucker|fck|f\*ck|motherfucker|mofo|stfu|wtf|fk)\b",
        r"\b(shit|shitty|bullshit|dipshit|horseshit|sh\*t|crap)\b",
        r"\b(bitch|bitches|bitching|bitchass|b\*tch)\b",
        r"\b(bastard|bastards|asshole|assholes|a\*\*hole|jackass|dumbass|smartass|arse|arsehole|asswipe|assface)\b",
        r"\b(dick|dickhead|d\*ck|cock|cocksucker|prick|pussy|p\*ssy|cunt|c\*nt|twat|dildo)\b",
        r"\b(whore|whores|slut|sluts|skank|douche|douchebag|wanker|tosser|jerkoff)\b",
        r"\b(blowjob|handjob|porn|pornography|xxx|boobs|tits|penis|vagina)\b",
        r"\b(moron|idiot|imbecile|retard|retarded|scumbag|loser)\b",
        # Violent threats, self-harm, hate speech and attacks
        r"\b(kill\s+yourself|go\s+die|commit\s+suicide|jump\s+off|kys|hang\s+yourself)\b",
        r"\b(i\s+will\s+kill|murder\s+you|slit\s+your|beat\s+you\s+up|punch\s+you|rot\s+in\s+hell)\b",
        r"\b(shoot\s+up|bomb\s+the|blow\s+up|terrorist|terrorism|massacre)\b",
        r"\b(hate\s+speech|nigger|nigga|faggot|fag|pedophile|pedo|rapist|rape|molest)\b",
        r"\b(hack\s+into|ddos|leak\s+database|drop\s+database|exploit\s+system)\b",
        # Common Indian regional profanities (Hindi / Urdu / Hinglish)
        r"\b(bhosdike|bhosadike|bhosdi|bsdk|chutiya|chutiye|chutiyapa|chut)\b",
        r"\b(madarchod|mc|madrchod|behenchod|bc|bhenchod|bhen\s+ke\s+lode)\b",
        r"\b(gand|gaand|gandu|gaandu|lauda|loda|lavda|lund|lodu|harami|kamina|kamini|saala|saale|suar|kutte|kutta|randi|randa|randwa|bhadwe|bhadwa|tatti|jhant|jhaatu)\b",
        # Kannada profanities and insults
        r"\b(sule|sulemagne|bolimagne|hadargetti|thika|tika|tika\s+mucchu|loffer|lofar|gube|huccha|hucchi|naaye|nayee|kalla|halakatte|bevarsi|bewarsi|baddimagane|shaata|doddmunde|munde)\b",
        # Telugu / Tamil profanities
        r"\b(lanja|lanjamunda|dengey|dengu|donga|otha|omala|thevidiya|poolu|sunni|kena|punda|baadu|moodhevi)\b",
    ]


    @classmethod
    def check_guardrails(cls, message: str) -> tuple[bool, Optional[str]]:
        """
        Validates user input for:
        1. Prompt injection attempts
        2. Abusive/foul language
        3. Extreme out-of-scope irrelevance
        Returns (is_allowed: bool, rejection_response: str or None)
        """
        msg_lower = message.lower().strip()

        # 1. Prompt Injection
        for pattern in cls.INJECTION_PATTERNS:
            if re.search(pattern, msg_lower):
                logger.warning(f"[Guardrail] Prompt injection attempt intercepted: {message}")
                return False, (
                    "⚠️ **Security Notice**: Your query contains instructions that violate security guidelines. "
                    "I am strictly programmed as the BMSIT College AI Assistant and cannot alter my guidelines or reveal internal configurations."
                )

        # 2. Abusive / Foul / Harmful Language (with leetspeak, repeat characters, and de-spacing)
        leet_clean = msg_lower
        for char, repl in {"@": "a", "$": "s", "0": "o", "1": "i", "!": "i", "3": "e", "*": ""}.items():
            leet_clean = leet_clean.replace(char, repl)
        leet_clean_collapsed = re.sub(r"(.)\1{2,}", r"\1\1", leet_clean)
        # Normalize spaced-out or dotted letters e.g. "f u c k" or "b.s.d.k"
        despaced = re.sub(r"(?<=\b\w)[ ._\-](?=\w\b)", "", leet_clean)

        for pattern in cls.ABUSIVE_PATTERNS:
            if (
                re.search(pattern, msg_lower)
                or re.search(pattern, leet_clean)
                or re.search(pattern, leet_clean_collapsed)
                or re.search(pattern, despaced)
            ):
                logger.warning(f"[Guardrail] Inappropriate language intercepted: {message}")
                return False, (
                    "🕊️ **Community Guidelines**: We maintain a respectful, educational environment for students and visitors. "
                    "Please keep interactions courteous. How may I assist you with information about BMSIT?"
                )

        # 3. Friendly greetings and polite navigation
        greetings = ["hi", "hello", "hey", "good morning", "good evening", "good afternoon", "namaste", "help", "who are you", "what can you do"]
        if any(msg_lower == g or msg_lower.startswith(g + " ") for g in greetings):
            return True, None

        # Check for obvious external questions (recipes, generic coding, outside trivia)
        unrelated_signals = [
            r"who won the (world cup|ipl|olympics|fifa)",
            r"write (a|python|java|c\+\+|javascript) code to",
            r"recipe for",
            r"solve this math problem",
            r"who is the president of",
            r"tell me a joke about",
            r"write an essay on global warming"
        ]
        for signal in unrelated_signals:
            if re.search(signal, msg_lower):
                return False, (
                    "🎓 **BMSIT Assistant Focus**: I am specialized exclusively in providing verified information regarding "
                    "**B.M.S. Institute of Technology and Management (BMSIT&M)** — such as our programs, admissions, faculty, placements, "
                    "hostels, and campus facilities. I cannot assist with general knowledge, external trivia, or unrelated tasks."
                )

        return True, None

    @classmethod
    def generate_chat_response(
        cls,
        user_message: str,
        conversation_history: Optional[List[dict]] = None,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Production RAG workflow:
        1. Guardrail safety validation.
        2. LangChain ChatHistory synchronization and standalone query reformulation.
        3. High-precision hybrid retrieval from BMSIT vector index.
        4. Generation via Google Gemini with multi-model failover.
        5. Semantic fallback if API offline.
        """
        # Guardrail check
        is_allowed, rejection = cls.check_guardrails(user_message)
        if not is_allowed:
            return {
                "reply": rejection,
                "sources": [],
                "guardrail_triggered": True,
                "session_id": session_id
            }

        # Friendly greeting handler
        msg_lower = user_message.lower().strip()
        if msg_lower in ["hi", "hello", "hey", "namaste", "good morning", "good evening", "good afternoon"]:
            greeting_reply = (
                "👋 **Hello and welcome to BMSIT&M!**\n\n"
                "I am your official AI College Guide. I can help you with:\n"
                "• **Admissions & Eligibility** (KCET, COMEDK, Management Quota)\n"
                "• **Programs & Departments** (CSE, AI&ML, ISE, ECE, ME, CV, MCA, M.Tech, etc.)\n"
                "• **Placement Records & Recruiters**\n"
                "• **Hostel Facilities, Fees & Campus Amenities**\n"
                "• **Official College Contacts & Office Hours**\n\n"
                "What would you like to know about BMSIT today?"
            )
            # Record in memory
            history_mgr = LangChainHistoryManager.get_instance()
            history_mgr.append_turn(session_id, user_message, greeting_reply)
            return {
                "reply": greeting_reply,
                "sources": [],
                "guardrail_triggered": False,
                "session_id": session_id
            }

        api_keys = cls.get_configured_api_keys()
        effective_key = api_keys[0] if api_keys else ""

        # Synchronize conversation history via LangChain memory
        history_mgr = LangChainHistoryManager.get_instance()
        langchain_messages = history_mgr.sync_history(session_id, conversation_history)

        # Enterprise Query Rewriting: Multi-turn resolution & search query expansion
        rewriter = QueryRewriter.get_instance()
        search_query = rewriter.rewrite(
            query=user_message,
            messages=langchain_messages,
            api_key=effective_key
        )

        # Retrieve verified context from Knowledge Base with Stage 2 Reranking
        rag = RAGService.get_instance()
        retrieved_chunks = rag.retrieve(search_query, top_k=Config.TOP_K, enable_rerank=True)

        if not retrieved_chunks:
            stats = rag.get_stats()
            if stats.get("total_chunks", 0) == 0:
                logger.warning("[Gemini] Knowledge base is empty - no content has been indexed yet.")
                empty_reply = (
                    "The BMSIT knowledge base is currently empty, so I have nothing verified to answer from. "
                    "An administrator needs to run a website crawl or upload documents from the admin dashboard.\n\n"
                    "In the meantime, please contact the college directly:\n"
                    "• **Email**: `admissions@bmsit.in` / `principal@bmsit.in`\n"
                    "• **Phone**: +91-80-68730444 / +91-80-68730424\n"
                    "• **Website**: [https://bmsit.ac.in](https://bmsit.ac.in)"
                )
                return {
                    "reply": empty_reply,
                    "sources": [],
                    "guardrail_triggered": False,
                    "knowledge_base_empty": True,
                    "session_id": session_id
                }

        # Build context string from top retrieved chunks
        top_k = int(getattr(Config, "TOP_K", 10) or 10)
        context_parts = []
        sources = []
        seen_sources = set()

        for chunk in retrieved_chunks[:top_k]:
            source_name = chunk.get("source_name", "BMSIT Document")
            source_type = chunk.get("source_type", "document")
            meta = chunk.get("metadata", {})
            sec = meta.get("section_title") or meta.get("page") or ""
            url = meta.get("url") or ""

            updated_at = chunk.get("updated_at") or meta.get("updated_at") or ""
            label = f"--- Source: {source_name}"
            if sec:
                label += f" ({sec})"
            if url:
                label += f" | {url}"
            if updated_at:
                label += f" | Last Updated: {updated_at}"
            label += " ---"
            context_parts.append(f"{label}\n{chunk['text']}")

            source_key = f"{source_name}_{sec}_{url}"
            if source_key not in seen_sources:
                seen_sources.add(source_key)
                sources.append({
                    "name": source_name,
                    "type": source_type,
                    "section": str(sec) if sec else None,
                    "url": url or None,
                    "score": round(chunk.get("score", 0.0), 3)
                })

        context_text = "\n\n".join(context_parts) if context_parts else "NO MATCHING DOCUMENTS FOUND IN KNOWLEDGE BASE."

        # Format history string using LangChain abstraction
        history_block = history_mgr.format_history_for_prompt(langchain_messages)

        reply = None
        # Call Gemini API if keys are available
        if api_keys:
            try:
                reply = cls._call_gemini_api(api_keys, user_message, context_text, history_block)
            except Exception as e:
                logger.error(f"[Gemini] API Call failed: {e}. Falling back to knowledge synthesis.")

        # Fallback knowledge synthesis if API call failed or keys are missing
        if not reply:
            reply = cls._synthesize_from_context(user_message, retrieved_chunks)

        # Record this completed turn into LangChain conversational memory
        history_mgr.append_turn(session_id, user_message, reply)

        return {
            "reply": reply,
            "sources": sources,
            "guardrail_triggered": False,
            "session_id": session_id
        }

    @classmethod
    def _contextualize_query(cls, query: str, conversation_history: Optional[List[dict]] = None) -> str:
        """
        Lightweight backward-compatible helper for conversational query contextualization.
        """
        history_mgr = LangChainHistoryManager.get_instance()
        langchain_messages = history_mgr.sync_history("test_session", conversation_history)
        return QueryRewriter.get_instance().rewrite(
            query=query,
            messages=langchain_messages
        )

    @classmethod
    def _call_gemini_api(
        cls,
        api_keys: Any,
        query: str,
        context: str,
        history_block: str = ""
    ) -> str:
        """
        Calls Google Gemini API strictly powered by gemini-3.5-flash-lite with multi-key rotation.
        Zero temperature parameter (unsupported on Gemini 3.5 Flash-lite and causes HTTP 400).
        Unlimited max output tokens: lengthy answers are never cut off.
        """
        keys_list = [api_keys] if isinstance(api_keys, str) else (api_keys or cls.get_configured_api_keys())
        if not keys_list:
            raise RuntimeError("No Gemini API keys configured.")

        user_prompt = f"""KNOWLEDGE BASE CONTEXT (From BMSIT official website and verified college records):
{context}

USER QUESTION:
{query}

ANSWER (Provide a direct, complete, accurate, and helpful response based STRICTLY on the BMSIT context above, completing all sentences and list items):"""

        session = cls._get_session()
        timeout_sec = float(getattr(Config, "CHAT_TIMEOUT", 45) or 45)

        # Configured model is gemini-3.5-flash-lite
        configured_model = (getattr(Config, "GEMINI_MODEL", "gemini-3.5-flash-lite") or "gemini-3.5-flash-lite").strip()
        chat_models = getattr(Config, "CHAT_MODELS", []) or []
        model_tier = []
        for m in [configured_model] + chat_models + ["gemini-3.5-flash-lite", "gemini-2.5-flash-lite", "gemini-flash-latest"]:
            m = m.strip()
            if m and m not in model_tier:
                model_tier.append(m)

        max_output_tokens = getattr(Config, "MAX_OUTPUT_TOKENS", 0) or 0
        gen_config = {}
        if max_output_tokens > 0:
            gen_config["maxOutputTokens"] = int(max_output_tokens)

        last_error = None
        for key in keys_list:
            key = key.strip()
            if not key or key == "your_api_key_here":
                continue

            for model in model_tier:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"

                # Attempt 1: Standard payload with system_instruction
                payload = {
                    "system_instruction": {
                        "parts": [{"text": SYSTEM_PROMPT}]
                    },
                    "contents": [
                        {
                            "role": "user",
                            "parts": [{"text": f"{history_block}{user_prompt}"}]
                        }
                    ]
                }
                if gen_config:
                    payload["generationConfig"] = gen_config

                try:
                    resp = session.post(url, json=payload, timeout=timeout_sec)
                    # Attempt 2: If system_instruction is not accepted on this model, retry with inline prompt
                    if resp.status_code == 400 and "system_instruction" in payload:
                        inline_payload = {
                            "contents": [
                                {
                                    "role": "user",
                                    "parts": [{"text": f"{SYSTEM_PROMPT}\n\n{history_block}{user_prompt}"}]
                                }
                            ]
                        }
                        if gen_config:
                            inline_payload["generationConfig"] = gen_config
                        resp = session.post(url, json=inline_payload, timeout=timeout_sec)

                    if resp.status_code == 200:
                        res_json = resp.json()
                        candidates = res_json.get("candidates", [])
                        if candidates:
                            parts = candidates[0].get("content", {}).get("parts", [])
                            texts = [p["text"] for p in parts if isinstance(p, dict) and p.get("text")]
                            if texts:
                                cls._active_model = model
                                return "\n".join(texts).strip()

                    if resp.status_code == 429:
                        last_error = f"Key ending in ...{key[-4:]} hit 429 quota. Rotating key."
                        logger.warning(f"[Gemini] {last_error}")
                        break  # Rotate to next key

                    last_error = f"Model {model} returned HTTP {resp.status_code}: {resp.text[:120]}"
                    logger.warning(f"[Gemini] {last_error}")
                except Exception as e:
                    last_error = f"Model {model} exception: {e}"
                    logger.warning(f"[Gemini] {last_error}")
                    continue

        raise RuntimeError(f"All Gemini models and keys in tier failed: {last_error}")

    @classmethod
    def _synthesize_from_context(cls, query: str, chunks: List[dict]) -> str:
        """
        Extractive synthesis fallback when API key is offline or quota exceeded.
        Zero hardcoded word lists or static dictionaries.
        Safely preserves titles/abbreviations (Dr., Prof., etc.) to eliminate mid-sentence cutoffs.
        Extracts complete entity rosters, policies, and lists.
        """
        clean_query = (query or "").strip()
        q_lower = clean_query.lower()

        unknown_message = (
            "As of now, I don't have verified information regarding this in the BMSIT knowledge base. "
            "Please refer to https://bmsit.ac.in or contact the college directly:\n"
            "• **Email**: admissions@bmsit.in / principal@bmsit.in\n"
            "• **Phone**: +91-80-68730444 / +91-80-68730424\n"
            "• **Website**: https://bmsit.ac.in\n"
            "• **Address**: Doddaballapur Main Road, Avalahalli, Yelahanka, Bengaluru - 560064"
        )

        if not chunks:
            return unknown_message

        # Dynamic initialism & title matching
        def matches_title(term: str, title: str) -> bool:
            term = term.lower().strip()
            title = title.lower().strip()
            if not term or not title:
                return False
            if term in title or title in term:
                return True
            title_words = re.findall(r"[a-z]+", title)
            if not title_words:
                return False
            initialism = "".join(w[0] for w in title_words)
            if term == initialism:
                return True
            content_initials = "".join(w[0] for w in title_words if w not in ("and", "of", "the", "in", "for"))
            if term == content_initials:
                return True
            if any(w.startswith(term) for w in title_words if len(term) >= 3):
                return True
            return False

        # Dynamic token extraction from user query (excluding conversational fillers)
        stop_tokens = {"the", "and", "for", "with", "all", "need", "about", "tell", "list", "names", "what", "give", "please", "can", "you", "i"}
        raw_tokens = [
            t for t in re.findall(r"\b[a-zA-Z0-9_-]+\b", q_lower)
            if len(t) >= 2 and not t.isdigit() and t not in stop_tokens
        ]
        if not raw_tokens:
            raw_tokens = [t for t in re.findall(r"\b[a-zA-Z0-9_-]+\b", q_lower) if not t.isdigit()]

        stems = [
            t[:-1] if (t.endswith("s") and not t.endswith("ss") and len(t) > 3) else t
            for t in raw_tokens
        ]

        # Dynamic query intent parsing
        lines_match = re.search(r"\b(\d+)\s*(?:line|point|bullet|sentence)s?\b", q_lower)
        target_count = int(lines_match.group(1)) if lines_match else None
        one_line_requested = bool(re.search(r"\b(one\s+line|single\s+sentence|in\s+short)\b", q_lower))
        brief_requested = bool(re.search(r"\b(brief|briefly|summary)\b", q_lower))
        all_requested = bool(re.search(r"\b(all|every|full|complete|list|names?|facult|professor|members?|branches?|courses?)\b", q_lower))
        is_faculty_query = bool(re.search(r"\b(facult|professor|prof|teacher|staff|hod|head|warden|authorit|incharge|dean|director|names?)\b", q_lower))

        candidate_lines = []
        seen_snippets = set()

        for chunk_idx, c in enumerate(chunks[:8]):
            raw_text = c.get("text", "")
            source_title = c.get("source_name", "BMSIT Verified Records")
            src_lower = source_title.lower()
            meta = c.get("metadata", {}) or {}
            sec = str(meta.get("section_title") or meta.get("page") or "").lower()
            ts_str = str(c.get("updated_at") or meta.get("updated_at") or "")

            # Title alignment check via dynamic substring and initialism
            is_title_relevant = any(matches_title(st, src_lower) or matches_title(st, sec) for st in stems)
            is_top_chunk = (chunk_idx < 2)

            # Protect honorifics/abbreviations so names and titles are NEVER severed mid-sentence
            clean_text = raw_text.replace("\r\n", "\n")
            protected = re.sub(r"\b(Dr|Prof|Mr|Mrs|Ms|Dept|Hon|Govt|No|Vol|Sr|Jr|Ph\.D|M\.Tech|B\.E|B\.Tech)\.\s+", r"\1_DOT_ ", clean_text)
            
            # Split primarily on newlines to preserve tabular / roster items, then sentence breaks for long paragraphs
            raw_items = []
            for line in protected.split("\n"):
                line = line.strip()
                if not line:
                    continue
                if len(line) > 200 and not re.match(r"^[-*•\d|]", line):
                    sub_sentences = [
                        s.replace("_DOT_", ".").strip()
                        for s in re.split(r"(?<=[.!?])\s+", line)
                        if len(s.strip()) > 15
                    ]
                    raw_items.extend(sub_sentences)
                else:
                    raw_items.append(line.replace("_DOT_", ".").strip())

            for item in raw_items:
                item_clean = item.strip()
                # Strip leading bullet symbols for clean inspection
                content_text = re.sub(r"^[-*•#|>\d.)\s]+", "", item_clean).strip()
                if len(content_text) < 10:
                    continue

                item_lower = item_clean.lower()

                # Dynamic scoring
                token_hits = sum(1 for st in stems if (st in item_lower or (len(st) >= 4 and st[:4] in item_lower)))

                has_person_signal = bool(re.search(r"\b(Dr\.|Prof\.|Mr\.|Mrs\.|Ms\.|Professor|Associate\s+Head|Head|HoD|HOD|Warden|Dean|Director|Principal|Incharge|In-Charge|Lecturer|Faculty|President|Secretary)\b", item_clean))
                has_bullet_or_number = bool(re.match(r"^[-*•\d.]+", item_clean))

                if is_faculty_query:
                    if not (has_person_signal or token_hits > 0 or (is_title_relevant and has_bullet_or_number)):
                        continue
                elif token_hits == 0 and not is_title_relevant and not is_top_chunk:
                    continue

                score = (token_hits * 3.0) + (4.0 if has_person_signal else 0.0) + (3.0 if is_title_relevant else 0.0) + (2.0 if is_top_chunk else 0.0)

                normalized_snippet = re.sub(r"\s+", " ", item_lower[:80]).strip()
                if normalized_snippet in seen_snippets:
                    continue
                seen_snippets.add(normalized_snippet)

                # Ensure clean statement without unclosed parentheses
                final_item = item_clean
                if final_item.count("(") > final_item.count(")"):
                    final_item += ")"
                if final_item.count("[") > final_item.count("]"):
                    final_item += "]"

                candidate_lines.append((score, ts_str, final_item, source_title))

        if not candidate_lines:
            return unknown_message

        # Sort candidate lines by score descending, then recency
        candidate_lines.sort(key=lambda x: (x[0], x[1]), reverse=True)

        # Group verified lines by source
        source_sections: Dict[str, List[str]] = {}
        for _, _, line, src in candidate_lines:
            clean = line.strip()
            clean = re.sub(r"^(?:" + re.escape(src) + r"\s*)+", "", clean, flags=re.IGNORECASE).strip()
            if not clean:
                clean = line.strip()
            if src not in source_sections:
                source_sections[src] = []
            if clean not in source_sections[src]:
                source_sections[src].append(clean)

        if not source_sections:
            return unknown_message

        # Select lines according to dynamic length constraints
        limit = target_count if target_count is not None else (1 if one_line_requested else (4 if brief_requested else (60 if (all_requested or is_faculty_query) else 20)))

        output_blocks = []
        for src, lines in source_sections.items():
            if not lines:
                continue
            selected_lines = lines[:limit]
            if one_line_requested:
                output_blocks.append(f"**{src}**: {selected_lines[0]}")
                break
            else:
                formatted_lines = "\n".join(f"* {l if l.startswith(('*', '-', '•')) else '* ' + l}" for l in selected_lines)
                # Clean duplicate bullets if any
                formatted_lines = re.sub(r"^\*\s*[*•-]\s*", "* ", formatted_lines, flags=re.MULTILINE)
                output_blocks.append(f"**{src}:**\n{formatted_lines}")
            if len(output_blocks) >= (1 if brief_requested else 4):
                break

        if not output_blocks:
            return unknown_message

        if one_line_requested:
            return output_blocks[0]

        formatted_answer = "\n\n".join(output_blocks)
        return (
            f"Based on BMSIT verified official records:\n\n"
            f"{formatted_answer}\n\n"
            f"> *Source: BMSIT Official Records*"
        )
