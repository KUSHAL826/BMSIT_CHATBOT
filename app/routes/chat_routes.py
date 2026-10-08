import logging

from flask import Blueprint, jsonify, render_template, request

from app.services.gemini_service import GeminiService
from app.services.rag_service import RAGService

logger = logging.getLogger(__name__)

chat_bp = Blueprint("chat", __name__, url_prefix="")


@chat_bp.route("/chatbot")
def chatbot_page():
    """Public standalone chatbot, safe to link from the college website."""
    return render_template("chatbot.html")


@chat_bp.route("/health")
@chat_bp.route("/healthz")
def health():
    """Liveness plus a summary of what the knowledge base currently holds."""
    stats = RAGService.get_instance().get_stats()
    return jsonify({
        "status": "ok",
        "chunks": stats["total_chunks"],
        "vectors": stats["total_vectors"],
        "sources": stats["indexed_sources_count"],
        "active_embedding_provider": stats["active_provider"],
        "pending_reembed": stats["needs_reembed"],
    })


@chat_bp.route("/api/chat", methods=["POST"])
def process_chat_message():
    """
    Guardrails, then hybrid RAG retrieval, then answer generation with citations.
    Supports LangChain session chat history.
    """
    raw_data = request.get_json(silent=True)
    if not isinstance(raw_data, dict):
        return jsonify({"error": "Invalid request payload. Expected a JSON object."}), 400

    raw_msg = raw_data.get("message")
    if not isinstance(raw_msg, str):
        return jsonify({"error": "Invalid message: must be a text string."}), 400

    message = raw_msg.strip()
    if not message:
        return jsonify({"error": "Empty message."}), 400
    if len(message) > 2000:
        return jsonify({"error": "Message too long. Please keep it under 2000 characters."}), 400

    raw_history = raw_data.get("history")
    history = raw_history if isinstance(raw_history, list) else []

    raw_sid = raw_data.get("session_id")
    session_id = str(raw_sid).strip() if raw_sid is not None and str(raw_sid).strip() else None

    try:
        return jsonify(GeminiService.generate_chat_response(
            user_message=message,
            conversation_history=history,
            session_id=session_id
        ))

    except Exception as e:
        logger.exception("Error processing chat message: %s", e)
        return jsonify({
            "reply": "Something went wrong while checking the BMSIT knowledge base. Please try again.",
            "sources": [],
            "guardrail_triggered": False,
        }), 500


@chat_bp.route("/api/chat/clear", methods=["POST"])
def clear_chat_session():
    """Clears in-memory LangChain chat history for a session."""
    data = request.json or {}
    session_id = (data.get("session_id") or "").strip()
    if session_id:
        from app.services.langchain_history import LangChainHistoryManager
        LangChainHistoryManager.get_instance().clear_session(session_id)
    return jsonify({"status": "cleared"})

