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
    """
    data = request.json or {}
    message = (data.get("message") or "").strip()
    history = data.get("history", [])

    if not message:
        return jsonify({"error": "Empty message"}), 400
    if len(message) > 2000:
        return jsonify({"error": "Message too long. Please keep it under 2000 characters."}), 400

    try:
        return jsonify(GeminiService.generate_chat_response(
            user_message=message, conversation_history=history
        ))
    except Exception as e:
        logger.exception("Error processing chat message: %s", e)
        return jsonify({
            "reply": "Something went wrong while checking the BMSIT knowledge base. Please try again.",
            "sources": [],
            "guardrail_triggered": False,
        }), 500
