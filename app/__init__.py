import logging
import os
import sys

from flask import Flask

from app.config import Config


def _configure_logging():
    """
    Sends application logs to stdout at the configured level.

    Without this, every logger.info/warning in the ingestion pipeline was
    discarded and a run could only be diagnosed by guessing.
    """
    level = getattr(logging, Config.LOG_LEVEL.upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(level)

    if not any(getattr(h, "_bmsit", False) for h in root.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S"
        ))
        handler._bmsit = True
        root.addHandler(handler)

    # Werkzeug's per-request lines are noise at INFO.
    logging.getLogger("werkzeug").setLevel(logging.WARNING)


def create_app():
    Config.validate()
    Config.ensure_directories()
    _configure_logging()

    app = Flask(
        __name__,
        template_folder=str(Config.BASE_DIR / "templates"),
        static_folder=str(Config.BASE_DIR / "static"),
    )
    app.config["SECRET_KEY"] = Config.SECRET_KEY
    app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024  # 64 MB upload ceiling

    from app.routes.admin_routes import admin_bp
    from app.routes.chat_routes import chat_bp

    app.register_blueprint(admin_bp)
    app.register_blueprint(chat_bp)

    # Load the vector index, then bring any pre-existing data up to date.
    from app.services.rag_service import RAGService
    rag = RAGService.get_instance()
    logging.getLogger(__name__).info(
        "Knowledge base ready: %s chunk(s) across %s source(s).",
        rag.get_stats()["total_chunks"], rag.get_stats()["indexed_sources_count"],
    )

    from app.services.migration import run_migrations
    try:
        run_migrations()
    except Exception as e:
        logging.getLogger(__name__).warning("Migration warning: %s", e)

    # Start the daily job once, in the serving process only.
    if not app.debug or os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        from app.services.scheduler_service import SchedulerService
        try:
            SchedulerService.start()
        except Exception as e:
            logging.getLogger(__name__).warning("Scheduler did not start: %s", e)

    return app
