from datetime import timedelta
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

    log = logging.getLogger(__name__)
    log.info("Configuration: %s", Config.source_summary())
    for warning in Config.warnings():
        log.warning("Configuration: %s", warning)

    app = Flask(
        __name__,
        template_folder=str(Config.BASE_DIR / "templates"),
        static_folder=str(Config.BASE_DIR / "static"),
    )
    app.config["SECRET_KEY"] = Config.SECRET_KEY
    app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024  # 64 MB upload ceiling
    app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(
        minutes=int(getattr(Config, "ADMIN_SESSION_TIMEOUT_MINUTES", 20) or 20)
    )
    app.config["SESSION_COOKIE_HTTPONLY"] = True
    app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

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

        # Automated background delta scrape on server startup / container deployment
        auto_scrape = getattr(Config, "AUTO_SCRAPE_ON_STARTUP", False)
        if auto_scrape:
            import threading
            import time
            from app.services.scheduler_service import execute_bmsit_scrape_and_index

            def _background_startup_crawl():
                time.sleep(15)  # allow web server to bind port and serve traffic first
                try:
                    logging.getLogger(__name__).info("[Startup] Initiating automated background delta scrape & embedding...")
                    execute_bmsit_scrape_and_index(is_scheduled=True, trigger_label="Startup Auto-Crawl")
                except Exception as ex:
                    logging.getLogger(__name__).warning("[Startup] Auto-crawl error: %s", ex)

            threading.Thread(target=_background_startup_crawl, daemon=True, name="StartupAutoCrawl").start()

    return app
