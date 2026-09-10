"""
Daily cron layer.

The scheduler owns only the trigger. All ingestion logic lives in
`ingestion_pipeline`, so a manual "Scrape Now" and the 7:00 AM IST job run
exactly the same delta path.
"""
import logging
from zoneinfo import ZoneInfo

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from app.config import Config
from app.services.ingestion_pipeline import run_website_delta_ingestion
from app.services.storage import StorageService

logger = logging.getLogger(__name__)

scheduler = BackgroundScheduler(daemon=True)

JOB_ID = "bmsit_daily_delta_scrape"


def execute_bmsit_scrape_and_index(is_scheduled=False):
    """Kept as the public entry point used by routes, tests and the cron job."""
    return run_website_delta_ingestion(is_scheduled=is_scheduled)


class SchedulerService:
    @classmethod
    def start(cls):
        if scheduler.running:
            return

        settings = StorageService.load_settings()
        schedule_time = settings.get("schedule_time", Config.SCHEDULE_TIME)
        timezone = settings.get("schedule_timezone", Config.SCHEDULE_TIMEZONE)

        try:
            hour, minute = [int(part) for part in str(schedule_time).split(":")[:2]]
        except Exception:
            hour, minute, schedule_time = 7, 0, "07:00"

        scheduler.add_job(
            func=execute_bmsit_scrape_and_index,
            args=[True],
            trigger=CronTrigger(hour=hour, minute=minute, timezone=ZoneInfo(timezone)),
            id=JOB_ID,
            replace_existing=True,
            name=f"Daily BMSIT delta crawl at {schedule_time} {timezone}",
            misfire_grace_time=3600,
            coalesce=True,
            max_instances=1,
        )
        scheduler.start()
        logger.info("[Scheduler] Daily delta crawl scheduled at %s %s.", schedule_time, timezone)

    @classmethod
    def reschedule(cls):
        """Applies a changed schedule without restarting the server."""
        if not scheduler.running:
            cls.start()
            return cls.get_status()

        settings = StorageService.load_settings()
        schedule_time = settings.get("schedule_time", Config.SCHEDULE_TIME)
        timezone = settings.get("schedule_timezone", Config.SCHEDULE_TIMEZONE)
        try:
            hour, minute = [int(part) for part in str(schedule_time).split(":")[:2]]
        except Exception:
            hour, minute = 7, 0
        scheduler.reschedule_job(
            JOB_ID, trigger=CronTrigger(hour=hour, minute=minute, timezone=ZoneInfo(timezone))
        )
        return cls.get_status()

    @classmethod
    def get_status(cls):
        job = scheduler.get_job(JOB_ID) if scheduler.running else None
        settings = StorageService.load_settings()
        return {
            "is_running": scheduler.running,
            "next_run_time": str(job.next_run_time) if job and job.next_run_time else "Not scheduled",
            "schedule": (
                f"Every day at {settings.get('schedule_time', Config.SCHEDULE_TIME)} "
                f"({settings.get('schedule_timezone', Config.SCHEDULE_TIMEZONE)})"
            ),
        }
