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


def execute_bmsit_scrape_and_index(is_scheduled=False, trigger_label=None):
    """Kept as the public entry point used by routes, tests and the cron job."""
    return run_website_delta_ingestion(is_scheduled=is_scheduled, trigger_label=trigger_label)


class SchedulerService:
    @classmethod
    def get_next_friday_run(cls):
        from datetime import datetime, timedelta
        import zoneinfo
        try:
            tz = zoneinfo.ZoneInfo("Asia/Kolkata")
            now = datetime.now(tz)
            # Friday is weekday 4 (Monday=0 ... Friday=4)
            days_ahead = (4 - now.weekday()) % 7
            target_today = now.replace(hour=12, minute=43, second=0, microsecond=0)
            if days_ahead == 0 and now >= target_today:
                days_ahead = 7
            next_friday = (now + timedelta(days=days_ahead)).replace(hour=12, minute=43, second=0, microsecond=0)
            return next_friday.strftime("%Y-%m-%d %H:%M:%S%z")
        except Exception:
            return "Next Friday at 12:43 PM IST"

    @classmethod
    def start(cls):
        if not getattr(Config, "ENABLE_CLOUD_SCHEDULER", False):
            logger.info("[Scheduler] Cloud Cron is disabled. Delta scraping runs via GitHub Actions on Fridays.")
            return

        if scheduler.running:
            return

        settings = StorageService.load_settings()
        schedule_time = settings.get("schedule_time", Config.SCHEDULE_TIME)
        timezone = settings.get("schedule_timezone", Config.SCHEDULE_TIMEZONE)

        try:
            hour, minute = [int(part) for part in str(schedule_time).split(":")[:2]]
        except Exception:
            hour, minute, schedule_time = 12, 43, "12:43"

        scheduler.add_job(
            func=execute_bmsit_scrape_and_index,
            args=[True],
            trigger=CronTrigger(day_of_week="fri", hour=hour, minute=minute, timezone=ZoneInfo(timezone)),
            id=JOB_ID,
            replace_existing=True,
            name=f"Friday BMSIT delta crawl at {schedule_time} {timezone}",
            misfire_grace_time=3600,
            coalesce=True,
            max_instances=1,
        )
        scheduler.start()
        logger.info("[Scheduler] Friday delta crawl scheduled at %s %s.", schedule_time, timezone)

    @classmethod
    def reschedule(cls):
        """Applies a changed schedule without restarting the server."""
        if not getattr(Config, "ENABLE_CLOUD_SCHEDULER", False):
            return cls.get_status()

        if not scheduler.running:
            cls.start()
            return cls.get_status()

        settings = StorageService.load_settings()
        schedule_time = settings.get("schedule_time", Config.SCHEDULE_TIME)
        timezone = settings.get("schedule_timezone", Config.SCHEDULE_TIMEZONE)
        try:
            hour, minute = [int(part) for part in str(schedule_time).split(":")[:2]]
        except Exception:
            hour, minute = 12, 43
        scheduler.reschedule_job(
            JOB_ID, trigger=CronTrigger(day_of_week="fri", hour=hour, minute=minute, timezone=ZoneInfo(timezone))
        )
        return cls.get_status()

    @classmethod
    def get_status(cls):
        job = scheduler.get_job(JOB_ID) if scheduler.running else None
        settings = StorageService.load_settings()
        schedule_time = settings.get("schedule_time", Config.SCHEDULE_TIME)
        timezone = settings.get("schedule_timezone", Config.SCHEDULE_TIMEZONE)
        sched_label = f"Every Friday at {schedule_time} ({timezone})"

        if job and job.next_run_time:
            next_run = str(job.next_run_time)
        else:
            next_run = cls.get_next_friday_run()

        return {
            "is_running": scheduler.running,
            "next_run_time": next_run,
            "schedule": sched_label,
            "mode": "cloud_cron" if scheduler.running else "github_actions",
        }
