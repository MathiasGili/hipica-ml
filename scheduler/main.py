"""Lightweight nightly scheduler.

Runs once a day (default 06:30 UY time) and pre-warms the API cache by
calling ``/predict_program`` for today + tomorrow on every configured
racetrack. The actual scraping, OCR and prediction happen inside the
API container — this scheduler is a dumb HTTP client.

Configuration via env:

* ``API_URL``         (default ``http://api:8000``)
* ``RACETRACK_IDS``   comma-separated, default ``1`` (Maroñas)
* ``CRON_HOUR``       default ``6``
* ``CRON_MINUTE``     default ``30``
* ``RUN_ON_START``    ``true``/``false``, default ``true`` — also run
                      immediately at container start so logs show
                      something on day-zero.
"""
from __future__ import annotations

import logging
import os
import time
from datetime import date, timedelta

import requests
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

logger = logging.getLogger("hipica.scheduler")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)

API_URL = os.getenv("API_URL", "http://api:8000")
RACETRACK_IDS = [int(x.strip()) for x in os.getenv("RACETRACK_IDS", "1").split(",") if x.strip()]
CRON_HOUR = int(os.getenv("CRON_HOUR", "6"))
CRON_MINUTE = int(os.getenv("CRON_MINUTE", "30"))
RUN_ON_START = os.getenv("RUN_ON_START", "true").lower() == "true"


def _wait_for_api(timeout: int = 120) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{API_URL}/health", timeout=3)
            if r.status_code == 200:
                logger.info("API is healthy: %s", r.json())
                return True
        except requests.RequestException:
            pass
        time.sleep(2)
    logger.error("API did not become healthy within %ds", timeout)
    return False


def _scrape_one(racetrack_id: int, race_date: date) -> None:
    payload = {
        "race_date": race_date.isoformat(),
        "racetrack_id": racetrack_id,
        "force_refresh": False,
    }
    try:
        r = requests.post(f"{API_URL}/predict_program", json=payload, timeout=180)
        if r.status_code == 200:
            body = r.json()
            logger.info(
                "rt=%d  date=%s  → %d races (model=%s v%s)",
                racetrack_id, race_date, body["n_races"],
                body["model_name"], body["model_version"] or "—",
            )
        elif r.status_code == 404:
            logger.info("rt=%d  date=%s  → no Programa published (skip)",
                        racetrack_id, race_date)
        else:
            logger.warning("rt=%d  date=%s  → HTTP %d %s",
                           racetrack_id, race_date, r.status_code, r.text[:200])
    except requests.RequestException as exc:
        logger.error("rt=%d  date=%s  → network error: %s",
                     racetrack_id, race_date, exc)


def daily_job() -> None:
    """Pre-warm cache for today + tomorrow on every configured racetrack."""
    today = date.today()
    targets = [today, today + timedelta(days=1)]
    logger.info("Running daily job for tracks=%s targets=%s",
                RACETRACK_IDS, [t.isoformat() for t in targets])
    for rt in RACETRACK_IDS:
        for d in targets:
            _scrape_one(rt, d)


def main() -> None:
    logger.info("Hipica scheduler starting | API=%s tracks=%s cron=%02d:%02d",
                API_URL, RACETRACK_IDS, CRON_HOUR, CRON_MINUTE)
    if not _wait_for_api():
        # Don't crash; the cron will keep trying.
        logger.warning("Continuing despite API health timeout")

    scheduler = BlockingScheduler(timezone="America/Montevideo")
    scheduler.add_job(
        daily_job,
        CronTrigger(hour=CRON_HOUR, minute=CRON_MINUTE),
        id="daily_program_scrape",
        max_instances=1,
        coalesce=True,
    )

    if RUN_ON_START:
        try:
            daily_job()
        except Exception as exc:  # noqa: BLE001
            logger.exception("Initial run failed: %s", exc)

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler shutting down")


if __name__ == "__main__":
    main()
