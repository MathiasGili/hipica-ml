"""Scraper for hipica.maronas.com.uy.

The public site at https://hipica.maronas.com.uy/ is an AngularJS SPA that
talks to a REST endpoint hosted on Azure App Service. This module replays
the same JSON calls the browser makes, so we never need to render JavaScript.

Two endpoints are used:

* ``POST /XTurfRestService.svc/GetRacingCalendarHistory`` — returns the list
  of race days for a given racetrack and date range.
* ``POST /XTurfRestService.svc/GetDocuments`` (with ``documentType=2``,
  i.e. ``Tabulada``) — returns the URI of the per-race-day Tabulada file.

The Tabulada is then downloaded as Excel via
``GET /XTurfResourcesService.svc/GetRacingDocument?...&exportFormat=Excel``.

Files are stored under ``data/raw/<racetrack>/Tabulada_<RT>_<YYYYMMDD>.xls``.
The scraper is idempotent: existing files are skipped unless ``force=True``.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ..config import (
    DOC_TYPE_TABULADA,
    MARONAS_RESOURCES_URL,
    MARONAS_REST_URL,
    RACETRACKS,
    ScraperSettings,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RaceDay:
    """A single racing date returned by GetRacingCalendarHistory."""

    racetrack_id: int
    date: datetime          # midnight on the racing date
    has_program: bool


class MaronasScraper:
    """Thin client over the XTurf REST service.

    Parameters
    ----------
    settings:
        Optional :class:`ScraperSettings`. Defaults are read from env vars.
    session:
        Optional pre-configured ``requests.Session`` (useful for tests).
    """

    def __init__(
        self,
        settings: ScraperSettings | None = None,
        session: requests.Session | None = None,
    ) -> None:
        self.settings = settings or ScraperSettings()
        self.session = session or requests.Session()
        self.session.headers.update(
            {
                "Content-Type": "application/json",
                "Accept": "application/json, */*",
                "User-Agent": "hipica-ml/0.1 (academic scraper)",
            }
        )

    # ------------------------------------------------------------------ helpers
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        retry=retry_if_exception_type(requests.RequestException),
        reraise=True,
    )
    def _post(self, method: str, payload: dict) -> dict | list:
        url = f"{MARONAS_REST_URL}/{method}"
        resp = self.session.post(url, json=payload, timeout=self.settings.http_timeout)
        resp.raise_for_status()
        # The Azure-hosted service occasionally emits a UTF-8 BOM, which
        # breaks ``resp.json()``. Decode defensively with utf-8-sig.
        text = resp.content.decode("utf-8-sig")
        import json as _json
        return _json.loads(text)

    # ------------------------------------------------------------------ calendar
    def list_race_days(
        self,
        racetrack_id: int | None = None,
        date_from: str | datetime | None = None,
        date_to: str | datetime | None = None,
    ) -> list[RaceDay]:
        """Return the racing dates for ``racetrack_id`` between the given range.

        Replicates ``racingNavigationService.GetRacingCalendarHistory`` in the
        public AngularJS bundle.
        """
        rt = racetrack_id or self.settings.racetrack_id
        d_from = _to_iso(date_from or self.settings.date_from)
        d_to = _to_iso(date_to or self.settings.date_to)

        payload = {
            "raceTrackId": rt,
            "from": d_from,
            "to": d_to,
            "onlyWithProgramData": True,
            "nextDatesFirst": False,
            "sessionInfo": None,
        }
        raw = self._post("GetRacingCalendarHistory", payload)
        days: list[RaceDay] = []
        for item in raw or []:
            try:
                date = datetime.fromisoformat(item["Date"].replace("Z", ""))
            except Exception:  # noqa: BLE001 — defensive against API quirks
                logger.warning("Skipping malformed calendar entry: %r", item)
                continue
            days.append(
                RaceDay(
                    racetrack_id=rt,
                    date=date,
                    has_program=bool(item.get("IsProgramPublished")),
                )
            )
        days.sort(key=lambda d: d.date)
        logger.info(
            "Calendar: %s -> %d race days between %s and %s",
            RACETRACKS.get(rt, rt),
            len(days),
            d_from,
            d_to,
        )
        return days

    # ------------------------------------------------------------------ tabulada
    def _get_tabulada_uri(self, day: RaceDay) -> str | None:
        payload = {
            "documentType": DOC_TYPE_TABULADA,
            "raceTrackId": day.racetrack_id,
            "periodId": None,
            "date": day.date.strftime("%Y-%m-%dT00:00:00"),
            "raceOrdinal": None,
            "withBinaryData": False,
            "sessionInfo": None,
        }
        docs = self._post("GetDocuments", payload)
        if not docs:
            return None
        return docs[0].get("Uri")

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        retry=retry_if_exception_type(requests.RequestException),
        reraise=True,
    )
    def _download_excel(self, uri: str, dest: Path) -> int:
        # The published Uri starts with "/XTurfResourcesService.svc/...".
        # Strip the prefix because we already have it in ``MARONAS_RESOURCES_URL``.
        path_only = uri.split("/XTurfResourcesService.svc", 1)[-1]
        url = f"{MARONAS_RESOURCES_URL}{path_only}&exportFormat=Excel"
        with self.session.get(url, stream=True, timeout=self.settings.http_timeout) as resp:
            resp.raise_for_status()
            dest.parent.mkdir(parents=True, exist_ok=True)
            written = 0
            with dest.open("wb") as fh:
                for chunk in resp.iter_content(chunk_size=64 * 1024):
                    if chunk:
                        fh.write(chunk)
                        written += len(chunk)
            return written

    def fetch_tabulada(self, day: RaceDay, force: bool = False) -> Path | None:
        """Download the Tabulada Excel for ``day`` if not already on disk."""
        track = RACETRACKS.get(day.racetrack_id, f"RT{day.racetrack_id}")
        slug = day.date.strftime("%Y%m%d")
        dest = (
            self.settings.raw_dir
            / track.replace(" ", "_")
            / f"Tabulada_RT{day.racetrack_id}_{slug}.xls"
        )
        if dest.exists() and not force:
            logger.debug("Skipping %s — already downloaded", dest.name)
            return dest

        uri = self._get_tabulada_uri(day)
        if not uri:
            logger.warning("No Tabulada published for %s on %s", track, slug)
            return None
        size = self._download_excel(uri, dest)
        logger.info("Downloaded %s (%.0f KB)", dest, size / 1024)
        return dest

    # ------------------------------------------------------------------ pipeline
    def run(
        self,
        racetrack_id: int | None = None,
        date_from: str | datetime | None = None,
        date_to: str | datetime | None = None,
        force: bool = False,
    ) -> list[Path]:
        """Download every Tabulada in ``[date_from, date_to]``.

        Returns the list of paths actually written or already present.
        """
        days = self.list_race_days(racetrack_id, date_from, date_to)
        if not days:
            return []

        paths: list[Path] = []
        with ThreadPoolExecutor(max_workers=self.settings.max_workers) as pool:
            futures = {pool.submit(self.fetch_tabulada, d, force): d for d in days}
            for fut in as_completed(futures):
                day = futures[fut]
                try:
                    p = fut.result()
                    if p is not None:
                        paths.append(p)
                except Exception as exc:  # noqa: BLE001
                    logger.error("Failed to fetch %s: %s", day, exc)
        paths.sort()
        return paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _to_iso(value: str | datetime) -> str:
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%dT00:00:00")
    # Accepts "YYYY-MM-DD" or full ISO
    if "T" not in value:
        value = f"{value}T00:00:00"
    return value


def _parse_iter_dates(values: Iterable[str]) -> list[datetime]:
    return [datetime.fromisoformat(v.replace("Z", "")) for v in values]


def main() -> None:  # pragma: no cover — CLI entrypoint
    """``python -m src.ingestion.scraper`` — runs the scraper end-to-end."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    parser = argparse.ArgumentParser(description="Maroñas Tabulada scraper")
    parser.add_argument("--racetrack", type=int, default=None, help="Race-track id (1=Maroñas)")
    parser.add_argument("--from", dest="date_from", type=str, default=None)
    parser.add_argument("--to", dest="date_to", type=str, default=None)
    parser.add_argument("--force", action="store_true", help="Re-download even if file exists")
    args = parser.parse_args()

    scraper = MaronasScraper()
    written = scraper.run(
        racetrack_id=args.racetrack,
        date_from=args.date_from,
        date_to=args.date_to,
        force=args.force,
    )
    print(f"Done — {len(written)} Tabuladas in {scraper.settings.raw_dir}")


if __name__ == "__main__":  # pragma: no cover
    main()
