"""Tests for src.ingestion.program.

The full pipeline (scrape + OCR) requires network and Tesseract; these
tests cover the defensive bits we can hit without either.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from src.ingestion.program import fetch_program_xls
from src.ingestion.scraper import RaceDay


@pytest.fixture()
def fake_scraper():
    """A MaronasScraper-like stub that doesn't touch the network."""

    class _StubScraper:
        # Methods used by fetch_program_xls
        def _post(self, method, payload):  # noqa: ARG002
            return [{"Uri": "/XTurfResourcesService.svc/GetRacingDocument?stub=1"}]

        def _download_excel(self, uri, dest):  # noqa: ARG002
            dest.parent.mkdir(parents=True, exist_ok=True)
            # Write an HTML error page (what Maroñas sends when a Programa
            # isn't published yet) instead of a real BIFF .xls.
            dest.write_bytes(b"\r\n\r\n\r\n<!DOCTYPE html><html>oops</html>")
            return dest.stat().st_size

    return _StubScraper()


def test_fetch_program_returns_none_when_server_sends_html(tmp_path, fake_scraper):
    """The Maroñas service sometimes hands back an HTML error page with a
    .xls extension. The parser must detect this (via OLE2 BIFF magic) and
    return None — not crash downstream with XLRDError.
    """
    day = RaceDay(racetrack_id=1, date=datetime(2026, 6, 18), has_program=True)
    out = fetch_program_xls(day, dest_dir=tmp_path, scraper=fake_scraper)

    assert out is None, "HTML-as-xls must be rejected"
    # The defensive branch also cleans up the bogus file.
    assert not (tmp_path / "Maroñas" / "Programa_RT1_20260618.xls").exists()


def test_fetch_program_returns_path_for_valid_ole2_header(tmp_path):
    """If the downloaded file starts with the OLE2 magic, the parser
    trusts it (xlrd validation happens later, downstream).
    """
    class _Stub:
        def _post(self, method, payload):  # noqa: ARG002
            return [{"Uri": "/stub"}]

        def _download_excel(self, uri, dest):  # noqa: ARG002
            dest.parent.mkdir(parents=True, exist_ok=True)
            # Real BIFF8 .xls files start with this magic; the rest can be junk
            # for this test — we only validate the head detection.
            dest.write_bytes(b"\xd0\xcf\x11\xe0" + b"\x00" * 1024)
            return dest.stat().st_size

    day = RaceDay(racetrack_id=1, date=datetime(2026, 6, 19), has_program=True)
    out = fetch_program_xls(day, dest_dir=tmp_path, scraper=_Stub())

    assert out is not None
    assert out.name == "Programa_RT1_20260619.xls"
    assert out.exists()
