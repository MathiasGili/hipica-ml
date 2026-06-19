"""Programa parser — extracts upcoming/recent race-day cards from Maroñas.

The Programa is the pre-race document: it lists every horse entered in
each race of a given day, but does NOT include the race distance as
plain text. The distance lives inside an embedded image badge
(``XXXX mts``), which we extract via OCR.

Pipeline:

1. ``fetch_program_xls(day)`` — replays the same REST call as the
   scraper but with ``documentType=1``.
2. ``_xls_to_xlsx(path)`` — runs LibreOffice headless to convert the
   Crystal-Reports ``.xls`` into ``.xlsx`` so we can read the embedded
   shapes/images via openpyxl/zipfile.
3. ``_extract_distances(xlsx_path, dump_dir)`` — pulls the badge
   images from ``xl/media/`` in race order (anchor row in
   ``xl/drawings/drawing1.xml``) and OCRs them with Tesseract.
4. ``_parse_entries(xls_path)`` — iterates the sheet and groups rows
   into per-race blocks; offsets matching the live layout
   (col 0 post#, col 2 horse name, col 11 kg, col 13 track preference,
   col 14 sex, col 15 age, col 16 jockey).

Public entry point: :func:`fetch_program` returns a list of
:class:`RaceCard` ready to feed into ``/predict_batch``.
"""
from __future__ import annotations

import logging
import re
import subprocess
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree as ET

import xlrd
from PIL import Image, ImageOps

from ..config import MARONAS_RESOURCES_URL, RACETRACKS
from .scraper import MaronasScraper, RaceDay

logger = logging.getLogger(__name__)

DOC_TYPE_PROGRAM: int = 1

# Matches "1100 mts", "(2000mts)", "1.600 mt", "1100 m"
_DIST_RE = re.compile(r"(\d{3,4})\s*mt", re.IGNORECASE)
_VALID_DISTANCES = range(800, 3001)

# OOXML namespaces
_NS_XDR = "http://schemas.openxmlformats.org/drawingml/2006/spreadsheetDrawing"
_NS_A = "http://schemas.openxmlformats.org/drawingml/2006/main"
_NS_R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


# ---------------------------------------------------------------------------
@dataclass
class HorseCard:
    horse_name: str
    post_position: int
    kg: float
    horse_age: int | None
    sex_code: str | None  # 'M' or 'H'
    jockey_name: str | None


@dataclass
class RaceCard:
    race_index: int            # 1-based within the day
    distance_m: int
    post_time: str | None      # "HH:MM" if present in the sheet
    entries: list[HorseCard] = field(default_factory=list)


# ---------------------------------------------------------------------------
def fetch_program_xls(
    day: RaceDay,
    dest_dir: Path,
    scraper: MaronasScraper | None = None,
    force: bool = False,
) -> Path | None:
    """Download the Programa (.xls) for ``day`` if not already on disk."""
    s = scraper or MaronasScraper()
    track = RACETRACKS.get(day.racetrack_id, f"RT{day.racetrack_id}")
    slug = day.date.strftime("%Y%m%d")
    dest = dest_dir / track.replace(" ", "_") / f"Programa_RT{day.racetrack_id}_{slug}.xls"

    if dest.exists() and not force:
        return dest

    payload = {
        "documentType": DOC_TYPE_PROGRAM,
        "raceTrackId": day.racetrack_id,
        "periodId": None,
        "date": day.date.strftime("%Y-%m-%dT00:00:00"),
        "raceOrdinal": None,
        "withBinaryData": False,
        "sessionInfo": None,
    }
    docs = s._post("GetDocuments", payload)
    if not docs:
        logger.warning("No Programa published for %s on %s", track, slug)
        return None
    s._download_excel(docs[0]["Uri"], dest)
    logger.info("Downloaded %s", dest.name)
    return dest


# ---------------------------------------------------------------------------
def _xls_to_xlsx(xls_path: Path, out_dir: Path | None = None) -> Path:
    """Convert ``.xls`` → ``.xlsx`` via LibreOffice headless.

    LibreOffice exports the Crystal-Reports BIFF8 stream as proper OOXML,
    which gives us access to the embedded images (the Programa stores
    distance badges as images, not as cells).
    """
    out_dir = out_dir or xls_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "libreoffice", "--headless", "--convert-to", "xlsx",
        "--outdir", str(out_dir), str(xls_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        raise RuntimeError(
            f"LibreOffice conversion failed: {proc.stderr.strip() or proc.stdout.strip()}"
        )
    xlsx = out_dir / (xls_path.stem + ".xlsx")
    if not xlsx.exists():
        raise RuntimeError(f"LibreOffice produced no file at {xlsx}")
    return xlsx


def _ocr_distance(img_path: Path) -> int | None:
    """OCR a single badge image, voting across thresholds + PSMs."""
    try:
        import pytesseract  # local import; only the API container needs it
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "pytesseract is required to read distance badges. "
            "Install it and the system 'tesseract-ocr' package."
        ) from exc

    img = Image.open(img_path).convert("L")
    w, h = img.size
    # Crop to central 60% (the badge sits in the middle, ignoring track ring)
    crop = img.crop((int(w * 0.15), int(h * 0.28), int(w * 0.85), int(h * 0.78)))
    crop = crop.resize((crop.width * 3, crop.height * 3), Image.LANCZOS)

    candidates: list[int] = []
    for variant in (crop, ImageOps.invert(crop)):
        for thr in (110, 130, 150, 170):
            bw = variant.point(lambda p: 0 if p < thr else 255, mode="1")
            for psm in (7, 8, 11):
                txt = pytesseract.image_to_string(bw, config=f"--psm {psm}")
                for m in _DIST_RE.finditer(txt):
                    d = int(m.group(1))
                    if d in _VALID_DISTANCES:
                        candidates.append(d)
    if not candidates:
        return None
    return Counter(candidates).most_common(1)[0][0]


def _extract_distances(xlsx_path: Path, dump_dir: Path) -> list[int | None]:
    """Return the per-race distances in the order they appear in the sheet."""
    dump_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(xlsx_path) as zf:
        # Dump all media to disk
        media_paths: dict[str, Path] = {}
        for name in zf.namelist():
            if name.startswith("xl/media/"):
                fname = Path(name).name
                dest = dump_dir / fname
                dest.write_bytes(zf.read(name))
                media_paths[fname] = dest

        # Map rId → media filename via drawing1 rels
        rels_xml = zf.read("xl/drawings/_rels/drawing1.xml.rels")
        rels_root = ET.fromstring(rels_xml)
        rid_to_media = {r.attrib["Id"]: Path(r.attrib["Target"]).name for r in rels_root}

        # Walk drawing1 to find anchors → row + image
        xml = zf.read("xl/drawings/drawing1.xml")
        root = ET.fromstring(xml)
        anchors = root.findall(f".//{{{_NS_XDR}}}twoCellAnchor")
        anchors += root.findall(f".//{{{_NS_XDR}}}oneCellAnchor")

        badges: list[tuple[int, Path]] = []
        for anchor in anchors:
            frm = anchor.find(f"{{{_NS_XDR}}}from")
            row = int(frm.find(f"{{{_NS_XDR}}}row").text) if frm is not None else -1
            blip = anchor.find(f".//{{{_NS_A}}}blip")
            if blip is None:
                continue
            rid = blip.attrib.get(f"{{{_NS_R}}}embed")
            media = rid_to_media.get(rid)
            if not media:
                continue
            path = media_paths.get(media)
            if path is None:
                continue
            w, h = Image.open(path).size
            # Distance badges in current Maroñas Programa template are ~972x520.
            # Anything smaller is a chaquetilla (jockey colours) or the track logo.
            if w >= 800 and h >= 400:
                badges.append((row, path))

    badges.sort(key=lambda pair: pair[0])
    distances = [_ocr_distance(p) for _, p in badges]
    logger.info("Extracted %d distances via OCR: %s", len(distances), distances)
    return distances


# ---------------------------------------------------------------------------
# Sheet parsing — finds horse blocks per race
# ---------------------------------------------------------------------------
def _parse_entries(xls_path: Path) -> list[RaceCard]:
    """Walk the .xls sheet and emit one :class:`RaceCard` per race header."""
    book = xlrd.open_workbook(str(xls_path), ignore_workbook_corruption=True)
    sh = book.sheet_by_index(0)

    # 1) Find race headers ("PREMIO: ..." in column 7)
    premio_rows = [r for r in range(sh.nrows)
                   if isinstance(sh.cell_value(r, 7), str)
                   and sh.cell_value(r, 7).strip().upper().startswith("PREMIO:")]

    races: list[RaceCard] = []
    for race_idx, header_row in enumerate(premio_rows, start=1):
        # Post time sits a couple of rows above the PREMIO line, in col 25.
        post_time: str | None = None
        for r in range(max(0, header_row - 5), header_row + 1):
            v = sh.cell_value(r, 25)
            if isinstance(v, str) and re.fullmatch(r"\d{1,2}:\d{2}", v.strip()):
                post_time = v.strip()
                break

        # Entry rows live below header_row, until the next PREMIO (or EOF).
        next_header = premio_rows[race_idx] if race_idx < len(premio_rows) else sh.nrows
        entries: list[HorseCard] = []
        for r in range(header_row + 1, next_header):
            try:
                post_v = sh.cell_value(r, 0)
                horse_v = sh.cell_value(r, 2)
                if not isinstance(horse_v, str) or not horse_v.strip():
                    continue
                # post# can be int (1, 2, ...) or "1a"/"1b" for "co-runners"
                if isinstance(post_v, (int, float)):
                    post = int(post_v)
                elif isinstance(post_v, str) and post_v.strip()[:1].isdigit():
                    post = int(re.match(r"(\d+)", post_v.strip()).group(1))
                else:
                    continue
                kg = sh.cell_value(r, 11)
                if not isinstance(kg, (int, float)) or not (40 <= kg <= 70):
                    continue
                sex_v = sh.cell_value(r, 14)
                age_v = sh.cell_value(r, 15)
                jockey_v = sh.cell_value(r, 16)

                sex_code = None
                if isinstance(sex_v, str) and sex_v.strip():
                    s = sex_v.strip().lower()
                    sex_code = "M" if s.startswith("m") else "H"

                age = None
                if isinstance(age_v, (int, float)):
                    age = int(age_v)
                elif isinstance(age_v, str) and age_v.strip().isdigit():
                    age = int(age_v.strip())

                jockey = None
                if isinstance(jockey_v, str) and jockey_v.strip():
                    jockey = jockey_v.strip()

                entries.append(
                    HorseCard(
                        horse_name=horse_v.strip(),
                        post_position=post,
                        kg=float(kg),
                        horse_age=age,
                        sex_code=sex_code,
                        jockey_name=jockey,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("Row r=%d skipped: %s", r, exc)
                continue

        races.append(RaceCard(
            race_index=race_idx,
            distance_m=0,  # filled afterwards from OCR
            post_time=post_time,
            entries=entries,
        ))

    return races


# ---------------------------------------------------------------------------
def parse_program(xls_path: Path, ocr_dump_dir: Path | None = None) -> list[RaceCard]:
    """Parse a Programa .xls into a list of :class:`RaceCard`.

    Side-effect: dumps the OCR-input images under ``ocr_dump_dir`` (defaults
    to ``<xls>.ocr_imgs/``).
    """
    races = _parse_entries(xls_path)
    if not races:
        return []

    work_dir = ocr_dump_dir or xls_path.with_suffix(".ocr_imgs")
    xlsx = _xls_to_xlsx(xls_path, out_dir=work_dir.parent)
    distances = _extract_distances(xlsx, dump_dir=work_dir)

    # Pad/truncate to match number of races.
    if len(distances) < len(races):
        distances = list(distances) + [None] * (len(races) - len(distances))
    for race, dist in zip(races, distances):
        race.distance_m = dist or 0

    return races


# ---------------------------------------------------------------------------
def fetch_program(
    racetrack_id: int,
    race_date: datetime,
    raw_dir: Path,
    scraper: MaronasScraper | None = None,
    force: bool = False,
) -> list[RaceCard]:
    """End-to-end: download (if needed) and parse the Programa for one day."""
    s = scraper or MaronasScraper()
    day = RaceDay(racetrack_id=racetrack_id, date=race_date, has_program=True)
    xls = fetch_program_xls(day, dest_dir=raw_dir, scraper=s, force=force)
    if xls is None:
        return []
    return parse_program(xls)


def list_upcoming_program_days(
    racetrack_id: int,
    days_ahead: int = 14,
    scraper: MaronasScraper | None = None,
) -> list[RaceDay]:
    """List the next ``days_ahead`` race days that have a Programa published."""
    from datetime import timedelta
    s = scraper or MaronasScraper()
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    return [
        d for d in s.list_race_days(
            racetrack_id=racetrack_id,
            date_from=today.strftime("%Y-%m-%d"),
            date_to=(today + timedelta(days=days_ahead)).strftime("%Y-%m-%d"),
        )
        if d.has_program
    ]


def list_program_days(
    racetrack_id: int,
    days_back: int = 7,
    days_ahead: int = 14,
    scraper: MaronasScraper | None = None,
) -> list[RaceDay]:
    """List recent + upcoming race days that have a Programa published."""
    from datetime import timedelta
    s = scraper or MaronasScraper()
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    return [
        d for d in s.list_race_days(
            racetrack_id=racetrack_id,
            date_from=(today - timedelta(days=days_back)).strftime("%Y-%m-%d"),
            date_to=(today + timedelta(days=days_ahead)).strftime("%Y-%m-%d"),
        )
        if d.has_program
    ]
