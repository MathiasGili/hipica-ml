"""Parse Maroñas Tabulada `.xls` files into a long-form race-history DataFrame.

The Tabulada is a Crystal Reports export with one block per horse running a
given race-day. Each block contains:

* a *leader* line with the horse's name, current-race post position, kg,
  jockey and cuidador;
* a year-stats / career-stats / pedigree section;
* a tabular **race history** for that horse with one row per past run, with
  columns ``Fecha | Hip | Pos. | Participantes | Dist. | Tiempo | ... | Kg | ...``.

This module focuses on the **history rows** because they are already
labelled (we know the finishing position of the horse in each past race).
Cross-joining the histories from many tabuladas — after dedup — gives a
long-form labelled dataset suitable for both EDA and training.

Per-block leader metadata (current race ordinal, post position, jockey
today, etc.) is *not* needed to label past races, so we keep it simple here
and let the training notebook pull richer features straight from the
long-form DataFrame.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd
import xlrd

from ..config import RAW_DIR, RACETRACKS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Column layout of the per-horse history table (0-indexed, observed empirically
# from a 2025-01-26 Maroñas Tabulada, 63 columns wide). Other Tabuladas use
# the same Crystal Reports template, so these offsets are stable across files.
# ---------------------------------------------------------------------------
HIST_COL_DATE = 1            # "Fecha"   — DD/MM/YY
HIST_COL_TRACK_ABBR = 4      # "Hip"     — MRÑ / L.PD / SINT / ...
HIST_COL_POS = 6             # "Pos."    — finishing position of THIS horse
HIST_COL_WINNER = 7          # "Participantes" — winner name
HIST_COL_KG = 18             # "Kg"      — kg carried by this horse
HIST_COL_DISTANCE = 22       # "Dist."   — meters
HIST_COL_TOTAL_TIME = 25     # "T.Final" — total time of this horse
HIST_COL_DIVIDEND = 28       # "Divid."  — win dividend
HIST_COL_JOCKEY = 30         # "Jockey"
HIST_COL_BODY_WEIGHT = 33    # "Peso"    — body weight (kg of the horse)

# Racetrack abbreviation -> id (mirrors RACETRACKS in config.py)
TRACK_ABBR_TO_ID: dict[str, int] = {
    "MRÑ": 1, "MRN": 1, "MRO": 1, "MAR": 1,
    "L.PD": 13, "LPD": 13, "L PD": 13, "L.P": 13,
    "COL": 4, "MEL": 16, "PAY": 21, "FLD": 9, "RCO": 22, "FLS": 8,
}

# Pos. column may contain "1", "2", "12", "DSC" (descalificado),
# "RTD" (retirado), "PE" (perdió experiencia), etc.
_POS_RE = re.compile(r"^\s*(\d{1,2})\s*$")
_DATE_RE = re.compile(r"^\s*(\d{2})/(\d{2})/(\d{2,4})\s*$")
_NUM_RE = re.compile(r"[-+]?\d*\.?\d+")
# Block-leader meta lives a few rows below the leader (typically +7 but it
# varies). The age cell looks like ``"6 a."`` and the sex cell next to it
# looks like ``"m."`` or ``"h."``.
_AGE_RE = re.compile(r"^\s*(\d{1,2})\s*a\.?\s*$")
_SEX_RE = re.compile(r"^\s*([mh])\.?\s*$", re.IGNORECASE)


@dataclass(frozen=True)
class ParsedFile:
    """Outcome of parsing a single Tabulada file."""

    path: Path
    rows: list[dict]
    n_blocks: int
    n_history: int
    n_skipped: int


class RawTabuladaLoader:
    """Parse one or many Tabulada `.xls` files into a long-form DataFrame.

    Usage
    -----
    >>> loader = RawTabuladaLoader()
    >>> df = loader.load_dir(Path("data/raw"))
    >>> df.head()

    The output schema is::

        horse_name        : str
        race_date         : datetime64[ns]   (date of the past race)
        racetrack_id      : int
        racetrack_abbr    : str
        finish_pos        : int              (1, 2, 3, ... — the LABEL source)
        in_trifecta       : int              (1 if finish_pos in {1,2,3})
        winner_name       : str
        kg                : float            (kg carried in that race)
        distance_m        : int
        total_time_s      : float            (parsed "1'34''37" -> 94.37)
        dividend          : float
        jockey            : str
        body_weight_kg    : float
        source_file       : str              (Tabulada the row was harvested from)
    """

    # ------------------------------------------------------------------ public
    def load_file(self, path: Path) -> ParsedFile:
        try:
            wb = xlrd.open_workbook(str(path))
        except Exception as exc:  # noqa: BLE001
            logger.error("Cannot open %s: %s", path, exc)
            return ParsedFile(path=path, rows=[], n_blocks=0, n_history=0, n_skipped=0)

        sheet = wb.sheet_by_index(0)
        rows: list[dict] = []
        n_blocks = 0
        n_skipped = 0

        for current_horse, leader_meta, hist_block in self._iter_blocks(sheet):
            n_blocks += 1
            # leader_meta describes the horse on the TARGET race (the one in
            # the current Tabulada). For the historical rows we emit:
            #   - sex_code: truly constant per horse, safe to attach.
            #   - horse_age_at_race: leader_age minus years between the
            #     Tabulada date and the historical race date. Never larger
            #     than leader_age itself (rookies get NaN).
            #   - post_position: NOT attached. The leader's post is "today"
            #     and is unrelated to the post the horse held in past races.
            #     At serving time the API supplies post_position from the
            #     current entry; at training time it stays NaN (and the
            #     median imputer downstream handles it).
            tabulada_date = self._tabulada_date_from_filename(path.name)
            for raw in hist_block:
                parsed = self._parse_history_row(raw, current_horse, path.name)
                if parsed is None:
                    n_skipped += 1
                    continue
                parsed["sex_code"] = leader_meta.get("sex_code")
                if (
                    leader_meta.get("horse_age") is not None
                    and tabulada_date is not None
                    and parsed.get("race_date") is not None
                ):
                    years_back = max(0, tabulada_date.year - parsed["race_date"].year)
                    parsed["horse_age"] = max(2, int(leader_meta["horse_age"]) - years_back)
                else:
                    parsed["horse_age"] = None
                parsed["post_position"] = None
                rows.append(parsed)

        logger.info(
            "%s: %d blocks, %d history rows (%d skipped)",
            path.name, n_blocks, len(rows), n_skipped,
        )
        return ParsedFile(
            path=path,
            rows=rows,
            n_blocks=n_blocks,
            n_history=len(rows),
            n_skipped=n_skipped,
        )

    def load_dir(self, root: Path | None = None, pattern: str = "**/*.xls") -> pd.DataFrame:
        root = root or RAW_DIR
        files = sorted(Path(root).glob(pattern))
        logger.info("Loading %d Tabulada files from %s", len(files), root)
        all_rows: list[dict] = []
        for f in files:
            all_rows.extend(self.load_file(f).rows)
        if not all_rows:
            return self._empty_frame()
        df = pd.DataFrame(all_rows)
        return self._postprocess(df)

    # ------------------------------------------------------------------ blocks
    def _iter_blocks(self, sheet) -> Iterator[tuple[str, dict, list[list]]]:
        """Yield ``(horse_name, leader_meta, history_rows)`` for each block.

        ``leader_meta`` is a dict with keys ``post_position``, ``horse_age``,
        ``sex_code`` extracted from the block leader row and the
        ``coat / sex / age`` row that sits a few lines below it.

        A block starts at a *leader* row (col 1 = small int, col 4 = horse
        name, col 30 = kg) and ends at the next leader row. Within a block,
        the history rows are everything that comes after the ``"Fecha"``
        header row.
        """
        leader_indices = self._find_leader_rows(sheet)
        if not leader_indices:
            return

        # add sentinel for the last block
        leader_indices.append(sheet.nrows)

        for i in range(len(leader_indices) - 1):
            start = leader_indices[i]
            end = leader_indices[i + 1]
            horse_name = self._cell(sheet, start, 4)
            if not horse_name:
                continue
            leader_meta = self._extract_leader_meta(sheet, start, end)
            # find the "Fecha" header inside this block
            header_row = self._find_header_row(sheet, start, end)
            if header_row is None:
                continue
            history_rows = [
                [self._cell(sheet, r, c) for c in range(sheet.ncols)]
                for r in range(header_row + 1, end)
            ]
            yield horse_name, leader_meta, history_rows

    @staticmethod
    def _find_leader_rows(sheet) -> list[int]:
        """Identify horse-block leader rows.

        Leader rows look like::

            col 1 : '1'  '2'  ...
            col 4 : horse name
            col 30: kg as float
            col 36: jockey
        """
        out: list[int] = []
        for r in range(sheet.nrows):
            v1 = sheet.cell_value(r, 1) if sheet.ncols > 1 else ""
            v4 = sheet.cell_value(r, 4) if sheet.ncols > 4 else ""
            v30 = sheet.cell_value(r, 30) if sheet.ncols > 30 else ""
            if not isinstance(v4, str) or not v4.strip():
                continue
            # col 1 should be a small integer (post position 1..20)
            try:
                pos = int(float(v1))
            except (TypeError, ValueError):
                continue
            if not 1 <= pos <= 25:
                continue
            # col 30 should be a numeric kg
            try:
                kg = float(v30)
            except (TypeError, ValueError):
                continue
            if not 40 <= kg <= 70:
                continue
            out.append(r)
        return out

    @staticmethod
    def _find_header_row(sheet, start: int, end: int) -> int | None:
        for r in range(start, end):
            try:
                v = str(sheet.cell_value(r, HIST_COL_DATE)).strip().lower()
            except IndexError:
                continue
            if v == "fecha":
                return r
        return None

    @staticmethod
    def _tabulada_date_from_filename(name: str) -> datetime | None:
        """Parse ``Tabulada_RT1_20250126.xls`` -> ``datetime(2025, 1, 26)``."""
        m = re.search(r"_(\d{4})(\d{2})(\d{2})\.xls$", name, re.IGNORECASE)
        if not m:
            return None
        try:
            return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return None

    @staticmethod
    def _extract_leader_meta(sheet, start: int, end: int) -> dict:
        """Pull post_position, sex_code, horse_age from a block.

        - ``post_position`` is at col 1 of the leader row (``start``).
        - ``sex_code`` and ``horse_age`` live a few rows below the leader on
          the line that contains an age cell like ``"6 a."``. Crystal Reports
          varies the offset between blocks, so we scan forward up to 12
          rows.
        """
        meta = {"post_position": None, "horse_age": None, "sex_code": None}

        # post position from the leader row
        try:
            meta["post_position"] = int(float(sheet.cell_value(start, 1)))
        except (TypeError, ValueError):
            pass

        # walk forward looking for an "age" cell in the typical columns
        scan_end = min(start + 12, end)
        for r in range(start + 1, scan_end):
            for col_age in (10, 11, 12):
                if col_age >= sheet.ncols:
                    continue
                age_cell = str(sheet.cell_value(r, col_age)).strip()
                m = _AGE_RE.match(age_cell)
                if not m:
                    continue
                meta["horse_age"] = int(m.group(1))
                # sex sits 1 column to the left of age
                for col_sex in (col_age - 1, col_age - 2):
                    if col_sex < 0:
                        break
                    sex_cell = str(sheet.cell_value(r, col_sex)).strip()
                    sm = _SEX_RE.match(sex_cell)
                    if sm:
                        meta["sex_code"] = sm.group(1).upper()
                        break
                return meta
        return meta

    @staticmethod
    def _cell(sheet, r: int, c: int):
        if c >= sheet.ncols or r >= sheet.nrows:
            return ""
        return sheet.cell_value(r, c)

    # -------------------------------------------------------------- row parse
    def _parse_history_row(self, raw: list, horse_name: str, source: str) -> dict | None:
        date_raw = str(raw[HIST_COL_DATE]).strip() if len(raw) > HIST_COL_DATE else ""
        m = _DATE_RE.match(date_raw)
        if not m:
            return None
        dd, mm, yy = m.groups()
        year = int(yy)
        if year < 100:
            year += 2000 if year < 80 else 1900
        try:
            race_date = datetime(year, int(mm), int(dd))
        except ValueError:
            return None

        pos_raw = str(raw[HIST_COL_POS]).strip() if len(raw) > HIST_COL_POS else ""
        pos_match = _POS_RE.match(pos_raw)
        if not pos_match:
            # DSC, RTD, etc. — keep the row but mark pos as missing
            finish_pos = None
        else:
            finish_pos = int(pos_match.group(1))

        track_abbr = str(raw[HIST_COL_TRACK_ABBR]).strip() if len(raw) > HIST_COL_TRACK_ABBR else ""
        racetrack_id = TRACK_ABBR_TO_ID.get(track_abbr, None)

        return {
            "horse_name": _normalize_name(horse_name),
            "race_date": race_date,
            "racetrack_id": racetrack_id,
            "racetrack_abbr": track_abbr or None,
            "finish_pos": finish_pos,
            "in_trifecta": int(finish_pos in (1, 2, 3)) if finish_pos is not None else None,
            "winner_name": _normalize_name(_safe_str(raw, HIST_COL_WINNER)),
            "kg": _safe_float(raw, HIST_COL_KG),
            "distance_m": _safe_int(raw, HIST_COL_DISTANCE),
            "total_time_s": _parse_time_str(_safe_str(raw, HIST_COL_TOTAL_TIME)),
            "dividend": _safe_float(raw, HIST_COL_DIVIDEND),
            "jockey": _normalize_name(_safe_str(raw, HIST_COL_JOCKEY)),
            "body_weight_kg": _safe_float(raw, HIST_COL_BODY_WEIGHT),
            "source_file": source,
        }

    # ------------------------------------------------------------ post-process
    @staticmethod
    def _empty_frame() -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "horse_name", "race_date", "racetrack_id", "racetrack_abbr",
                "finish_pos", "in_trifecta", "winner_name", "kg", "distance_m",
                "total_time_s", "dividend", "jockey", "body_weight_kg",
                "source_file",
                "post_position", "horse_age", "sex_code",
            ]
        )

    @staticmethod
    def _postprocess(df: pd.DataFrame) -> pd.DataFrame:
        # Same horse-day-race may appear in many Tabuladas (each Tabulada
        # repeats the rival horses' histories). De-dup conservatively on
        # (horse, date, racetrack, distance).
        df = df.sort_values(["horse_name", "race_date"]).reset_index(drop=True)
        df = df.drop_duplicates(
            subset=["horse_name", "race_date", "racetrack_id", "distance_m"],
            keep="first",
        )
        df["race_date"] = pd.to_datetime(df["race_date"])
        return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _safe_str(raw: list, idx: int) -> str:
    if len(raw) <= idx:
        return ""
    return str(raw[idx]).strip()


def _safe_float(raw: list, idx: int) -> float | None:
    s = _safe_str(raw, idx).replace(",", ".")
    if not s or s in {"-", "—"}:
        return None
    m = _NUM_RE.search(s)
    return float(m.group(0)) if m else None


def _safe_int(raw: list, idx: int) -> int | None:
    v = _safe_float(raw, idx)
    return int(v) if v is not None else None


def _parse_time_str(s: str) -> float | None:
    """Parse '1\\'34\\'\\'37' or '34\\'\\'37' into seconds (94.37 / 34.37)."""
    if not s or s in {"-", "—"}:
        return None
    s = s.replace("''", '"').replace("'", "m").replace('"', ".")
    # Now formats look like '1m34.37' or '34.37'
    if "m" in s:
        try:
            mins, rest = s.split("m", 1)
            return float(mins) * 60 + float(rest)
        except ValueError:
            return None
    try:
        return float(s)
    except ValueError:
        return None


def _normalize_name(name: str) -> str:
    if not name:
        return ""
    return " ".join(name.strip().split()).upper()


# ---------------------------------------------------------------------------
# Convenience builder used by training notebooks and tests
# ---------------------------------------------------------------------------
def build_long_form_dataset(
    raw_dir: Path | None = None,
    cache_path: Path | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Parse every Tabulada under ``raw_dir`` and return a long-form DataFrame.

    If ``cache_path`` is given, the result is cached as parquet so subsequent
    calls are instant.
    """
    raw_dir = raw_dir or RAW_DIR
    if cache_path and use_cache and cache_path.exists():
        logger.info("Loading cached dataset from %s", cache_path)
        return pd.read_parquet(cache_path)

    df = RawTabuladaLoader().load_dir(raw_dir)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path)
        logger.info("Cached %d rows to %s", len(df), cache_path)
    return df


__all__ = [
    "RawTabuladaLoader",
    "ParsedFile",
    "build_long_form_dataset",
    "TRACK_ABBR_TO_ID",
]
