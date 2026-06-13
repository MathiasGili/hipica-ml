"""Temporal train/test splitter — the *only* split helper used in the project.

Random splits are explicitly disabled: in horse-racing data, two rows from
the same race day share latent context (track condition, weather, judge
decisions) that the model would otherwise leak across the train/test
boundary. Splitting strictly by race date prevents that.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Literal

import pandas as pd

from ..config import RACE_DATE_COL

logger = logging.getLogger(__name__)


def temporal_train_test_split(
    df: pd.DataFrame,
    cutoff: str | datetime | pd.Timestamp | None = None,
    test_size: float | None = 0.2,
    date_col: str = RACE_DATE_COL,
    strategy: Literal["cutoff", "quantile"] = "quantile",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Split ``df`` into ``(train, test, cutoff_date)`` ordered by ``date_col``.

    Parameters
    ----------
    df:
        Long-form DataFrame with a date column.
    cutoff:
        Explicit cutoff date. Rows with ``date < cutoff`` go to train, the
        rest to test. When ``None``, ``test_size`` is used to derive a date
        quantile.
    test_size:
        Fraction of *rows* (not days) reserved for test. Ignored if
        ``cutoff`` is given.
    date_col:
        Column name to split on. Must be parseable as datetime.
    strategy:
        - ``"cutoff"``: requires ``cutoff``.
        - ``"quantile"``: pick the date that puts ``1 - test_size`` of rows
          on the train side.

    Returns
    -------
    train, test, cutoff_date
    """
    if date_col not in df.columns:
        raise KeyError(f"Column {date_col!r} not in DataFrame")

    work = df.copy()
    work[date_col] = pd.to_datetime(work[date_col])
    work = work.sort_values(date_col).reset_index(drop=True)

    if strategy == "cutoff":
        if cutoff is None:
            raise ValueError("strategy='cutoff' needs an explicit cutoff date")
        cutoff_ts = pd.Timestamp(cutoff)
    else:
        if test_size is None or not 0 < test_size < 1:
            raise ValueError("test_size must be in (0, 1) for quantile strategy")
        cutoff_ts = work[date_col].quantile(1 - test_size, interpolation="nearest")

    train = work[work[date_col] < cutoff_ts].reset_index(drop=True)
    test = work[work[date_col] >= cutoff_ts].reset_index(drop=True)

    logger.info(
        "Temporal split @ %s | train=%d (%.0f..%s)  test=%d (%s..%s)",
        cutoff_ts.date() if hasattr(cutoff_ts, "date") else cutoff_ts,
        len(train),
        train[date_col].min().year if len(train) else 0,
        train[date_col].max().date() if len(train) else "—",
        len(test),
        test[date_col].min().date() if len(test) else "—",
        test[date_col].max().date() if len(test) else "—",
    )
    return train, test, pd.Timestamp(cutoff_ts)
