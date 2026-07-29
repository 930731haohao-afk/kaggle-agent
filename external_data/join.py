"""Leakage-guarded joins for external data (v5 Plan C).

Rule (knowledge/task_priors.md): an external value may be joined to a prediction
row only if the value's date/year is <= the row's date/year minus `lag`.
`lag=0` matches the convention used by historical solutions for these
competitions (same-period macro values); `lag=1` is the strictly-available
variant. Every join writes a JSON log with the source snapshot and match report
so the join is auditable and replayable.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd


class LeakageError(AssertionError):
    pass


def merge_year_safe(df: pd.DataFrame, ext: pd.DataFrame, *, df_country: str,
                    df_year: str, ext_key: str = "iso3", ext_year: str = "year",
                    value_col: str = "value", out_col: str, mapping: dict[str, str],
                    lag: int = 0, log_path: str | Path | None = None,
                    meta: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Join a yearly external table onto df with the leakage guard.

    For each row, uses the most recent ext value with year <= row_year - lag.
    Countries missing from `mapping` get NaN and are listed in the report.
    Returns (df_with_out_col, report).
    """
    d = df.reset_index(drop=True).copy()
    d["_orig_order"] = range(len(d))
    d["_iso3"] = d[df_country].map(mapping)

    e = ext[[ext_key, ext_year, value_col]].dropna().copy()
    e[ext_year] = e[ext_year].astype(int)
    e[value_col] = pd.to_numeric(e[value_col])  # raise on non-numeric — bad source data must surface
    n_dup = int(e.duplicated([ext_key, ext_year]).sum())
    # duplicate (key, year) rows: keep last, count in report (masks nothing silently)
    e = e.drop_duplicates([ext_key, ext_year], keep="last").sort_values([ext_key, ext_year])

    if e.empty:
        out = df.copy(); out[out_col] = float("nan")
        report = {"out_col": out_col, "lag": lag, "rows": int(len(df)), "rows_matched": 0,
                  "rows_unmatched": int(len(df)), "unmatched_countries": sorted(df[df_country].unique().tolist()),
                  "ext_year_range": None, "ext_duplicates_dropped": 0, "source_meta": meta or {},
                  "joined_at": datetime.now().isoformat(timespec="seconds"), "leakage_check": "n/a (empty external table)"}
        if log_path:
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            Path(log_path).write_text(json.dumps(report, indent=2, ensure_ascii=False))
        return out, report

    # as-of style: for each (iso3, row_year) take latest ext year <= row_year - lag
    d["_eff_year"] = d[df_year].astype(int) - lag
    e = e.rename(columns={ext_key: "_iso3", ext_year: "_ext_year", value_col: out_col})
    e["_used_year"] = e["_ext_year"]

    merged = pd.merge_asof(
        d.sort_values("_eff_year"),
        e.rename(columns={"_ext_year": "_eff_year"}).sort_values("_eff_year"),
        on="_eff_year", by="_iso3", direction="backward",
    ).sort_values("_orig_order").reset_index(drop=True)

    # leakage check: the ext year actually used must respect the rule
    got = merged.dropna(subset=["_used_year"])
    viol = got[got["_used_year"] > got[df_year].astype(int) - lag]
    if len(viol):
        raise LeakageError(f"{len(viol)} rows received external values from the future (lag={lag})")

    unmatched_countries = sorted(d.loc[d["_iso3"].isna(), df_country].unique().tolist())
    report = {
        "out_col": out_col,
        "lag": lag,
        "rows": int(len(d)),
        "rows_matched": int(merged[out_col].notna().sum()),
        "rows_unmatched": int(merged[out_col].isna().sum()),
        "unmatched_countries": unmatched_countries,
        "ext_year_range": [int(e["_ext_year"].min()), int(e["_ext_year"].max())],
        "ext_duplicates_dropped": n_dup,
        "source_meta": meta or {},
        "joined_at": datetime.now().isoformat(timespec="seconds"),
        "leakage_check": "passed",
    }
    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(log_path).write_text(json.dumps(report, indent=2, ensure_ascii=False))

    out = merged.drop(columns=["_iso3", "_eff_year", "_used_year", "_orig_order", "country_y"], errors="ignore")
    return out, report


def merge_holiday_flags(df: pd.DataFrame, hol: pd.DataFrame, *, df_country: str,
                        df_date: str, out_col: str = "is_holiday",
                        log_path: str | Path | None = None,
                        meta: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Add a holiday indicator column. Holidays are deterministic future — no leakage."""
    d = df.copy()
    ser = pd.to_datetime(d[df_date])
    if ser.dt.tz is not None:
        ser = ser.dt.tz_localize(None)
    d["_date"] = ser.dt.normalize()
    key = set(zip(hol["country"], hol["date"]))
    d[out_col] = [int((c, t) in key) for c, t in zip(d[df_country], d["_date"])]
    report = {
        "out_col": out_col,
        "rows": int(len(d)),
        "holiday_rows": int(d[out_col].sum()),
        "source_meta": meta or {},
        "joined_at": datetime.now().isoformat(timespec="seconds"),
        "leakage_check": "n/a (deterministic calendar)",
    }
    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(log_path).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return d.drop(columns=["_date"]), report
