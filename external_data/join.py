"""Leakage-guarded joins for external data (v5 Plan C).

Rule (knowledge/task_priors.md): an external value may be joined to a prediction
row only if the value's period is <= the row's period minus `lag` (periods of
the join's granularity). `lag=0` = same-period values (historical-solution
convention); `lag=1` = strictly earlier periods. Violations raise LeakageError.
Every join can write a JSON audit log (source snapshot, match report, lag) so
joins are auditable and replayable.

Input contract (strict, after adversarial review): time columns are datetimes
unless an explicit `*_time_unit` says otherwise. Numeric time columns without a
declared unit raise TypeError — silent epoch/ordinal misparses are the #1 way
to leak the future while the guard reports "passed".
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

# plausible ranges for declared ordinal units — reject anything else loudly
_MONTH_ORD_RANGE = (1800 * 12, 2200 * 12)                     # year*12+month
_DAY_ORD_RANGE = (657072, 803533)                              # 1800-01-01 .. 2200-12-31


class LeakageError(AssertionError):
    pass


def _assert_no_future(got: pd.DataFrame, used_col: str, row_col: str, lag: int,
                      label: str) -> None:
    """Defense-in-depth invariant: every used external period respects the rule.

    Unreachable via merge_asof(direction="backward") by construction, but kept
    (and unit-tested directly) so any future refactor that breaks the backward
    guarantee fails loudly instead of leaking silently.
    """
    viol = got[got[used_col] > got[row_col] - lag]
    if len(viol):
        raise LeakageError(f"{len(viol)} rows received external values from the future ({label})")


def _to_period(series: pd.Series, freq: str, unit: str, side: str) -> pd.Series:
    """Convert a time column to int64 period ordinals under an explicit unit."""
    if unit == "datetime":
        if pd.api.types.is_numeric_dtype(series):
            raise TypeError(
                f"{side} time column is numeric but *_time_unit='datetime'; "
                f"declare 'month_ordinal'/'day_ordinal' explicitly — refusing to guess")
        t = pd.to_datetime(series)
        if getattr(t.dt, "tz", None) is not None:
            t = t.dt.tz_localize(None)
        if freq == "M":
            return (t.dt.year * 12 + t.dt.month).astype("int64")
        return t.dt.normalize().map(pd.Timestamp.toordinal).astype("int64")
    if unit == "month_ordinal":
        if freq != "M":
            raise ValueError("month_ordinal only valid with freq='M'")
        v = series.astype("int64")
        lo, hi = _MONTH_ORD_RANGE
        bad = v[(v < lo) | (v > hi)]
        if len(bad):
            raise ValueError(
                f"{side} month ordinals out of plausible range [{lo},{hi}]: e.g. {bad.iloc[0]} "
                f"— plain years or YYYYMM ints are not month ordinals (year*12+month)")
        return v
    if unit == "day_ordinal":
        if freq != "D":
            raise ValueError("day_ordinal only valid with freq='D'")
        v = series.astype("int64")
        lo, hi = _DAY_ORD_RANGE
        bad = v[(v < lo) | (v > hi)]
        if len(bad):
            raise ValueError(
                f"{side} day ordinals out of plausible range [{lo},{hi}]: e.g. {bad.iloc[0]}")
        return v
    raise ValueError(f"unknown time unit '{unit}' (datetime|month_ordinal|day_ordinal)")


def _mapping_collisions(d: pd.DataFrame, df_key: str, mapped_col: str) -> dict[str, list[str]]:
    """mapped-key -> [distinct df keys] for every mapped key hit by >1 df key."""
    pairs = d[[df_key, mapped_col]].dropna().drop_duplicates()
    counts = pairs.groupby(mapped_col)[df_key].apply(list)
    return {str(k): sorted(map(str, v)) for k, v in counts.items() if len(v) > 1}


def _write_log(report: dict, log_path: str | Path | None) -> None:
    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(log_path).write_text(json.dumps(report, indent=2, ensure_ascii=False))


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
    e[ext_year] = e[ext_year].astype("int64")
    e[value_col] = pd.to_numeric(e[value_col])  # raise on non-numeric — bad source data must surface
    n_dup = int(e.duplicated([ext_key, ext_year]).sum())
    # duplicate (key, year) rows: keep last, count in report (masks nothing silently)
    e = e.drop_duplicates([ext_key, ext_year], keep="last").sort_values([ext_key, ext_year])

    unmatched_countries = sorted(d.loc[d["_iso3"].isna(), df_country].astype(str).unique().tolist())
    keys_not_in_ext = sorted(set(d["_iso3"].dropna()) - set(e[ext_key])) if len(e) else []
    collisions = _mapping_collisions(d, df_country, "_iso3")

    if e.empty:
        out = df.copy(); out[out_col] = float("nan")
        report = {"out_col": out_col, "lag": lag, "rows": int(len(df)), "rows_matched": 0,
                  "rows_unmatched": int(len(df)), "unmatched_countries": unmatched_countries,
                  "keys_not_in_ext": [], "mapping_collisions": collisions,
                  "ext_year_range": None, "ext_duplicates_dropped": 0, "source_meta": meta or {},
                  "joined_at": datetime.now().isoformat(timespec="seconds"),
                  "leakage_check": "n/a (empty external table)"}
        _write_log(report, log_path)
        return out, report

    d["_eff_year"] = d[df_year].astype("int64") - lag
    e = e.rename(columns={ext_key: "_iso3", ext_year: "_ext_year", value_col: out_col})
    e["_used_year"] = e["_ext_year"]

    merged = pd.merge_asof(
        d.sort_values("_eff_year"),
        e.rename(columns={"_ext_year": "_eff_year"}).sort_values("_eff_year"),
        on="_eff_year", by="_iso3", direction="backward",
    ).sort_values("_orig_order").reset_index(drop=True)

    got = merged.dropna(subset=["_used_year"]).copy()
    got["_row_year"] = got[df_year].astype("int64")
    _assert_no_future(got, "_used_year", "_row_year", lag, f"yearly, lag={lag}")

    report = {
        "out_col": out_col,
        "lag": lag,
        "rows": int(len(d)),
        "rows_matched": int(merged[out_col].notna().sum()),
        "rows_unmatched": int(merged[out_col].isna().sum()),
        "unmatched_countries": unmatched_countries,
        "keys_not_in_ext": [str(x) for x in keys_not_in_ext],
        "mapping_collisions": collisions,
        "ext_year_range": [int(e["_ext_year"].min()), int(e["_ext_year"].max())],
        "ext_duplicates_dropped": n_dup,
        "source_meta": meta or {},
        "joined_at": datetime.now().isoformat(timespec="seconds"),
        "leakage_check": "passed",
    }
    _write_log(report, log_path)
    out = merged.drop(columns=["_iso3", "_eff_year", "_used_year", "_orig_order", "country_y"],
                      errors="ignore")
    return out, report


def merge_period_safe(df: pd.DataFrame, ext: pd.DataFrame, *, freq: str,
                      df_key: str, df_time: str, ext_key: str, ext_time: str,
                      value_cols: list[str], lag: int = 0,
                      mapping: dict[str, str] | None = None,
                      df_time_unit: str = "datetime", ext_time_unit: str = "datetime",
                      coarsen: str | None = None,
                      log_path: str | Path | None = None,
                      meta: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Leakage-guarded as-of join at monthly ("M") or daily ("D") granularity.

    ext values join a row only if ext period <= row period - lag (periods of
    `freq`). Numeric time columns must declare their unit ('month_ordinal' /
    'day_ordinal') — undeclared numeric time raises TypeError.

    If ext is finer-grained than `freq` (several distinct times collapse into
    one period), you must pass `coarsen` in {'first','last','mean','max','min'};
    the collapse is recorded in the report. Note: coarsen='last' + lag=0 admits
    within-period values dated after the row — prefer lag>=1 or coarsen='first'
    when row dates fall inside the period.
    """
    if freq not in ("M", "D"):
        raise ValueError("freq must be 'M' or 'D'")

    d = df.reset_index(drop=True).copy()
    d["_orig_order"] = range(len(d))
    d["_key"] = d[df_key].map(mapping) if mapping else d[df_key]

    e = ext[[ext_key, ext_time] + value_cols].dropna(subset=[ext_key, ext_time]).copy()
    e = e.dropna(subset=value_cols, how="all")            # all-NaN rows must not shadow real values
    e = e.rename(columns={ext_key: "_key"})
    for c in value_cols:
        e[c] = pd.to_numeric(e[c])
    e["_period"] = _to_period(e[ext_time], freq, ext_time_unit, "ext")

    finer_grained = bool(e.groupby(["_key", "_period"])[ext_time].nunique().gt(1).any())
    if finer_grained and coarsen is None:
        raise ValueError(
            f"external data is finer-grained than freq='{freq}' (multiple distinct times per "
            "period); pass coarsen='first'|'last'|'mean'|'max'|'min' explicitly — refusing to pick silently")
    agg = coarsen or "last"
    n_rows_before = len(e)
    # collapse to one row per (key, period); groupby first/last skip NaN per column
    e = (e.sort_values("_period")
           .groupby(["_key", "_period"], as_index=False)
           .agg({c: agg for c in value_cols}))
    n_collapsed = n_rows_before - len(e)
    e = e.sort_values("_period")

    d["_row_period"] = _to_period(d[df_time], freq, df_time_unit, "df")
    d["_eff_period"] = d["_row_period"] - lag
    e["_used_period"] = e["_period"]

    merged = pd.merge_asof(
        d.sort_values("_eff_period"),
        e.rename(columns={"_period": "_eff_period"}).sort_values("_eff_period"),
        on="_eff_period", by="_key", direction="backward",
    ).sort_values("_orig_order").reset_index(drop=True)

    got = merged.dropna(subset=["_used_period"])
    _assert_no_future(got, "_used_period", "_row_period", lag, f"freq={freq}, lag={lag}")

    unmatched_keys = sorted(d.loc[d["_key"].isna(), df_key].astype(str).unique().tolist())
    keys_not_in_ext = sorted(set(d["_key"].dropna()) - set(e["_key"])) if len(e) else []
    any_match = merged[value_cols].notna().any(axis=1)
    report = {
        "value_cols": value_cols, "freq": freq, "lag": lag,
        "rows": int(len(d)),
        "rows_matched": int(any_match.sum()),
        "rows_matched_by_col": {c: int(merged[c].notna().sum()) for c in value_cols},
        "rows_unmatched": int((~any_match).sum()),
        "unmatched_keys": unmatched_keys,
        "keys_not_in_ext": [str(x) for x in keys_not_in_ext],
        "mapping_collisions": _mapping_collisions(d, df_key, "_key"),
        "ext_rows_collapsed": n_collapsed,
        "coarsen": coarsen,
        "source_meta": meta or {},
        "joined_at": datetime.now().isoformat(timespec="seconds"),
        "leakage_check": "passed",
    }
    if coarsen == "last" and lag == 0 and finer_grained:
        report["warning"] = ("coarsen='last' with lag=0 admits within-period values dated "
                             "after the row; prefer lag>=1 or coarsen='first'")
    _write_log(report, log_path)
    out = merged.drop(columns=["_key", "_row_period", "_eff_period", "_used_period", "_orig_order"],
                      errors="ignore")
    return out, report


def merge_holiday_flags(df: pd.DataFrame, hol: pd.DataFrame, *, df_country: str,
                        df_date: str, out_col: str = "is_holiday",
                        log_path: str | Path | None = None,
                        meta: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Add a holiday indicator column via exact (country, date) lookup.

    Holidays are a deterministic calendar — there is no lag parameter and no
    LeakageError path; knowing future holidays is not leakage.
    """
    d = df.copy()
    ser = pd.to_datetime(d[df_date])
    if ser.dt.tz is not None:
        ser = ser.dt.tz_localize(None)
    d["_date"] = ser.dt.normalize()
    key = set(zip(hol["country"], hol["date"]))
    d[out_col] = [int((c, t) in key) for c, t in zip(d[df_country], d["_date"])]
    countries_without_calendar = sorted(set(d[df_country]) - set(hol["country"]))
    report = {
        "out_col": out_col,
        "rows": int(len(d)),
        "holiday_rows": int(d[out_col].sum()),
        "countries_without_calendar": [str(x) for x in countries_without_calendar],
        "source_meta": meta or {},
        "joined_at": datetime.now().isoformat(timespec="seconds"),
        "leakage_check": "n/a (deterministic calendar)",
    }
    _write_log(report, log_path)
    return d.drop(columns=["_date"]), report
