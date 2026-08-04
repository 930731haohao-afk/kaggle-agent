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


def _key_list(values) -> list[str]:
    """Sorted, stringified, de-duplicated keys for a report.

    `astype(str)` does NOT reliably turn a missing value into "nan" (pandas keeps NA), so
    sorting a set that mixes real keys with NA raises TypeError -- from inside the REPORT
    builder, discarding a join that had already succeeded (2026-08-04 adversarial
    verification). Stringify per element and sort the strings.
    """
    return sorted({str(v) for v in values if pd.notna(v)})


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

    unmatched_countries = _key_list(d.loc[d["_iso3"].isna(), df_country])
    keys_not_in_ext = _key_list(set(d["_iso3"].dropna()) - set(e[ext_key])) if len(e) else []
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

    unmatched_keys = _key_list(d.loc[d["_key"].isna(), df_key])
    keys_not_in_ext = _key_list(set(d["_key"].dropna()) - set(e["_key"])) if len(e) else []
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


def merge_lookup_safe(df: pd.DataFrame, ext: pd.DataFrame, *, df_key: str,
                      ext_key: str, value_cols: list[str], out_prefix: str = "",
                      log_path: str | Path | None = None,
                      meta: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Join a STATIC attribute table onto df by an exact code/id match.

    The `lookup` join key class: a code column in the competition data (a CPC classification
    code, an ISO code, a product id) resolved against a reference table that has no time
    axis. There is no lag and no LeakageError path, because there is no period to be on the
    wrong side of -- but the same non-negotiables as every other join apply: row count and
    row ORDER are preserved, keys that do not resolve are REPORTED rather than silently
    filled, and duplicate keys in the reference table are counted rather than quietly
    resolved (2026-08-03 source-layer extension).

    Returns (df_with_new_columns, report).
    """
    if not value_cols:
        raise ValueError("value_cols must name at least one column to bring across")
    missing = [c for c in (ext_key, *value_cols) if c not in ext.columns]
    if missing:
        raise ValueError(f"reference table lacks {missing} (has {list(ext.columns)})")
    if df_key not in df.columns:
        raise ValueError(f"competition data has no key column {df_key!r}")
    if ext.empty:
        # A reference table with zero rows joins to nothing and yields an all-NaN column.
        # That is indistinguishable downstream from "the source was useless", when what
        # actually happened is that the fetcher returned nothing (cache miss, network off,
        # upstream schema change). Refuse rather than degrade quietly.
        raise ValueError(
            f"reference table for key {ext_key!r} is empty; a lookup join against zero rows "
            f"produces an all-NaN column that reads downstream as a useless source rather "
            f"than as a fetch failure")

    d = df.reset_index(drop=True).copy()
    d["_orig_order"] = range(len(d))
    e = ext[[ext_key, *value_cols]].copy()
    e[ext_key] = e[ext_key].astype(str)
    n_dup = int(e.duplicated([ext_key]).sum())
    # keep last, count it: a reference table with duplicate codes is a source problem the
    # report must surface, not something the join decides quietly
    e = e.drop_duplicates([ext_key], keep="last")

    rename = {c: f"{out_prefix}{c}" for c in value_cols}
    out_cols = list(rename.values())
    e = e.rename(columns=rename)

    # A key that is MISSING must never join. astype(str) turns NaN/None into the string "nan",
    # so a null key on both sides matched itself and quietly imported that row's values
    # (2026-08-04, a regression introduced by the previous fix). Keep the null mask explicitly.
    left_null = d[df_key].isna()
    left_keys = d[df_key].astype(str).mask(left_null, other=pd.NA)
    e = e.rename(columns={ext_key: "_k"})
    e = e[e["_k"].notna() & (e["_k"].astype(str) != "nan")]

    # An explicit match indicator, under a name that cannot collide with a real column.
    # Using "the first value column is null" as a proxy for "the key did not resolve"
    # conflates a key ABSENT from the reference with a key that resolved to a row whose first
    # value happens to be null; but a fixed sentinel name collided with a competition column
    # called _matched, and with an output column of that name (2026-08-04).
    taken = set(d.columns) | set(e.columns) | set(out_cols)
    mcol = "_matched"
    while mcol in taken:
        mcol += "_"
    e[mcol] = True
    merged = d.assign(_k=left_keys).merge(e, on="_k", how="left")
    merged = merged.sort_values("_orig_order").reset_index(drop=True)
    assert len(merged) == len(df), (
        f"lookup join changed the row count ({len(df)} -> {len(merged)}); duplicate keys "
        f"survived de-duplication")

    unresolved_mask = merged[mcol].isna()
    # str() every key before sorting: under pandas 3 astype(str) preserves NA rather than
    # producing 'nan', and sorted() on a list mixing str with NA raises TypeError -- which
    # discarded a join that had already succeeded, from inside the REPORT builder
    # (2026-08-04 adversarial verification).
    unresolved_keys = _key_list(left_keys[unresolved_mask.to_numpy()].unique())
    if unresolved_mask.all() and len(df):
        # Same failure as an empty table, reached by a different route: a non-empty reference
        # whose key space does not overlap the competition's (wrong column, wrong code
        # vintage, string-vs-int keys). Partial misses are normal and get reported; a total
        # miss is a wiring error.
        raise ValueError(
            f"lookup join on {df_key!r} -> {ext_key!r} resolved 0 of {len(df)} rows. The "
            f"reference table has {len(ext)} rows but shares no keys with the data "
            f"(data e.g. {left_keys.unique()[:3].tolist()}, "
            f"reference e.g. {e['_k'].unique()[:3].tolist() if '_k' in e else ext[ext_key].astype(str).unique()[:3].tolist()}). "
            f"This is a wiring error, not a weak feature.")
    out = merged.drop(columns=["_k", "_orig_order", mcol])

    report = {
        "out_cols": out_cols,
        "join_key_class": "lookup",
        "leakage_rule": "static (no time axis)",
        "rows": int(len(df)),
        "rows_matched": int((~unresolved_mask).sum()),
        "rows_unmatched": int(unresolved_mask.sum()),
        "unresolved_keys": unresolved_keys[:50],
        "unresolved_key_count": len(unresolved_keys),
        # Per-column nulls, because a key resolving does not mean every value arrived: a
        # second-or-later value column that is entirely null used to be booked as a realized
        # feature with nothing anywhere saying it was empty.
        "null_by_column": {c: int(merged[c].isna().sum()) for c in out_cols},
        "all_null_columns": [c for c in out_cols if merged[c].isna().all()],
        "ext_duplicates_dropped": n_dup,
        "ext_rows": int(len(ext)),
        "rows_with_no_key": int(left_null.sum()),
        "source_meta": meta or {},
        "joined_at": datetime.now().isoformat(timespec="seconds"),
    }
    _write_log(report, log_path)
    return out, report


def merge_holiday_flags(df: pd.DataFrame, hol: pd.DataFrame, *, df_country: str | None,
                        df_date: str, out_col: str = "is_holiday",
                        log_path: str | Path | None = None,
                        meta: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Add a holiday indicator column via exact (country, date) lookup.

    Holidays are a deterministic calendar — there is no lag parameter and no
    LeakageError path; knowing future holidays is not leakage.

    `df_country=None` is the DATE-ONLY form: any date-indexed competition can take a single
    calendar without having a country panel. Requiring a country column was an implementation
    binding, not a property of the data, and it is one reason the injection layer only ever
    fired on country panels (2026-08-03 source-layer extension). In that form `hol` must
    carry exactly one country, so the calendar being applied is unambiguous.
    """
    def _normalize_dates(s: pd.Series, what: str) -> pd.Series:
        # Refuse numerics rather than guessing. pd.to_datetime(20240101) is 1970-01-01 plus
        # 20240101 NANOSECONDS, so an int-encoded calendar silently collapses to one instant:
        # every row matches or none does, and the report looks clean either way. The sibling
        # helper _to_period already refuses this input; this one used to guess (2026-08-04).
        if pd.api.types.is_numeric_dtype(s):
            raise ValueError(
                f"{what} date column is numeric ({s.dtype}); pass real dates or parsed "
                f"Timestamps. Interpreting integers as datetimes collapses them all to the "
                f"epoch and produces an all-or-nothing match with a clean-looking report.")
        out = pd.to_datetime(s)
        if getattr(out.dt, "tz", None) is not None:
            out = out.dt.tz_localize(None)
        return out.dt.normalize()

    d = df.copy()
    d["_date"] = _normalize_dates(d[df_date], "competition")
    # The CALENDAR side needs the same normalization. It used to be compared raw, so a
    # calendar whose dates are strings, datetime.date objects, or tz-aware Timestamps matched
    # nothing at all and the function returned an all-zero flag with a clean report -- the
    # feature looked complete and was constant (2026-08-04 adversarial verification).
    hol = hol.copy()
    hol["date"] = _normalize_dates(hol["date"], "calendar")
    # NaT must never join. `pd.NaT in {pd.NaT}` is True by IDENTITY, so once both sides were
    # normalized a calendar row with a blank date started flagging every data row with a
    # missing date as a holiday -- a fabricated 1 where the pre-fix code gave a correct 0, and
    # the fabricated count also defeats apply.py's all-zero join-failure guard (2026-08-04).
    n_nat_cal = int(hol["date"].isna().sum())
    hol = hol[hol["date"].notna()]
    if df_country is None:
        cals = _key_list(hol["country"]) if "country" in hol.columns else ["<none>"]
        if "country" in hol.columns and len(cals) != 1:
            raise ValueError(
                f"date-only holiday join needs exactly one calendar in `hol`, got {cals}. "
                f"With several, which one applies to a row is undefined -- pass df_country "
                f"and let the country column decide.")
        key = set(hol["date"])
        d[out_col] = [0 if pd.isna(t) else int(t in key) for t in d["_date"]]
        countries_without_calendar = []
    else:
        key = set(zip(hol["country"], hol["date"]))
        d[out_col] = [0 if pd.isna(t) else int((c, t) in key)
                      for c, t in zip(d[df_country], d["_date"])]
        countries_without_calendar = sorted(
            str(c) for c in set(d[df_country]) - set(hol["country"]) if pd.notna(c))
    report = {
        "out_col": out_col,
        "rows": int(len(d)),
        "holiday_rows": int(d[out_col].sum()),
        "countries_without_calendar": [str(x) for x in countries_without_calendar],
        "rows_with_no_date": int(d["_date"].isna().sum()),
        "calendar_rows_with_no_date": n_nat_cal,
        "source_meta": meta or {},
        "joined_at": datetime.now().isoformat(timespec="seconds"),
        "leakage_check": "n/a (deterministic calendar)",
    }
    _write_log(report, log_path)
    return d.drop(columns=["_date"]), report
