"""Injection-operator dispatcher — the execution half of the shared contract.

The judgment layer (Stage 0.5) emits typed operators from
`knowledge/injection_operators.md`; this module realizes the ones it implements and
records the rest in a coverage ledger. An idea is never dropped in silence: that
silence is precisely how a correct "GDP as a level covariate" judgment turned into a
plain feature join on s3e19 and cost the run its point.

Realized here: join_feature, ratio_target, log_offset, trend_term, flag_feature.
Recorded-unrealized: sample_weight (no per-row weight hook in the evaluators),
split_policy and postprocess (honored by the evaluator, not by this module).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .join import merge_holiday_flags, merge_year_safe
from .sources import fetch_holidays, fetch_worldbank, resolve_iso3

# operators this module can realize; everything else lands in the ledger
REALIZED = {"join_feature", "ratio_target", "log_offset", "trend_term", "flag_feature"}
EVALUATOR_OWNED = {"split_policy", "postprocess"}      # honored elsewhere, not a gap
_WB_PREFIX = "worldbank:"


def _fetch_covariate(source: str, years: list[int], countries: list[str]):
    """Return (frame, mapping, meta) for a whitelisted yearly country covariate."""
    if not source.startswith(_WB_PREFIX):
        raise ValueError(f"source '{source}' is not a whitelisted yearly covariate")
    key = source[len(_WB_PREFIX):]
    mapping, unmatched = resolve_iso3(countries)
    if unmatched:
        raise ValueError(f"countries not resolvable to ISO3, refusing to guess: {unmatched}")
    frame, meta = fetch_worldbank(key, min(years) - 1, max(years))
    return frame, mapping, meta


def apply_operators(train: pd.DataFrame, test: pd.DataFrame, ideas: list[dict], *,
                    country_col: str = "country", date_col: str = "date",
                    target_col: str = "num_sold",
                    ledger_path: str | Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Realize typed injection operators on (train, test).

    Returns (train2, test2, plan) where `plan` carries the coverage ledger plus the
    target transform the evaluator must apply:
        plan["target_transform"] = {"kind": "ratio_log", "covariate_col": "gdp_pc"}
    means fit on log(target / covariate) and invert with exp(pred) * covariate.
    """
    tr, te = train.copy(), test.copy()
    years = sorted(set(pd.to_datetime(tr[date_col]).dt.year) |
                   set(pd.to_datetime(te[date_col]).dt.year))
    countries = sorted(set(tr[country_col]) | set(te[country_col]))
    plan: dict = {"proposed": len(ideas), "realized": [], "unrealized": [],
                  "target_transform": None, "added_columns": [], "sources": {}}

    for idea in ideas:
        op = idea.get("operator")
        params = idea.get("params") or {}
        if op in EVALUATOR_OWNED:
            plan["realized"].append({"operator": op, "note": "honored by the evaluator"})
            continue
        if op not in REALIZED:
            plan["unrealized"].append({"operator": op,
                                       "reason": f"not implemented in apply.py (known set: {sorted(REALIZED)})"})
            continue
        try:
            if op in ("join_feature", "ratio_target", "log_offset"):
                src = params["source"]
                out_col = (params.get("as") or ["cov"])[0] if op == "join_feature" else "cov_level"
                frame, mapping, meta = _fetch_covariate(src, years, countries)
                lag = (params.get("join") or {}).get("lag", 0)
                for name, df in (("train", tr), ("test", te)):
                    df["_year"] = pd.to_datetime(df[date_col]).dt.year
                joined = {}
                for name, df in (("train", tr), ("test", te)):
                    out, rep = merge_year_safe(df, frame, df_country=country_col,
                                               df_year="_year", out_col=out_col,
                                               mapping=mapping, lag=lag, meta=meta)
                    if rep["rows_matched"] < rep["rows"]:
                        # carry forward is required when the test period post-dates coverage
                        if not params.get("carry_forward"):
                            raise ValueError(
                                f"{name}: {rep['rows_unmatched']} rows unmatched and "
                                f"carry_forward is not set")
                        latest = frame.sort_values("year").groupby("iso3")["value"].last()
                        iso = out[country_col].map(mapping)
                        out[out_col] = out[out_col].fillna(iso.map(latest))
                        plan.setdefault("carry_forward_applied", []).append(name)
                    joined[name] = out.drop(columns=["_year"], errors="ignore")
                tr, te = joined["train"], joined["test"]
                plan["sources"][src] = meta.get("snapshot")
                if op == "join_feature":
                    plan["added_columns"].append(out_col)
                else:
                    plan["target_transform"] = {
                        "kind": "ratio_log" if params.get("space", "log") == "log" else "ratio_linear",
                        "covariate_col": out_col}
                plan["realized"].append({"operator": op, "column": out_col})

            elif op == "trend_term":
                unit, deg = params.get("unit", "year"), int(params.get("degree", 1))
                if unit != "year":
                    raise ValueError("only unit='year' is implemented")
                base = pd.to_datetime(tr[date_col]).dt.year
                centre = float(base.mean()) if params.get("centered", True) else 0.0
                for df in (tr, te):
                    y = pd.to_datetime(df[date_col]).dt.year - centre
                    for d in range(1, deg + 1):
                        col = "year_c" if d == 1 else f"year_c{d}"
                        df[col] = y ** d
                cols = ["year_c"] + [f"year_c{d}" for d in range(2, deg + 1)]
                plan["added_columns"] += cols
                plan["realized"].append({"operator": op, "columns": cols, "centre": centre})

            elif op == "flag_feature":
                as_cols = params.get("as") or ["is_holiday"]
                hol, meta = fetch_holidays(countries, years)
                for name in ("train", "test"):
                    df = tr if name == "train" else te
                    out, _ = merge_holiday_flags(df, hol, df_country=country_col,
                                                 df_date=date_col, out_col=as_cols[0], meta=meta)
                    if name == "train":
                        tr = out
                    else:
                        te = out
                plan["added_columns"].append(as_cols[0])
                plan["sources"]["holidays"] = meta.get("snapshot")
                extra = [k for k in ("window", "per_name") if params.get(k)]
                if extra:
                    plan["unrealized"].append({"operator": op, "params_subset": extra,
                                               "reason": "only a boolean same-day flag is implemented"})
                plan["realized"].append({"operator": op, "column": as_cols[0]})
        except Exception as e:  # noqa: BLE001 — a failed operator is recorded, never hidden
            plan["unrealized"].append({"operator": op, "reason": f"{type(e).__name__}: {e}"})

    plan["realized_count"] = len(plan["realized"])
    if ledger_path:
        Path(ledger_path).parent.mkdir(parents=True, exist_ok=True)
        Path(ledger_path).write_text(json.dumps(plan, indent=2, ensure_ascii=False))
    return tr, te, plan
