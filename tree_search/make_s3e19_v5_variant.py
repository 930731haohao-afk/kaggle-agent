"""Build an s3e19 v5 external-data variant: augmented workspace + generated evaluator.

Creates competitions/playground-series-s3e19-v5-<variant>/data/ containing the
digit-exact processed CSVs PLUS whitelisted external columns, then writes
tree_search/eval_s3e19_v5_<variant>.py by transforming eval_s3e19.py source
(variant _COMP_DIR, extended FEATURE_COLS, separate cache dir) so every CV
subtlety (unique-date TimeSeriesSplit, IDX mask, postprocess-in-metric) is
inherited byte-identically instead of re-implemented.

Variants:
  gdp      — + gdp_pc (World Bank, country x year, lag=0)
  gdp_hol  — + gdp_pc, is_holiday (holidays pkg, exact date)

Validation contract: running the variant evaluator on a config that DROPS the
external columns must reproduce the baseline evaluator's score digit-exactly.

Usage: VIRTUAL_ENV= uv run python3 tree_search/make_s3e19_v5_variant.py <variant>
"""
import json
import os
import re
import shutil
import sys

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "external_data"))
from join import merge_holiday_flags, merge_year_safe  # noqa: E402
from sources import fetch_holidays, fetch_worldbank, resolve_iso3  # noqa: E402

COMP = "playground-series-s3e19"
BASE_DIR = os.path.join(_ROOT, "competitions", COMP)
EXT_COLS = {"gdp": ["gdp_pc"], "gdp_hol": ["gdp_pc", "is_holiday"]}


def build(variant: str) -> None:
    ext_cols = EXT_COLS[variant]
    vcomp = f"{COMP}-v5-{variant}"
    vdir = os.path.join(_ROOT, "competitions", vcomp, "data")
    os.makedirs(vdir, exist_ok=True)

    raw = {split: pd.read_csv(os.path.join(BASE_DIR, "data", f"{split}.csv"),
                              usecols=["id", "date", "country"])
           for split in ("train", "test")}
    countries = sorted(raw["train"]["country"].unique())
    mapping, unmatched = resolve_iso3(countries)
    assert not unmatched, f"unmapped countries: {unmatched}"

    years = []
    for split in ("train", "test"):
        years += pd.to_datetime(raw[split]["date"]).dt.year.unique().tolist()

    # The rules gate is binding on EVERY fetch path, not only apply_operators. This builder
    # called fetch_worldbank/fetch_holidays directly, so it reached external data with no
    # verdict at all -- the exact bypass the gate exists to close (2026-08-07 re-verification).
    import json as _json
    _vpath = os.path.join(BASE_DIR, "rules_verdict.json")
    if not os.path.exists(_vpath):
        raise SystemExit(
            f"no rules verdict at {_vpath}. Run `python3 external_data/rules_gate.py "
            f"<rules.txt> --record-to {_vpath}` first; this builder fetches external data "
            f"and absence of a verdict is not permission.")
    _v = _json.load(open(_vpath))
    if _v.get("verdict") != "permitted":
        raise SystemExit(f"rules verdict is {_v.get('verdict')!r}; external data is off for "
                         f"this competition and this builder must not run.")
    gdp, gmeta = fetch_worldbank("gdp_per_capita", min(years) - 1, max(years))
    hol, hmeta = fetch_holidays(countries, sorted(set(years)))

    for split in ("train", "test"):
        proc = pd.read_csv(os.path.join(BASE_DIR, "data", f"{split}_processed.csv"))
        aug = proc.merge(raw[split][["id", "country"]], on="id", how="left")
        aug["_year"] = pd.to_datetime(aug["date"]).dt.year
        aug, grep_ = merge_year_safe(
            aug, gdp, df_country="country", df_year="_year", out_col="gdp_pc",
            mapping=mapping, lag=0, meta=gmeta,
            log_path=os.path.join(vdir, f"join_gdp_{split}.log.json"))
        assert grep_["rows_matched"] == grep_["rows"], grep_
        if "is_holiday" in ext_cols:
            aug, _ = merge_holiday_flags(
                aug, hol, df_country="country", df_date="date", meta=hmeta,
                log_path=os.path.join(vdir, f"join_holiday_{split}.log.json"))
        aug = aug.drop(columns=["country", "_year"])
        assert list(aug.columns[:len(proc.columns)]) == list(proc.columns)
        assert len(aug) == len(proc)
        aug.to_csv(os.path.join(vdir, f"{split}_processed.csv"), index=False)
    # raw files: symlink so any fallback path still resolves
    for name in ("train.csv", "test.csv", "sample_submission.csv"):
        src = os.path.join(BASE_DIR, "data", name)
        dst = os.path.join(vdir, name)
        if os.path.exists(src) and not os.path.exists(dst):
            os.symlink(src, dst)

    # generate the variant evaluator from the baseline source
    src = open(os.path.join(_HERE, "eval_s3e19.py")).read()
    src, n1 = re.subn(
        r'_COMP_DIR = os\.path\.join\(_REPO_ROOT, "competitions", "playground-series-s3e19"\)',
        f'_COMP_DIR = os.path.join(_REPO_ROOT, "competitions", "{vcomp}")', src)
    src, n2 = re.subn(
        r'CACHE_DIR = os\.path\.join\(_HERE, "cache_s3e19"\)',
        f'CACHE_DIR = os.path.join(_HERE, "cache_s3e19_v5_{variant}")', src)
    ext_list = ", ".join(f'"{c}"' for c in ext_cols)
    src, n3 = re.subn(
        r'(FEATURE_COLS = \[[^\]]+\])',
        r"\1" + f"\nFEATURE_COLS = FEATURE_COLS + [{ext_list}]  # v5 external columns", src)
    assert (n1, n2, n3) == (1, 1, 1), (n1, n2, n3)
    out_eval = os.path.join(_HERE, f"eval_s3e19_v5_{variant}.py")
    open(out_eval, "w").write(src)
    print(json.dumps({"variant": variant, "workspace": vdir, "evaluator": out_eval,
                      "ext_cols": ext_cols, "gdp_snapshot": gmeta["snapshot"]}, indent=2))


if __name__ == "__main__":
    build(sys.argv[1])
