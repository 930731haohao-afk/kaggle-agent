"""
Deterministic cross-competition benchmark table builder (Phase C-1).

For each of the 10 playground-series competitions run this weekend, extracts
three tier scores:
  tier1 = generic baseline (competitions/run_competition.py batch script)
  tier2 = best skill-pipeline score BEFORE the "Phase B 自我改進迭代" commit
  tier3 = final best score (after Phase B, i.e. current state)

Every experiments.json is loaded and normalized via the EXISTING adapter
(collect.py's normalize()/build_facts()), imported by path so the adapter
logic is never duplicated here. Tier boundaries that are not mechanically
derivable from the schema alone (which experiment_id belongs to which
"phase") are hardcoded below in TIER_CONFIG / SPECIAL CASES, per the task
brief — but every numeric value placed in the output still comes verbatim
from the competition's experiments.json (verified with asserts).

Usage (from project root):
    uv run python3 docs/scripts/build_benchmark_table.py
Writes docs/benchmark_facts.json.
"""
import importlib.util
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
COLLECT_PATH = os.path.join(
    ROOT, ".claude", "skills", "kaggle-report", "assets", "collect.py")
COMPETITIONS_DIR = os.path.join(ROOT, "competitions")
OUT_PATH = os.path.join(ROOT, "docs", "benchmark_facts.json")

# ---------------------------------------------------------------------------
# Load collect.py by path (no duplication of its adapter logic).
# ---------------------------------------------------------------------------
_spec = importlib.util.spec_from_file_location("kaggle_report_collect", COLLECT_PATH)
collect = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(collect)


def by_id(experiments, exp_id):
    matches = [e for e in experiments if e.get("experiment_id") == exp_id]
    if not matches:
        raise KeyError(f"experiment_id {exp_id} not found")
    return matches[0]


def best(experiments, ids, direction):
    pool = [e for e in experiments if e.get("experiment_id") in ids and e.get("score") is not None]
    if not pool:
        raise ValueError(f"no scored experiments among ids={ids}")
    fn = min if direction == "minimize" else max
    return fn(pool, key=lambda e: e["score"])


def relative_pct(tier1, tier3, direction):
    if tier1 is None or tier3 is None:
        return None
    if direction == "minimize":
        return (tier1 - tier3) / tier1 * 100.0
    return (tier3 - tier1) / tier1 * 100.0


# ---------------------------------------------------------------------------
# Per-competition tier-boundary map (hardcoded where automation is ambiguous;
# every id below maps to an experiment_id that literally exists in that
# competition's experiments.json — see comments for the git-history evidence
# used to draw each boundary).
# ---------------------------------------------------------------------------
# "phase_a_ids": experiment_ids present in experiments.json as of the
#   "完整 skill 流程" commit (git log --oneline -- <path>, first commit).
#   tier1 is always id 1 in this group (confirmed model starts "generic ").
#   tier2 = best among phase_a_ids EXCLUDING id 1.
#   tier3 = best among ALL ids (phase_a + Phase B).
STANDARD_COMPS = {
    "s3e1":  {"phase_a_ids": [1, 2, 3]},
    "s3e3":  {"phase_a_ids": [1, 2, 3]},
    "s3e5":  {"phase_a_ids": [1, 2, 3]},
    "s3e7":  {"phase_a_ids": [1, 2, 3]},
    "s3e9":  {"phase_a_ids": [1, 2, 3, 4]},
    "s3e11": {"phase_a_ids": [1, 2, 3]},
    "s3e14": {"phase_a_ids": [1, 2, 3]},
}


def build_standard_row(comp_key, cfg):
    comp_dir = os.path.join(COMPETITIONS_DIR, f"playground-series-{comp_key}")
    facts = collect.build_facts(comp_dir)
    exps = facts["experiments"]
    direction = exps[0]["direction"]
    metric = exps[0]["metric"]

    tier1_e = by_id(exps, 1)
    assert (tier1_e["source_format"] == "generic_batch"
            or str(tier1_e.get("model", "")).startswith("generic ")), \
        f"{comp_key}: experiment_id 1 is not a generic-batch entry"

    phase_a_ids = cfg["phase_a_ids"]
    tier2_e = best(exps, [i for i in phase_a_ids if i != 1], direction)
    all_ids = [e["experiment_id"] for e in exps]
    tier3_e = best(exps, all_ids, direction)

    return {
        "comp": comp_key,
        "metric": metric,
        "direction": direction,
        "tier1": {"value": tier1_e["score"], "experiment_id": tier1_e["experiment_id"],
                   "label": "generic batch (run_competition.py)"},
        "tier2": {"value": tier2_e["score"], "experiment_id": tier2_e["experiment_id"],
                   "label": "skill pipeline best, pre-Phase-B (Phase A)"},
        "tier3": {"value": tier3_e["score"], "experiment_id": tier3_e["experiment_id"],
                   "label": "final best (post-Phase-B self-improvement)"},
        "relative_pct": relative_pct(tier1_e["score"], tier3_e["score"], direction),
        "note": None,
    }


# ---------------------------------------------------------------------------
# s3e19 — CV-scheme guard (knowledge/experience.md 依 CV 設計節):
# the generic baseline (exp #1) and the diagnostic re-run (exp #3) both use
# plain shuffled 5-fold KFold; the actual skill pipeline (exp #2, #4-#7) uses
# TimeSeriesSplit because test is a strictly future period. Comparing across
# schemes is explicitly forbidden by experience.md ("絕不跨 CV 方案比較分數").
# We therefore report TWO rows: the honest TimeSeriesSplit trajectory (used
# as the main-table row) and a same-CV-scheme diagnostic pair (reported
# separately, not blended into the main tier1->tier3 comparison).
# ---------------------------------------------------------------------------
def build_s3e19_rows():
    comp_dir = os.path.join(COMPETITIONS_DIR, "playground-series-s3e19")
    facts = collect.build_facts(comp_dir)
    exps = facts["experiments"]
    direction = exps[0]["direction"]
    metric = exps[0]["metric"]

    gen = by_id(exps, 1)
    assert gen["source_format"] == "generic_batch" or str(gen.get("model", "")).startswith("generic ")
    diag = by_id(exps, 3)
    ts_phase_a = by_id(exps, 2)
    ts_all_ids = [2, 4, 5, 6, 7]
    ts_best = best(exps, ts_all_ids, direction)

    ts_row = {
        "comp": "s3e19",
        "metric": metric,
        "direction": direction,
        "tier1": None,  # no generic entry exists under TimeSeriesSplit CV — not comparable
        "tier2": {"value": ts_phase_a["score"], "experiment_id": ts_phase_a["experiment_id"],
                   "label": "skill pipeline best, pre-Phase-B (Phase A, TimeSeriesSplit CV)"},
        "tier3": {"value": ts_best["score"], "experiment_id": ts_best["experiment_id"],
                   "label": "final best (post-Phase-B, TimeSeriesSplit CV)"},
        "relative_pct": relative_pct(ts_phase_a["score"], ts_best["score"], direction),
        "relative_pct_basis": "tier2->tier3 (no comparable generic tier1 under this CV scheme)",
        "note": ("Primary/honest row: test period is strictly future, so TimeSeriesSplit is the "
                 "only valid CV scheme here. The generic-batch entry (exp #1, 5.31891) used plain "
                 "KFold and is NOT tier1-comparable to this row — see the separate CV-scheme "
                 "diagnostic pair below."),
    }
    diag_row = {
        "comp": "s3e19-kfold-diagnostic",
        "metric": metric,
        "direction": direction,
        "tier1": {"value": gen["score"], "experiment_id": gen["experiment_id"],
                   "label": "generic batch (run_competition.py), plain KFold"},
        "tier2": {"value": diag["score"], "experiment_id": diag["experiment_id"],
                   "label": "skill features/models, SAME plain KFold (diagnostic only)"},
        "tier3": {"value": diag["score"], "experiment_id": diag["experiment_id"],
                   "label": "same as tier2 — never iterated further under this CV scheme"},
        "relative_pct": relative_pct(gen["score"], diag["score"], direction),
        "note": ("Same-CV-scheme sanity check only (knowledge/experience.md CV-design guard): "
                 "isolates the skill's feature/model gain from the CV-scheme optimism gap. Not a "
                 "real production trajectory — the actual submission path is the TimeSeriesSplit "
                 "row above."),
    }
    return [ts_row, diag_row]


# ---------------------------------------------------------------------------
# s3e16 — rounded-OOF guard (knowledge/experience.md MAE/整數目標 節):
# the decision metric for this competition is rounded-to-integer OOF MAE
# (validated against real Kaggle LB, gap ~0.006). collect.py's generic
# adapters take the raw (pre-round) score field, so the tier2/tier3 rounded
# values (1.33812) are not exposed by the normalized "score" field for the
# original skill run — they are pulled directly (and verified present,
# verbatim) from the experiments.json notes text where they were logged.
# Phase B (exp #3, #4) tried Optuna-tuning + seed-bagging per the validated
# cross-comp recipe, but BOTH rounds made the rounded score worse (documented
# in-file) — so tier3 == tier2 here: Phase B ran but did not improve.
# ---------------------------------------------------------------------------
def build_s3e16_row():
    comp_dir = os.path.join(COMPETITIONS_DIR, "playground-series-s3e16")
    facts = collect.build_facts(comp_dir)
    exps = facts["experiments"]
    raw_text = open(os.path.join(comp_dir, "experiments.json"), encoding="utf-8").read()

    tier1_e = by_id(exps, 2)
    assert tier1_e["source_format"] == "generic_batch"
    assert tier1_e["score"] == 1.35441

    rounded_best = "1.33812"
    assert rounded_best in raw_text, "rounded best score not found verbatim in experiments.json"
    # Phase B (exp #3 notes: rounded 1.33850; exp #4 notes: rounded 1.33893) both worse -> stop.
    for worse_val in ("1.33850", "1.33893"):
        assert worse_val in raw_text

    return {
        "comp": "s3e16",
        "metric": exps[0]["metric"],
        "direction": "minimize",
        "tier1": {"value": tier1_e["score"], "experiment_id": tier1_e["experiment_id"],
                   "label": "generic batch (run_competition.py), rounded MAE"},
        "tier2": {"value": float(rounded_best), "experiment_id": 1,
                   "label": "skill pipeline best, pre-Phase-B (rounded OOF MAE, from experiments.json "
                            "notes text — raw/unrounded blend_oof_mae=1.35589)"},
        "tier3": {"value": float(rounded_best), "experiment_id": 1,
                   "label": "final best — Phase B ran (exp #3/#4: Optuna-tune + seed-bag) but BOTH "
                            "rounds made rounded OOF MAE worse (1.33850, 1.33893); stopped per "
                            "2-non-improving-round protocol, best remains unchanged"},
        "relative_pct": relative_pct(tier1_e["score"], float(rounded_best), "minimize"),
        "note": ("Decision metric is ROUNDED OOF MAE (validated against real Kaggle LB, CV-LB gap "
                 "~0.006), not the raw blend_oof_mae exposed by collect.py's generic adapter. "
                 "Phase B attempted the validated Optuna-tune+seed-bag recipe but it did not survive "
                 "the rounding step (raw OOF improved 1.35589->1.35533, rounded OOF worsened "
                 "1.33812->1.33893) — a documented boundary case, not a missing iteration."),
    }


# ---------------------------------------------------------------------------
# s3e20 — no generic-batch script was ever run for this competition (it is a
# 2023-closed legacy competition first solved in Session 8, 2026-02-14 —
# outside the weekend run_competition.py batch scope). exp #1-#2 use a
# single time-based split (2019-2020 train / 2021 val) with no CatBoost and
# no location-week TE; they are NOT CV-comparable to exp #3 onward, which
# switch to Leave-One-Year-Out CV. Per the task brief, we use exp #3 (GBDT
# blend WITHOUT the location-week TE member, but under the SAME LOYO CV as
# the final result) as a proxy tier1 anchor -- clearly flagged as not a true
# generic-batch entry -- rather than silently treating it as one.
# ---------------------------------------------------------------------------
def build_s3e20_row():
    comp_dir = os.path.join(COMPETITIONS_DIR, "playground-series-s3e20")
    facts = collect.build_facts(comp_dir)
    exps = facts["experiments"]
    direction = "minimize"

    tier1_e = by_id(exps, 3)
    assert tier1_e["score"] == 28.3424
    tier2_e = by_id(exps, 4)
    assert tier2_e["score"] == 22.6488
    tier3_e = best(exps, [4, 5, 6, 7, 8], direction)

    return {
        "comp": "s3e20",
        "metric": exps[0]["metric"],
        "direction": direction,
        "tier1": {"value": tier1_e["score"], "experiment_id": tier1_e["experiment_id"],
                   "label": "PROXY baseline: 3-model GBDT blend WITHOUT location-week TE, same "
                            "Leave-One-Year-Out CV as tier2/3 (NOT a real generic-batch entry — see note)"},
        "tier2": {"value": tier2_e["score"], "experiment_id": tier2_e["experiment_id"],
                   "label": "skill pipeline best, pre-Phase-B (adds pure location-week TE, weight "
                            "search gives it 100%)"},
        "tier3": {"value": tier3_e["score"], "experiment_id": tier3_e["experiment_id"],
                   "label": "final best (post-Phase-B TE denoising: shrinkage + COVID-downweight + "
                            "neighbor-week smoothing)"},
        "relative_pct": relative_pct(tier1_e["score"], tier3_e["score"], direction),
        "note": ("No run_competition.py generic-batch script was ever executed for s3e20 — it is a "
                 "closed (2023) legacy competition first solved in an earlier session (2026-02-14), "
                 "outside this weekend's 10-competition batch scope. Its own exp #1-#2 (RMSE 38.51, "
                 "33.21) use a single time-based split with no CatBoost/TE and are NOT CV-comparable "
                 "to exp #3 onward (Leave-One-Year-Out); they are excluded rather than mislabeled as "
                 "tier1. tier1 here is a same-CV-scheme PROXY (GBDT blend before the TE insight), "
                 "not a generic-batch entry — treat this tier1->tier3 comparison as an approximation."),
    }


def main():
    rows = []
    for comp_key, cfg in STANDARD_COMPS.items():
        rows.append(build_standard_row(comp_key, cfg))
    rows.extend(build_s3e19_rows())
    rows.append(build_s3e16_row())
    rows.append(build_s3e20_row())

    out = {
        "generated_by": "docs/scripts/build_benchmark_table.py",
        "source": "collect.py (imported by path) over each competitions/<name>/experiments.json",
        "tier_definitions": {
            "tier1": "generic baseline (competitions/run_competition.py batch run)",
            "tier2": "best skill-pipeline score BEFORE the Phase B self-improvement commit",
            "tier3": "final best score (after Phase B, current state)",
        },
        "rows": rows,
    }
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"wrote {OUT_PATH}: {len(rows)} rows")
    for r in rows:
        t1 = r["tier1"]["value"] if r["tier1"] else None
        t3 = r["tier3"]["value"]
        print(f"  {r['comp']:24s} tier1={t1} tier2={r['tier2']['value']} tier3={t3} "
              f"rel%={r['relative_pct']}")


if __name__ == "__main__":
    main()
