"""
Deterministic cross-competition benchmark table builder (Phase C-1;
extended Phase G-1b with a tier4 tree-search column).

For each of the 10 playground-series competitions run this weekend, extracts
four tier scores:
  tier1 = generic baseline (competitions/run_competition.py batch script)
  tier2 = best skill-pipeline score BEFORE the "Phase B 自我改進迭代" commit
  tier3 = final best score after Phase B linear self-improvement iteration,
          EXCLUDING any tree-search entry (i.e. frozen at the Phase C-1
          benchmark's original meaning, even though tree-search entries were
          appended to experiments.json afterwards by Phase G-1a/G-1b)
  tier4 = best score after tree-search harvest (Phase G-1a/G-1b), i.e. the
          best over ALL experiment_ids including tree-search entries

Every experiments.json is loaded and normalized via the EXISTING adapter
(collect.py's normalize()/build_facts()), imported by path so the adapter
logic is never duplicated here. Tier boundaries that are not mechanically
derivable from the schema alone (which experiment_id belongs to which
"phase") are hardcoded below in TIER_CONFIG / SPECIAL CASES, per the task
brief — but every numeric value placed in the output still comes verbatim
from the competition's experiments.json (verified with asserts).

Tree-search entries are identified mechanically: every tree-search log_v2
entry logged by Phase G-1a/G-1b starts its "model" field with "tree-search"
(verified convention across all 9 harvested comps) — see is_tree_entry().
s3e9's tree search only TIED the linear best (never beat it), so no
tree-search entry was ever appended to its experiments.json; tier4 == tier3
there (12.070034), which is not a bug but the honest sweep-verdict recorded
in STATUS.md/weekend-plan.md (Phase E-1 "v2 12.070034 精確追平線性").

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


def is_tree_entry(e):
    """Mechanical detector for a Phase G-1a/G-1b tree-search log_experiment_v2
    entry: every one of them was logged with a "model" string starting with
    "tree-search" (verified convention, all 9 harvested comps)."""
    return str(e.get("model", "")).lower().startswith("tree-search")


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
    non_tree_ids = [e["experiment_id"] for e in exps if not is_tree_entry(e)]
    tier3_e = best(exps, non_tree_ids, direction)
    tier4_e = best(exps, all_ids, direction)
    has_tree = any(is_tree_entry(e) for e in exps)

    return {
        "comp": comp_key,
        "metric": metric,
        "direction": direction,
        "tier1": {"value": tier1_e["score"], "experiment_id": tier1_e["experiment_id"],
                   "label": "generic batch (run_competition.py)"},
        "tier2": {"value": tier2_e["score"], "experiment_id": tier2_e["experiment_id"],
                   "label": "skill pipeline best, pre-Phase-B (Phase A)"},
        "tier3": {"value": tier3_e["score"], "experiment_id": tier3_e["experiment_id"],
                   "label": "final best (post-Phase-B self-improvement, linear iteration only, "
                            "excludes tree-search entries)"},
        "tier4": {"value": tier4_e["score"], "experiment_id": tier4_e["experiment_id"],
                   "label": ("best after tree-search harvest (Phase G-1a/G-1b)" if has_tree
                             else "no tree-search entry beat the linear best -- tie with tier3")},
        "relative_pct": relative_pct(tier1_e["score"], tier3_e["score"], direction),
        "relative_pct_tier1_tier4": relative_pct(tier1_e["score"], tier4_e["score"], direction),
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
    # tier4: same TimeSeriesSplit trajectory + any tree-search entry that also
    # used TimeSeriesSplit CV (mechanically checked, not assumed) — s3e19's
    # single tree-search entry (id 8) does, per its own cv.strategy field.
    ts_tree_ids = [e["experiment_id"] for e in exps
                   if is_tree_entry(e)
                   and "timeseriessplit" in str(e.get("cv", {}).get("strategy", "")).lower()]
    ts_tier4_ids = ts_all_ids + ts_tree_ids
    ts_tier4_best = best(exps, ts_tier4_ids, direction)

    ts_row = {
        "comp": "s3e19",
        "metric": metric,
        "direction": direction,
        "tier1": None,  # no generic entry exists under TimeSeriesSplit CV — not comparable
        "tier2": {"value": ts_phase_a["score"], "experiment_id": ts_phase_a["experiment_id"],
                   "label": "skill pipeline best, pre-Phase-B (Phase A, TimeSeriesSplit CV)"},
        "tier3": {"value": ts_best["score"], "experiment_id": ts_best["experiment_id"],
                   "label": "final best (post-Phase-B, TimeSeriesSplit CV, linear iteration only, "
                            "excludes tree-search entries)"},
        "tier4": {"value": ts_tier4_best["score"], "experiment_id": ts_tier4_best["experiment_id"],
                   "label": "best after tree-search harvest (Phase G-1b, node #17, same "
                            "TimeSeriesSplit CV scheme)"},
        "relative_pct": relative_pct(ts_phase_a["score"], ts_best["score"], direction),
        "relative_pct_tier1_tier4": None,
        "relative_pct_tier2_tier4": relative_pct(ts_phase_a["score"], ts_tier4_best["score"], direction),
        "relative_pct_basis": "tier2->tier3/tier4 (no comparable generic tier1 under this CV scheme)",
        "note": ("Primary/honest row: test period is strictly future, so TimeSeriesSplit is the "
                 "only valid CV scheme here. The generic-batch entry (exp #1, 5.31891) used plain "
                 "KFold and is NOT tier1-comparable to this row — see the separate CV-scheme "
                 "diagnostic pair below. This CV-scheme caveat now covers tier2, tier3, AND tier4: "
                 "the tier4 tree-search entry (Phase D-5, node #17, 9.75707) also ran under the "
                 "same honest TimeSeriesSplit CV and carries the SAME fold-5 double-dip caveat as "
                 "tier2/tier3 (see competitions/playground-series-s3e19/REPORT.md 3c and STATUS.md "
                 "Appendix), PLUS its own scale-parameter/seed-selection OOF-fitting caveat — the "
                 "true expected 2022 SMAPE is best read as 'meaningfully below 10.02', not "
                 "literally 9.76."),
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
        "tier4": {"value": diag["score"], "experiment_id": diag["experiment_id"],
                   "label": "same as tier2/tier3 — no tree-search was ever run under this "
                            "diagnostic plain-KFold CV scheme (tree-search only ran on the honest "
                            "TimeSeriesSplit trajectory, see the row above)"},
        "relative_pct": relative_pct(gen["score"], diag["score"], direction),
        "relative_pct_tier1_tier4": relative_pct(gen["score"], diag["score"], direction),
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

    # Phase G-1a tree-search entry (id 5): its "score" field is ALREADY the
    # rounded-integer OOF MAE (same v2-schema convention as id 3/4 — see its
    # own notes: "top-level score = rounded, ensemble.score = raw"), so it can
    # be read straight from the normalized score field, no raw-text pull-out
    # needed like tier2/tier3 above.
    tree_e = next((e for e in exps if is_tree_entry(e)), None)
    assert tree_e is not None and tree_e["score"] == 1.33563, \
        "s3e16 tree-search entry missing or score drifted from the expected 1.33563"
    tier4_value = min(float(rounded_best), tree_e["score"])

    # Tree-search node #15 (tier4) was rebuilt into a real test submission and
    # submitted to Kaggle on 2026-07-06 — pulled mechanically (not hand-listed)
    # from experiments.json #5's own "leaderboard" field, same as champion_e
    # below for experiment_id 1's 2026-07-03 submission.
    champion_e = by_id(exps, 1)
    tier4_leaderboard = tree_e.get("leaderboard")
    tier2_leaderboard = champion_e.get("leaderboard")

    return {
        "comp": "s3e16",
        "metric": exps[0]["metric"],
        "direction": "minimize",
        "tier1": {"value": tier1_e["score"], "experiment_id": tier1_e["experiment_id"],
                   "label": "generic batch (run_competition.py), rounded MAE"},
        "tier2": {"value": float(rounded_best), "experiment_id": 1,
                   "label": "skill pipeline best, pre-Phase-B (rounded OOF MAE, from experiments.json "
                            "notes text — raw/unrounded blend_oof_mae=1.35589)",
                   "leaderboard": tier2_leaderboard},
        "tier3": {"value": float(rounded_best), "experiment_id": 1,
                   "label": "final best — Phase B ran (exp #3/#4: Optuna-tune + seed-bag) but BOTH "
                            "rounds made rounded OOF MAE worse (1.33850, 1.33893); stopped per "
                            "2-non-improving-round protocol, best remains unchanged. Linear "
                            "iteration only, excludes the tree-search entry below."},
        "tier4": {"value": tier4_value, "experiment_id": tree_e["experiment_id"],
                   "label": "best after tree-search harvest (Phase G-1a, node #15, ROUNDED OOF MAE — "
                            "this is a raw-vs-rounded INVERSION case: the node's raw OOF MAE "
                            "(1.35712) is WORSE than the linear champion's raw 1.35589, but its "
                            "ROUNDED OOF MAE (1.33563) is better; decision metric on this comp is "
                            "the rounded score, per this comp's own boundary-case finding). Rebuilt "
                            "into a real test submission and submitted to Kaggle on 2026-07-06 "
                            "(sub_tree_best_1.33563_20260706_115200.csv) — no longer CV-only, see "
                            "leaderboard field.",
                   "leaderboard": tier4_leaderboard},
        "relative_pct": relative_pct(tier1_e["score"], float(rounded_best), "minimize"),
        "relative_pct_tier1_tier4": relative_pct(tier1_e["score"], tier4_value, "minimize"),
        "note": ("Decision metric is ROUNDED OOF MAE (validated against real Kaggle LB, CV-LB gap "
                 "~0.006), not the raw blend_oof_mae exposed by collect.py's generic adapter. "
                 "Phase B attempted the validated Optuna-tune+seed-bag recipe but it did not survive "
                 "the rounding step (raw OOF improved 1.35589->1.35533, rounded OOF worsened "
                 "1.33812->1.33893) — a documented boundary case, not a missing iteration. tier4's "
                 "tree-search entry is likewise rounded-MAE-only-comparable: raw OOF is worse than "
                 "the linear champion, rounded OOF is better — see tier4 label. tier1/tier2/tier3 "
                 "report OOF metrics only (though the tier2/tier3 champion experiment #1 was itself "
                 "submitted to Kaggle on 2026-07-03, Public 1.34356/Private 1.34075 — see "
                 "tier2.leaderboard). tier4 is now ALSO real-LB-validated: the tree-search entry "
                 "(experiment #5) was rebuilt and submitted on 2026-07-06, Public 1.34315/Private "
                 "1.33859 (see tier4.leaderboard) — improving on experiment #1's LB on BOTH boards, "
                 "the first external validation of this comp's tree-search recipe."),
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
    tree_ids = [e["experiment_id"] for e in exps if is_tree_entry(e)]
    tier4_e = best(exps, [4, 5, 6, 7, 8] + tree_ids, direction)

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
                            "neighbor-week smoothing). Linear iteration only, excludes the "
                            "tree-search entry below."},
        "tier4": {"value": tier4_e["score"], "experiment_id": tier4_e["experiment_id"],
                   "label": "best after tree-search harvest (Phase G-1b, node #28, pure-structural "
                            "JOINT lineage — NOT the 21.0332 GBDT-blend node, which STATUS.md flags "
                            "as low-confidence CV-noise, see competitions/playground-series-s3e20/"
                            "STATUS.md Appendix 'Phase E-4 樹搜尋 v2')"},
        "relative_pct": relative_pct(tier1_e["score"], tier3_e["score"], direction),
        "relative_pct_tier1_tier4": relative_pct(tier1_e["score"], tier4_e["score"], direction),
        "note": ("No run_competition.py generic-batch script was ever executed for s3e20 — it is a "
                 "closed (2023) legacy competition first solved in an earlier session (2026-02-14), "
                 "outside this weekend's 10-competition batch scope. Its own exp #1-#2 (RMSE 38.51, "
                 "33.21) use a single time-based split with no CatBoost/TE and are NOT CV-comparable "
                 "to exp #3 onward (Leave-One-Year-Out); they are excluded rather than mislabeled as "
                 "tier1. tier1 here is a same-CV-scheme PROXY (GBDT blend before the TE insight), "
                 "not a generic-batch entry — treat this tier1->tier3/tier4 comparison as an "
                 "approximation. Late Submission for this competition is closed; all tiers (incl. "
                 "tier4) are CV-only and were never submitted to Kaggle."),
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
            "tier3": "final best score after Phase B linear self-improvement iteration "
                     "(excludes tree-search entries, frozen at the Phase C-1 benchmark meaning)",
            "tier4": "best score after tree-search harvest (Phase G-1a/G-1b), i.e. best over "
                     "ALL experiment_ids including tree-search entries",
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
        t4 = r["tier4"]["value"] if r.get("tier4") else None
        print(f"  {r['comp']:24s} tier1={t1} tier2={r['tier2']['value']} tier3={t3} tier4={t4} "
              f"rel%={r['relative_pct']} rel%(1->4)={r.get('relative_pct_tier1_tier4')}")


if __name__ == "__main__":
    main()
