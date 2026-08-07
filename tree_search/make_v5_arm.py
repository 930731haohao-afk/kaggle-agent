"""Comp-agnostic v5 arm builder — driven by the dossier's typed injection operators.

Replaces the column-only `make_s3e19_v5_variant.py`: that builder could express exactly
one operator (`join_feature`) and so silently truncated a correct `ratio_target` judgment
into a plain feature join, which a GBDT cannot extrapolate (s3e19: 2022 predicted at 0.96x
the 2021 level, 48.24 SMAPE, while the experience library already held the ratio recipe
worth -2.6 SMAPE). Here the operator set drives both the data and the target.

What it does, per arm:
  1. run the operators through external_data.apply -> augmented raw frames + a plan
     (added columns, an optional target transform, and a coverage ledger)
  2. write an augmented workspace whose processed CSVs are the baseline's PLUS the new
     columns, so a config that drops them reproduces the baseline digit-for-digit
  3. generate an evaluator from the baseline evaluator by patching: comp dir, cache dir,
     FEATURE_COLS, and -- when the plan asks for it -- the target transform
     (fit on log(y/c), invert with exp(pred)*c instead of log1p/expm1)

Usage:
  VIRTUAL_ENV= uv run python3 tree_search/make_v5_arm.py <comp> <arm-name> <ideas.json>
"""
from __future__ import annotations

import json
import os
import re
import sys

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
from external_data.apply import apply_operators  # noqa: E402

# baseline evaluator + id column per competition
def _slug(comp: str) -> str:
    """Short, filesystem-safe tag for generated module and cache names."""
    return comp.replace("playground-series-", "").replace("tabular-", "").strip("-").replace("-", "")


BASE = {
    "playground-series-s3e19": ("eval_s3e19", "id"),
    "tabular-playground-series-sep-2022": ("eval_sep22", "row_id"),
    "playground-series-s5e1": ("eval_s5e1", "id"),
}


def _patch_target_transform(src: str, cov_col: str, kind: str) -> str:
    """Make the generated evaluator fit on the ratio and invert with the covariate."""
    if kind not in ("ratio_log", "ratio_linear"):
        raise ValueError(f"unknown target-transform kind {kind!r}")
    inv = "np.exp" if kind == "ratio_log" else ""
    # The FIT space must match the inversion: a ratio_linear arm that fits log(y/c) and then
    # inverts without exp trains on one target and predicts another (2026-08-03 audit).
    fit_expr = ("np.log(_train[TARGET].to_numpy(np.float64) / _COV_TR)" if kind == "ratio_log"
                else "_train[TARGET].to_numpy(np.float64) / _COV_TR")
    anchor = "_y = _train[TARGET].to_numpy(np.float64)"
    assert src.count(anchor) == 1
    src = src.replace(anchor, anchor + f'''

# ---- v5 target transform ({kind}) -------------------------------------------------
# Fit on the covariate-normalized target so the test-period LEVEL comes from the
# covariate instead of from a piecewise-constant model that cannot extrapolate.
_COV_TR = _train["{cov_col}"].to_numpy(np.float64)
_COV_TE = _test["{cov_col}"].to_numpy(np.float64)
assert np.all(_COV_TR > 0) and np.all(_COV_TE > 0), "covariate must be strictly positive"
_Y_RATIO = {fit_expr}


def _invert_va(p, va_mask):
    return {inv}(p) * _COV_TR[va_mask]


def _invert_test(p):
    return {inv}(p) * _COV_TE
# -----------------------------------------------------------------------------------''')

    # train on the ratio instead of log1p(y), EVERYWHERE _y_log is used as the fit target:
    # the lgb/cat fold loops, the sparse-linear family, and the per-fold target encoding.
    # Patching only the first left a mixed-target arm whose linear member and encodings were
    # fit on log1p(y) while the trees fit the ratio (2026-08-03 audit).
    n_y = src.count("y_tr, y_va = _y_log[tr_mask], _y_log[va_mask]")
    assert n_y >= 1, "no training target lines found"
    src = src.replace("y_tr, y_va = _y_log[tr_mask], _y_log[va_mask]",
                      "y_tr, y_va = _Y_RATIO[tr_mask], _Y_RATIO[va_mask]")
    n_other = 0
    for frm, to in (("esup.run_linear(params, Xdf, Xtestdf, cat_feats, _FOLDS, _y_log,",
                     "esup.run_linear(params, Xdf, Xtestdf, cat_feats, _FOLDS, _Y_RATIO,"),
                    ("X_tr, X_va, Xtestdf, _y_log[tr_mask],",
                     "X_tr, X_va, Xtestdf, _Y_RATIO[tr_mask],")):
        n_other += src.count(frm)
        src = src.replace(frm, to)
    # the linear family inverts with expm1 too; align it with the ratio inversion
    # Use the SAME fold-aware pair the GBDT path uses. The old rewrite passed a test-side
    # lambda for the OOF as well, which broadcast fold-sized predictions against the test
    # covariate (crash) or cancelled to ratio space (wrong units) (2026-08-07 round-3).
    n_lin_inv = src.count("invert=np.expm1)")
    src = src.replace("invert=np.expm1)",
                      "invert=_invert_test, invert_va=_invert_va)")
    if n_other or n_lin_inv:
        print(f"   ratio target also applied to {n_other} non-GBDT fit site(s), "
              f"{n_lin_inv} linear inversion(s)")
    # invert with the covariate instead of expm1. The test-frame operand is whatever the
    # baseline evaluator names it: `Xtestdf` before the per-fold target-encoding refactor,
    # `Xte_fold` after (2026-07-31). Matching on the frame name is what silently broke when
    # the evaluators were refactored, so the operand is now discovered rather than assumed.
    n_va = src.count("oof[va_mask] = np.expm1(m.predict(X_va))")
    te_from = re.findall(r"pred \+= np\.expm1\(m\.predict\((\w+)\)\) / N_SPLITS", src)
    n_te = len(te_from)
    assert n_va >= 1 and n_te >= 1, (
        "no inversion lines found -- the baseline evaluator's fold-loop shape changed; "
        "update _patch_target_transform's anchors before building arms")
    src = src.replace("oof[va_mask] = np.expm1(m.predict(X_va))",
                      "oof[va_mask] = _invert_va(m.predict(X_va), va_mask)")
    src = re.sub(r"pred \+= np\.expm1\(m\.predict\((\w+)\)\) / N_SPLITS",
                 r"pred += _invert_test(m.predict(\1)) / N_SPLITS", src)
    print(f"   target transform patched: {n_y} fit sites, {n_va} val + {n_te} test inversions")
    return src


def _patch_sample_weight(src: str, wcol: str) -> str:
    """Pass per-row weights into every fold fit, so a down-weighted regime actually
    influences training instead of being recorded in the plan and then ignored."""
    anchor = "_y = _train[TARGET].to_numpy(np.float64)"
    assert src.count(anchor) == 1
    src = src.replace(anchor, anchor +
                      f'\n_ROW_W = _train["{wcol}"].to_numpy(np.float64)   # v5 sample_weight operator')
    lgb_from = "        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],"
    lgb_to = "        m.fit(X_tr, y_tr, sample_weight=_ROW_W[tr_mask], eval_set=[(X_va, y_va)],"
    cat_from = "        m.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=esr)"
    cat_to = ("        m.fit(X_tr, y_tr, sample_weight=_ROW_W[tr_mask], "
              "eval_set=(X_va, y_va), early_stopping_rounds=esr)")
    n_lgb, n_cat = src.count(lgb_from), src.count(cat_from)
    # Both families must be patched: `n_lgb + n_cat >= 1` passed while one whole family
    # trained unweighted, so a down-weighted regime still drove half the pool
    # (2026-08-03 audit). A family that is absent from this evaluator is fine; a family that
    # is present but unpatched is not.
    has_lgb = "lgb.LGBMRegressor(" in src or "LGBMClassifier(" in src
    has_cat = "CatBoostRegressor(" in src or "CatBoostClassifier(" in src
    missing = [fam for fam, present, patched in
               (("lgb", has_lgb, n_lgb), ("cat", has_cat, n_cat)) if present and not patched]
    assert not missing, (f"sample_weight could not be patched into {missing} -- their fit "
                         f"call shape changed; update _patch_sample_weight's anchors")
    assert n_lgb + n_cat >= 1, "no fit sites found for sample_weight"
    src = src.replace(lgb_from, lgb_to).replace(cat_from, cat_to)
    # the sparse-linear family fits through eval_support.run_linear, which takes no weights:
    # say so rather than silently leaving that member unweighted
    if "esup.run_linear(" in src:
        print("   NOTE: the sparse-linear family does not take sample weights; its member "
              "trains unweighted (recorded, not silent)")
    print(f"   sample_weight patched into {n_lgb} lgb + {n_cat} cat fit sites")
    return src


# Which operator DEFINES each arm kind: an arm whose defining operator did not realize is a
# baseline clone wearing the arm's name, which is precisely the artifact the 2026-08-07
# re-verification found this module emitting (arm="ratio", target_transform=null, ledger
# realized=[], evaluator byte-identical to the baseline).
ARM_DEFINING_OP = {"ratio": "ratio_target", "log": "log_offset", "logoffset": "log_offset",
                   "featurejoin": "join_feature", "join": "join_feature",
                   "gdp": "join_feature", "gdp_hol": "flag_feature"}


def _defining_op_for(arm: str):
    """Longest-match lookup. Truncating at the first '_' made 'gdp_hol' resolve to 'gdp'
    (the wrong operator) and its own dict entry unreachable (2026-08-07 round-3)."""
    if arm in ARM_DEFINING_OP:
        return ARM_DEFINING_OP[arm]
    for cand in (arm.split("-")[0], arm.split("_")[0], arm.split("-")[0].split("_")[0]):
        if cand in ARM_DEFINING_OP:
            return ARM_DEFINING_OP[cand]
    return None


def build(comp: str, arm: str, ideas: list[dict]) -> dict:
    base_mod, id_col = BASE[comp]
    base_dir = os.path.join(_ROOT, "competitions", comp)
    vcomp = f"{comp}-v5-{arm}"
    vdir = os.path.join(_ROOT, "competitions", vcomp, "data")
    os.makedirs(vdir, exist_ok=True)

    # The rules gate in apply_operators is binding, and this is its production caller: the
    # verdict Stage 0.5 recorded lives in the competition workspace. Read it from there rather
    # than growing a parameter every call site would have to thread through -- and if Stage 0.5
    # never ran the gate, apply_operators refuses, which is the correct outcome
    # (2026-08-07: build() had no way to supply a verdict, so every external-op arm died at
    # construction with the gate's ValueError).
    rules_verdict_path = os.path.join(base_dir, "rules_verdict.json")

    raw_tr = pd.read_csv(os.path.join(base_dir, "data", "train.csv"), parse_dates=["date"])
    raw_te = pd.read_csv(os.path.join(base_dir, "data", "test.csv"), parse_dates=["date"])
    aug_tr, aug_te, plan = apply_operators(
        raw_tr, raw_te, ideas, target_col="num_sold",
        ledger_path=os.path.join(vdir, "injection_ledger.json"),
        rules_verdict=rules_verdict_path, expected_competition=comp)

    # THE ARM MUST BE WHAT ITS NAME SAYS. If the operator that defines this arm kind was
    # requested but not realized (a precondition refused it, a gate stopped it), refuse to
    # emit the arm rather than shipping a baseline clone under the arm's name -- nothing
    # downstream re-checks, and the search would race "ratio vs join" where one lane is
    # secretly the baseline.
    defining = _defining_op_for(arm)
    if defining is None and len(ideas) == 1:
        defining = ideas[0].get("operator")
    if defining and any(i.get("operator") == defining for i in ideas):
        realized_ops = {r.get("operator") for r in plan.get("realized", [])}
        if defining not in realized_ops:
            reasons = [u.get("reason", "")[:120] for u in plan.get("unrealized", [])
                       if u.get("operator") == defining]
            raise RuntimeError(
                f"arm {arm!r} is defined by operator {defining!r}, which was requested but "
                f"NOT realized ({reasons or 'no reason recorded'}). Refusing to emit a "
                f"baseline clone under this arm's name.")
    new_cols = list(plan["added_columns"])
    tt = plan.get("target_transform")
    if tt:
        new_cols.append(tt["covariate_col"])
    if plan.get("weight_column"):
        new_cols.append(plan["weight_column"])

    # augmented processed CSVs: baseline columns first (byte-identical), then the new ones
    for split, aug in (("train", aug_tr), ("test", aug_te)):
        proc = pd.read_csv(os.path.join(base_dir, "data", f"{split}_processed.csv"))
        add = aug[[id_col] + new_cols]
        out = proc.merge(add, on=id_col, how="left")
        assert len(out) == len(proc), "join changed row count"
        assert list(out.columns[:len(proc.columns)]) == list(proc.columns), "baseline columns moved"
        assert out[new_cols].notna().all().all(), f"{split}: new columns contain NaN"
        out.to_csv(os.path.join(vdir, f"{split}_processed.csv"), index=False)
    for name in ("train.csv", "test.csv", "sample_submission.csv"):
        src_p, dst_p = os.path.join(base_dir, "data", name), os.path.join(vdir, name)
        if os.path.exists(src_p) and not os.path.exists(dst_p):
            os.symlink(src_p, dst_p)

    # generate the arm's evaluator from the baseline evaluator
    src = open(os.path.join(_HERE, f"{base_mod}.py")).read()
    src, n1 = re.subn(r'"competitions", "' + re.escape(comp) + r'"',
                      f'"competitions", "{vcomp}"', src)
    src, n2 = re.subn(r'CACHE_DIR = os\.path\.join\(_HERE, "[^"]+"\)',
                      f'CACHE_DIR = os.path.join(_HERE, "cache_{arm}_{_slug(comp)}")', src)
    feat_cols = [c for c in new_cols
                 if c != plan.get("weight_column")
                 and not (tt and c == tt["covariate_col"])]
    if feat_cols:
        lst = ", ".join(f'"{c}"' for c in feat_cols)
        src, n3 = re.subn(r'(FEATURE_COLS = \[[^\]]+\])',
                          r"\1" + f"\nFEATURE_COLS = FEATURE_COLS + [{lst}]  # v5 operator columns",
                          src)
        assert n3 == 1, "FEATURE_COLS patch failed"
    assert n1 >= 1 and n2 == 1, (n1, n2)
    if tt:
        src = _patch_target_transform(src, tt["covariate_col"], tt["kind"])
    if plan.get("weight_column"):
        src = _patch_sample_weight(src, plan["weight_column"])
    ev_path = os.path.join(_HERE, f"eval_{arm}_{_slug(comp)}.py")
    open(ev_path, "w").write(src)

    manifest = {"comp": comp, "arm": arm, "workspace": vdir, "evaluator": ev_path,
                "feature_columns_added": feat_cols, "target_transform": tt,
                "ledger": plan}
    json.dump(manifest, open(os.path.join(vdir, "arm_manifest.json"), "w"), indent=2)
    print(json.dumps({k: manifest[k] for k in
                      ("arm", "evaluator", "feature_columns_added", "target_transform")}, indent=2))
    print(f"   ledger: {plan['realized_count']} realized / {plan['proposed']} proposed, "
          f"{len(plan['unrealized'])} unrealized")
    return manifest


if __name__ == "__main__":
    comp, arm, ideas_path = sys.argv[1], sys.argv[2], sys.argv[3]
    build(comp, arm, json.load(open(ideas_path)))
