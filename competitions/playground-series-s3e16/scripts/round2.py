"""Round 2: seed bagging — add LGB_tuned with seed=2024 (same hyperparams) as 5th member.

Recipe (knowledge/experience.md): seed bagging after Optuna tuning is the
cheapest residual gain — a single extra 5-fold run, same hyperparams,
different random_state.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import pool_lib as P  # noqa: E402

BEST_JSON = f"{P.COMP}/scripts/lgb_optuna_best.json"
best = json.load(open(BEST_JSON))
tuned_params = dict(best["best_params"])
tuned_params["n_estimators"] = 3000

oof_lgb, pred_lgb, m_lgb, t_lgb = P.get_or_train("LGB", P.train_lgb)
oof_xgb, pred_xgb, m_xgb, t_xgb = P.get_or_train("XGB", P.train_xgb)
oof_cat, pred_cat, m_cat, t_cat = P.get_or_train("CAT", P.train_cat)
oof_lgbT, pred_lgbT, m_lgbT, t_lgbT = P.get_or_train(
    "LGB_tuned", lambda: P.train_lgb(params_override=tuned_params))
oof_lgbS, pred_lgbS, m_lgbS, t_lgbS = P.get_or_train(
    "LGB_tuned_seed2024", lambda: P.train_lgb(seed=2024, params_override=tuned_params))

names = ["LGB", "XGB", "CAT", "LGB_tuned", "LGB_tuned_seed2024"]
oofs = [oof_lgb, oof_xgb, oof_cat, oof_lgbT, oof_lgbS]
preds = [pred_lgb, pred_xgb, pred_cat, pred_lgbT, pred_lgbS]
maes = dict(zip(names, [m_lgb, m_xgb, m_cat, m_lgbT, m_lgbS]))

oof_stack = np.stack(oofs, axis=1)
pred_stack = np.stack(preds, axis=1)
w, s = P.weight_search(oof_stack)
final_oof = oof_stack @ w
final_pred = pred_stack @ w
round_mae = P.mae(np.round(final_oof))
print(f"\nBEST weights {dict(zip(names, np.round(w,2)))}")
print(f"Blend raw OOF MAE = {s:.5f}  rounded OOF MAE = {round_mae:.5f}")

use_round = round_mae < s
BEST_ROUNDED = 1.33812
prev = json.load(open(f"{P.COMP}/scripts/round1_result.json")) if os.path.exists(f"{P.COMP}/scripts/round1_result.json") else None
prev_round_mae = prev["blend_round_mae"] if prev else BEST_ROUNDED
improved_vs_orig = round_mae < BEST_ROUNDED
improved_vs_prev = round_mae < prev_round_mae
print(f"use_round={use_round}  improved_vs_orig_best({BEST_ROUNDED})={improved_vs_orig}  improved_vs_round1({prev_round_mae})={improved_vs_prev}")

result = dict(names=names, maes=maes, weights=dict(zip(names, [round(float(x), 3) for x in w])),
              blend_raw_mae=round(float(s), 5), blend_round_mae=round(float(round_mae), 5),
              use_round=bool(use_round), improved_vs_orig=bool(improved_vs_orig),
              improved_vs_round1=bool(improved_vs_prev))
with open(f"{P.COMP}/scripts/round2_result.json", "w") as f:
    json.dump(result, f, indent=2)

if improved_vs_orig and improved_vs_prev:
    path = P.write_submission(final_pred, use_round, stamp_prefix="sub_round2")
    print(f"Wrote new best submission: {path}")
    result["submission"] = os.path.basename(path)
    with open(f"{P.COMP}/scripts/round2_result.json", "w") as f:
        json.dump(result, f, indent=2)
