"""Round 1: Optuna fold-proxy tuned LGB ADDED to pool (LGB, XGB, CAT, LGB_tuned)."""
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
print("Tuned params (fold0 proxy):", tuned_params, "fold0_mae=", best["best_fold0_mae"])

oof_lgb, pred_lgb, m_lgb, t_lgb = P.get_or_train("LGB", P.train_lgb)
oof_xgb, pred_xgb, m_xgb, t_xgb = P.get_or_train("XGB", P.train_xgb)
oof_cat, pred_cat, m_cat, t_cat = P.get_or_train("CAT", P.train_cat)
oof_lgbT, pred_lgbT, m_lgbT, t_lgbT = P.get_or_train(
    "LGB_tuned", lambda: P.train_lgb(params_override=tuned_params))

names = ["LGB", "XGB", "CAT", "LGB_tuned"]
oofs = {n: o for n, o in zip(names, [oof_lgb, oof_xgb, oof_cat, oof_lgbT])}
preds = {n: p for n, p in zip(names, [pred_lgb, pred_xgb, pred_cat, pred_lgbT])}
maes = {n: m for n, m in zip(names, [m_lgb, m_xgb, m_cat, m_lgbT])}

oof_stack = np.stack([oofs[n] for n in names], axis=1)
pred_stack = np.stack([preds[n] for n in names], axis=1)
w, s = P.weight_search(oof_stack)
final_oof = oof_stack @ w
final_pred = pred_stack @ w
round_mae = P.mae(np.round(final_oof))
print(f"\nBEST weights {dict(zip(names, np.round(w,2)))}")
print(f"Blend raw OOF MAE = {s:.5f}  rounded OOF MAE = {round_mae:.5f}")

use_round = round_mae < s
BEST_ROUNDED = 1.33812
improved = round_mae < BEST_ROUNDED
print(f"use_round={use_round}  improved_over_{BEST_ROUNDED}={improved}")

result = dict(names=names, maes=maes, weights=dict(zip(names, [round(float(x), 3) for x in w])),
              blend_raw_mae=round(float(s), 5), blend_round_mae=round(float(round_mae), 5),
              use_round=bool(use_round), improved=bool(improved))
with open(f"{P.COMP}/scripts/round1_result.json", "w") as f:
    json.dump(result, f, indent=2)

if improved:
    path = P.write_submission(final_pred, use_round, stamp_prefix="sub_round1")
    print(f"Wrote new best submission: {path}")
    result["submission"] = os.path.basename(path)
    with open(f"{P.COMP}/scripts/round1_result.json", "w") as f:
        json.dump(result, f, indent=2)
