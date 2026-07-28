"""Baselines: structural Ridge (several variants) + GDP-LGB + blend."""
import sys, os, json, time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import common

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "experiment_log",
    "/home/tjyen/ai_agents/kaggle/.claude/skills/kaggle-agent/assets/utils/experiment_log.py")
experiment_log = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(experiment_log)

COMP = "/home/tjyen/ai_agents/kaggle/competitions/tabular-playground-series-jan-2022"

CONFIGS = {
    "ridge_base": {"model": "ridge", "alpha": 0.1, "fourier_k": 4, "fourier_product": False,
                   "holidays": False},
    "ridge_fp": {"model": "ridge", "alpha": 0.1, "fourier_k": 8, "fourier_product": True,
                 "holidays": False},
    "ridge_hol": {"model": "ridge", "alpha": 0.1, "fourier_k": 8, "fourier_product": True,
                  "holidays": True, "hol_lo": -5, "hol_hi": 10},
    "lgb_gdp": {"model": "lgb", "num_leaves": 31, "lr": 0.05, "n_estimators": 800},
}

results = {}
for name, cfg in CONFIGS.items():
    t0 = time.time()
    r = common.evaluate(cfg, return_preds=True)
    dt = time.time() - t0
    results[name] = r
    print(f"{name:12s} fold2017 {r['fold_2017']:.4f}  fold2018 {r['fold_2018']:.4f}  "
          f"mean {r['mean']:.4f}  ({dt:.1f}s)", flush=True)
    experiment_log.log_experiment_v2(
        COMP, model=name, metric="SMAPE", direction="minimize", score=r["mean"],
        features=json.dumps({k: v for k, v in cfg.items()}),
        notes=f"fold2017={r['fold_2017']:.4f} fold2018={r['fold_2018']:.4f} expanding-year folds")

# blend best ridge + lgb on val preds (grid weight, per-fold honest view)
import pandas as pd
tr, _ = common.load_data()
tr["year"] = tr["date"].dt.year
best_w, best_s = None, 1e9
for w in np.arange(0, 1.01, 0.05):
    ss = []
    for vy in [2017, 2018]:
        a = tr.loc[tr["year"] == vy, "num_sold"].values
        p = w * results["ridge_hol"]["preds"][vy].values + (1 - w) * results["lgb_gdp"]["preds"][vy].values
        ss.append(common.smape(a, np.round(p)))
    m = np.mean(ss)
    if m < best_s:
        best_s, best_w, best_folds = m, w, ss
print(f"blend ridge_hol*{best_w:.2f}+lgb*{1-best_w:.2f}: mean {best_s:.4f} "
      f"(2017 {best_folds[0]:.4f}, 2018 {best_folds[1]:.4f})")
experiment_log.log_experiment_v2(
    COMP, model="blend_ridge_lgb", metric="SMAPE", direction="minimize", score=best_s,
    features=f"w_ridge={best_w:.2f}",
    notes=f"grid blend ridge_hol+lgb_gdp, fold2017={best_folds[0]:.4f} fold2018={best_folds[1]:.4f}")
