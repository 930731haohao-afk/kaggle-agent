"""Modeling — playground-series-s3e5 (wine quality, QWK metric).

Approach (justified in scripts/eda.py step 7):
  - Regression head (LightGBM / XGBoost / CatBoost), NOT multiclass classification.
    QWK penalizes distance^2 between predicted/true ordinal class; a regressor's
    continuous output naturally respects class ordering, and pools signal across
    the abundant mid-range classes (5/6) to also place the rare extremes (3/8).
  - 5-fold StratifiedKFold on the raw `quality` label (imbalanced ordinal target).
  - OOF weight-search blend across the 3 models, scored via a properly
    OPTIMIZED ROUNDER (Nelder-Mead cutpoints tuned directly on OOF QWK) rather
    than naive round-to-nearest-integer.
  - Logs two experiments for comparison: naive-round blend vs optimized-rounder
    blend, to make the value of threshold optimization explicit (self-improvement
    strategy: Verifiable Rewards).

Run: uv run python3 competitions/playground-series-s3e5/scripts/train.py
"""
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
SUB_DIR = os.path.join(ROOT, "submissions")
TARGET, IDC = "quality", "Id"
SEED, N_SPLITS = 42, 5


class OptimizedRounder:
    """Tune per-class cutpoints on a continuous prediction vector to maximize QWK.
    Standard trick for ordinal-regression + QWK (Nelder-Mead on cutpoint vector).
    """

    def __init__(self, low: int, high: int):
        self.low, self.high = low, high
        self.coef_ = np.arange(low + 0.5, high, 1.0)  # initial guess: midpoints

    def _to_classes(self, x, coef):
        coef = np.sort(coef)
        return np.clip(np.digitize(x, coef) + self.low, self.low, self.high)

    def _loss(self, coef, x, y):
        return -cohen_kappa_score(y, self._to_classes(x, coef), weights="quadratic")

    def fit(self, x, y):
        res = minimize(self._loss, self.coef_, args=(x, y), method="Nelder-Mead",
                        options={"xatol": 1e-4, "fatol": 1e-4, "maxiter": 2000})
        self.coef_ = np.sort(res.x)
        return self

    def predict(self, x):
        return self._to_classes(x, self.coef_)


def qwk(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def main():
    train = pd.read_csv(os.path.join(DATA, "train_processed.csv"))
    test = pd.read_csv(os.path.join(DATA, "test_processed.csv"))
    feat_cols = [c for c in train.columns if c not in (IDC, TARGET)]
    X = train[feat_cols].to_numpy(np.float32)
    y = train[TARGET].to_numpy(int)
    Xtest = test[feat_cols].to_numpy(np.float32)
    low, high = int(y.min()), int(y.max())
    print(f"train={X.shape}  test={Xtest.shape}  target range=[{low},{high}]  features={len(feat_cols)}")

    import lightgbm as lgb
    import xgboost as xgb
    from catboost import CatBoostRegressor

    def make_models():
        return {
            "LGB": lgb.LGBMRegressor(objective="regression", n_estimators=1500, learning_rate=0.03,
                                      num_leaves=31, max_depth=6, subsample=0.8, subsample_freq=1,
                                      colsample_bytree=0.7, reg_lambda=1.0, min_child_samples=15,
                                      random_state=SEED, n_jobs=-1, verbose=-1),
            "XGB": xgb.XGBRegressor(objective="reg:squarederror", n_estimators=1500, learning_rate=0.03,
                                     max_depth=5, subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0,
                                     min_child_weight=5, random_state=SEED, n_jobs=-1, tree_method="hist"),
            "CAT": CatBoostRegressor(loss_function="RMSE", iterations=1500, learning_rate=0.03,
                                      depth=6, l2_leaf_reg=3.0, random_seed=SEED, verbose=False),
        }

    NAMES = ["LGB", "XGB", "CAT"]
    oof = {n: np.zeros(len(y)) for n in NAMES}
    pred = {n: np.zeros(len(Xtest)) for n in NAMES}
    folds = list(StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED).split(X, y))

    t0 = time.time()
    for f, (tr, va) in enumerate(folds):
        for n, m in make_models().items():
            m.fit(X[tr], y[tr])
            oof[n][va] = m.predict(X[va])
            pred[n] += m.predict(Xtest) / len(folds)
    train_time = time.time() - t0
    print(f"5-fold x 3 models trained in {train_time:.1f}s")

    # per-model QWK (naive round, for a quick per-model comparison)
    per_model_naive = {n: qwk(y, np.clip(np.round(oof[n]), low, high).astype(int)) for n in NAMES}
    print("Per-model QWK (naive round): " + "  ".join(f"{n}={per_model_naive[n]:.5f}" for n in NAMES))

    # ---- Blend weight search: score each candidate weight via the OPTIMIZED ROUNDER ----
    oofs = np.stack([oof[n] for n in NAMES], axis=1)
    preds_stack = np.stack([pred[n] for n in NAMES], axis=1)
    best_w, best_qwk, best_coef = None, -1e18, None
    t1 = time.time()
    for w0 in np.arange(0, 1.001, 0.1):
        for w1 in np.arange(0, 1.001 - w0, 0.1):
            w2 = 1 - w0 - w1
            if w2 < -1e-9:
                continue
            w = np.array([w0, w1, w2])
            blend_oof = oofs @ w
            rounder = OptimizedRounder(low, high).fit(blend_oof, y)
            s = qwk(y, rounder.predict(blend_oof))
            if s > best_qwk:
                best_qwk, best_w, best_coef = s, w, rounder.coef_.copy()
    search_time = time.time() - t1
    print(f"Blend weight search done in {search_time:.1f}s")
    print(f"BEST blend weights {dict(zip(NAMES, np.round(best_w, 2)))}"
          f" -> optimized-rounder OOF QWK = {best_qwk:.5f}")
    print(f"Optimized cutpoints: {np.round(best_coef, 3)}")

    # naive-round QWK for the SAME best blend weights (comparison experiment)
    best_blend_oof = oofs @ best_w
    naive_qwk = qwk(y, np.clip(np.round(best_blend_oof), low, high).astype(int))
    print(f"Same blend, naive round -> OOF QWK = {naive_qwk:.5f}  "
          f"(optimized rounder gain: {best_qwk - naive_qwk:+.5f})")

    # ---- Log experiment: naive-round blend (comparison / "before" experiment) ----
    import importlib.util
    _spec = importlib.util.spec_from_file_location(
        "experiment_log", os.path.join(ROOT, "..", "..", ".claude/skills/kaggle-agent/assets/utils/experiment_log.py"))
    experiment_log = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(experiment_log)

    experiment_log.log_experiment_v2(
        ROOT,
        model="LGB+XGB+CAT regression blend (naive round)",
        metric="quadratic_weighted_kappa",
        direction="maximize",
        score=round(float(naive_qwk), 5),
        cv=dict(scheme=f"{N_SPLITS}fold", n_splits=N_SPLITS, seed=SEED, strategy="StratifiedKFold on quality"),
        base_models=[dict(name=n, score=round(float(per_model_naive[n]), 5)) for n in NAMES],
        ensemble=dict(weights=dict(zip(NAMES, [round(float(x), 2) for x in best_w])),
                      score=round(float(naive_qwk), 5)),
        postprocess=["round_to_nearest_int", f"clip[{low},{high}]"],
        features=list(feat_cols),
        notes="Regression blend, same weights as experiment 3, but rounded naively (baseline "
              "for measuring the optimized-rounder gain). Not submitted.",
    )

    # ---- Final model: same blend + optimized rounder ----
    final_oof_class = OptimizedRounder(low, high).fit(best_blend_oof, y).predict(best_blend_oof)
    final_qwk = qwk(y, final_oof_class)
    assert abs(final_qwk - best_qwk) < 1e-9

    exp_id = experiment_log.log_experiment_v2(
        ROOT,
        model="LGB+XGB+CAT regression blend (optimized-rounder)",
        metric="quadratic_weighted_kappa",
        direction="maximize",
        score=round(float(best_qwk), 5),
        cv=dict(scheme=f"{N_SPLITS}fold", n_splits=N_SPLITS, seed=SEED, strategy="StratifiedKFold on quality"),
        base_models=[dict(name=n, score=round(float(per_model_naive[n]), 5)) for n in NAMES],
        ensemble=dict(weights=dict(zip(NAMES, [round(float(x), 2) for x in best_w])),
                      score=round(float(best_qwk), 5)),
        postprocess=[f"OptimizedRounder(cutpoints={[round(float(c), 3) for c in best_coef]})",
                     f"clip[{low},{high}]"],
        features=list(feat_cols),
        notes=f"Final model for submission. Optimized-rounder gain over naive round on the same "
              f"blend: {best_qwk - naive_qwk:+.5f}. Baseline (generic pipeline, experiment #1): 0.47871.",
    )
    print(f"Logged experiment_id={exp_id} (final, for submission)")

    # ---- Retrain each model on FULL training data for the submission ----
    print("Retraining on full data for submission...")
    t2 = time.time()
    full_pred = {}
    for n, m in make_models().items():
        m.fit(X, y)
        full_pred[n] = m.predict(Xtest)
    full_time = time.time() - t2
    print(f"Full retrain done in {full_time:.1f}s")

    final_blend_test = np.stack([full_pred[n] for n in NAMES], axis=1) @ best_w
    rounder_final = OptimizedRounder(low, high).fit(best_blend_oof, y)  # thresholds fixed from OOF
    final_test_class = rounder_final.predict(final_blend_test)

    ss = pd.read_csv(os.path.join(DATA, "sample_submission.csv"))
    sub = pd.DataFrame({ss.columns[0]: test[IDC].astype(int), ss.columns[1]: final_test_class.astype(int)})
    assert sub.shape == ss.shape, f"shape mismatch {sub.shape} vs {ss.shape}"
    assert sub.isnull().sum().sum() == 0
    assert (sub[ss.columns[0]] == ss[ss.columns[0]]).all(), "ID mismatch/order"
    assert sub[ss.columns[1]].between(low, high).all()

    os.makedirs(SUB_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(SUB_DIR, f"sub_blend_optround_{best_qwk:.5f}_{stamp}.csv")
    sub.to_csv(path, index=False)
    print(f"Wrote submission: {os.path.basename(path)}  ({len(sub)} rows)")
    print(f"Test prediction distribution:\n{sub[ss.columns[1]].value_counts().sort_index()}")
    print(f"Train target distribution:\n{pd.Series(y).value_counts().sort_index()}")

    print(f"\nTotal script time: {time.time() - t0:.1f}s "
          f"(train {train_time:.1f}s + search {search_time:.1f}s + full-retrain {full_time:.1f}s)")
    print(f"\nFINAL: OOF QWK = {best_qwk:.5f}  vs baseline 0.47871  "
          f"(delta = {best_qwk - 0.47871:+.5f})")


if __name__ == "__main__":
    main()
