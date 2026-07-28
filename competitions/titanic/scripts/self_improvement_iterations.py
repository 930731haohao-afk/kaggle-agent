"""
Self-Improvement Iterations for Titanic Competition
=====================================================
Applies: Best-of-N, Verifiable Rewards, Reflexion
Focus: Reduce CV-LB gap (overfitting) rather than maximizing CV.
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import lightgbm as lgb
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

COMPETITION_DIR = "competitions/titanic"
TRAIN_FILE = "data/train_processed.csv"
TARGET_COL = "Survived"
ID_COL = "PassengerId"
EVAL_METRIC = "accuracy"
N_FOLDS = 5
RANDOM_SEED = 42

train = pd.read_csv(os.path.join(COMPETITION_DIR, TRAIN_FILE))
y = train[TARGET_COL].astype(int)
X = train.drop(columns=[TARGET_COL, ID_COL])
feature_names = X.columns.tolist()

print(f"Training data: {X.shape[0]} rows, {X.shape[1]} features")
print(f"Features: {feature_names}")
print(f"Target distribution: {dict(y.value_counts())}\n")


def log_experiment(model_name, params, scores, notes=""):
    """Log experiment to experiments.json."""
    exp_file = os.path.join(COMPETITION_DIR, "experiments.json")
    with open(exp_file, "r") as f:
        experiments = json.load(f)

    exp_id = len(experiments) + 1
    experiment = {
        "experiment_id": exp_id,
        "timestamp": datetime.now().isoformat(),
        "model": model_name,
        "features": TRAIN_FILE,
        "n_features": len(feature_names),
        "params": {k: str(v) for k, v in params.items()},
        "cv_strategy": f"{N_FOLDS}-fold-stratified",
        "cv_scores": [round(s, 6) for s in scores],
        "cv_mean": round(float(np.mean(scores)), 6),
        "cv_std": round(float(np.std(scores)), 6),
        "eval_metric": EVAL_METRIC,
        "notes": notes
    }
    experiments.append(experiment)

    with open(exp_file, "w") as f:
        json.dump(experiments, f, indent=2)

    return exp_id, float(np.mean(scores)), float(np.std(scores))


def cv_evaluate(model, model_name, params, notes="", use_repeated=False):
    """Run CV and log results. Returns (mean, std, scores)."""
    if use_repeated:
        kf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=RANDOM_SEED)
        cv_label = "5x3-repeated-stratified"
    else:
        kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
        cv_label = f"{N_FOLDS}-fold-stratified"

    scores = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model_clone = clone_model(model, params)
        model_clone.fit(X_tr, y_tr)
        y_pred = model_clone.predict(X_val)
        score = accuracy_score(y_val, y_pred)
        scores.append(score)

    mean_score = np.mean(scores)
    std_score = np.std(scores)

    # Log only the first N_FOLDS scores for consistency
    log_scores = scores[:N_FOLDS] if use_repeated else scores
    exp_id, _, _ = log_experiment(model_name, params, log_scores, notes)

    print(f"  {model_name}: CV={mean_score:.6f} (+/- {std_score:.6f}) [#{exp_id}]")
    return mean_score, std_score, scores


def clone_model(model, params):
    """Create a fresh model instance."""
    return model.__class__(**model.get_params())


# ============================================================
# VERIFIABLE REWARDS: Track baseline
# ============================================================
BASELINE_SCORE = 0.616162  # Majority class
PREVIOUS_BEST = 0.843990   # XGBoost
PREVIOUS_LB = 0.75358

print("=" * 60)
print("VERIFIABLE REWARDS — Tracking")
print("=" * 60)
print(f"  Baseline (majority class): {BASELINE_SCORE:.6f}")
print(f"  Previous best CV:          {PREVIOUS_BEST:.6f}")
print(f"  Previous best LB:          {PREVIOUS_LB:.5f}")
print(f"  CV-LB gap:                 {PREVIOUS_BEST - PREVIOUS_LB:.4f} (overfitting signal)")
print()


# ============================================================
# ITERATION 1: CatBoost (new model family)
# ============================================================
print("=" * 60)
print("ITERATION 1: CatBoost (Best-of-N — new model family)")
print("=" * 60)
print("Hypothesis: CatBoost has strong built-in regularization")
print("and handles categoricals natively, which may generalize better.")
print()

try:
    from catboost import CatBoostClassifier

    cat_params = {
        "iterations": 300,
        "learning_rate": 0.05,
        "depth": 5,
        "l2_leaf_reg": 5.0,
        "random_seed": RANDOM_SEED,
        "verbose": 0
    }
    cat_model = CatBoostClassifier(**cat_params)

    kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    cat_scores = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = CatBoostClassifier(**cat_params)
        model.fit(X_tr, y_tr, verbose=0)
        y_pred = model.predict(X_val)
        score = accuracy_score(y_val, y_pred)
        cat_scores.append(score)
        print(f"  Fold {fold+1}: {score:.6f}")

    cat_mean = np.mean(cat_scores)
    cat_std = np.std(cat_scores)
    exp_id, _, _ = log_experiment("CatBoost", cat_params, cat_scores,
                                   "CatBoost with regularization (l2=5.0)")
    delta = cat_mean - PREVIOUS_BEST
    print(f"\n  CatBoost: CV={cat_mean:.6f} (+/- {cat_std:.6f}) [#{exp_id}]")
    print(f"  Reward: {delta:+.6f} vs previous best")
    print(f"  Signal: {'IMPROVEMENT' if delta > 0 else 'FLAT' if abs(delta) < 0.001 else 'DEGRADATION'}")
except ImportError:
    print("  CatBoost not installed. Skipping.")
    cat_mean = 0
    cat_scores = []


# ============================================================
# ITERATION 2: Feature Selection + XGBoost
# ============================================================
print()
print("=" * 60)
print("ITERATION 2: Feature-Selected XGBoost (reduce overfitting)")
print("=" * 60)
print("Hypothesis: Fewer features on small data = better generalization.")
print("Selecting top features by mutual information with target.")
print()

# Find top features using mutual information
selector = SelectKBest(mutual_info_classif, k=10)
selector.fit(X, y)
mi_scores = pd.Series(selector.scores_, index=X.columns).sort_values(ascending=False)
print("Feature mutual information with target:")
for feat, score in mi_scores.items():
    marker = " <<<" if score >= mi_scores.iloc[9] else ""
    print(f"  {feat:20s} {score:.4f}{marker}")

top_features = mi_scores.head(10).index.tolist()
print(f"\nSelected {len(top_features)} features: {top_features}")

X_selected = X[top_features]

xgb_params = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "max_depth": 4,  # Shallower than before (was 5)
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.5,  # More regularization
    "reg_lambda": 1.0,
}

kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
xgb_sel_scores = []
for fold, (train_idx, val_idx) in enumerate(kf.split(X_selected, y)):
    X_tr = X_selected.iloc[train_idx]
    X_val = X_selected.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = xgb.XGBClassifier(**xgb_params, random_state=RANDOM_SEED,
                               use_label_encoder=False, verbosity=0, eval_metric="logloss")
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_val)
    score = accuracy_score(y_val, y_pred)
    xgb_sel_scores.append(score)
    print(f"  Fold {fold+1}: {score:.6f}")

xgb_sel_mean = np.mean(xgb_sel_scores)
xgb_sel_std = np.std(xgb_sel_scores)
log_params = {**xgb_params, "n_features_selected": 10, "features": str(top_features)}
exp_id, _, _ = log_experiment("XGBoost-FeatureSelected", log_params, xgb_sel_scores,
                               f"XGBoost with top 10 MI features, shallower trees, more regularization")
delta = xgb_sel_mean - PREVIOUS_BEST
print(f"\n  XGBoost-FeatureSelected: CV={xgb_sel_mean:.6f} (+/- {xgb_sel_std:.6f}) [#{exp_id}]")
print(f"  Reward: {delta:+.6f} vs previous best")
print(f"  Signal: {'IMPROVEMENT' if delta > 0 else 'FLAT' if abs(delta) < 0.001 else 'DEGRADATION'}")


# ============================================================
# ITERATION 3: Heavily Regularized GBM (prevent overfitting)
# ============================================================
print()
print("=" * 60)
print("ITERATION 3: Heavily Regularized LightGBM")
print("=" * 60)
print("Hypothesis: Strong regularization + early stopping reduces gap.")
print()

lgb_reg_params = {
    "n_estimators": 500,
    "learning_rate": 0.01,  # Much slower learning
    "num_leaves": 15,       # Fewer leaves = less complex
    "max_depth": 4,         # Shallow
    "min_child_samples": 20,  # Higher = more regularization
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "reg_alpha": 1.0,       # Strong L1
    "reg_lambda": 2.0,      # Strong L2
}

kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
lgb_reg_scores = []
for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = lgb.LGBMClassifier(verbosity=-1, random_state=RANDOM_SEED, **lgb_reg_params)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(50, verbose=False)])
    y_pred = model.predict(X_val)
    score = accuracy_score(y_val, y_pred)
    lgb_reg_scores.append(score)
    print(f"  Fold {fold+1}: {score:.6f}")

lgb_reg_mean = np.mean(lgb_reg_scores)
lgb_reg_std = np.std(lgb_reg_scores)
exp_id, _, _ = log_experiment("LightGBM-HeavyReg", lgb_reg_params, lgb_reg_scores,
                               "LightGBM with heavy regularization + early stopping")
delta = lgb_reg_mean - PREVIOUS_BEST
print(f"\n  LightGBM-HeavyReg: CV={lgb_reg_mean:.6f} (+/- {lgb_reg_std:.6f}) [#{exp_id}]")
print(f"  Reward: {delta:+.6f} vs previous best")
print(f"  Signal: {'IMPROVEMENT' if delta > 0 else 'FLAT' if abs(delta) < 0.001 else 'DEGRADATION'}")


# ============================================================
# ITERATION 4: Soft Voting Ensemble (diversity for generalization)
# ============================================================
print()
print("=" * 60)
print("ITERATION 4: Soft Voting Ensemble (top diverse models)")
print("=" * 60)
print("Hypothesis: Ensemble of diverse models reduces variance")
print("and should generalize better than any single model.")
print()

kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
ensemble_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # Diverse ensemble members
    lr = LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_SEED)
    rf = RandomForestClassifier(n_estimators=200, max_depth=8,
                                 min_samples_split=5, random_state=RANDOM_SEED, n_jobs=-1)
    xgb_m = xgb.XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=5,
                                subsample=0.8, colsample_bytree=0.8,
                                random_state=RANDOM_SEED, use_label_encoder=False,
                                verbosity=0, eval_metric="logloss")
    lgb_m = lgb.LGBMClassifier(verbosity=-1, random_state=RANDOM_SEED,
                                 n_estimators=300, learning_rate=0.05,
                                 num_leaves=31, max_depth=6)

    ensemble = VotingClassifier(
        estimators=[("lr", lr), ("rf", rf), ("xgb", xgb_m), ("lgb", lgb_m)],
        voting="soft"
    )
    ensemble.fit(X_tr, y_tr)
    y_pred = ensemble.predict(X_val)
    score = accuracy_score(y_val, y_pred)
    ensemble_scores.append(score)
    print(f"  Fold {fold+1}: {score:.6f}")

ens_mean = np.mean(ensemble_scores)
ens_std = np.std(ensemble_scores)
ens_params = {"models": "LR+RF+XGB+LGB", "voting": "soft"}
exp_id, _, _ = log_experiment("SoftVotingEnsemble", ens_params, ensemble_scores,
                               "4-model soft voting: LR + RF + XGBoost + LightGBM")
delta = ens_mean - PREVIOUS_BEST
print(f"\n  SoftVotingEnsemble: CV={ens_mean:.6f} (+/- {ens_std:.6f}) [#{exp_id}]")
print(f"  Reward: {delta:+.6f} vs previous best")
print(f"  Signal: {'IMPROVEMENT' if delta > 0 else 'FLAT' if abs(delta) < 0.001 else 'DEGRADATION'}")


# ============================================================
# ITERATION 5: GradientBoosting (sklearn) with conservative params
# ============================================================
print()
print("=" * 60)
print("ITERATION 5: sklearn GradientBoosting (conservative)")
print("=" * 60)
print("Hypothesis: sklearn's GBM is known to generalize well on small")
print("datasets with careful regularization (no GPU shortcuts).")
print()

gb_params = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "max_depth": 3,           # Very shallow
    "min_samples_split": 10,
    "min_samples_leaf": 5,
    "subsample": 0.8,
    "max_features": "sqrt",
}

kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
gb_scores = []
for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = GradientBoostingClassifier(**gb_params, random_state=RANDOM_SEED)
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_val)
    score = accuracy_score(y_val, y_pred)
    gb_scores.append(score)
    print(f"  Fold {fold+1}: {score:.6f}")

gb_mean = np.mean(gb_scores)
gb_std = np.std(gb_scores)
exp_id, _, _ = log_experiment("GradientBoosting-conservative", gb_params, gb_scores,
                               "sklearn GBM with shallow trees + strong regularization")
delta = gb_mean - PREVIOUS_BEST
print(f"\n  GradientBoosting: CV={gb_mean:.6f} (+/- {gb_std:.6f}) [#{exp_id}]")
print(f"  Reward: {delta:+.6f} vs previous best")
print(f"  Signal: {'IMPROVEMENT' if delta > 0 else 'FLAT' if abs(delta) < 0.001 else 'DEGRADATION'}")


# ============================================================
# SUMMARY — Autonomous Iteration Report
# ============================================================
print()
print("=" * 60)
print("AUTONOMOUS ITERATION SUMMARY")
print("=" * 60)

results = [
    ("Previous best: XGBoost", PREVIOUS_BEST, 0.006675, "—", "Baseline"),
]

# Collect all iteration results
iter_results = []
if cat_scores:
    iter_results.append(("CatBoost", cat_mean, cat_std, cat_scores))
iter_results.append(("XGBoost-FeatureSelected", xgb_sel_mean, xgb_sel_std, xgb_sel_scores))
iter_results.append(("LightGBM-HeavyReg", lgb_reg_mean, lgb_reg_std, lgb_reg_scores))
iter_results.append(("SoftVotingEnsemble", ens_mean, ens_std, ensemble_scores))
iter_results.append(("GradientBoosting", gb_mean, gb_std, gb_scores))

print(f"\n{'#':<4} {'Model':<30} {'CV Mean':<10} {'CV Std':<10} {'Delta':<10} {'Signal':<12}")
print("-" * 76)
print(f"{'0':<4} {'XGBoost (prev best)':<30} {PREVIOUS_BEST:<10.6f} {0.006675:<10.6f} {'—':<10} {'Baseline':<12}")

best_new = PREVIOUS_BEST
best_name = "XGBoost (prev best)"
for i, (name, mean, std, scores) in enumerate(iter_results, 1):
    delta = mean - PREVIOUS_BEST
    if abs(delta) < 0.001:
        signal = "FLAT"
    elif delta > 0:
        signal = "IMPROVEMENT"
    else:
        signal = "DEGRADATION"

    print(f"{i:<4} {name:<30} {mean:<10.6f} {std:<10.6f} {delta:<+10.6f} {signal:<12}")

    if mean > best_new:
        best_new = mean
        best_name = name

print(f"\nBest: {best_name} (CV {best_new:.6f})")
print(f"Overall improvement: {best_new - PREVIOUS_BEST:+.6f} vs previous best")
print(f"\nNote: With 891 samples and a 9% CV-LB gap, a model with lower CV")
print(f"but lower variance (std) may actually perform BETTER on the leaderboard.")
print(f"Consider submitting the model with lowest fold variance for comparison.")
