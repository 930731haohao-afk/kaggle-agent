"""Tier3 self-improvement iteration for playground-series-s4e1 (Bank Churn, ROC-AUC).

Usage: uv run python3 competitions/playground-series-s4e1/scripts/05_iterate.py <round>
  r1_unbalance_ablation -- apples-to-apples test of knowledge/experience.md's AUC prior
                           ("remove imbalance weighting improves AUC ranking") on THIS
                           large (165k row) cross-season dataset, holding features/folds
                           fixed vs tier2's LGB (drops is_unbalance already). The Feb
                           baseline (STATUS.md exp #2) used is_unbalance=True and scored
                           CV 0.89653 -- higher than tier2's no-imbalance LGB (0.89365) on
                           the SAME 28-feature family. This round isolates whether that
                           gap is really about is_unbalance, or about the fold-safe
                           surname-TE correction (Feb's version had a subtle cross-fold
                           leak -- see scripts/features.py docstring) inflating the old
                           number. Trains is_unbalance=True with the CORRECTED (fold-safe)
                           features + canonical folds, both logged for direct comparison.
  r2_optuna_lgb          -- fold-0-proxy Optuna tuning of whichever LGB variant wins r1,
                            added to the pool (not replacing).
  r3_seed_bag            -- seed-bagging the strongest pool member(s).

Every round loads solo_LGB/XGB/CAT caches from scripts/cache/*.npz (tier2 output) so
untouched members are never retrained; only re-runs what the round actually changes.
"""
import importlib.util
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import make_folds, build_all, TARGET, ID  # noqa: E402

optuna.logging.set_verbosity(optuna.logging.WARNING)

_spec = importlib.util.spec_from_file_location(
    "experiment_log", ".claude/skills/kaggle-agent/assets/utils/experiment_log.py")
experiment_log = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(experiment_log)

COMP = "competitions/playground-series-s4e1"
DATA = f"{COMP}/data"
SUB = f"{COMP}/submissions"
CACHE = f"{COMP}/scripts/cache"
SEED, N_SPLITS = 42, 5

train_raw = pd.read_csv(f"{DATA}/train.csv")
test_raw = pd.read_csv(f"{DATA}/test.csv")
y = train_raw[TARGET].to_numpy(np.int64)
FOLDS = make_folds(y)
Xtr_df, Xte_df, FEATS = build_all(train_raw, test_raw, y, FOLDS)
X = Xtr_df[FEATS].to_numpy(np.float32)
Xtest = Xte_df[FEATS].to_numpy(np.float32)


def auc(yt, p):
    return float(roc_auc_score(yt, p))


def load_cache(name):
    d = np.load(os.path.join(CACHE, f"solo_{name}.npz"))
    return d["oof"], d["pred"], float(d["auc"])


def save_cache(name, oof, pred, score):
    np.savez(os.path.join(CACHE, f"solo_{name}.npz"), oof=oof, pred=pred, auc=score)


def run_lgb(params=None, fold_subset=None):
    import lightgbm as lgb
    p = dict(objective="binary", metric="auc", learning_rate=0.05, num_leaves=63,
              max_depth=-1, min_child_samples=30, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, verbose=-1, n_jobs=-1, seed=SEED)
    if params:
        p.update(params)
    folds = fold_subset if fold_subset is not None else FOLDS
    oof = np.zeros(len(y)); pred = np.zeros(len(Xtest)); best_iters = []
    for tr, va in folds:
        dtr = lgb.Dataset(X[tr], label=y[tr], feature_name=FEATS)
        dva = lgb.Dataset(X[va], label=y[va], feature_name=FEATS, reference=dtr)
        m = lgb.train(p, dtr, num_boost_round=2000, valid_sets=[dva],
                       callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        oof[va] = m.predict(X[va])
        pred += m.predict(Xtest) / len(folds)
        best_iters.append(m.best_iteration)
    return oof, pred, best_iters


def weight_search(oof_dict):
    names = list(oof_dict)
    oofs = np.stack([oof_dict[n] for n in names], 1)
    rng = np.random.default_rng(SEED)
    cands = [np.eye(len(names))[i] for i in range(len(names))]
    cands.append(np.full(len(names), 1.0 / len(names)))
    cands += list(rng.dirichlet(np.ones(len(names)), size=800))
    best_w, best_s = None, -1e18
    for w in cands:
        s = auc(y, oofs @ w)
        if s > best_s:
            best_s, best_w = s, w
    conc = np.clip(best_w, 1e-3, None) * 200.0
    for w in rng.dirichlet(conc, size=300):
        s = auc(y, oofs @ w)
        if s > best_s:
            best_s, best_w = s, w
    return names, best_w, best_s


ROUND = sys.argv[1] if len(sys.argv) > 1 else "r1_unbalance_ablation"

if ROUND == "r1_unbalance_ablation":
    t0 = time.time()
    oof, pred, best_iters = run_lgb(params={"is_unbalance": True})
    wall = time.time() - t0
    s_bal = auc(y, oof)
    oof_noimb, _, s_noimb = load_cache("LGB")
    print(f"LGB is_unbalance=True : OOF AUC={s_bal:.6f}  wall={wall:.1f}s  mean_best_iter={np.mean(best_iters):.0f}")
    print(f"LGB no-imbalance (tier2, cached): OOF AUC={s_noimb:.6f}")
    delta = s_bal - s_noimb
    print(f"delta (imbalance - no_imbalance) = {delta:+.6f}")

    experiment_log.log_experiment_v2(
        COMP, model="LGB is_unbalance=True (tier3 r1 ablation)", metric="auc", direction="maximize",
        score=round(s_bal, 6),
        cv=dict(scheme=f"{N_SPLITS}fold_stratified", n_splits=N_SPLITS, seed=SEED),
        features=FEATS,
        notes=(f"Apples-to-apples re-test of knowledge/experience.md AUC-ranking prior "
               f"(s3e3: removing imbalance weighting improved AUC) on s4e1's 165k-row "
               f"cross-season churn data, SAME corrected fold-safe features/folds as "
               f"tier2's no-imbalance LGB (cached score {s_noimb:.6f}). Delta={delta:+.6f}. "
               f"{'Prior HOLDS: is_unbalance hurts AUC here too.' if delta < 0 else 'Prior DOES NOT TRANSFER: is_unbalance helps AUC on this large cross-season dataset (opposite of the small-sample s3e3 evidence).'}"),
    )

    # decide winner for the pool: keep whichever LGB variant scores higher
    if s_bal > s_noimb:
        save_cache("LGB", oof, pred, s_bal)
        print("Adopting is_unbalance=True LGB as the pool's LGB member (higher OOF AUC).")
        winner_note = "is_unbalance=True"
    else:
        print("Keeping tier2's no-imbalance LGB as the pool's LGB member (higher OOF AUC).")
        winner_note = "no_imbalance"

    # re-run blend with the (possibly updated) LGB member
    oof_lgb, pred_lgb, _ = load_cache("LGB")
    oof_xgb, pred_xgb, s_xgb = load_cache("XGB")
    oof_cat, pred_cat, s_cat = load_cache("CAT")
    names, w, s_blend = weight_search({"LGB": oof_lgb, "XGB": oof_xgb, "CAT": oof_cat})
    print(f"BEST blend (post r1) {dict(zip(names, np.round(w, 3)))} -> AUC={s_blend:.6f}")

    final_test = np.stack([pred_lgb, pred_xgb, pred_cat], 1) @ w
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sub_path = os.path.join(SUB, f"tier3_r1_blend_{s_blend:.5f}_{stamp}.csv")
    pd.DataFrame({ID: test_raw[ID], TARGET: final_test}).to_csv(sub_path, index=False)
    print(f"wrote {sub_path}")
    np.savez(os.path.join(CACHE, "blend_tier3_r1.npz"), oof=oof_lgb * w[0] + oof_xgb * w[1] + oof_cat * w[2],
             pred=final_test, weights=w, names=np.array(names))

    experiment_log.log_experiment_v2(
        COMP, model="tier3 r1 LGB+XGB+CAT blend (LGB=" + winner_note + ")", metric="auc", direction="maximize",
        score=round(s_blend, 6),
        cv=dict(scheme=f"{N_SPLITS}fold_stratified", n_splits=N_SPLITS, seed=SEED),
        base_models=[dict(name=n, score=round(float(auc(y, o)), 6)) for n, o in
                     zip(names, [oof_lgb, oof_xgb, oof_cat])],
        ensemble=dict(weights=dict(zip(names, [round(float(x), 3) for x in w])), score=round(s_blend, 6)),
        features=FEATS, submission=os.path.basename(sub_path),
        notes=f"tier3 round1: is_unbalance ablation resolved (LGB={winner_note}), blend re-weighted.",
    )
    print("RESULT " + json.dumps({"s_bal": s_bal, "s_noimb": float(s_noimb), "delta": delta,
                                    "blend": s_blend, "winner": winner_note}))

elif ROUND == "r2_optuna_lgb":
    oof_lgb, pred_lgb, s_lgb_base = load_cache("LGB")
    print(f"tuning target: current pool LGB OOF AUC={s_lgb_base:.6f}")
    fold0_tr, fold0_va = FOLDS[0]

    def objective(trial):
        params = {
            "is_unbalance": trial.suggest_categorical("is_unbalance", [True, False]),
            "num_leaves": trial.suggest_int("num_leaves", 7, 255, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
        import lightgbm as lgb
        p = dict(objective="binary", metric="auc", n_jobs=-1, verbose=-1, seed=SEED, bagging_freq=1)
        p.update(params)
        dtr = lgb.Dataset(X[fold0_tr], label=y[fold0_tr], feature_name=FEATS)
        dva = lgb.Dataset(X[fold0_va], label=y[fold0_va], feature_name=FEATS, reference=dtr)
        m = lgb.train(p, dtr, num_boost_round=2000, valid_sets=[dva],
                       callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        return auc(y[fold0_va], m.predict(X[fold0_va]))

    t0 = time.time()
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=50, timeout=600, show_progress_bar=False)
    print(f"Optuna: {len(study.trials)} trials in {time.time()-t0:.1f}s, best fold0 AUC={study.best_value:.6f}")
    print("best params:", study.best_params)

    oof_tuned, pred_tuned, best_iters = run_lgb(params=study.best_params)
    s_tuned = auc(y, oof_tuned)
    print(f"Full 5-fold OOF AUC (tuned LGB) = {s_tuned:.6f}  vs pool LGB {s_lgb_base:.6f}")
    save_cache("LGB_tuned", oof_tuned, pred_tuned, s_tuned)
    with open(os.path.join(CACHE, "lgb_tuned_params.json"), "w") as f:
        json.dump(study.best_params, f, indent=2)

    experiment_log.log_experiment_v2(
        COMP, model="LGB Optuna-tuned (tier3 r2, fold-0 proxy, 50 trials)", metric="auc", direction="maximize",
        score=round(s_tuned, 6),
        cv=dict(scheme=f"{N_SPLITS}fold_stratified", n_splits=N_SPLITS, seed=SEED),
        features=FEATS,
        notes=(f"Optuna TPE 50 trials on fold-0 proxy ({time.time()-t0:.1f}s), best fold0 AUC "
               f"{study.best_value:.6f}, params={study.best_params}. Full-5fold OOF {s_tuned:.6f} "
               f"vs untuned pool LGB {s_lgb_base:.6f} (delta {s_tuned - s_lgb_base:+.6f}). "
               f"Added to pool (not replacing), per experience.md convention."),
    )

    oof_xgb, pred_xgb, _ = load_cache("XGB")
    oof_cat, pred_cat, _ = load_cache("CAT")
    pool = {"LGB": oof_lgb, "LGB_tuned": oof_tuned, "XGB": oof_xgb, "CAT": oof_cat}
    names, w, s_blend = weight_search(pool)
    print(f"BEST blend (post r2, 4-way) {dict(zip(names, np.round(w, 3)))} -> AUC={s_blend:.6f}")
    preds = {"LGB": pred_lgb, "LGB_tuned": pred_tuned, "XGB": pred_xgb, "CAT": pred_cat}
    final_test = np.stack([preds[n] for n in names], 1) @ w
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sub_path = os.path.join(SUB, f"tier3_r2_blend_{s_blend:.5f}_{stamp}.csv")
    pd.DataFrame({ID: test_raw[ID], TARGET: final_test}).to_csv(sub_path, index=False)
    print(f"wrote {sub_path}")
    np.savez(os.path.join(CACHE, "blend_tier3_r2.npz"),
             oof=np.stack([pool[n] for n in names], 1) @ w, pred=final_test, weights=w, names=np.array(names))
    experiment_log.log_experiment_v2(
        COMP, model="tier3 r2 4-way blend (+LGB_tuned)", metric="auc", direction="maximize",
        score=round(s_blend, 6),
        cv=dict(scheme=f"{N_SPLITS}fold_stratified", n_splits=N_SPLITS, seed=SEED),
        base_models=[dict(name=n, score=round(float(auc(y, pool[n])), 6)) for n in names],
        ensemble=dict(weights=dict(zip(names, [round(float(x), 3) for x in w])), score=round(s_blend, 6)),
        features=FEATS, submission=os.path.basename(sub_path),
        notes="tier3 round2: Optuna fold-0-proxy tuned LGB added to pool (not replacing).",
    )
    print("RESULT " + json.dumps({"s_tuned": s_tuned, "s_lgb_base": float(s_lgb_base), "blend": s_blend}))

elif ROUND == "r3_seed_bag":
    with open(os.path.join(CACHE, "lgb_tuned_params.json")) as f:
        tuned_params = json.load(f)
    oof_lgb, pred_lgb, _ = load_cache("LGB")
    oof_tuned, pred_tuned, s_tuned = load_cache("LGB_tuned")
    oof_xgb, pred_xgb, _ = load_cache("XGB")
    oof_cat, pred_cat, _ = load_cache("CAT")

    seed_params = dict(tuned_params); seed_params["seed"] = 2024
    oof_seed, pred_seed, _ = run_lgb(params=seed_params)
    s_seed = auc(y, oof_seed)
    print(f"LGB_tuned seed=2024: OOF AUC={s_seed:.6f}  (tuned seed=42: {s_tuned:.6f})")
    save_cache("LGB_tuned_seed2024", oof_seed, pred_seed, s_seed)

    experiment_log.log_experiment_v2(
        COMP, model="LGB_tuned seed=2024 (tier3 r3 seed-bag)", metric="auc", direction="maximize",
        score=round(s_seed, 6),
        cv=dict(scheme=f"{N_SPLITS}fold_stratified", n_splits=N_SPLITS, seed=SEED),
        features=FEATS,
        notes=f"Seed-bagging member: same Optuna-tuned params as r2's LGB_tuned, random_state=2024. Added to pool.",
    )

    pool = {"LGB": oof_lgb, "LGB_tuned": oof_tuned, "LGB_tuned_s2024": oof_seed, "XGB": oof_xgb, "CAT": oof_cat}
    names, w, s_blend = weight_search(pool)
    print(f"BEST blend (post r3, 5-way) {dict(zip(names, np.round(w, 3)))} -> AUC={s_blend:.6f}")
    preds = {"LGB": pred_lgb, "LGB_tuned": pred_tuned, "LGB_tuned_s2024": pred_seed, "XGB": pred_xgb, "CAT": pred_cat}
    final_test = np.stack([preds[n] for n in names], 1) @ w
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sub_path = os.path.join(SUB, f"tier3_r3_blend_{s_blend:.5f}_{stamp}.csv")
    pd.DataFrame({ID: test_raw[ID], TARGET: final_test}).to_csv(sub_path, index=False)
    print(f"wrote {sub_path}")
    np.savez(os.path.join(CACHE, "blend_tier3_r3.npz"),
             oof=np.stack([pool[n] for n in names], 1) @ w, pred=final_test, weights=w, names=np.array(names))
    experiment_log.log_experiment_v2(
        COMP, model="tier3 r3 5-way blend (+LGB_tuned seed-bag)", metric="auc", direction="maximize",
        score=round(s_blend, 6),
        cv=dict(scheme=f"{N_SPLITS}fold_stratified", n_splits=N_SPLITS, seed=SEED),
        base_models=[dict(name=n, score=round(float(auc(y, pool[n])), 6)) for n in names],
        ensemble=dict(weights=dict(zip(names, [round(float(x), 3) for x in w])), score=round(s_blend, 6)),
        features=FEATS, submission=os.path.basename(sub_path),
        notes="tier3 round3: seed-bagged Optuna-tuned LGB (seed=2024) added to pool.",
    )
    print("RESULT " + json.dumps({"s_seed": s_seed, "s_tuned": float(s_tuned), "blend": s_blend}))
else:
    raise ValueError(f"unknown round {ROUND!r}")
