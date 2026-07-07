"""Stage 3 self-improvement iteration for playground-series-s5e10 (accident_risk, RMSE).

Usage: uv run python3 competitions/playground-series-s5e10/scripts/05_iterate.py <round>

  r1_interaction_prior -- controlled check of TWO experience-library priors before
        trusting either (cross-season rule: a prior must survive a small controlled
        comparison on THIS comp's metric or be dropped, honestly logged either way):
        (A) "trees learn multiplicative interactions themselves; explicit product
            columns add collinearity, not information" (s3e9 evidence) -- directly
            contradicts Feb's STATUS.md claim that the interaction features are key.
            Test: same-params LGB on the 12 encoded raw features vs stage 2's cached
            27-feature LGB, same folds, both scored via rmse_clip.
        (B) "snap predictions to the target's value grid" (s3e14 evidence, earned on
            MAE) -- the target here sits on a 0.01 grid (98 unique values), but RMSE
            is a squared-error metric whose optimal point prediction is a conditional
            mean, not a grid point. Test: zero-cost snap vs clip on every cached OOF.
        Winner of (A) keeps the pool's LGB slot; blend re-weighted afterwards.
  r2_optuna_lgb -- fold-0-proxy Optuna tuning of whichever LGB variant won r1,
        objective = fold-0 clipped-RMSE directly (the s3e7/s3e14/s3e1/s3e11
        fold-proxy recipe; winner re-validated on the full 5 folds). Added to the
        pool (not replacing), per knowledge/experience.md.
  r3_seed_bag -- seed-bagging the tuned model (seed 2024) -- cheapest residual gain
        per knowledge/experience.md's Optuna section.
  r4_seed_bag2 -- optional second seed (777); run only while the stop rule ("two
        consecutive rounds without blend improvement -> stop") has not fired.

Every round loads solo_* caches from scripts/cache/*.npz (stage 2 output) so
untouched members are never retrained; only re-runs what the round changes.
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import (make_folds, build_all, rmse_clip, snap_to_grid,  # noqa: E402
                      INTERACTION_FEATURES, TARGET, ID)

optuna.logging.set_verbosity(optuna.logging.WARNING)

_spec = importlib.util.spec_from_file_location(
    "experiment_log", ".claude/skills/kaggle-agent/assets/utils/experiment_log.py")
experiment_log = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(experiment_log)

COMP = "competitions/playground-series-s5e10"
DATA = f"{COMP}/data"
SUB = f"{COMP}/submissions"
CACHE = f"{COMP}/scripts/cache"
SEED, N_SPLITS = 42, 5

train_raw = pd.read_csv(f"{DATA}/train.csv")
test_raw = pd.read_csv(f"{DATA}/test.csv")
y = train_raw[TARGET].to_numpy(np.float64)
FOLDS = make_folds(y)
Xtr_df, Xte_df, FEATS = build_all(train_raw, test_raw)


def matrices(feats):
    return Xtr_df[feats].to_numpy(np.float32), Xte_df[feats].to_numpy(np.float32)


X, Xtest = matrices(FEATS)
BASE_FEATS = [c for c in FEATS if c not in INTERACTION_FEATURES]


def load_cache(name):
    d = np.load(os.path.join(CACHE, f"solo_{name}.npz"))
    return d["oof"], d["pred"], float(d["rmse"])


def save_cache(name, oof, pred, score):
    np.savez(os.path.join(CACHE, f"solo_{name}.npz"), oof=oof, pred=pred, rmse=score)


def run_lgb(params=None, feats=None, fold_subset=None):
    import lightgbm as lgb
    p = dict(objective="regression", metric="rmse", learning_rate=0.05, num_leaves=63,
             max_depth=-1, min_child_samples=30, feature_fraction=0.8,
             bagging_fraction=0.8, bagging_freq=1, reg_alpha=0.1, reg_lambda=0.1,
             verbose=-1, n_jobs=-1, seed=SEED)
    if params:
        p.update(params)
    feats = feats or FEATS
    Xf, Xtf = matrices(feats)
    folds = fold_subset if fold_subset is not None else FOLDS
    oof = np.zeros(len(y)); pred = np.zeros(len(Xtf)); best_iters = []
    for tr, va in folds:
        dtr = lgb.Dataset(Xf[tr], label=y[tr], feature_name=feats)
        dva = lgb.Dataset(Xf[va], label=y[va], feature_name=feats, reference=dtr)
        m = lgb.train(p, dtr, num_boost_round=3000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
        oof[va] = m.predict(Xf[va])
        pred += m.predict(Xtf) / len(folds)
        best_iters.append(m.best_iteration)
    return oof, pred, best_iters


def weight_search(oof_dict):
    """Dirichlet random search + corner/uniform seeds + concentration refinement,
    scored DIRECTLY against clipped-OOF RMSE (never raw) -- same convention as
    s4e11's 05_iterate.py generalized to this comp's metric."""
    names = list(oof_dict)
    oofs = np.stack([oof_dict[n] for n in names], 1)
    rng = np.random.default_rng(SEED)
    cands = [np.eye(len(names))[i] for i in range(len(names))]
    cands.append(np.full(len(names), 1.0 / len(names)))
    cands += list(rng.dirichlet(np.ones(len(names)), size=800))
    best_w, best_s = None, 1e18
    for w in cands:
        s = rmse_clip(y, oofs @ w)
        if s < best_s:
            best_s, best_w = s, w
    conc = np.clip(best_w, 1e-3, None) * 200.0
    for w in rng.dirichlet(conc, size=300):
        s = rmse_clip(y, oofs @ w)
        if s < best_s:
            best_s, best_w = s, w
    return names, best_w, best_s


def emit_blend(tag, pool_oof, pool_pred, note):
    names, w, s_blend = weight_search(pool_oof)
    print(f"BEST blend ({tag}, {len(names)}-way) "
          f"{dict(zip(names, np.round(w, 3)))} -> RMSE={s_blend:.6f}")
    final_test = np.clip(np.stack([pool_pred[n] for n in names], 1) @ w, 0, 1)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sub_path = os.path.join(SUB, f"stage3_{tag}_blend_{s_blend:.5f}_{stamp}.csv")
    pd.DataFrame({ID: test_raw[ID], TARGET: final_test}).to_csv(sub_path, index=False)
    print(f"wrote {sub_path}")
    np.savez(os.path.join(CACHE, f"blend_stage3_{tag}.npz"),
             oof=np.stack([pool_oof[n] for n in names], 1) @ w,
             pred=np.stack([pool_pred[n] for n in names], 1) @ w,
             weights=w, names=np.array(names))
    experiment_log.log_experiment_v2(
        COMP, model=f"stage3 {tag} {len(names)}-way blend", metric="RMSE",
        direction="minimize", score=round(s_blend, 6),
        cv=dict(scheme=f"{N_SPLITS}fold", n_splits=N_SPLITS, seed=SEED),
        base_models=[dict(name=n, score=round(rmse_clip(y, pool_oof[n]), 6)) for n in names],
        ensemble=dict(weights=dict(zip(names, [round(float(x), 3) for x in w])),
                      score=round(s_blend, 6)),
        features=FEATS, postprocess=["clip[0,1]"],
        submission=os.path.basename(sub_path), notes=note)
    return names, w, s_blend


ROUND = sys.argv[1] if len(sys.argv) > 1 else "r1_interaction_prior"

if ROUND == "r1_interaction_prior":
    # --- prior (A): explicit interactions on/off, same params/folds/metric ---
    t0 = time.time()
    oof_base, pred_base, best_iters = run_lgb(feats=BASE_FEATS)
    wall = time.time() - t0
    s_base = rmse_clip(y, oof_base)
    oof_full, _, s_full = load_cache("LGB")
    delta = s_base - s_full
    print(f"LGB base-{len(BASE_FEATS)}-features : OOF RMSE={s_base:.6f}  wall={wall:.1f}s "
          f"mean_best_iter={np.mean(best_iters):.0f}")
    print(f"LGB full-{len(FEATS)}-features (stage2, cached): OOF RMSE={s_full:.6f}")
    print(f"delta (base - full) = {delta:+.6f}  ({'interactions HELP' if delta > 0 else 'interactions do NOT help'})")

    # --- prior (B): snap-to-0.01-grid, zero training cost ---
    d = np.load(os.path.join(CACHE, "blend_stage2.npz"))
    blend_oof = d["oof"]
    s_clip = rmse_clip(y, blend_oof)
    s_snap = rmse_clip(y, snap_to_grid(blend_oof))
    print(f"snap-to-grid check on stage2 blend OOF: clip={s_clip:.6f}  snap={s_snap:.6f} "
          f"delta={s_snap - s_clip:+.6f}")

    experiment_log.log_experiment_v2(
        COMP, model=f"LGB base-{len(BASE_FEATS)}-features (stage3 r1 prior check)",
        metric="RMSE", direction="minimize", score=round(s_base, 6),
        cv=dict(scheme=f"{N_SPLITS}fold", n_splits=N_SPLITS, seed=SEED),
        features=BASE_FEATS, postprocess=["clip[0,1]"],
        notes=(f"Prior check A: experience-library prior 'explicit product/poly columns add "
               f"nothing for GBDT' (s3e9) vs Feb's claim that interactions are key. Same "
               f"params/folds LGB, 12 raw encoded features {s_base:.6f} vs 27-feature stage2 "
               f"LGB {s_full:.6f}, delta {delta:+.6f} -> "
               + ("prior CONFIRMED here: interaction block adds nothing; adopting the "
                  "12-feature LGB as the pool member."
                  if delta < 0 else
                  "prior REJECTED here (interactions genuinely help at this scale/metric); "
                  "keeping the 27-feature set. ")
               + f" Prior check B (zero-cost): snap-to-0.01-grid (s3e14 MAE prior) on stage2 "
                 f"blend OOF: clip {s_clip:.6f} vs snap {s_snap:.6f}, delta "
                 f"{s_snap - s_clip:+.6f} -> "
               + ("snap ADOPTED." if s_snap < s_clip else
                  "snap REJECTED for RMSE (squared error wants the conditional mean, not a "
                  "grid point); post-processing stays clip-only.")),
    )

    if s_base < s_full:
        save_cache("LGB", oof_base, pred_base, s_base)
        winner_note = "base12"
        winner_feats = BASE_FEATS
    else:
        winner_note = "full27"
        winner_feats = FEATS
    # marker consumed by r2/r3/r4 and by tree_search/run_s5e10_v3.py's root config,
    # so every later stage trains the SAME feature variant the r1 check adopted.
    with open(os.path.join(CACHE, "pool_lgb_variant.json"), "w") as f:
        json.dump({"variant": winner_note, "feats": winner_feats}, f, indent=2)
    print(f"pool LGB member: {winner_note}")

    oof_lgb, pred_lgb, _ = load_cache("LGB")
    oof_xgb, pred_xgb, _ = load_cache("XGB")
    oof_cat, pred_cat, _ = load_cache("CAT")
    emit_blend("r1", {"LGB": oof_lgb, "XGB": oof_xgb, "CAT": oof_cat},
               {"LGB": pred_lgb, "XGB": pred_xgb, "CAT": pred_cat},
               f"stage3 round1: both priors resolved (A: LGB={winner_note}; B: snap "
               f"rejected/adopted per r1 prior-check entry), blend re-weighted with "
               f"Dirichlet search (finer than stage2's 0.05 grid).")

elif ROUND == "r2_optuna_lgb":
    oof_lgb, pred_lgb, s_lgb_base = load_cache("LGB")
    print(f"tuning target: current pool LGB OOF RMSE={s_lgb_base:.6f}")
    fold0_tr, fold0_va = FOLDS[0]
    # tune the SAME feature variant the r1 prior check adopted
    with open(os.path.join(CACHE, "pool_lgb_variant.json")) as f:
        _variant = json.load(f)
    POOL_FEATS = _variant["feats"]
    Xp, _ = matrices(POOL_FEATS)
    print(f"tuning on r1-adopted variant: {_variant['variant']} ({len(POOL_FEATS)} features)")

    def objective(trial):
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 15, 511, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 14),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 200),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
        import lightgbm as lgb
        p = dict(objective="regression", metric="rmse", n_jobs=-1, verbose=-1,
                 seed=SEED, bagging_freq=1)
        p.update(params)
        dtr = lgb.Dataset(Xp[fold0_tr], label=y[fold0_tr], feature_name=POOL_FEATS)
        dva = lgb.Dataset(Xp[fold0_va], label=y[fold0_va], feature_name=POOL_FEATS, reference=dtr)
        m = lgb.train(p, dtr, num_boost_round=3000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
        return rmse_clip(y[fold0_va], m.predict(Xp[fold0_va]))

    t0 = time.time()
    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=40, timeout=1500, show_progress_bar=False)
    tune_wall = time.time() - t0
    print(f"Optuna: {len(study.trials)} trials in {tune_wall:.1f}s, "
          f"best fold0 RMSE={study.best_value:.6f}")
    print("best params:", study.best_params)

    oof_tuned, pred_tuned, best_iters = run_lgb(params=study.best_params, feats=POOL_FEATS)
    s_tuned = rmse_clip(y, oof_tuned)
    print(f"Full 5-fold OOF RMSE (tuned LGB) = {s_tuned:.6f} vs pool LGB {s_lgb_base:.6f}")
    save_cache("LGB_tuned", oof_tuned, pred_tuned, s_tuned)
    with open(os.path.join(CACHE, "lgb_tuned_params.json"), "w") as f:
        json.dump(study.best_params, f, indent=2)

    experiment_log.log_experiment_v2(
        COMP, model="LGB Optuna-tuned (stage3 r2, fold-0 proxy, direct-metric objective)",
        metric="RMSE", direction="minimize", score=round(s_tuned, 6),
        cv=dict(scheme=f"{N_SPLITS}fold", n_splits=N_SPLITS, seed=SEED),
        features=POOL_FEATS, postprocess=["clip[0,1]"],
        notes=(f"Optuna TPE {len(study.trials)} trials on fold-0 proxy ({tune_wall:.1f}s), "
               f"objective = fold-0 clipped RMSE DIRECTLY, best fold0 {study.best_value:.6f}, "
               f"params={study.best_params}. Full-5fold OOF {s_tuned:.6f} vs untuned pool LGB "
               f"{s_lgb_base:.6f} (delta {s_tuned - s_lgb_base:+.6f}). Added to pool (not "
               f"replacing), per experience-library convention."),
    )

    oof_xgb, pred_xgb, _ = load_cache("XGB")
    oof_cat, pred_cat, _ = load_cache("CAT")
    emit_blend("r2", {"LGB": oof_lgb, "LGB_tuned": oof_tuned, "XGB": oof_xgb, "CAT": oof_cat},
               {"LGB": pred_lgb, "LGB_tuned": pred_tuned, "XGB": pred_xgb, "CAT": pred_cat},
               "stage3 round2: Optuna fold-0-proxy tuned LGB (direct-metric objective) "
               "added to pool (not replacing).")

elif ROUND in ("r3_seed_bag", "r4_seed_bag2"):
    new_seed = 2024 if ROUND == "r3_seed_bag" else 777
    member = f"LGB_tuned_seed{new_seed}"
    with open(os.path.join(CACHE, "lgb_tuned_params.json")) as f:
        tuned_params = json.load(f)
    with open(os.path.join(CACHE, "pool_lgb_variant.json")) as f:
        POOL_FEATS = json.load(f)["feats"]
    seed_params = dict(tuned_params); seed_params["seed"] = new_seed
    oof_seed, pred_seed, _ = run_lgb(params=seed_params, feats=POOL_FEATS)
    s_seed = rmse_clip(y, oof_seed)
    _, _, s_tuned = load_cache("LGB_tuned")
    print(f"{member}: OOF RMSE={s_seed:.6f}  (tuned seed=42: {s_tuned:.6f})")
    save_cache(member, oof_seed, pred_seed, s_seed)

    experiment_log.log_experiment_v2(
        COMP, model=f"{member} (stage3 {ROUND[:2]} seed-bag)", metric="RMSE",
        direction="minimize", score=round(s_seed, 6),
        cv=dict(scheme=f"{N_SPLITS}fold", n_splits=N_SPLITS, seed=SEED),
        features=POOL_FEATS, postprocess=["clip[0,1]"],
        notes=(f"Seed-bagging member: same Optuna-tuned params as r2's LGB_tuned, "
               f"seed={new_seed}. Added to pool."),
    )

    pool_oof, pool_pred = {}, {}
    for n in ["LGB", "LGB_tuned", "LGB_tuned_seed2024", "LGB_tuned_seed777", "XGB", "CAT"]:
        path = os.path.join(CACHE, f"solo_{n}.npz")
        if os.path.exists(path):
            o, p_, _ = load_cache(n)
            pool_oof[n], pool_pred[n] = o, p_
    emit_blend(ROUND[:2], pool_oof, pool_pred,
               f"stage3 {ROUND}: seed-bagged Optuna-tuned LGB (seed={new_seed}) added to pool.")
else:
    raise ValueError(f"unknown round {ROUND!r}")
