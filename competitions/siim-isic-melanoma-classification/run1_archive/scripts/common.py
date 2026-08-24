"""Shared pieces: folds, metadata features, metric, experiment logging."""
from __future__ import annotations

import importlib.util
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

COMP = Path(__file__).resolve().parents[1]
DATA = COMP / "data"
CACHE = COMP / "cache"
ART = COMP / "artifacts"
SEED = 42
N_FOLDS = 5

_spec = importlib.util.spec_from_file_location(
    "experiment_log",
    str(COMP.parents[1] / ".claude/skills/kaggle-agent/assets/utils/experiment_log.py"))
experiment_log = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(experiment_log)


def log_experiment(model: str, score: float, features, params: dict | None = None,
                   fold_scores=None, notes: str = "", ensemble: dict | None = None,
                   submission: str | None = None, cv_scheme: str | None = None):
    """Thin adapter onto the canonical v2 logger (its signature takes `cv`, not `cv_scheme`)."""
    cv = {"scheme": cv_scheme or f"StratifiedKFold({N_FOLDS}, shuffle=True, seed={SEED}), image-level",
          "n_splits": N_FOLDS}
    if fold_scores is not None:
        cv["fold_scores"] = [round(float(v), 6) for v in fold_scores]
    if params:
        notes = (notes + " | params=" + json.dumps(params, default=str)).strip(" |")
    return experiment_log.log_experiment_v2(
        str(COMP), model=model, metric="roc_auc", direction="maximize", score=score,
        cv=cv, features=list(features), ensemble=ensemble, submission=submission, notes=notes)


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def auc(y: np.ndarray, p: np.ndarray) -> float:
    return float(roc_auc_score(y, p))


def load_meta() -> tuple[pd.DataFrame, pd.DataFrame]:
    """train/test metadata joined with cached per-image stats, plus fold ids."""
    tr = pd.read_csv(DATA / "train.csv")
    te = pd.read_csv(DATA / "test.csv")
    tr = tr.merge(pd.read_csv(CACHE / "train_stats.csv"), on="image_name", how="left", validate="1:1")
    te = te.merge(pd.read_csv(CACHE / "test_stats.csv"), on="image_name", how="left", validate="1:1")
    assert tr.orig_w.notna().all() and te.orig_w.notna().all()

    # Image-level StratifiedKFold mirrors the graded split (all 1457 test patients occur in train).
    tr["fold"] = -1
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for k, (_, vi) in enumerate(skf.split(tr, tr.target)):
        tr.loc[vi, "fold"] = k
    assert (tr.fold >= 0).all()
    return tr, te


# ---------------------------------------------------------------- tabular features
_SITES = ["torso", "lower extremity", "upper extremity", "head/neck", "palms/soles", "oral/genital"]
_STAT_COLS = ["r_mean", "g_mean", "b_mean", "r_std", "g_std", "b_std", "orig_px"]


def patient_context(tr: pd.DataFrame, te: pd.DataFrame) -> pd.DataFrame:
    """Label-free patient aggregates over train+test combined (no leakage: no target used)."""
    both = pd.concat([tr, te], ignore_index=True)
    both["orig_px"] = both.orig_w * both.orig_h
    g = both.groupby("patient_id")
    pat = g.size().rename("pat_n_img").to_frame()
    pat["pat_n_sites"] = g.anatom_site_general_challenge.nunique()
    agg = g[_STAT_COLS].agg(["mean", "std"])
    agg.columns = [f"pat_{c}_{s}" for c, s in agg.columns]
    return pat.join(agg).reset_index()


def base_feats(df: pd.DataFrame, pat: pd.DataFrame) -> pd.DataFrame:
    d = df.reset_index(drop=True)
    f = pd.DataFrame(index=d.index)
    f["sex"] = d.sex.map({"male": 1, "female": 0}).astype("float32")
    f["age"] = d.age_approx.astype("float32")
    for s in _SITES:
        f["site_" + s.replace("/", "_").replace(" ", "_")] = (d.anatom_site_general_challenge == s).astype("float32")
    f["site_na"] = d.anatom_site_general_challenge.isna().astype("float32")
    f["orig_w"] = d.orig_w.astype("float32")
    f["orig_h"] = d.orig_h.astype("float32")
    f["orig_px"] = (d.orig_w * d.orig_h).astype("float32")
    f["orig_ar"] = (d.orig_w / d.orig_h).astype("float32")
    for c in ["r_mean", "g_mean", "b_mean", "r_std", "g_std", "b_std"]:
        f[c] = d[c].astype("float32")
    j = d[["patient_id"]].merge(pat, on="patient_id", how="left")
    f["pat_n_img"] = j.pat_n_img.to_numpy("float32")
    f["pat_n_sites"] = j.pat_n_sites.to_numpy("float32")
    # "ugly duckling": deviation of this lesion from the same patient's other lesions
    for c in _STAT_COLS:
        raw = (d.orig_w * d.orig_h if c == "orig_px" else d[c]).to_numpy("float32")
        mu = j[f"pat_{c}_mean"].to_numpy("float32")
        sd = j[f"pat_{c}_std"].to_numpy("float32")
        f[f"ud_{c}"] = raw - mu
        f[f"udz_{c}"] = (raw - mu) / np.where(sd > 1e-6, sd, np.nan)
    return f


def patient_te(tr: pd.DataFrame, te: pd.DataFrame, prior_w: float = 5.0):
    """Fold-safe patient-level target encoding.

    Returns (te_tr, te_val, te_te) where
      te_tr  : (n_train, N_FOLDS)  value for row i while TRAINING split k
                                   (leave-one-out inside folds != k; NaN if row i is in fold k)
      te_val : (n_train,)          value for row i while it is VALIDATED (stats from folds != its own)
      te_te  : (n_test,  N_FOLDS)  value for a test row under split k (stats from folds != k)
    Smoothed toward the global positive rate with weight `prior_w`.
    """
    y = tr.target.to_numpy("float64")
    pid = tr.patient_id.to_numpy()
    folds = tr.fold.to_numpy()
    tpid = te.patient_id.to_numpy()
    g = y.mean()

    te_tr = np.full((len(tr), N_FOLDS), np.nan)
    te_val = np.full(len(tr), np.nan)
    te_te = np.full((len(te), N_FOLDS), np.nan)
    for k in range(N_FOLDS):
        m = folds != k
        s = pd.Series(y[m]).groupby(pd.Series(pid[m])).agg(["sum", "size"])
        ti = np.where(m)[0]
        a = s.reindex(pid[ti])
        te_tr[ti, k] = (a["sum"].to_numpy() - y[ti] + prior_w * g) / (a["size"].to_numpy() - 1.0 + prior_w)
        vi = np.where(~m)[0]
        b = s.reindex(pid[vi])
        te_val[vi] = (np.nan_to_num(b["sum"].to_numpy()) + prior_w * g) / \
                     (np.nan_to_num(b["size"].to_numpy()) + prior_w)
        c = s.reindex(tpid)
        te_te[:, k] = (np.nan_to_num(c["sum"].to_numpy()) + prior_w * g) / \
                      (np.nan_to_num(c["size"].to_numpy()) + prior_w)
    assert np.isfinite(te_val).all() and np.isfinite(te_te).all()
    return te_tr, te_val, te_te


def build_tabular(tr: pd.DataFrame, te: pd.DataFrame):
    pat = patient_context(tr, te)
    return base_feats(tr, pat), base_feats(te, pat)
