"""Shared plumbing for the run2 siim-isic pipeline: folds, metadata features, logging."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

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


def log_exp(**kw) -> int:
    return experiment_log.log_experiment_v2(str(COMP), **kw)


def load_meta() -> tuple[pd.DataFrame, pd.DataFrame]:
    """train/test metadata joined with image sizes and prep-time image statistics."""
    tr = pd.read_csv(DATA / "train.csv")
    te = pd.read_csv(DATA / "test.csv")
    # `diagnosis` and `benign_malignant` are train-only restatements of the target.
    tr = tr.drop(columns=["diagnosis", "benign_malignant"])
    for df, split in ((tr, "train"), (te, "test")):
        for f in (f"{split}_sizes.csv", f"r2_{split}_stats.csv"):
            add = pd.read_csv(CACHE / f)
            cols = [c for c in add.columns if c != "image_name"]
            merged = df[["image_name"]].merge(add, on="image_name", how="left")
            for c in cols:
                df[c] = pd.to_numeric(merged[c].values, errors="coerce")
    return tr, te


def make_folds(tr: pd.DataFrame) -> pd.DataFrame:
    """Two fold columns.

    `fold` (primary): plain StratifiedKFold on images. EDA established that mle-bench
    re-split the original competition train set RANDOMLY BY IMAGE — all 1457 test
    patients also occur in train, per-patient test share 0.124 vs overall 0.125. So the
    graded test set is iid-by-image and an image-level split is the regime-matched CV.

    `gfold` (secondary): StratifiedGroupKFold on patient_id. Reported alongside as the
    honest generalisation estimate (what a genuinely unseen patient would score).
    """
    tr = tr.copy()
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    tr["fold"] = -1
    for k, (_, va) in enumerate(skf.split(tr, tr.target)):
        tr.loc[tr.index[va], "fold"] = k
    sgkf = StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=SEED)
    tr["gfold"] = -1
    for k, (_, va) in enumerate(sgkf.split(tr, tr.target, groups=tr.patient_id)):
        tr.loc[tr.index[va], "gfold"] = k
    return tr


SITES = ["head/neck", "lower extremity", "oral/genital", "palms/soles", "torso",
         "upper extremity"]
STAT_COLS = [c for c in pd.read_csv(CACHE / "r2_train_stats.csv", nrows=1).columns
             if c != "image_name"]


def build_features(tr: pd.DataFrame, te: pd.DataFrame,
                   use_res: bool = True, use_patient: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Metadata + image-statistic feature frames.

    Patient-relative features encode the 'ugly duckling' sign: a lesion is suspicious
    relative to the OTHER moles of the same patient, not in absolute terms. They are
    computed from features only (never the target), so they are safe under any split and
    are computed over train+test jointly, which is legitimate transductive preprocessing.
    """
    n_tr = len(tr)
    df = pd.concat([tr.drop(columns=["target"]), te], ignore_index=True)

    df["sex_n"] = df.sex.map({"male": 1, "female": 0})
    df["age"] = df.age_approx
    df["site"] = df.anatom_site_general_challenge.map(
        {s: i for i, s in enumerate(SITES)}).fillna(-1).astype(int)
    df["site_na"] = df.anatom_site_general_challenge.isna().astype(int)
    feats = ["sex_n", "age", "site", "site_na"]

    if use_res:
        df["area"] = np.log1p(df.width.astype(float) * df.height)
        df["aspect"] = df.width / df.height
        df["long_side"] = np.log1p(df[["width", "height"]].max(1))
        res = df.width.astype(str) + "x" + df.height.astype(str)
        top = res.value_counts()
        df["res_id"] = res.map({r: i for i, r in enumerate(top.index[:20])}).fillna(-1).astype(int)
        feats += ["area", "aspect", "long_side", "res_id"]

    feats += STAT_COLS

    if use_patient:
        g = df.groupby("patient_id")
        df["pat_n"] = g.image_name.transform("size")
        df["pat_age_min"] = g.age.transform("min")
        df["pat_site_nunique"] = g.site.transform("nunique")
        feats += ["pat_n", "pat_age_min", "pat_site_nunique"]
        # ugly-duckling: z-score of this lesion's appearance within its own patient
        for c in ["core_m_r", "core_m_g", "core_m_b", "gray_m", "gray_contrast",
                  "contrast_r", "contrast_b", "gray_s", "grad_m"]:
            mu, sd = g[c].transform("mean"), g[c].transform("std")
            df[f"ud_{c}"] = (df[c] - mu) / (sd + 1e-6)
            df[f"pm_{c}"] = mu
            feats += [f"ud_{c}", f"pm_{c}"]

    return df.iloc[:n_tr].reset_index(drop=True), df.iloc[n_tr:].reset_index(drop=True), feats


def oof_auc(y: np.ndarray, p: np.ndarray) -> float:
    return float(roc_auc_score(y, p))


def per_fold_auc(y: np.ndarray, p: np.ndarray, folds: np.ndarray) -> list[float]:
    return [float(roc_auc_score(y[folds == k], p[folds == k])) for k in range(N_FOLDS)]


def save_pred(tag: str, oof: np.ndarray, pred: np.ndarray) -> None:
    ART.mkdir(exist_ok=True)
    np.save(ART / f"oof_{tag}.npy", oof)
    np.save(ART / f"pred_{tag}.npy", pred)


def load_pred(tag: str) -> tuple[np.ndarray, np.ndarray]:
    return np.load(ART / f"oof_{tag}.npy"), np.load(ART / f"pred_{tag}.npy")


def write_json(name: str, obj) -> None:
    (COMP / name).write_text(json.dumps(obj, indent=2, default=float))
