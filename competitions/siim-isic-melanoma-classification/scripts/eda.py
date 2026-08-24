"""Stage 1 EDA — siim-isic-melanoma-classification (run2).

Verifies the Stage 0.5 dossier hypotheses:
  * is the train/test split disjoint by patient_id?
  * how strong is the class imbalance and how is it distributed over patients?
  * which metadata columns are usable (and which are train-only leakage)?
  * are raw image dimensions a site/scanner fingerprint that correlates with the target?
Prints everything to stdout and writes a machine-readable summary to eda_summary.json.
"""
from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

COMP = Path(__file__).resolve().parents[1]
DATA = COMP / "data"
SEED = 42


def jpeg_size(path: Path) -> tuple[int, int]:
    """Read width/height from the JPEG header without decoding pixels."""
    with Image.open(path) as im:
        return im.size


def collect_sizes(names: pd.Series, split: str) -> pd.DataFrame:
    paths = [DATA / "jpeg" / split / f"{n}.jpg" for n in names]
    with ThreadPoolExecutor(max_workers=16) as ex:
        sizes = list(ex.map(jpeg_size, paths))
    w, h = zip(*sizes)
    return pd.DataFrame({"image_name": names.values, "width": w, "height": h})


def main() -> None:
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    sub = pd.read_csv(DATA / "sample_submission.csv")
    out: dict = {}

    print("=" * 70)
    print("1. SHAPES / COLUMNS")
    print("=" * 70)
    print(f"train {train.shape}  cols={list(train.columns)}")
    print(f"test  {test.shape}  cols={list(test.columns)}")
    print(f"sample_submission {sub.shape} cols={list(sub.columns)}")
    train_only = set(train.columns) - set(test.columns) - {"target"}
    print(f"train-only columns (leakage risk, must be dropped): {sorted(train_only)}")
    out["shapes"] = {"train": list(train.shape), "test": list(test.shape)}
    out["train_only_cols"] = sorted(train_only)

    print()
    print("=" * 70)
    print("2. TARGET")
    print("=" * 70)
    pos = int(train["target"].sum())
    rate = train["target"].mean()
    print(f"positives {pos} / {len(train)} = {rate:.4%}  imbalance 1:{(1 - rate) / rate:.1f}")
    print("\ndiagnosis x target (train-only column):")
    print(train.groupby(["diagnosis", "target"]).size().unstack(fill_value=0))
    print("\nbenign_malignant vs target crosstab:")
    print(pd.crosstab(train["benign_malignant"], train["target"]))
    out["target"] = {"positives": pos, "n": len(train), "pos_rate": float(rate)}

    print()
    print("=" * 70)
    print("3. PATIENT STRUCTURE (dossier hypothesis: disjoint patients)")
    print("=" * 70)
    tr_pat, te_pat = set(train.patient_id), set(test.patient_id)
    overlap = tr_pat & te_pat
    print(f"train patients {len(tr_pat)}, test patients {len(te_pat)}, overlap {len(overlap)}")
    ipp = train.groupby("patient_id").size()
    print(f"images per train patient: min {ipp.min()} median {ipp.median():.0f} "
          f"mean {ipp.mean():.1f} max {ipp.max()}")
    pat_pos = train.groupby("patient_id")["target"].sum()
    print(f"patients with >=1 positive: {(pat_pos > 0).sum()} / {len(pat_pos)} "
          f"({(pat_pos > 0).mean():.2%})")
    print(f"positives concentrated in patients: top-10 positive patients hold "
          f"{pat_pos.nlargest(10).sum()} of {pos} positives")
    out["patients"] = {
        "train": len(tr_pat), "test": len(te_pat), "overlap": len(overlap),
        "imgs_per_patient_mean": float(ipp.mean()), "imgs_per_patient_max": int(ipp.max()),
        "patients_with_positive": int((pat_pos > 0).sum()),
    }

    print()
    print("=" * 70)
    print("4. METADATA COLUMNS")
    print("=" * 70)
    for col in ["sex", "age_approx", "anatom_site_general_challenge"]:
        print(f"\n--- {col} ---")
        print(f"missing train {train[col].isna().mean():.3%}  test {test[col].isna().mean():.3%}")
        if col == "age_approx":
            print(train[col].describe().to_string())
            bins = pd.cut(train[col], bins=[0, 30, 45, 55, 65, 75, 100])
            print(train.groupby(bins, observed=True)["target"].agg(["size", "mean"]).to_string())
        else:
            g = train.groupby(col, dropna=False)["target"].agg(["size", "mean"])
            g["test_share"] = test[col].value_counts(dropna=False, normalize=True)
            print(g.to_string())
        unseen = set(test[col].dropna().unique()) - set(train[col].dropna().unique())
        if unseen:
            print(f"!! categories in test not seen in train: {unseen}")
    out["missing"] = {c: float(train[c].isna().mean()) for c in
                      ["sex", "age_approx", "anatom_site_general_challenge"]}

    print()
    print("=" * 70)
    print("5. IMAGE DIMENSIONS (scanner/site fingerprint check)")
    print("=" * 70)
    tr_sz = collect_sizes(train.image_name, "train")
    te_sz = collect_sizes(test.image_name, "test")
    tr = train.merge(tr_sz, on="image_name")
    tr["res"] = tr.width.astype(str) + "x" + tr.height.astype(str)
    te_sz["res"] = te_sz.width.astype(str) + "x" + te_sz.height.astype(str)
    top = tr.res.value_counts().head(12)
    g = tr.groupby("res")["target"].agg(["size", "mean"]).loc[top.index]
    g["test_share"] = te_sz.res.value_counts(normalize=True).reindex(top.index).fillna(0)
    g["train_share"] = top / len(tr)
    print("top-12 resolutions (train count, train positive rate, shares):")
    print(g.to_string())
    print(f"\ndistinct resolutions: train {tr.res.nunique()}  test {te_sz.res.nunique()}")
    unseen_res = set(te_sz.res) - set(tr.res)
    print(f"resolutions in test unseen in train: {len(unseen_res)} "
          f"covering {te_sz.res.isin(unseen_res).mean():.2%} of test rows")
    # how much signal is in resolution alone?
    from sklearn.metrics import roc_auc_score
    res_rate = tr.groupby("res")["target"].mean()
    print(f"in-sample AUC of resolution-group mean alone (OPTIMISTIC, no CV): "
          f"{roc_auc_score(tr.target, tr.res.map(res_rate)):.4f}")
    out["resolution"] = {
        "n_train": int(tr.res.nunique()), "n_test": int(te_sz.res.nunique()),
        "test_rows_with_unseen_res": float(te_sz.res.isin(unseen_res).mean()),
    }
    tr_sz.to_csv(COMP / "cache" / "train_sizes.csv", index=False)
    te_sz.to_csv(COMP / "cache" / "test_sizes.csv", index=False)

    print()
    print("=" * 70)
    print("6. VALIDATION STRATEGY CHECK")
    print("=" * 70)
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = np.zeros(len(train), dtype=int)
    for k, (_, va) in enumerate(sgkf.split(train, train.target, groups=train.patient_id)):
        folds[va] = k
    train["fold"] = folds
    print("fold sizes / positive counts (StratifiedGroupKFold on patient_id):")
    print(train.groupby("fold")["target"].agg(["size", "sum", "mean"]).to_string())
    leak = train.groupby("patient_id")["fold"].nunique().max()
    print(f"max distinct folds per patient (must be 1): {leak}")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    f2 = np.zeros(len(train), dtype=int)
    for k, (_, va) in enumerate(skf.split(train, train.target)):
        f2[va] = k
    print(f"plain StratifiedKFold would split {(train.groupby('patient_id')[[]].size().index.isin(pd.Series(f2).groupby(train.patient_id).nunique().pipe(lambda s: s[s > 1]).index)).sum()} "
          f"patients across folds -> leakage")
    train[["image_name", "patient_id", "target", "fold"]].to_csv(COMP / "cache" / "folds.csv", index=False)
    out["folds"] = {"scheme": "StratifiedGroupKFold(5, patient_id)",
                    "max_folds_per_patient": int(leak),
                    "per_fold_pos": train.groupby("fold")["target"].sum().tolist()}

    (COMP / "eda_summary.json").write_text(json.dumps(out, indent=2))
    print("\nwrote eda_summary.json, cache/folds.csv, cache/{train,test}_sizes.csv")


if __name__ == "__main__":
    main()
