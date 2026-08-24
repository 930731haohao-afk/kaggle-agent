"""Stage 1 EDA for siim-isic-melanoma-classification (mle-bench prepared split)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

COMP = Path(__file__).resolve().parents[1]
DATA = COMP / "data"

pd.set_option("display.width", 200)


def main() -> None:
    tr = pd.read_csv(DATA / "train.csv")
    te = pd.read_csv(DATA / "test.csv")
    ss = pd.read_csv(DATA / "sample_submission.csv")

    print("=" * 70)
    print(f"train {tr.shape}  test {te.shape}  sample_sub {ss.shape}")
    print("train cols:", list(tr.columns))
    print("test  cols:", list(te.columns))
    print("sample_sub head:\n", ss.head(3))

    print("\n--- TARGET ---")
    print(tr["target"].value_counts())
    pos = tr["target"].mean()
    print(f"positive rate = {pos:.5f}  imbalance 1:{(1 - pos) / pos:.1f}")
    print("\nbenign_malignant x target:\n", pd.crosstab(tr["benign_malignant"], tr["target"]))
    print("\ndiagnosis value counts:\n", tr["diagnosis"].value_counts())
    print("\ndiagnosis x target:\n", pd.crosstab(tr["diagnosis"], tr["target"]))

    print("\n--- PATIENTS (group structure) ---")
    print(f"train unique patients: {tr.patient_id.nunique()}  test: {te.patient_id.nunique()}")
    overlap = set(tr.patient_id) & set(te.patient_id)
    print(f"PATIENT OVERLAP train/test: {len(overlap)}  <-- decides group split necessity")
    ipp = tr.groupby("patient_id").size()
    print("images per patient train: mean %.1f median %d min %d max %d"
          % (ipp.mean(), ipp.median(), ipp.min(), ipp.max()))
    ippt = te.groupby("patient_id").size()
    print("images per patient test : mean %.1f median %d min %d max %d"
          % (ippt.mean(), ippt.median(), ippt.min(), ippt.max()))
    # patient-level positive concentration
    ppat = tr.groupby("patient_id")["target"].agg(["sum", "size"])
    print(f"patients with >=1 positive: {(ppat['sum'] > 0).sum()} / {len(ppat)}")
    print("target rate within positive-patients: %.4f"
          % (ppat.loc[ppat["sum"] > 0, "sum"].sum() / ppat.loc[ppat["sum"] > 0, "size"].sum()))

    print("\n--- IMAGE NAME OVERLAP ---")
    print("image_name dup in train:", tr.image_name.duplicated().sum())
    print("train/test image_name overlap:", len(set(tr.image_name) & set(te.image_name)))
    print("sample_sub ids == test ids (order):", (ss.image_name.values == te.image_name.values).all())

    print("\n--- METADATA COLUMNS ---")
    for c in ["sex", "age_approx", "anatom_site_general_challenge"]:
        print(f"\n[{c}] train nulls={tr[c].isna().sum()} test nulls={te[c].isna().sum()}")
        vc_tr = tr[c].value_counts(dropna=False, normalize=True).head(12)
        vc_te = te[c].value_counts(dropna=False, normalize=True).head(12)
        comp = pd.concat([vc_tr.rename("train"), vc_te.rename("test")], axis=1)
        print(comp)
        if c != "age_approx":
            print("target rate per category:")
            print(tr.groupby(c, dropna=False)["target"].agg(["mean", "size"]))
    print("\nage_approx target rate by bin:")
    print(tr.groupby(pd.cut(tr.age_approx, bins=[0, 30, 40, 50, 60, 70, 100]),
                     observed=False)["target"].agg(["mean", "size"]))

    print("\n--- IMAGE FILE SANITY ---")
    jt = DATA / "jpeg" / "train"
    jte = DATA / "jpeg" / "test"
    missing_tr = [n for n in tr.image_name.values[:2000] if not (jt / f"{n}.jpg").exists()]
    missing_te = [n for n in te.image_name.values if not (jte / f"{n}.jpg").exists()]
    print(f"missing jpg (first 2000 train): {len(missing_tr)}   missing test jpg: {len(missing_te)}")
    rng = np.random.default_rng(42)
    sample = rng.choice(tr.image_name.values, 40, replace=False)
    sizes = []
    for n in sample:
        with Image.open(jt / f"{n}.jpg") as im:
            sizes.append(im.size)
    sizes_arr = np.array(sizes)
    print("sampled 40 train jpg sizes: unique =", np.unique(sizes_arr, axis=0)[:10])
    print("w: min %d max %d | h: min %d max %d"
          % (sizes_arr[:, 0].min(), sizes_arr[:, 0].max(), sizes_arr[:, 1].min(), sizes_arr[:, 1].max()))
    sample_te = rng.choice(te.image_name.values, 20, replace=False)
    st = []
    for n in sample_te:
        with Image.open(jte / f"{n}.jpg") as im:
            st.append(im.size)
    st = np.array(st)
    print("test  w: min %d max %d | h: min %d max %d"
          % (st[:, 0].min(), st[:, 0].max(), st[:, 1].min(), st[:, 1].max()))

    # summary artifact
    out = {
        "n_train": int(len(tr)), "n_test": int(len(te)),
        "pos_rate": float(pos), "n_pos": int(tr.target.sum()),
        "patients_train": int(tr.patient_id.nunique()),
        "patients_test": int(te.patient_id.nunique()),
        "patient_overlap": int(len(overlap)),
        "diagnosis_in_test": False,
    }
    (COMP / "eda_summary.json").write_text(json.dumps(out, indent=2))
    print("\nwrote eda_summary.json:", out)


if __name__ == "__main__":
    main()
