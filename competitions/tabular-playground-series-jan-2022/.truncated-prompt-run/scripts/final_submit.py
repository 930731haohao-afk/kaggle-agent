"""Final model for tabular-playground-series-jan-2022.

Recipe (validated on expanding-year CV, primary fold 2018):
  ridge  — log1p target; dummies country/store/product/dow, eoy(1..18) x country,
           Fourier k=1..6 x product, holiday-name x offset(-5..+10) dummies,
           log_gdp + linear year_c; alpha=1.0        fold2018 SMAPE 4.1855
  lgb_gdp — LightGBM on log(num_sold/gdp_pc)         fold2018 SMAPE 5.795
  blend 0.95*ridge + 0.05*lgb_gdp, round to int      fold2018 SMAPE ~4.179
Trains on all 2015-2018, predicts 2019 test, writes submission.csv.
"""
import importlib.util

import numpy as np
import pandas as pd
import holidays as hol
import lightgbm as lgb
from sklearn.linear_model import Ridge

COMP = "competitions/tabular-playground-series-jan-2022"
OFF = range(-5, 11)
W_RIDGE = 0.95
SEED = 42

_spec = importlib.util.spec_from_file_location(
    "experiment_log", ".claude/skills/kaggle-agent/assets/utils/experiment_log.py")
experiment_log = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(experiment_log)

tr = pd.read_parquet(f"{COMP}/train_feat.parquet")
te = pd.read_parquet(f"{COMP}/test_feat.parquet")

CALS = {
    "Finland": hol.Finland(years=range(2014, 2021)),
    "Norway": hol.Norway(years=range(2014, 2021)),
    "Sweden": hol.Sweden(years=range(2014, 2021)),
}
hol_maps = {}
for c, cal in CALS.items():
    m = {}
    for d, name in cal.items():
        nm = name.split(";")[0].strip()
        for o in OFF:
            m[(pd.Timestamp(d) + pd.Timedelta(days=o), o)] = nm
    hol_maps[c] = m
ALL_NAMES = sorted({nm.split(";")[0].strip() for cal in CALS.values() for nm in cal.values()})


def design(df: pd.DataFrame) -> pd.DataFrame:
    cols = {}
    base = pd.get_dummies(df[["country", "store", "product", "dow"]].astype(str))
    for v in range(1, 19):
        for c in df.country.cat.categories:
            cols[f"eoy_{v}_{c}"] = ((df.eoy_idx == v) & (df.country == c)).astype(int).values
    for k in range(1, 7):
        s = np.sin(2 * np.pi * k * df.doy / 365.0)
        co = np.cos(2 * np.pi * k * df.doy / 365.0)
        for p in df["product"].cat.categories:
            m = (df["product"] == p).astype(int).values
            cols[f"sin{k}_{p}"] = s.values * m
            cols[f"cos{k}_{p}"] = co.values * m
    for nm in ALL_NAMES:
        for o in OFF:
            cols[f"hol_{nm}_{o}"] = np.zeros(len(df))
    for c, hmap in hol_maps.items():
        idx = np.where((df.country == c).values)[0]
        for i, dt in zip(idx, df.date.iloc[idx]):
            for o in OFF:
                nm = hmap.get((dt, o))
                if nm is not None:
                    cols[f"hol_{nm}_{o}"][i] = 1.0
    cols["log_gdp"] = df.log_gdp.values
    cols["year_c"] = (df.year - 2015).values
    return pd.concat([base, pd.DataFrame(cols, index=df.index)], axis=1).astype(float)


# --- ridge on full train ---
Xtr, Xte = design(tr), design(te)
Xte = Xte.reindex(columns=Xtr.columns, fill_value=0.0)
ridge = Ridge(alpha=1.0)
ridge.fit(Xtr, np.log1p(tr.num_sold))
pred_ridge = np.clip(np.expm1(ridge.predict(Xte)), 1, None)

# --- lgb_gdp on full train (fixed rounds ~ CV best iters 90-229, use 180) ---
FEATS = ["country", "store", "product", "year", "month", "day", "dow", "doy",
         "is_weekend", "sin1", "cos1", "sin2", "cos2", "sin3", "cos3", "sin4", "cos4",
         "is_holiday", "holiday_win2", "eoy_idx"]
LGB_PARAMS = dict(objective="regression", learning_rate=0.05, num_leaves=31,
                  min_child_samples=20, feature_fraction=0.9, bagging_fraction=0.9,
                  bagging_freq=1, lambda_l2=1.0, num_threads=10, deterministic=True,
                  force_row_wise=True, seed=SEED, verbosity=-1)
bst = lgb.train(LGB_PARAMS, lgb.Dataset(tr[FEATS], np.log(tr.num_sold / tr.gdp_pc)),
                num_boost_round=180)
pred_lgb = np.clip(np.exp(bst.predict(te[FEATS])) * te.gdp_pc.values, 1, None)

# --- blend + round ---
pred = np.round(W_RIDGE * pred_ridge + (1 - W_RIDGE) * pred_lgb).astype(int)

# --- format & validate ---
sample = pd.read_csv(f"{COMP}/data/sample_submission.csv")
sub = pd.DataFrame({"row_id": te.row_id.values, "num_sold": pred})
sub = sample[["row_id"]].merge(sub, on="row_id")
assert sub.shape == sample.shape and list(sub.columns) == list(sample.columns)
assert (sub.row_id.values == sample.row_id.values).all()
assert sub.num_sold.isnull().sum() == 0 and (sub.num_sold > 0).all()
sub.to_csv(f"{COMP}/submission.csv", index=False)
sub.to_csv(f"{COMP}/submissions/submission_ridge95_lgb05_cv4.179_20260728.csv", index=False)

print("submission rows:", len(sub))
print("pred stats:", sub.num_sold.describe().round(1).to_dict())
print("train 2018 mean:", round(tr[tr.year == 2018].num_sold.mean(), 1),
      "| pred 2019 mean:", round(sub.num_sold.mean(), 1))

experiment_log.log_experiment_v2(
    COMP, model="blend_final(ridge_v4*0.95+lgb_gdp*0.05, rounded)",
    metric="SMAPE", direction="minimize", score=4.1793,
    cv={"strategy": "expanding-year (val 2016/2017/2018), primary fold 2018",
        "scores": [5.0154, 4.8401, 4.1793],
        "mean": 4.678},
    ensemble={"models": ["ridge_v4", "lgb_gdp"], "weights": [0.95, 0.05],
              "search": "grid on fold-2018 OOF"},
    postprocess=["clip>=1", "round to int"],
    submission="submissions/submission_ridge95_lgb05_cv4.179_20260728.csv",
    notes="final_submit.py: retrained on all 2015-2018; ridge holiday-name x offset(-5..10) "
          "design + log_gdp + year_c; World Bank GDP pc incl. 2019")
print("logged final experiment")
