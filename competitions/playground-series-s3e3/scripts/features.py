"""Feature engineering for playground-series-s3e3 (Employee Attrition).

Fit-on-train, transform-both. No target leakage: categorical encodings are
frequency-based (computed on train only) plus label encoding for trees.
"""
import numpy as np
import pandas as pd

CAT_COLS = ["BusinessTravel", "Department", "EducationField", "Gender",
            "JobRole", "MaritalStatus", "OverTime"]
DROP_COLS = ["EmployeeCount", "StandardHours", "Over18"]  # constant, zero-variance

SATISFACTION_COLS = ["EnvironmentSatisfaction", "JobSatisfaction",
                     "RelationshipSatisfaction", "WorkLifeBalance", "JobInvolvement"]


def _freq_encode(train_col: pd.Series, col: pd.Series) -> pd.Series:
    freq = train_col.value_counts(normalize=True)
    return col.map(freq).fillna(0.0)


def fit_encoders(train: pd.DataFrame) -> dict:
    """Compute train-only statistics needed for encoding (label maps, freq maps)."""
    enc = {"label_maps": {}, "freq_maps": {}}
    for c in CAT_COLS:
        cats = sorted(train[c].unique())
        enc["label_maps"][c] = {v: i for i, v in enumerate(cats)}
        enc["freq_maps"][c] = train[c].value_counts(normalize=True).to_dict()
    return enc


def build_features(df: pd.DataFrame, enc: dict) -> pd.DataFrame:
    out = df.copy()
    out = out.drop(columns=[c for c in DROP_COLS if c in out.columns], errors="ignore")

    # Label + frequency encode categoricals (unseen category -> -1 / 0.0)
    for c in CAT_COLS:
        lmap = enc["label_maps"][c]
        fmap = enc["freq_maps"][c]
        out[f"{c}_le"] = out[c].map(lmap).fillna(-1).astype(int)
        out[f"{c}_freq"] = out[c].map(fmap).fillna(0.0)
    out = out.drop(columns=CAT_COLS)

    # Tenure ratios (guard div-by-zero)
    out["role_tenure_ratio"] = out["YearsInCurrentRole"] / out["YearsAtCompany"].replace(0, np.nan)
    out["mgr_tenure_ratio"] = out["YearsWithCurrManager"] / out["YearsAtCompany"].replace(0, np.nan)
    out["promo_ratio"] = out["YearsSinceLastPromotion"] / out["YearsAtCompany"].replace(0, np.nan)
    out["company_tenure_ratio"] = out["YearsAtCompany"] / out["TotalWorkingYears"].replace(0, np.nan)
    for c in ["role_tenure_ratio", "mgr_tenure_ratio", "promo_ratio", "company_tenure_ratio"]:
        out[c] = out[c].fillna(0.0)

    # Income features
    out["income_per_joblevel"] = out["MonthlyIncome"] / out["JobLevel"].replace(0, np.nan)
    out["income_per_year_worked"] = out["MonthlyIncome"] / (out["TotalWorkingYears"] + 1)
    out["income_per_joblevel"] = out["income_per_joblevel"].fillna(out["MonthlyIncome"])

    # Satisfaction composite
    out["satisfaction_avg"] = out[SATISFACTION_COLS].mean(axis=1)
    out["satisfaction_min"] = out[SATISFACTION_COLS].min(axis=1)

    # OverTime x JobLevel interaction (OverTime already label/freq encoded above)
    out["overtime_joblevel"] = out["OverTime_le"] * out["JobLevel"]

    # Age-related
    out["age_at_join"] = out["Age"] - out["TotalWorkingYears"]
    out["companies_per_year"] = out["NumCompaniesWorked"] / (out["Age"] - 18).clip(lower=1)

    return out


def feature_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in ("id", "Attrition")]
