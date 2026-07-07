"""Feature engineering + LightGBM baseline for playground-series-s4e1 (Bank Churn)."""
import pandas as pd
import numpy as np
import os
import json
import time
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import LabelEncoder

data_dir = "competitions/playground-series-s4e1/data"
script_dir = "competitions/playground-series-s4e1/scripts"
exp_file = "competitions/playground-series-s4e1/experiments.json"

train = pd.read_csv(os.path.join(data_dir, "train.csv"))
test = pd.read_csv(os.path.join(data_dir, "test.csv"))

print("=" * 70)
print("FEATURE ENGINEERING + BASELINE")
print("=" * 70)

# ============================================================
# 1. FEATURE ENGINEERING
# ============================================================
target_col = 'Exited'
drop_cols = ['id', target_col, 'CustomerId', 'Surname']

def engineer_features(df, surname_target_map=None, is_train=True):
    """Build features for bank churn prediction."""
    out = df.copy()

    # --- Encode Gender ---
    out['is_female'] = (out['Gender'] == 'Female').astype(int)

    # --- Geography dummies ---
    for geo in ['Germany', 'Spain']:
        out[f'geo_{geo}'] = (out['Geography'] == geo).astype(int)

    # --- Age features ---
    out['age_sq'] = out['Age'] ** 2
    out['age_decade'] = out['Age'] // 10
    out['age_40_60'] = ((out['Age'] >= 40) & (out['Age'] <= 60)).astype(int)  # peak churn age

    # --- Balance features ---
    out['has_balance'] = (out['Balance'] > 0).astype(int)
    out['balance_salary_ratio'] = out['Balance'] / out['EstimatedSalary'].clip(lower=1)

    # --- NumOfProducts features ---
    out['products_gt2'] = (out['NumOfProducts'] > 2).astype(int)
    out['products_eq1'] = (out['NumOfProducts'] == 1).astype(int)

    # --- Interaction features ---
    out['age_x_products'] = out['Age'] * out['NumOfProducts']
    out['age_x_balance'] = out['Age'] * out['Balance']
    out['age_x_active'] = out['Age'] * out['IsActiveMember']
    out['geo_gender'] = out['Geography'].astype(str) + '_' + out['Gender'].astype(str)
    out['active_x_balance'] = out['IsActiveMember'] * out['Balance']
    out['active_x_products'] = out['IsActiveMember'] * out['NumOfProducts']
    out['credit_age_ratio'] = out['CreditScore'] / out['Age'].clip(lower=1)

    # --- Surname target encoding ---
    if surname_target_map is not None:
        out['surname_target_enc'] = out['Surname'].map(surname_target_map)
        global_mean = surname_target_map.get('__global_mean__', 0.5)
        out['surname_target_enc'] = out['surname_target_enc'].fillna(global_mean)

    return out


# Build surname target encoding with smoothing (avoid leakage using OOF)
print("Building surname target encoding (OOF)...")
n_folds_te = 5
skf_te = StratifiedKFold(n_splits=n_folds_te, shuffle=True, random_state=99)
train['surname_target_enc'] = np.nan
global_mean = train[target_col].mean()

for fold, (tr_idx, val_idx) in enumerate(skf_te.split(train, train[target_col])):
    tr = train.iloc[tr_idx]
    surname_stats = tr.groupby('Surname')[target_col].agg(['mean', 'count'])
    smoothing = 20  # smoothing parameter
    surname_stats['smoothed'] = (surname_stats['mean'] * surname_stats['count'] + global_mean * smoothing) / (surname_stats['count'] + smoothing)
    mapping = surname_stats['smoothed'].to_dict()
    train.loc[val_idx, 'surname_target_enc'] = train.loc[val_idx, 'Surname'].map(mapping).fillna(global_mean)

# Full mapping for test
surname_stats_full = train.groupby('Surname')[target_col].agg(['mean', 'count'])
surname_stats_full['smoothed'] = (surname_stats_full['mean'] * surname_stats_full['count'] + global_mean * smoothing) / (surname_stats_full['count'] + smoothing)
surname_map_full = surname_stats_full['smoothed'].to_dict()
surname_map_full['__global_mean__'] = global_mean

# Engineer features
train_fe = engineer_features(train, surname_target_map=surname_map_full, is_train=True)
test_fe = engineer_features(test, surname_target_map=surname_map_full, is_train=False)

# Label encode remaining string columns
label_encoders = {}
string_cols = [c for c in train_fe.select_dtypes(include='object').columns if c not in drop_cols]
for col in string_cols:
    le = LabelEncoder()
    combined = pd.concat([train_fe[col].astype(str), test_fe[col].astype(str)])
    le.fit(combined)
    train_fe[col] = le.transform(train_fe[col].astype(str))
    test_fe[col] = le.transform(test_fe[col].astype(str))
    label_encoders[col] = le

# Define feature columns
feature_cols = [c for c in train_fe.columns if c not in drop_cols]
print(f"\nFeatures ({len(feature_cols)}): {feature_cols}")

X = train_fe[feature_cols].values
y = train_fe[target_col].values
X_test = test_fe[feature_cols].values

print(f"X shape: {X.shape}, y shape: {y.shape}")
print(f"X_test shape: {X_test.shape}")

# ============================================================
# 2. MAJORITY CLASS BASELINE
# ============================================================
print("\n" + "=" * 70)
print("2. MAJORITY CLASS BASELINE")
print("=" * 70)

majority = int(pd.Series(y).mode()[0])
maj_acc = accuracy_score(y, [majority] * len(y))
# AUC for majority would be 0.5
print(f"Majority class: {majority}, Accuracy: {maj_acc:.5f}, AUC: 0.50000")

# ============================================================
# 3. LIGHTGBM 5-FOLD CV
# ============================================================
print("\n" + "=" * 70)
print("3. LIGHTGBM 5-FOLD CV")
print("=" * 70)

lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.05,
    'num_leaves': 63,
    'max_depth': -1,
    'min_child_samples': 30,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'verbose': -1,
    'n_jobs': -1,
    'seed': 42,
    'is_unbalance': True,
}

n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))
fold_scores = []
best_iterations = []

t0 = time.time()
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=feature_cols)
    dval = lgb.Dataset(X_val, label=y_val, feature_name=feature_cols, reference=dtrain)

    model = lgb.train(
        lgb_params, dtrain,
        num_boost_round=2000,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )

    val_pred = model.predict(X_val)
    oof_preds[val_idx] = val_pred
    test_preds += model.predict(X_test) / n_folds

    fold_auc = roc_auc_score(y_val, val_pred)
    fold_scores.append(fold_auc)
    best_iterations.append(model.best_iteration)
    print(f"  Fold {fold+1}: AUC={fold_auc:.5f}, best_iter={model.best_iteration}")

train_time = time.time() - t0

oof_auc = roc_auc_score(y, oof_preds)
print(f"\nOOF AUC: {oof_auc:.5f}")
print(f"Mean fold AUC: {np.mean(fold_scores):.5f} +/- {np.std(fold_scores):.5f}")
print(f"Mean best iteration: {np.mean(best_iterations):.0f}")
print(f"Training time: {train_time:.1f}s")

# Feature importance
importance = model.feature_importance(importance_type='gain')
feat_imp = sorted(zip(feature_cols, importance), key=lambda x: x[1], reverse=True)
print("\nTop 15 features by gain:")
for name, imp in feat_imp[:15]:
    print(f"  {name:<35} {imp:.1f}")

# ============================================================
# 4. LOG EXPERIMENTS
# ============================================================
print("\n" + "=" * 70)
print("4. EXPERIMENT LOG")
print("=" * 70)

with open(exp_file, 'r') as f:
    exp_data = json.load(f)

exp_data['experiments'].append({
    'id': 1,
    'name': 'Majority Class Baseline',
    'model': 'majority',
    'cv_score': 0.5,
    'cv_std': 0.0,
    'notes': f'AUC=0.5 (random), Accuracy={maj_acc:.5f}'
})

exp_data['experiments'].append({
    'id': 2,
    'name': 'LightGBM Baseline',
    'model': 'lightgbm',
    'params': lgb_params,
    'n_features': len(feature_cols),
    'cv_score': round(oof_auc, 5),
    'cv_std': round(np.std(fold_scores), 5),
    'cv_folds': [round(s, 5) for s in fold_scores],
    'mean_best_iteration': int(np.mean(best_iterations)),
    'training_time_sec': round(train_time, 1),
    'notes': f'{len(feature_cols)} features, stratified 5-fold, surname target encoding, is_unbalance=True'
})

with open(exp_file, 'w') as f:
    json.dump(exp_data, f, indent=2)
print("Experiments logged.")

# ============================================================
# 5. GENERATE SUBMISSION
# ============================================================
print("\n" + "=" * 70)
print("5. SUBMISSION")
print("=" * 70)

sub_dir = "competitions/playground-series-s4e1/submissions"
os.makedirs(sub_dir, exist_ok=True)

from datetime import datetime as dt
timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
sub_file = os.path.join(sub_dir, f"lgbm_baseline_{timestamp}.csv")

submission = pd.DataFrame({
    'id': test['id'],
    'Exited': test_preds  # probabilities for AUC
})

submission.to_csv(sub_file, index=False)
print(f"Submission saved: {sub_file}")
print(f"Submission shape: {submission.shape}")
print(f"Prediction stats: mean={test_preds.mean():.4f}, min={test_preds.min():.4f}, max={test_preds.max():.4f}")

# Validate
sample_sub = pd.read_csv(os.path.join(data_dir, "sample_submission.csv"))
assert submission.shape == sample_sub.shape, f"Shape mismatch: {submission.shape} vs {sample_sub.shape}"
assert list(submission.columns) == list(sample_sub.columns), f"Column mismatch"
print("Submission validated OK!")

print(f"\n{'='*70}")
print(f"SUMMARY: Majority AUC=0.5, LightGBM OOF AUC={oof_auc:.5f}")
print(f"{'='*70}")
