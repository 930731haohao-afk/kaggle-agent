"""Feature engineering + LightGBM baseline for playground-series-s4e11 (Depression)."""
import pandas as pd
import numpy as np
import os
import json
import time
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

data_dir = "competitions/playground-series-s4e11/data"
script_dir = "competitions/playground-series-s4e11/scripts"
exp_file = "competitions/playground-series-s4e11/experiments.json"

train = pd.read_csv(os.path.join(data_dir, "train.csv"))
test = pd.read_csv(os.path.join(data_dir, "test.csv"))

print("=" * 70)
print("FEATURE ENGINEERING + BASELINE")
print("=" * 70)

# ============================================================
# 1. FEATURE ENGINEERING
# ============================================================
target_col = 'Depression'
drop_cols = ['id', target_col]

def engineer_features(df, is_train=True):
    """Build features for depression prediction."""
    out = df.copy()

    # --- Clean Sleep Duration ---
    sleep_map = {
        'Less than 5 hours': 'less_than_5',
        '5-6 hours': '5_6',
        '7-8 hours': '7_8',
        'More than 8 hours': 'more_than_8',
    }
    out['Sleep Duration'] = out['Sleep Duration'].map(sleep_map).fillna('other')

    # --- Clean Dietary Habits ---
    diet_map = {
        'Healthy': 'Healthy',
        'Moderate': 'Moderate',
        'Unhealthy': 'Unhealthy',
    }
    out['Dietary Habits'] = out['Dietary Habits'].map(diet_map).fillna('Other')

    # --- Encode Working Professional or Student ---
    out['is_student'] = (out['Working Professional or Student'] == 'Student').astype(int)

    # --- Fill structured NaN with 0 (absent feature) ---
    # For professionals, academic features are NA -> fill with 0
    # For students, work features are NA -> fill with 0
    for col in ['Academic Pressure', 'Work Pressure', 'Study Satisfaction', 'Job Satisfaction']:
        out[col] = out[col].fillna(0)
    out['CGPA'] = out['CGPA'].fillna(0)

    # --- Encode suicidal thoughts ---
    out['suicidal_thoughts'] = (out['Have you ever had suicidal thoughts ?'] == 'Yes').astype(int)

    # --- Encode family history ---
    out['family_history'] = (out['Family History of Mental Illness'] == 'Yes').astype(int)

    # --- Encode Gender ---
    out['is_male'] = (out['Gender'] == 'Male').astype(int)

    # --- Financial Stress: fill NaN ---
    out['Financial Stress'] = out['Financial Stress'].fillna(out['Financial Stress'].median())

    # --- Age features ---
    out['age_decade'] = out['Age'] // 10
    out['is_young'] = (out['Age'] < 30).astype(int)

    # --- Interaction features ---
    out['pressure'] = out['Academic Pressure'] + out['Work Pressure']  # combined pressure
    out['satisfaction'] = out['Study Satisfaction'] + out['Job Satisfaction']  # combined satisfaction
    out['stress_x_pressure'] = out['Financial Stress'] * out['pressure']
    out['suicidal_x_student'] = out['suicidal_thoughts'] * out['is_student']
    out['age_x_pressure'] = out['Age'] * out['pressure']

    # --- Label encode high-cardinality categoricals ---
    # City, Profession, Degree, Name -> label encode
    # We'll handle Name by dropping (not predictive from EDA)
    out = out.drop(columns=['Name'], errors='ignore')

    return out


train_fe = engineer_features(train, is_train=True)
test_fe = engineer_features(test, is_train=False)

# Label encode remaining string columns
label_encoders = {}
string_cols = train_fe.select_dtypes(include='object').columns.tolist()
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
print(f"Majority class: {majority}, Accuracy: {maj_acc:.5f}")

# ============================================================
# 3. LIGHTGBM BASELINE
# ============================================================
print("\n" + "=" * 70)
print("3. LIGHTGBM 5-FOLD CV")
print("=" * 70)

lgb_params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
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

    fold_acc = accuracy_score(y_val, (val_pred >= 0.5).astype(int))
    fold_scores.append(fold_acc)
    best_iterations.append(model.best_iteration)
    print(f"  Fold {fold+1}: Accuracy={fold_acc:.5f}, best_iter={model.best_iteration}")

train_time = time.time() - t0

oof_acc = accuracy_score(y, (oof_preds >= 0.5).astype(int))
print(f"\nOOF Accuracy: {oof_acc:.5f}")
print(f"Mean fold accuracy: {np.mean(fold_scores):.5f} +/- {np.std(fold_scores):.5f}")
print(f"Mean best iteration: {np.mean(best_iterations):.0f}")
print(f"Training time: {train_time:.1f}s")

# Feature importance
importance = model.feature_importance(importance_type='gain')
feat_imp = sorted(zip(feature_cols, importance), key=lambda x: x[1], reverse=True)
print("\nTop 15 features by gain:")
for name, imp in feat_imp[:15]:
    print(f"  {name:<35} {imp:.1f}")

# ============================================================
# 4. THRESHOLD OPTIMIZATION
# ============================================================
print("\n" + "=" * 70)
print("4. THRESHOLD OPTIMIZATION")
print("=" * 70)

best_thresh = 0.5
best_acc = oof_acc
for thresh in np.arange(0.1, 0.9, 0.01):
    acc = accuracy_score(y, (oof_preds >= thresh).astype(int))
    if acc > best_acc:
        best_acc = acc
        best_thresh = thresh

print(f"Best threshold: {best_thresh:.2f}, Accuracy: {best_acc:.5f}")
print(f"Improvement over 0.5: +{(best_acc - oof_acc)*100:.3f}%")

# ============================================================
# 5. LOG EXPERIMENTS
# ============================================================
print("\n" + "=" * 70)
print("5. EXPERIMENT LOG")
print("=" * 70)

with open(exp_file, 'r') as f:
    exp_data = json.load(f)

# Majority baseline
exp_data['experiments'].append({
    'id': 1,
    'name': 'Majority Class Baseline',
    'model': 'majority',
    'cv_score': round(maj_acc, 5),
    'cv_std': 0.0,
    'notes': f'Predict class {majority} for all'
})

# LightGBM baseline
exp_data['experiments'].append({
    'id': 2,
    'name': 'LightGBM Baseline',
    'model': 'lightgbm',
    'params': lgb_params,
    'n_features': len(feature_cols),
    'cv_score': round(oof_acc, 5),
    'cv_std': round(np.std(fold_scores), 5),
    'cv_folds': fold_scores,
    'best_threshold': round(best_thresh, 2),
    'best_threshold_acc': round(best_acc, 5),
    'mean_best_iteration': int(np.mean(best_iterations)),
    'training_time_sec': round(train_time, 1),
    'notes': f'{len(feature_cols)} features, stratified 5-fold, is_unbalance=True'
})

with open(exp_file, 'w') as f:
    json.dump(exp_data, f, indent=2)
print("Experiments logged.")

# ============================================================
# 6. GENERATE SUBMISSION
# ============================================================
print("\n" + "=" * 70)
print("6. SUBMISSION")
print("=" * 70)

sub_dir = "competitions/playground-series-s4e11/submissions"
os.makedirs(sub_dir, exist_ok=True)

from datetime import datetime as dt
timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
sub_file = os.path.join(sub_dir, f"lgbm_baseline_{timestamp}.csv")

submission = pd.DataFrame({
    'id': test['id'],
    'Depression': (test_preds >= best_thresh).astype(int)
})

submission.to_csv(sub_file, index=False)
print(f"Submission saved: {sub_file}")
print(f"Submission shape: {submission.shape}")
print(f"Prediction distribution: {submission['Depression'].value_counts().to_dict()}")

# Validate
sample_sub = pd.read_csv(os.path.join(data_dir, "sample_submission.csv"))
assert submission.shape == sample_sub.shape, f"Shape mismatch: {submission.shape} vs {sample_sub.shape}"
assert list(submission.columns) == list(sample_sub.columns), f"Column mismatch"
print("Submission validated OK!")

print(f"\n{'='*70}")
print(f"SUMMARY: Majority={maj_acc:.5f}, LightGBM={oof_acc:.5f} (thresh={best_thresh:.2f}→{best_acc:.5f})")
print(f"{'='*70}")
