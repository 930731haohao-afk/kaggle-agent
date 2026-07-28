"""
Simple LLM Classification with Traditional ML
Uses LightGBM + TF-IDF features
"""
import pandas as pd
import numpy as np
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("LLM Classification with Traditional ML (LightGBM + TF-IDF)")
print("="*80)

# Load data
print("\nLoading data...")
train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')

print(f"Train: {len(train)} samples")
print(f"Test: {len(test)} samples")

# Parse JSON fields
def parse_json_field(field_str):
    try:
        parsed = json.loads(field_str)
        if isinstance(parsed, list):
            return " ".join(parsed)
        return str(parsed)
    except:
        return str(field_str)

print("\nParsing JSON fields...")
train['prompt_text'] = train['prompt'].apply(parse_json_field)
train['response_a_text'] = train['response_a'].apply(parse_json_field)
train['response_b_text'] = train['response_b'].apply(parse_json_field)

test['prompt_text'] = test['prompt'].apply(parse_json_field)
test['response_a_text'] = test['response_a'].apply(parse_json_field)
test['response_b_text'] = test['response_b'].apply(parse_json_field)

# Create combined text features
print("\nCreating text features...")
train['combined_text'] = (
    "PROMPT: " + train['prompt_text'] +
    " RESPONSE_A: " + train['response_a_text'] +
    " RESPONSE_B: " + train['response_b_text']
)

test['combined_text'] = (
    "PROMPT: " + test['prompt_text'] +
    " RESPONSE_A: " + test['response_a_text'] +
    " RESPONSE_B: " + test['response_b_text']
)

# Create labels
train['label'] = train.apply(
    lambda x: 0 if x['winner_model_a'] == 1 else (1 if x['winner_model_b'] == 1 else 2),
    axis=1
)

print(f"\nLabel distribution:")
print(train['label'].value_counts().sort_index())

# TF-IDF Vectorization
print("\nCreating TF-IDF features...")
tfidf = TfidfVectorizer(
    max_features=10000,  # Top 10k features
    ngram_range=(1, 2),  # Unigrams and bigrams
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

X_train_tfidf = tfidf.fit_transform(train['combined_text'])
X_test_tfidf = tfidf.transform(test['combined_text'])

print(f"TF-IDF shape: {X_train_tfidf.shape}")

# Split for validation
print("\nSplitting train/validation (80/20)...")
X_train, X_val, y_train, y_val = train_test_split(
    X_train_tfidf,
    train['label'],
    test_size=0.2,
    random_state=42,
    stratify=train['label']
)

print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}")

# Train LightGBM
print("\n" + "="*80)
print("Training LightGBM")
print("="*80)

lgb_params = {
    'objective': 'multiclass',
    'num_class': 3,
    'metric': 'multi_logloss',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': 6,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'verbose': 1,
    'device': 'gpu',  # Use GPU
    'gpu_platform_id': 0,
    'gpu_device_id': 0,
}

train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

print("\nTraining...")
model = lgb.train(
    lgb_params,
    train_data,
    num_boost_round=500,
    valid_sets=[train_data, val_data],
    valid_names=['train', 'val'],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=50)
    ]
)

# Validation predictions
print("\n" + "="*80)
print("Validation Results")
print("="*80)

val_preds = model.predict(X_val, num_iteration=model.best_iteration)
val_loss = log_loss(y_val, val_preds)

print(f"\nValidation Log Loss: {val_loss:.5f}")
print(f"Best iteration: {model.best_iteration}")

# Test predictions
print("\nGenerating test predictions...")
test_preds = model.predict(X_test_tfidf, num_iteration=model.best_iteration)

# Create submission
print("\nCreating submission...")
submission = pd.DataFrame({
    'id': test['id'],
    'winner_model_a': test_preds[:, 0],
    'winner_model_b': test_preds[:, 1],
    'winner_tie': test_preds[:, 2]
})

submission.to_csv('../submissions/submission_lightgbm.csv', index=False)

print("\n" + "="*80)
print("Submission Preview")
print("="*80)
print(submission)
print(f"\nProbability sums: {submission[['winner_model_a', 'winner_model_b', 'winner_tie']].sum(axis=1).values}")

print("\n" + "="*80)
print("✓ Training Completed Successfully!")
print("="*80)
print(f"Validation Log Loss: {val_loss:.5f}")
print(f"Submission saved to: ../submissions/submission_lightgbm.csv")
