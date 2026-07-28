# Linking Writing Processes to Writing Quality — Competition Status

## Competition Info
- **URL**: https://www.kaggle.com/c/linking-writing-processes-to-writing-quality
- **Problem**: Predict essay quality scores (0.5–6.0) from keystroke log data
- **Metric**: RMSE (minimize)
- **Type**: Code competition (notebook submission required)
- **Data**: 2,471 train essays (8.4M keystroke events), 3 test essays (placeholder)
- **Key constraint**: Text content anonymized — only writing process signals available

## Current Best Score
| CV RMSE | Public LB |
|---------|-----------|
| 0.648   | — (code competition, CSV rejected) |

## Pipeline Summary

### Features (66 total)
Aggregated from ~3,400 raw keystroke events per essay into essay-level features:
- **Length** (3): word count, cursor position, event count — strongest signals (corr 0.59–0.65)
- **Speed/IKI** (9): mean/median/P10/P25/P75/P90/std/IQR/skew of inter-key intervals
- **Burst** (6): count, mean/median/max/std burst length, words per burst
- **Revision** (5): backspace count/rate, remove count/rate, net efficiency
- **Pauses** (6): short/long pause counts/rates, total/mean pause duration, pause-to-active ratio
- **Activity** (6): input/remove/nonproduction counts and rates
- **Navigation** (4): arrow/click counts and rates
- **Special keys** (5): space/enter/shift/punctuation counts and rates
- **Writing phases** (16): events, input rate, IKI mean, word count delta per time quartile
- **Structure** (1): words per sentence
- **Productivity** (4): WPM, events/min, total time, action time stats
- Dropped: `n_paragraphs`, `approx_sentences` (zero importance)

### Models
All trained with 5-fold KFold, strong regularization for small dataset (2,471 samples).

| Rank | Model | CV RMSE |
|------|-------|---------|
| 1 | Ensemble (LGB=0.3, XGB=0.2, Cat=0.5) | 0.648 |
| 2 | Ensemble (equal avg) | 0.649 |
| 3 | CatBoost | 0.650 |
| 4 | LightGBM-tuned | 0.651 |
| 5 | XGBoost-tuned | 0.652 |
| 6 | LightGBM-default | 0.667 |
| 7 | Ridge | 0.685 |
| 8 | Mean predictor | 1.025 |

### Submission
- **Status**: Code competition — CSV submission rejected (400 error). Requires Kaggle notebook.
- **Submission file**: `submissions/submission_ensemble3_cv0.648_20260222_123217.csv` (generated locally)

## Key Observations
1. **Essay length dominates** — word count alone has 0.636 correlation with score
2. **Typing speed is #2 signal** — faster typists (lower IKI) score higher (-0.587 corr)
3. **Burst fluency captures writing skill** — mean burst length has 0.464 correlation
4. **Writing time is fixed** (~29 min for all essays) — productivity within the window matters, not duration
5. **Small dataset** (2,471) requires strong regularization — all GBMs used low learning rate (0.03), high reg
6. **All GBMs within 0.002** of each other — features matter more than model choice

## Potential Improvements
- **Kaggle notebook** — Convert pipeline to notebook format for actual submission
- **More granular features** — N-gram keystroke patterns, transition matrices between event types
- **Text reconstruction signals** — Word length distribution, sentence length patterns from space/enter
- **Revision sophistication** — Distinguish local corrections (1-2 backspace) from large deletions
- **Multi-seed ensembling** — Average across multiple random seeds for stability
- **Optuna tuning** — Systematic hyperparameter search

## Files
```
competitions/linking-writing-processes-to-writing-quality/
├── config.yaml
├── STATUS.md
├── experiments.json              # 8 experiments logged
├── data/
│   ├── train_logs.csv            # 463 MB, 8.4M rows
│   ├── train_scores.csv          # Target scores
│   ├── test_logs.csv             # 3 dummy essays
│   ├── sample_submission.csv
│   ├── train_processed.parquet   # 66 features per essay
│   └── test_processed.parquet
├── scripts/
│   ├── eda.py
│   ├── feature_engineering.py
│   ├── train.py
│   └── submit.py
└── submissions/
    └── submission_ensemble3_cv0.648_20260222_123217.csv
```
