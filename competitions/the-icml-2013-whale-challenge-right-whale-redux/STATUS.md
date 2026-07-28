# Right Whale Upcall Detection (ICML 2013)

**Last Updated**: 2026-02-22

## Competition Overview
- **URL**: https://www.kaggle.com/c/the-icml-2013-whale-challenge-right-whale-redux
- **Problem**: Binary classification — detect right whale upcalls in underwater audio recordings
- **Metric**: AUC-ROC (maximize)
- **Dataset**: 47,841 train + 25,468 test audio clips (AIFF format)
- **Deadline**: 2013 (closed)

## Data Characteristics
- **Audio format**: 2-second clips, 2000 Hz sample rate, mono (4000 samples each)
- **Class balance**: 11% positive (5,276 whale upcalls vs 42,565 negative)
- **Recording dates**: 4 days (2009-03-28 to 2009-03-31), positive rate 7.8%–12.4%
- **Label encoding**: Filename suffix `_0` (no whale) or `_1` (whale upcall)
- **Frequency range**: 0–1000 Hz (Nyquist limit at 2000 Hz sample rate)

## EDA Insights
- Whale upcalls concentrated in 50–200 Hz frequency band
- Most discriminative mel band: ~111 Hz (+1.23 dB difference)
- Whale clips have lower overall energy than negative clips (quieter background)
- MFCC 0 (overall energy) is the strongest single discriminator
- Low-frequency energy ratio (50-250 Hz): 0.566 positive/negative ratio

## Feature Engineering
**402 features per audio clip:**

| Category | Count | Description |
|----------|-------|-------------|
| Waveform stats | 5 | RMS, max abs, DC offset, std, zero-crossing rate |
| Band energies | 7 | Log energy in 7 frequency bands (0-50, 50-100, ..., 500-1000 Hz) |
| Spectral features | 3 | Spectral centroid, bandwidth, rolloff (85%) |
| Mel spectrogram | 256 | Mean, std, min, max per 64 mel bands |
| MFCC | 80 | Mean, std, min, max per 20 coefficients |
| Delta MFCC | 40 | Mean, std per 20 first-derivative coefficients |
| Temporal envelope | 8 | RMS + max per 4 time quarters |
| Energy ratios | 3 | Whale band energy, other energy, ratio |

## Experiment Results

| # | Model | Mean CV AUC | OOF AUC | Notes |
|---|-------|------------|---------|-------|
| 1 | LightGBM | 0.951 | 0.950 | is_unbalance=True |
| 2 | XGBoost | 0.951 | 0.950 | scale_pos_weight |
| 3 | CatBoost | 0.949 | 0.948 | Balanced weights |
| 4 | **Ensemble (LGB=0.4, XGB=0.5, Cat=0.1)** | — | **0.951** | Best OOF |
| 5 | Ensemble (equal avg) | — | 0.951 | Simple average |

**Top features**: Mel bands 85-90 (~100-200 Hz whale call range), band energy 50-100 Hz, temporal envelope features

## Submission
- **File**: `submissions/submission_ensemble3_cv0.951_20260222_135143.csv`
- **Format**: 25,468 rows with clip filename + probability
- **Predicted positive rate**: 8.2% (vs 11% in train)
- **Status**: Competition closed (ICML 2013) — API submission returns 400 error

## Validation Strategy
- StratifiedKFold (5 folds, shuffle=True, seed=42)
- All models very close (~0.95 AUC), suggesting the feature set captures most signal
- Low fold variance (~0.002) indicates stable features

## Key Learnings
- Audio classification via handcrafted features + GBM works well for whale detection
- Mel spectrogram statistics (64 bands x 4 stats) capture the bulk of the signal
- Whale upcalls have a characteristic frequency signature around 100-200 Hz
- Class imbalance (8:1) handled well by is_unbalance/scale_pos_weight
- All three GBMs perform very similarly — signal is well-captured by the features
- 2000 Hz sample rate limits analysis to 0-1000 Hz (sufficient for whale upcalls)

## Files
- `config.yaml` — competition metadata
- `experiments.json` — 4 experiment records
- `scripts/eda.py` — audio EDA
- `scripts/train.py` — full pipeline (feature extraction + 3 models + ensemble)
- `scripts/submit.py` — submission generation
- `data/train_features.npz` — cached train features (47,841 x 402)
- `data/test_features.npz` — cached test features (25,468 x 402)
- `data/best_weights.npy` — optimal ensemble weights
