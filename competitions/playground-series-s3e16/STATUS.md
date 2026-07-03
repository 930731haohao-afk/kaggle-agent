# STATUS — playground-series-s3e16 (Crab Age)

- **Task**: regression, predict `Age`; metric **MAE** (minimize); id col `id`
- **Data**: 74,051 train / 49,368 test; 8 features (1 cat `Sex` + 7 continuous size/weight); no missing, no dups
- **Rules**: no external data, no pretrained, no internet; daily submission limit 5

## Progress
- [x] Stage 0 Setup — config.yaml, data downloaded
- [x] Stage 1 EDA — `scripts/eda.py`
- [x] Stage 2 Feature engineering — `scripts/features.py`
- [x] Stage 3 Modeling + CV — `scripts/train.py` (LGB/XGB/CatBoost, MAE objective)
- [x] Stage 5 First submission generated — `submissions/sub_blend_20260703_091840.csv`
- [x] Submitted to Kaggle leaderboard — **Public 1.34356 / Private 1.34075** (2026-07-03)

## Leaderboard result (submission #1)
| | OOF MAE | Public LB | Private LB |
|-|---------|-----------|------------|
| Blend (rounded) | 1.33812 | **1.34356** | **1.34075** |

CV↔LB gap ~0.006 → **CV is trustworthy** (slightly optimistic, as expected). Safe to iterate on OOF.

> ⚠️ **Auth note**: valid Kaggle token is `KGAT_7...` in `~/.kaggle/kaggle_api_token.txt`,
> NOT the `KGAT_2...` in `~/.kaggle/kaggle.json` (that one 403s). Submit with:
> `export KAGGLE_API_TOKEN=$(cat ~/.kaggle/kaggle_api_token.txt | tr -d '[:space:]')`

## EDA key findings
1. `Sex='I'` (infant) → mean age 7.6 vs M 10.9 / F 11.3. Strong signal → `is_infant` flag.
2. `Shell Weight` best correlated with Age (Spearman 0.74); `Shucked Weight` weakest (0.61).
3. Severe collinearity (0.86–0.99) among size/weight; `Weight≈Shucked+Viscera+Shell` (r=0.993).
4. `Height==0` in 24 train rows (invalid) → treated as missing, imputed with median.
5. Train/test distributions near-identical (<0.4% mean diff) → no covariate shift.
6. Target integer 1–29, right-skewed (skew 1.09). **Rounding OOF preds improves MAE.**

## CV scheme (fixed first — matters more than the model)
- **5-fold StratifiedKFold on binned Age** (ages ≥20 merged into one bin), shuffle, seed=42.
- Fold MAE very stable (1.34–1.37) → CV is trustworthy.

## Feature engineering (24 features)
Sex one-hot + `is_infant`; part weight ratios; `weight_resid = Weight − Σparts`; size ratios
(diam/len, height/len, height/diam); `volume`, `density`, `shell_density`; `meat_to_shell`, `shucked_to_shell`.

## Model results (OOF MAE, 5-fold)
| Model | OOF MAE | rounded | time |
|-------|---------|---------|------|
| LightGBM (L1) | 1.35651 | 1.33885 | 31s |
| XGBoost (reg:absoluteerror) | 1.35763 | 1.34160 | 37s |
| CatBoost (MAE) | 1.36162 | 1.33846 | 24s |
| **Blend 0.6/0.3/0.1 (round)** | **1.35589** | **1.33812** | — |

Experiment log: `experiments.json` (#1).

## Iteration round — Phase B self-improvement (2026-07-03)

Goal: push CV further via the validated "Optuna fold-proxy → add to pool → seed bag" recipe
(knowledge/experience.md). Result: **no improvement — best stays 1.33812 rounded**. STOPPED after
2 consecutive non-improving rounds, per protocol.

| Round | Change | Raw OOF MAE | Rounded OOF MAE | vs best (1.33812) |
|-------|--------|-------------|------------------|--------------------|
| — | baseline (LGB/XGB/CAT 0.6/0.3/0.1) | 1.35589 | **1.33812** | — |
| 1 | + Optuna fold0-proxy tuned LGB (21/50 trials, 300s timeout) as 4th pool member | 1.35541 | 1.33850 | worse (+0.00038) |
| 2 | + seed-bagged tuned LGB (seed=2024) as 5th pool member | 1.35533 | 1.33893 | worse (+0.00081) |

Both rounds improved the **raw** OOF MAE (1.35589 → 1.35541 → 1.35533, a real but small gain from
tuning+bagging) but made the **rounded** OOF MAE worse. Root cause: LGB_tuned and
LGB_tuned_seed2024 only differ from the original LGB in depth/lr/reg (same objective, features,
folds) → low ensemble diversity → the small raw-MAE gain isn't large enough to shift the discrete
rounding boundary favorably. New insight logged to `knowledge/experience.md`: the
Optuna→pool→seed-bag recipe (previously validated on RMSE-scale targets s3e1/s3e7/s3e11/s3e14)
does **not** reliably transfer when the final metric is **rounded integer MAE** — decide on the
post-rounded score, not raw OOF, or the "improvement" can be illusory.
Scripts: `scripts/tune_lgb_optuna.py`, `scripts/pool_lib.py`, `scripts/round1.py`, `scripts/round2.py`.
Experiment log: `experiments.json` (#3, #4).

## Next ideas (updated after this iteration)
- Optuna+seed-bagging tried — didn't survive rounding (see above); not worth more trials on LGB alone.
- Untried: Tweedie/Poisson-objective LGB as a genuinely diverse pool member (different loss
  surface, not just different hyperparams) — `pool_lib.train_lgb_tweedie()` and `scripts/round3.py`
  are scaffolded but not yet run (stopped early per 2-non-improving-round rule).
- Untried: collinear feature pruning (Weight≈Shucked+Viscera+Shell, r 0.86–0.99) or a stacking
  meta-model on OOF.

## Reproduce
```bash
cd /home/tjyen/ai_agents/kaggle
uv run python3 competitions/playground-series-s3e16/scripts/eda.py
uv run python3 competitions/playground-series-s3e16/scripts/train.py
# submit (use the huangweihaohuang token; kaggle.json 403s):
export KAGGLE_API_TOKEN=$(cat ~/.kaggle/kaggle_api_token.txt | tr -d '[:space:]')
uv run kaggle competitions submit -c playground-series-s3e16 -f <submission.csv> -m "<msg>"
```
