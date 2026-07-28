# tabular-playground-series-jan-2022 — Competition Status

**Final CV (fold-2018, full-year-ahead): SMAPE 4.1793** — blend 0.95×ridge-decomposition + 0.05×LGB-gdp, rounded to int (expanding-year CV 2016/2017/2018: 5.0154 / 4.8401 / 4.1793, mean 4.678).
Winning lever chain: GDP-normalized level (World Bank gdp_pc incl. 2019) → structural ridge on log1p → per-holiday-name × offset(-5..+10) dummies (5.46 → 4.21).
Submission: `submission.csv` (= `submissions/submission_ridge95_lgb05_cv4.179_20260728.csv`), 6570 rows, id order per sample_submission.

## Competition Info
- https://www.kaggle.com/competitions/tabular-playground-series-jan-2022
- Forecast daily 2019 sales for 18 series (3 countries × 2 stores × 3 products), train 2015-2018 daily. Metric SMAPE.

## Validation
- Expanding year folds: train <Y, validate year Y for Y∈{2016,2017,2018}; fold 2018 primary (mirrors full-year-ahead extrapolation of the real test). Random KFold banned (interpolation optimism, per experience library).

## Models (fold-2018 SMAPE)
| model | 2018 | notes |
|---|---|---|
| blend_final (0.95 ridge + 0.05 lgb_gdp, rounded) | **4.1793** | submitted |
| ridge_v4 (holiday-name offsets -5..+10) | 4.1855 | best solo |
| ridge_v3 (offsets -2..+2) | 5.2305 | |
| ridge E_all (pooled holiday flags) | 5.4564 | |
| lgb_gdp (target log(num_sold/gdp_pc)) | 5.8028 | |
| ridge A_base | 5.5270 | |
| lgb_raw (log1p) | 8.4292 | trees can't extrapolate level |
| naive lag-364 | 10.0470 | baseline |

## Key Findings
- Data is clean multiplicative synthetic: store ratio 1.742 constant every year, product shares constant, dow effect identical across all dims.
- Country-year level ∝ GDP per capita (total/gdp_pc near-equal across countries each year, common ~+1.7%/yr drift → log_gdp + linear year_c in ridge; year_c ablation hurts badly in 2017: 4.83→6.02).
- Post-holiday +3..+7 day windows carry big elevated sales (post-Easter +34% miss with narrow window) — widening holiday-name offsets to -5..+10 was the largest single gain (5.23→4.19).
- month×product dummies hurt (Fourier k=6 sufficient); fixed shrink factor inconsistent across folds (level bias ≫ SMAPE asymmetry) — rejected.
- Tree search skipped per skill's small/cheap-eval fallback clause; manual iteration converged in 4 rounds.

## Files
- `scripts/eda.py`, `scripts/feature_engineering.py`, `scripts/train_v1.py`–`train_v3b.py`, `scripts/final_submit.py`
- `data_official/gdp.csv` (World Bank NY.GDP.PCAP.CD, FIN/NOR/SWE 2015-2019)
- `experiments.json` (13 entries), `train_feat.parquet` / `test_feat.parquet`

## Potential Improvements
- Per-country year_c slopes; smoothed holiday-offset coefficients (currently unregularized dummies share alpha=1)
- LGB on ridge residuals; seed-bag LGB member; November -5% residual (Fourier ringing) via targeted terms
