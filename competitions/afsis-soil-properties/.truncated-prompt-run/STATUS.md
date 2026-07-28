# afsis-soil-properties — STATUS

**Final CV: MCRMSE 0.44817** (nested/honest, GroupKFold-5 on spatial-signature site; full-OOF weight fit 0.43129, naive mean baseline 1.02421).
**Submission**: `submission.csv` — 727 rows, columns `PIDN,Ca,P,pH,SOC,Sand` in `sample_submission.csv` order, no nulls. CV-only, not submitted to Kaggle (competition closed 2014).
**Approach**: 32-member honest pool (Kernel Ridge RBF / Ridge / SVR / PLS over 14 spectral preprocessing variants) combined by per-target greedy forward selection with replacement.

---

## Competition
Africa Soil Property Prediction Challenge. Predict 5 soil properties (Ca, P, pH, SOC, Sand)
from 3,578 mid-infrared absorbance bands plus 15 spatial covariates and a Topsoil/Subsoil
flag. Metric: MCRMSE (mean of the 5 per-column RMSEs), minimize. Train 1,157 rows,
test 727 rows — p >> n.

## EDA findings that shaped the pipeline
- **Sites are paired, not independent.** The 15 spatial columns take only 580 distinct
  values across 1,157 training rows: 565 sites contribute a Topsoil/Subsoil pair. A plain
  KFold puts both halves of a site on either side of the split and leaks location, so all
  CV here is **GroupKFold(5) keyed on the rounded spatial signature** (fold sizes
  228/233/230/235/231).
- **Targets are heavily right-skewed** (skew: P 7.45, Ca 4.71, SOC 2.45, pH 0.87, Sand 0.39),
  yet log/winsor transforms hurt — see "what did not work".
- **P is the binding constraint.** Its per-target RMSE (0.90) is ~2.8x every other target's,
  and it barely beats its own standard deviation (0.995). MCRMSE improvements come mostly
  from the other four.
- Spectra are clean: no nulls, no duplicate spectra, train and test distributions match
  (mean absolute band difference 0.0118, global std 0.5082 vs 0.5086). The 15 spatial
  columns do shift mildly (LSTN -0.42 sd, ELEV +0.22 sd).
- The CO2 absorption band (2352–2380 cm⁻¹, 15 columns) is a known instrument artifact and
  is dropped from every variant.

## Pipeline
1. **Preprocessing variants** (14): raw, Savitzky-Golay 1st/2nd derivative at windows
   11/25/31/41, SNV, SNV+SG combinations, fingerprint region (<2500 cm⁻¹), and raw/SG
   augmented with the spatial columns + Depth.
2. **Model families**: Kernel Ridge RBF (closed form, eigendecomposition reused across the
   alpha grid), linear Ridge via SVD, SVR-RBF, PLS.
3. **Honest hyperparameter selection**: every KRR/Ridge member picks (gamma, alpha) per
   target by an *inner* CV restricted to the outer-training rows, so no member's OOF vector
   contains selection leakage. Test predictions use hyperparameters chosen by CV over the
   whole training set and refit on all rows.
4. **Blending**: per-target greedy forward selection with replacement. Plain greedy and
   bagged greedy (Caruana-style member subsampling) are compared on the **nested**
   leave-fold-out score, and the winner's weights are then fit on the full OOF.

## Score trajectory
| # | Step | MCRMSE |
|---|------|--------|
| 1 | mean predictor | 1.02421 |
| 2 | best single Ridge variant (raw + spatial) | 0.48668 |
| 3 | 7-variant Ridge greedy blend | 0.47787 |
| 4 | + PLS + SVR members (16), nested | 0.47338 |
| 5 | Kernel Ridge RBF single family (honest) | 0.46661 |
| 9 | **final 32-member honest blend, nested** | **0.44817** |

Nested per-fold scores: 0.42543 / 0.35546 / 0.45158 / 0.42476 / 0.52089 — wide spread, as
expected when folds are whole geographic site groups.

## What did not work (tested, reverted)
- **Log-shifted or winsorized targets.** Skew is real, but the metric is plain RMSE, so
  compressing the tail costs more than it buys (krrlog 0.47216 and krrwin 0.46725 vs
  krr 0.45244 on the same variant).
- **Laplacian-kernel KRR members.** Solo 0.485–0.558; adding them moved the pool's nested
  score from 0.44817 to 0.44903. Reverted.
- **Per-target shrink calibration** of the blend output. Optimal b landed in 0.96–1.04 and
  bought 0.0004 — noise. Dropped.
- **Equal-weight blending.** 0.48777 across the 7 Ridge variants, worse than the best single
  member (0.48668). Weights must be searched.

## Reproduce
```bash
cd /home/tjyen/ai_agents/kaggle
OMP_NUM_THREADS=10 uv run python3 competitions/afsis-soil-properties/scripts/eda.py
OMP_NUM_THREADS=10 uv run python3 competitions/afsis-soil-properties/scripts/iterate2.py   # Ridge/PLS/SVR members
OMP_NUM_THREADS=10 uv run python3 competitions/afsis-soil-properties/scripts/iterate3.py   # kernel ridge sweep
OMP_NUM_THREADS=10 uv run python3 competitions/afsis-soil-properties/scripts/iterate4.py   # refined KRR grid
OMP_NUM_THREADS=10 uv run python3 competitions/afsis-soil-properties/scripts/iterate6.py   # honest member pool
OMP_NUM_THREADS=10 uv run python3 competitions/afsis-soil-properties/scripts/final_blend.py
```
All randomness is seeded (fold assignment seed 42, bagging seed 42); the linear-algebra
paths are deterministic, so reruns reproduce the scores above exactly.
