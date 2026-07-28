# ML Specification Report — playground-series-s3e20

### Rwanda CO2 Emission Prediction · our from-scratch agent (structural location-week empirical-Bayes model + tree-search)

> *Figures grounded in the competition's `config.yaml`, `facts.json`, `eda_summary.json`, `STATUS.md`, and `experiments_tree_v3.json`, with external benchmarking from `/tmp/ext_facts.json`.*

## Overview

The task is to predict weekly CO2 emission over Rwanda from satellite-sensor readings (metric: RMSE, lower is
better). Unlike our other episodes, this is the one competition where **structure completely dominates**: a purely
structural per-(location, week) empirical-Bayes historical-mean model beats every gradient-boosted tree we throw at it. Our
committed champion is that structural model — no GBDT, no deep net — scoring **CV RMSE ≈ 21.15** (Leave-One-Year-Out); a
tree search then nudges the composed structural optimum to **21.0589**. The NVIDIA reproduce-agent, applying the technique
from a strong public kernel (`kacperrabczewski/rwanda-co2-step-by-step-guide`), reaches **19.648** — NVIDIA ahead.


**Why it matters.** Satellite-based emission estimates support climate-policy monitoring and carbon accounting where ground sensors are sparse — a pressing need across developing regions such as Rwanda.

---

## Data

**Purpose of Data.** Predict the continuous `emission` target for each (latitude, longitude, year, week) cell in Rwanda,
from satellite-derived atmospheric measurements — a **regression** Playground Series episode (S3E20). **Data Format** is
clean **tabular CSV**, entirely numeric. **Data Volume** is **79,023 training rows / 24,353 test rows**, with **74 numeric
features** and **0 categorical features** once the `ID_LAT_LON_YEAR_WEEK` key is set aside. The single most important
structural fact from EDA is that the training set is a **perfect balanced panel — 497 locations × 53 weeks × 3 years =
79,023 rows** — so every (location, week) cell is observed in every training year (2019–2021), and the test set is the same
grid for 2022.

**Data Quality** is mixed. There are **0 duplicate rows**, but missingness is heavy and structured: the entire
**`UvAerosolLayerHeight_*` sensor group is 99.4% missing** (78,584 / 79,023 rows), `NitrogenDioxide_*` is ~23% missing
(18,320 rows), and `SulphurDioxide_*` ~18% (14,609 rows). Several sensor-geometry columns also show a large **train↔test
mean shift** (e.g. `SulphurDioxide_sensor_azimuth_angle` −217%, `Formaldehyde_sensor_azimuth_angle` −135%), a warning that
raw sensor angles do not transfer across the year boundary. The target is **extremely right-skewed** (`emission` mean
81.94, median 45.59, min 0.0, max 3167.77, **skew 10.17, kurtosis 157.6**), which motivates a log1p transform in the tree
members. Crucially, the **satellite features carry almost no linear signal**: the strongest target correlation is merely
`longitude` (Pearson 0.103 / Spearman 0.291), and every sensor column has |Pearson| < 0.07. The predictive signal lives in
*which cell and which week* a row belongs to, not in the sensor readings — the finding that shapes the whole solution.

**Annotation Guidelines.** The label is the continuous `emission` value (≥ 0); submissions are real-valued predictions
scored by RMSE against the held-out 2022 grid. The provided `sample_submission.csv` baseline is simply the **mean of the
training target**.

**Feature Set.** The champion is **structural, so its "features" are the panel key, not the sensors**: it predicts each
test cell by an empirical-Bayes-shrunk historical mean of `emission` over that cell's (location, week) group across the
training years, with COVID-year down-weighting, neighbor-week smoothing, and per-year weighting (see the Models section). The 74 raw sensor
columns enter only the **GBDT diagnostic** (63 sensor columns + latitude/longitude/week, log1p target); the earlier tree
tiers additionally used loc-week and loc target encodings on top of the sensor block. Sensor-derived features were found to
add essentially nothing over the structural key.

**Splitting strategy.** The canonical scheme is **Leave-One-Year-Out (LOYO) CV, 3 folds (hold out 2019 / 2020 / 2021 in
turn), seed = 42**. This is the honest analogue of the train→test year gap (train 2019–2021 → test 2022) and replaced the
early time-based split (2019–2020 train / 2021 val) used in v1. Note that the automated `validation_hint` suggested a plain
KFold ("continuous i.i.d. target → standard KFold"); we overrode it because the task is a per-year panel, and LOYO is the
only split that measures cross-year generalization. **The leaderboard is closed and this run is OOF-only** — no test
predictions were generated for the structural champion, so all scores here are **LOYO CV**; the `leaderboard` field is
`null` (**not recorded**) and no public/private LB score exists for the committed champion.

## Models & Architecture

**Purpose of Architecture.** A regressor minimizing RMSE on 2022 Rwanda CO2 emission. **Architecture Type** is — honestly —
**a structural, non-parametric empirical-Bayes historical-mean model over the (location, week) panel**, *not* a deep
network and *not* a GBDT. The mapping to a DL/GBDT architecture is therefore **N/A**; the one GBDT we fit is retained only
as a diagnostic (below).

**Input Format** for the champion is the panel key (location, week, year) rather than a dense feature vector, so a
conventional **Input Dimension** count is **N/A** — the model consumes group identity, not the 74 sensor columns.

**Architecture Description.** For each (location, week) cell the model forms a historical mean of `emission` across the
training years and shrinks it toward broader means via empirical-Bayes (the committed root uses shrinkage α, COVID-year
down-weight `w2020`, neighbor-week smoothing weight `wnb`, and a smoothing window). The tree-search champion adds a per-year
weighting axis. The committed Phase-B root config (tree node #0) and the tree-search optimum (node #28) are:

| Axis | Root (node #0) | Tree-search optimum (node #28) |
|------|---------------:|-------------------------------:|
| EB shrinkage `alpha` | 1.0 | 0.99 |
| COVID down-weight `w2020` | 0.24 | 0.2 |
| neighbor-week smoothing `wnb` | 0.28 | 0.3 |
| smoothing `window` | ±1 | ±1 |
| `year_weights` (2019/2020/2021) | — (none) | 0.6 / 0.2 / 1.2 |
| LOYO OOF RMSE | 21.1487 | **21.0589** |

**Model Complexity** is not a trainable-parameter count (**N/A** for a structural model); it is **4 tuned scalar
hyperparameters** in the root, plus the new 3-value `year_weights` axis at the optimum. **The GBDT diagnostic** — a single
lightweight LightGBM (`n_estimators` = 400, `learning_rate` = 0.05, log1p target, 63 sensor + lat/lon/week features, LOYO
CV; tree node #9) — scored **solo RMSE 51.40**, far worse than the structural model, re-confirming across five prior Phase-B
runs that GBDTs earn ~0 blend weight against this signal.

## Training procedures

There is no gradient-based training loop for the champion; "training" is (a) computing the shrunk historical means and (b)
searching a handful of scalar weights. The improvement trajectory across our runs was:

| Stage | Configuration | CV scheme | RMSE |
|-------|---------------|-----------|-----:|
| v1 | LGB+XGB+Ridge ensemble on sensor features | time-based (2019–20 / 2021) | 38.51 |
| v1-final | LGB+XGB 50/50 on sensor features | time-based | 33.21 |
| v2 | LGB+XGB+CatBoost + loc-week target encoding | LOYO (3-fold) | 28.34 |
| v2+TE | + pure structural TE member added to blend | LOYO | 22.6488 |
| Phase-B | structural TE tuned (EB shrinkage / COVID / neighbor smoothing) | LOYO | 21.6317 → **21.1487** (root) |
| Tree-search | JOINT lineage: all 4 axes + new `year_weights`, co-optimized | LOYO | **21.0589** (node #28) |

Across v2 onward the blend weight on every GBDT member collapsed to 0 — the structural target-encoding member alone carried
the score. The **Loss Function / Optimization Algorithm / Learning Rate / Learning Rate Scheduler / Batch Size** are all
**N/A for the structural champion** (no gradient objective, no shrinkage schedule, no mini-batches); the only place these
apply is the GBDT diagnostic, whose loss is L2 on the log1p target with `learning_rate` = 0.05 — and it loses decisively.

The tree search co-optimized the structural axes rather than Phase-B's one-axis-at-a-time manual tuning: re-scanning the
individually-tuned axes (`w2020`, `wnb`) in isolation only **confirmed** Phase-B's optimum (an honest tie), but
**JOINT-moving all axes together plus the new `year_weights` axis** reached a composed optimum (21.0589) that the manual
protocol was structurally unable to find. The largest single-axis gain was the new `year_weights` idea (down-weight the
anomalous 2019/2020 years, up-weight 2021 as closest to the 2022 test year), which alone moved 21.0951 → 21.0808.

**Training Duration** for the entire tree sweep was **~33 s of wall-clock** over **38 evaluated nodes** (`our_wall_s` = 33,
`our_train_fits` = 38); an individual structural node fits in ~0.07 s (pure groupby, no model training). **Training Memory**
was **not recorded** — a 79k × 74 float panel is trivial on the single arm64 CPU machine and peak was not measured.
**Transfer Learning** is **N/A (no pretrained weights)**; its analogue is our cross-competition experience library that
carried the "structure-first, GBDT-as-diagnostic" prior into this episode. **Data Augmentation** is **N/A (tabular)**; the
structural analogues are neighbor-week smoothing and empirical-Bayes shrinkage.

**Reproducibility Standards** are strict: fixed LOYO fold seed = 42, digit-for-digit root verification (node #0 re-computed
to 21.1487 before the search began), and a faithful-replay contract. The recorded replay is **38/38 nodes exact, Δ = 0**
(`faithful_replay`), and the search state is marked `stopped` with `stop_reason` "faithful replay of committed tree".

## Inference procedures

**Decision Threshold** is **N/A** — RMSE is a regression metric, so predictions are the real-valued shrunk historical means
(clipped to ≥ 0 in the earlier tree tiers), never thresholded. **Inference Duration** and **Inference Memory** were **not
recorded**; structural scoring is a groupby lookup, effectively free on CPU. Note the champion search was **OOF-only** — no
2022 test predictions or submission file were produced (the competition is closed), so there is no submitted-inference
profile to report.

## Evaluation & Benchmarking

**Task for Performance Evaluation.** Predict 2022 weekly CO2 emission across the Rwanda grid; score by RMSE. **Performance
Metrics.** Our committed structural champion reaches **CV RMSE ≈ 21.15** (LOYO), improving 33.21 → 28.34 → 22.65 → 21.63 →
21.15 across the trajectory; the tree search then refines the structural optimum to **21.0589** (node #28). A sibling
**BLEND node (#37)** mixed the structural node with the GBDT diagnostic and scored 21.0332 — but the GBDT earns only
**2.17% weight** on 3-fold LOYO with the weight fit on the same OOF it is scored on, which `STATUS.md` explicitly flags as
low-confidence, within CV-noise, **not a robust conclusion**. We therefore report the purely-structural **21.0589 / 21.15**
as the honest result. No public/private LB figures exist for this champion (comp closed, OOF-only) — **not recorded**.

**Performance Benchmarking.** The comparator is the **NVIDIA reproduce-agent**, which applies the *technique* from a strong
public kernel (`repro_type = technique`):

| Agent | Approach | CV / kernel RMSE | Note |
|-------|----------|-----------------:|------|
| NVIDIA | reproduces technique of `kacperrabczewski/rwanda-co2-step-by-step-guide` | **19.648** | winner |
| Our agent | from-scratch structural location-week EB model + tree search | 21.15 | behind by ~7% |

NVIDIA leads (lower RMSE is better): 19.648 vs our 21.15, a gap of ~1.5 RMSE (~7% relative). The honest read: this episode
rewards a specific, well-known structural recipe for the Rwanda panel that the public kernel encodes tightly; our
from-scratch agent independently discovered the *same* structural regime (historical means dominate, GBDTs get ~0 weight)
and tuned it to 21.06–21.15, but did not close the last increment to the kernel's optimized 19.648. The load-bearing
finding — that a 4-scalar structural model *beats every GBDT ensemble* here, confirmed five separate times — is itself the
report's central result.
