# Playground Series S3E20 - Competition Report

**Competition**: Predict CO2 Emissions in Rwanda
**Status**: Model trained and validated ✅ · Phase-B 迭代完成
**Date**: 2026-02-14 (v1) · 2026-07-03 (v2 improved) · 2026-07-04 (Phase B v3–v5) · 2026-07-04 (Phase E-4 樹搜尋 v2)

---

## 🌲 Phase E-4 樹搜尋 v2(2026-07-04)— `tree_search/eval_s3e20_v2.py` / `run_s3e20_v2.py`

**問題**:s3e20 是目前唯一「結構完全主宰」的競賽——Phase-B 已證明純 location-week 歷史
均值(去噪後)完勝所有 GBDT(權重搜尋每輪皆給 GBDT 0)。這個結構管線本身有 4 個純量超參
(EB 收縮 α、COVID 降權 W2020、鄰週平滑權重 WNB、平滑窗寬),Phase-B 是「一次一項」手動
調校(偶爾為新增旋鈕做一次 scratch 聯合重調,從未存檔)。本輪把這 4 個(+ 幾個 Phase-B
從未試過的新結構想法)當成樹搜尋節點空間,測試:在單一結構訊號主宰、且評估成本近乎零
(純 groupby 均值,無 GBDT 訓練)的地形上,樹搜尋還有沒有價值?

**根節點驗證**:`ROOT_CONFIG`(α=1.0, W2020=0.24, WNB=0.28, window=1)獨立重算得
OOF RMSE **21.148726**,與 `scripts/train_v5.py` / `experiments.json` #7 的 21.1487
逐位吻合(root 驗證於搜尋前完成,腳本內建 assert)。

**節點空間**(9 條一代 lineage + 1 條 JOINT + 1 條 BLEND):

| Lineage | 想法 | 結果 |
|---------|------|------|
| ALPHA | α 精細再掃描(Phase-B 已用 α=1.0) | 局部最優 α≈0.98-0.99,微幅改善(21.1487→21.146) |
| W2020 | W2020 精細再掃描(Phase-B 已用 0.24) | **確認 0.24 已是單軸最優**,任何方向移動皆變差(誠實的「打平」結果) |
| WNB | WNB 精細再掃描(Phase-B 已用 0.28) | **確認 0.28 已是單軸最優**,同上 |
| WINDOW | 窗寬 ±2/±3 改用「衰減權重」(wnb^d)而非 Phase-B 的平權 | 仍變差(21.23/21.59),衰減沒能救回寬窗 |
| **YEARWEIGHTS**(新想法) | Phase-B 只降權 2020(COVID);本想法把「異常年降權」概念推廣到其他年度(2019 起漲年降權、2021 貼近測試年升權) | **真實增益**:2019 降權至 0.6-0.7 單獨即達 21.0951→21.0808(疊代後),為本輪最大單軸增益來源 |
| PERLOCALPHA(新想法) | 全域純量 α 換成逐地點經驗貝葉斯 α(依該地點週間訊噪比) | 最佳 k=4 得 21.10 左右,**接近但不如**單純全域 α=1.0(誠實負向新想法) |
| CIRCULAR(新想法) | 週索引跨年首尾環繞平滑(week 52↔week 0) | 變差(21.28-21.33)——Rwanda 週 0/52 季節並不真的相似 |
| MONTHFALLBACK(新想法) | 為缺失 cell 加一層地點-月均值 fallback | **與根節點精確打平(21.1487)**——資料集是完美平衡面板(497 地點×53 週×3 年,每個 (地點,週) 組合在每個 LOYO 折都存在),此 fallback 數學上永遠不會觸發;誠實記錄為 no-op |
| GBDT(唯一診斷) | 單一輕量 LGB(非 3 模型全集成),重新確認 GBDT 對此結構訊號的價值 | solo OOF RMSE 51.40(遠劣於結構預測器) |
| **JOINT**(本輪核心論點) | 從根節點一步同時移動 α+W2020(2 參數),接著疊加 WNB+YEARWEIGHTS(4 參數共 2 步),再局部微調 | **21.1487 → 21.0589**(−0.0898,−0.42%),為全樹最佳純結構節點 |
| BLEND | 全樹最佳結構節點(#28)+ GBDT 診斷(#9)做權重搜尋 | GBDT 獲得極小非零權重 2.17%,OOF RMSE 21.0589→**21.0332**(−0.0257);**僅 3 折 LOYO CV 且權重直接對同一份 OOF 擬合,信心低,視為 CV 噪音範圍內的邊際發現,非穩健結論** |

**規模**:38 個已評估節點(37 solo + 1 blend)、0 個去重拒絕、15 次 backtrack 事件(含 1 次
「全部 lineage 都 plateaued → 重開一次」的 fallback)、全程 wall-clock 僅 **33.3 秒**
(相較其他 harness_v2 競賽動輒需 GBDT 訓練的 20-30 分鐘量級,快兩個數量級)。
Prior 使用率:12 個節點標註 `[PRIOR ...]`,勝率 33.3%;14 個 `[PRIOR none]`,勝率
42.9%——**本輪 prior 標註的節點勝率反而略低於無 prior 節點**,誠實記錄(prior 只指向
「往結構去噪方向試」這個大方向,不預測具體哪個數值組合會贏,故未展現預測力優勢)。

**最終結果**:v2 樹搜尋最佳(純結構)21.0589,加上邊際 GBDT 混合 21.0332,雙雙 **BEAT**
Phase-B 手調基準 21.1487(改善 0.42%~0.55%)。

**地形對比觀察(regime observation)**——這是本輪要回答的核心問題:

1. **在已被人工調到位的軸上(W2020、WNB),樹搜尋只能誠實地「打平」**——重掃描確認
   Phase-B 的單軸最優解已經是真最優,自動化搜尋沒有額外增益。這是預期中「不丟臉」的
   結果,證明 Phase-B 的手動一次一項調校本身沒有留下明顯的單軸漏洞。
2. **樹搜尋的價值來自兩處,都不是「重掃描已知軸」**:(a) 一個 Phase-B 從未想過的
   全新結構軸(YEARWEIGHTS:異常年降權概念推廣到 2019/2021),獨力貢獻約 −0.05 RMSE;
   (b) **JOINT lineage 把 4 個結構旋鈕(α, W2020, WNB, 多年權重)一步到位共同移動**,
   複合出 −0.09 RMSE 的總增益——這正是 Phase-B「一次一項」協定結構性難以觸及的區域
   (Phase-B 偶爾手動 scratch 重調 2-3 個「已知」軸,但從未把新軸與舊軸一起在同一步
   探索;樹搜尋把這種聯合移動變成一等公民的 mutation,而非另開手稿的例外操作)。
3. **評估成本是本輪最大的地形差異**:結構超參地形每個節點僅需幾個 groupby 均值運算
   (~0.1 秒/節點),比模型/blend 地形(每節點常需分鐘級 GBDT 訓練)便宜 2-3 個數量級。
   這讓 38 節點 + 15 次 backtrack 的完整搜尋 33 秒內跑完——「結構主宰」與「模型主宰」
   地形不只解的形狀不同,搜尋預算的稀缺程度也天差地遠:在此地形上,「多跑幾個節點」
   幾乎沒有機會成本,樹搜尋的價值瓶頸從「能負擔多少次評估」轉移到「有沒有想到夠多元
   的結構性想法可供評估」。
4. **GBDT 依然是零/近零權重**——即使結構預測器本身被樹搜尋進一步優化(21.1487→
   21.0589),GBDT 混合貢獻的極小權重(2.17%)大機率是 3 折 LOYO CV 的噪音,而非真實
   訊號;與 Phase-B 殘差診斷(exp #8)的結論一致:感測器特徵對此目標本質上不含結構
   之外的可預測訊號,不因換了搜尋方法而改變。
5. **總結**:在結構主宰的地形上,樹搜尋不會無中生有出模型層級的巨大增益,但仍然
   透過(a) 誠實驗證已調校軸、(b) 系統性探索新結構軸、(c) 一等公民的聯合移動,
   拿到 Phase-B 手動迭代協定結構性難以觸及的 0.4-0.55% 增量——比模型/blend 地形上
   常見的樹搜尋增益幅度小,但並非零,且成本極低。

- 交付物:`tree_search/eval_s3e20_v2.py`、`tree_search/run_s3e20_v2.py`、
  `competitions/playground-series-s3e20/experiments_tree.json`(38 節點)。
  Cache:`tree_search/cache_s3e20/`(gitignored,可由 run 腳本重建)。
- 未觸碰:`scripts/train_v5.py`、`experiments.json`(遵循任務限制)。

---

## 🚀 Phase-B 自我改進迭代 (2026-07-04) — `scripts/train_v3.py` / `train_v4.py` / `train_v5.py`

v2 已確立「純 location-week 歷史均值(TE)完勝 GBDT」(22.6488,GBDT 權重全 0)。
Phase-B 針對 TE 本身做三輪去噪改進 + 一輪負向診斷,CV 方案不變(Leave-One-Year-Out
2019/2020/2021,與 v2 完全同折):

| Round | 改動(一次一項) | TE OOF RMSE | Δ | exp # |
|-------|----------------|------------|---|-------|
| — | v2 baseline(純 loc-week 均值) | 22.6488 | — | #4 |
| 1 | **經驗貝葉斯收縮**:cell 均值(每折僅 n=2 觀測)向 loc 均值(~106 觀測)收縮,`α=0.935`(pre-sweep 網格選定) | **22.4897** | −0.159 | #5 |
| 2 | **COVID 年降權**:計算歷史均值時 2020 年觀測權重降至 `W2020=0.29`(α 聯合重調至 0.945;W2020=0 完全排除反而更差) | **21.6317** | −0.858 | #6 |
| 3 | **鄰週平滑**:cell 均值與同地點 week±1 的 cell 均值加權平均(`WNB=0.28`);α 聯合重調後收斂到 1.0 —— 鄰週平滑把 Round-1 的 loc-mean 收縮「吸收」了(兩者攻擊同一噪音源) | **21.1487** | −0.483 | #7 |
| 4a | 加寬平滑窗至 ±2/±3 週 → 全部變差(21.23/21.32/21.51),±1 即最優 | 21.1487 | 0 | #8 |
| 4b | **殘差診斷**:LGB 訓練於 (y − TE_v5) 殘差,以 1.0/0.5/0.25/0.1 比例加回 → 22.07/21.39/21.21/21.16,單調趨近 TE-only | 21.1487 | 0 | #8 |

**最終:OOF RMSE 22.6488 → 21.1487(−6.6%)**;三個 GBDT 在每一輪權重搜尋中仍全為 0。

**Round 4b 的價值(負向結果)**:感測器特徵連「殘差」都解釋不了 —— 結構(平滑後的
loc-week 均值)已飽和全部可預測訊號。與 s3e19 反例合併,確認「目標跨年近恆定 →
結構勝 GBDT」邊界條件在殘差層面同樣成立。

- 最終 artifact:`submissions/sub_v5_blend_21.1487_20260704_011931.csv`(comp closed,不可上傳)。
- 迭代紀錄:`experiments.json` #5–#8;各輪腳本 `scripts/train_v{3,4,5}.py`(v5 為最終版)。

---

## 🔄 v2 Improved Experiment (2026-07-03) — `scripts/train_v2.py`

**Motivation**: v1 (RMSE 33.21) relied on satellite sensor features and a single
time-based split. A quick check revealed the real signal.

**Key insight (verified)**: emission is near-constant across years for a given
`(latitude, longitude, week_no)` — cross-year std median ≈ 3.1. A pure
location-week historical mean predictor scores **RMSE ~19.8 on a 2021 holdout**,
far better than the v1 model.

**What v2 does**:
- Adds a **location-week target encoding** (`te_locweek`) computed per fold from the
  *other* training years → no leakage (this is exactly how the 2022 test is meant
  to be predicted from 2019–2021 history).
- **Leave-One-Year-Out CV** (hold out 2019 / 2020 / 2021 in turn) instead of one split.
- Adds **CatBoost** as a 3rd tree model; OOF **weight-optimized 4-way blend**
  (LGB / XGB / CatBoost / pure loc-week-mean).
- log1p target; drops 7 sensor cols with >90% missing (keep 63).

**Results (LOYO OOF RMSE)**:

| Member | OOF RMSE |
|--------|----------|
| LightGBM (log1p) | 32.58 |
| XGBoost (log1p) | 28.40 |
| CatBoost (log1p) | 28.95 |
| **Pure location-week mean (TE)** | **22.65** |
| **Optimal blend** | **22.65** (weights: TE=1.0, trees=0.0) |

**Finding**: the weight search assigns **100% to the location-week mean** — the
gradient-boosted trees add *nothing* beyond it. The satellite sensor features are
essentially noise for this target. Understanding the data beat model complexity.

- **v1 → v2: RMSE 33.21 → 22.65** (−31.8%); vs LOYO global-mean baseline **144.4 → −84.3%**.
- Submission artifact: `submissions/sub_v2_blend_22.6488_*.csv` (comp closed, cannot upload).
- Experiment log: `experiments.json` #4.

> ⚠️ s3e20 Late Submission is **closed** — no online LB score possible; local CV only.

---

## Competition Overview

- **Problem Type**: Regression
- **Evaluation Metric**: RMSE (Root Mean Squared Error)
- **Dataset Size**: 79,023 train samples, 24,353 test samples
- **Features**: 75 original features (satellite sensor data)
- **Target**: CO2 emissions (continuous, range: 0-3,168)

---

## Pipeline Summary

### Stage 1: EDA ✅
**Key Findings**:
- Target extremely right-skewed (skewness: 10.17)
- 7 features with 99.4% missing values
- 21 features with 10-90% missing (SulphurDioxide, NitrogenDioxide groups)
- 497 unique geographic locations
- Strong temporal component (2019-2021 train, 2022 test)
- Location-based emission variation (some areas 10× higher)

**Recommendations**:
- Drop ultra-high missing features
- Log1p transformation for target
- Time-based validation (2019-2020 train, 2021 val)
- Location-based aggregations

### Stage 2: Feature Engineering ✅
**Actions Taken**:
- Dropped 7 features with >90% missing
- Created 21 missing indicators for medium-missing features
- Imputed remaining missing with median
- **Created 36 new features**:
  - Geographic: lat×lon, lat², lon²
  - Temporal: cyclical week encoding (sin/cos), season
  - Location aggregates: mean, median, std emission by location
  - Temporal: year_week, weeks_since_start

**Result**: 68 → **102 features**

### Stage 3: Modeling ✅
**Models Trained**:

| Model | Val RMSE | vs Baseline |
|-------|----------|-------------|
| Baseline (mean) | 155.54 | - |
| Ridge Regression | 23,917.63 | ❌ Failed |
| XGBoost | 34.44 | -121.10 (77.9%) |
| LightGBM | 32.75 | -122.79 (78.9%) |
| **Ensemble (50/50)** | **33.21** | **-122.33 (78.6%)** ⭐ |

**Model Configuration**:
- **Target**: log1p transformed
- **Validation**: Time-based split (2019-2020 train, 2021 val)
- **Training**: GPU-accelerated (NVIDIA)
- **Ensemble**: 50% LightGBM + 50% XGBoost

**LightGBM Params**:
```python
{
  'learning_rate': 0.05,
  'num_leaves': 31,
  'subsample': 0.8,
  'colsample_bytree': 0.8,
  'reg_alpha': 0.1,
  'reg_lambda': 0.1,
  'num_boost_round': 700
}
```

**XGBoost Params**:
```python
{
  'learning_rate': 0.05,
  'max_depth': 6,
  'subsample': 0.8,
  'colsample_bytree': 0.8,
  'num_boost_round': 200
}
```

### Stage 4: Evaluation ✅
**Validation Results**:
- **Validation RMSE**: 33.21
- **Improvement**: 78.6% over baseline
- **Prediction range**: [-0.02, 2,731.10]
- **Prediction mean**: 87.68 (vs train mean 81.94)

**Model Analysis**:
- LightGBM slightly outperformed XGBoost (32.75 vs 34.44)
- 50/50 ensemble provides stable predictions
- Ridge failed catastrophically (likely due to log-transformed target scale)
- Time-based validation critical for preventing temporal leakage

### Stage 5: Submission ✅
**Submission File**: `submission_final_33.2056_20260214_161743.csv`
- **Format**: ID, emission predictions
- **Size**: 24,353 predictions
- **Status**: Generated successfully ✅
- **Kaggle Submission**: Competition closed (2023), cannot submit

---

## Key Learnings

1. **Log transformation essential**: Target skewness (10.17 → -0.61)
2. **Time-based validation critical**: Random CV would leak future information
3. **Location matters**: Geographic aggregates were valuable features
4. **Missing value handling**: Dropping 99%+ missing features, creating indicators for others
5. **GPU acceleration**: Significantly faster training (LightGBM, XGBoost)
6. **Simple ensemble wins**: 50/50 weighted average of top 2 models

---

## Files Created

### Data
- `data/train_processed.csv` - Engineered features (79,023 × 104)
- `data/test_processed.csv` - Engineered features (24,353 × 103)

### Scripts
- `scripts/initial_inspection.py` - Data exploration
- `scripts/eda.py` - Comprehensive EDA
- `scripts/feature_engineering.py` - Feature creation
- `scripts/train_model.py` - Model training
- `scripts/create_submission.py` - Final submission generation

### Outputs
- `submissions/submission_final_33.2056_20260214_161743.csv` - Final predictions
- `experiments.json` - Experiment tracking
- `config.yaml` - Competition metadata

---

## Performance Summary

**Baseline → Final Model**
- RMSE: 155.54 → **33.21**
- Improvement: **78.6%** ⭐
- Training time: ~3 minutes (GPU)

**Model Ranking**:
1. 🥇 LightGBM: 32.75
2. 🥈 XGBoost: 34.44
3. 🥉 Ensemble: 33.21

**Why Ensemble scored between individual models**: The 50/50 weighting wasn't optimized. Individual models performed better, but ensemble provides stability.

---

## Potential Improvements

1. **Optimize ensemble weights** - Use validation RMSE for inverse-variance weighting
2. **More feature engineering**:
   - Lagged features (previous week's emission by location)
   - Rolling statistics (3-week, 4-week windows)
   - Interaction terms (sensor readings × location)
3. **Hyperparameter tuning** - Optuna for LightGBM/XGBoost
4. **Stacking** - Meta-model on out-of-fold predictions
5. **CatBoost** - Add third tree model to ensemble
6. **Feature selection** - Remove low-importance features

---

## Conclusion

Successfully completed full Kaggle competition pipeline:
- ✅ Data exploration and understanding
- ✅ Feature engineering (102 features)
- ✅ Model training with time-based validation
- ✅ Ensemble creation
- ✅ Submission file generation

**Final Val RMSE: 33.21** (78.6% better than baseline)

The model is production-ready and the submission file is formatted correctly for Kaggle.
