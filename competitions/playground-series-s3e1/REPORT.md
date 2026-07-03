# 競賽分析報告:playground-series-s3e1

> 產生方式:kaggle-report skill(數字來自 facts.json,敘述由 agent 撰寫)
> 素材等級:full | 產生日期:2026-07-03

## 1. 競賽目的

**What**:本競賽要求根據加州人口普查區塊(block group)的 8 個統計特徵(收入中位數、屋齡、
平均房間數、平均臥室數、人口、平均入住人數、緯度、經度),預測該區塊的房屋價值中位數
(`MedHouseVal`)——這是經典 California Housing 資料集的 Kaggle Playground 版本
(competition.notes: "Aygun et al. Nature 2026 Kaggle Playground benchmark (Season 3 Episode 1).")。

**Why**:目標欄位是連續型實數(房價中位數),用 RMSE(均方根誤差,minimize)作為評估指標是
合理的——RMSE 以原始單位(房價)呈現誤差幅度,並對較大誤差給予更高的懲罰,適合房價這類
不允許出現離譜誤差的迴歸任務。

| 項目 | 值 |
|------|-----|
| 競賽 | playground-series-s3e1 |
| URL | https://www.kaggle.com/competitions/playground-series-s3e1 |
| 問題型別 | regression |
| 評估指標 | rmse(minimize) |
| 目標欄位 | MedHouseVal |
| id 欄位 | id |

## 2. 資料規格

本場已完整執行 EDA(素材等級:full)。特徵工程前的原始特徵數為 8(experiments[0].n_features
= 8);最終採用模型(best)使用 24 個特徵(best.n_features = 24)。

**特別規則**(competition.special_rules):
- 不允許外部資料(external_data_allowed: false)
- 不允許預訓練模型(pretrained_models_allowed: false)
- 不允許存取網路(internet_access_allowed: false)
- 每日提交上限:5 次(daily_submission_limit: 5)

## 3. 模型規格

最佳實驗(best, experiment_id=2)為 LGB+XGB+CAT weight-searched blend (engineered features)。
各 base model 的 OOF 分數(best.base_models):

| 模型 | OOF RMSE |
|------|----------|
| LGB | 0.56109 |
| XGB | 0.56297 |
| CAT | 0.56188 |

**Ensemble 權重**(best.ensemble):LGB=0.45、XGB=0.15、CAT=0.4,ensemble score = 0.558768。

**選型理由**:三個樹模型(LightGBM、XGBoost、CatBoost)個別 OOF 分數相近(0.561–0.563 區
間),顯示模型間存在互補的預測誤差,適合以 OOF 網格搜尋(grid search)尋找凸組合
權重做加權平均;搜尋結果偏重 LGB(0.45)與 CAT(0.4),XGB 權重較低(0.15),但三者皆非零,
代表三個模型都對 ensemble 有貢獻。

> 註:best 是以 OOF score 最小者選出(本場 metric 為 rmse,minimize),與是否已提交至 Kaggle
> 無關——本場未提交至 Kaggle(見第 6 節),因此沒有 leaderboard 分數可與 best 對照。

## 4. 訓練規格

**CV 方案**(best.cv):

| scheme | n_splits | shuffle | seed |
|--------|----------|---------|------|
| KFold | 5 | true | 42 |

**為何用此 CV**:目標為連續值且無自然分層依據,資料列彼此獨立(每列為一個普查區塊,無
時間或群組結構),故標準 KFold 為合理選擇,無需 Stratified/Group/Time-based 切分。

**Objective 與關鍵超參**(best.base_models[].params):

| 模型 | objective/loss | 關鍵超參 |
|------|----------------|----------|
| LGB | 未於 params 中記錄 objective 字串,但 model 名稱標示為 RMSE 目標 | n_estimators=2000, learning_rate=0.03, num_leaves=63, min_child_samples=20, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1, random_state=42 |
| XGB | 未於 params 中記錄 objective 字串 | n_estimators=2000, learning_rate=0.03, max_depth=7, min_child_weight=5, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, random_state=42, tree_method=hist |
| CAT | 未於 params 中記錄 loss_function 字串 | iterations=2000, learning_rate=0.03, depth=8, l2_leaf_reg=3.0, random_seed=42 |

## 5. 推論程序

**後處理步驟**(best.postprocess):`clip_to_train_target_range`——將預測值裁切至訓練集
目標欄位的觀察範圍內。

**Submission 格式**:id 欄位為 `id`,目標欄位為 `MedHouseVal`(competition.id_column /
target_column)。

**Submission 檔名**(best.submission):`sub_lgb_xgb_cat_blend_0.55877_20260703_182947.csv`

## 6. 評估指標

**指標定義**:RMSE(Root Mean Squared Error)= 預測誤差平方之平均值的平方根,誤差單位與
目標欄位相同(房價中位數)。

**分數總表**:

| 實驗 | 說明 | OOF RMSE |
|------|------|----------|
| experiment_id=1 | baseline: generic LGB+XGB+CAT blend | 0.56166 |
| experiment_id=2 (best) | LGB+XGB+CAT weight-searched blend (engineered features) | 0.558768 |
| experiment_id=3 | LGB single-model probe: +smoothed geo_cluster_50 target-encoding | 0.561017 |

facts.json 的 `leaderboard` 欄位為 null,且 `missing` 列表包含 `"leaderboard"`——**無紀錄**。
本場並未提交至 Kaggle,因此無 Public/Private LB 分數可供比對,亦無法計算 CV↔LB gap。

best(experiment_id=2)相對 baseline(experiment_id=1)的改善幅度:

```
0.56166 - 0.558768 = 0.002892   (絕對改善)
0.002892 / 0.56166 = 0.0051497...
0.0051497 * 100 ≈ 0.515%        (相對改善百分比)
```

## 7. 實驗軌跡

| experiment_id | timestamp | score | source_format |
|---------------|-----------|-------|----------------|
| 1 | 2026-07-03T12:02:12 | 0.56166 | generic_batch |
| 2 | 2026-07-03T18:29:47 | 0.558768 | v2 |
| 3 | 2026-07-03T18:31:02 | 0.561017 | v2 |

**突破點**:分數躍升發生在 experiment_id=2——由 baseline 的 8 個原始特徵改為 24 個工程化
特徵(新增 households、bedroom_ratio、rooms_per_person、log 轉換、與主要城市的距離、
geo_cluster 等),並以 OOF 網格搜尋權重取代固定權重,使分數由 0.56166 降至 0.558768。

experiment_id=3 為一個反思(reflexion)實驗:假設「以較細緻的 50-cluster 平滑 target
encoding 取代 experiment 2 中粗略的 25-cluster geo_cluster id,應能進一步改善分數」,因
EDA 階段的快速 LGB 模型顯示 Longitude/Latitude 為最重要的兩個原始特徵。實驗結果顯示
LGB 單模型 OOF 從 0.56109 變為 0.56102(delta -0.00007,屬雜訊等級,未帶來實質改善),
故未被採納進最終特徵集(experiment 2 的特徵集維持為最佳)。

facts.json 的 `unparsed` 列表為空——無法解析之紀錄:無。

## 8. 重現指令

```bash
cd /home/tjyen/ai_agents/kaggle

# Stage 1: EDA
uv run python3 competitions/playground-series-s3e1/scripts/eda.py

# Stage 2: Feature engineering (writes train_processed.csv / test_processed.csv)
uv run python3 competitions/playground-series-s3e1/scripts/features.py

# Stage 3: Modeling (LGB/XGB/CatBoost, 5-fold KFold, OOF weight-searched blend,
# writes submission CSV to submissions/ and logs to experiments.json)
uv run python3 competitions/playground-series-s3e1/scripts/train.py

# Stage 3b (optional, informational only — probe not adopted into final model):
uv run python3 competitions/playground-series-s3e1/scripts/tune_geo_te.py

# Report generation
uv run python3 .claude/skills/kaggle-report/assets/collect.py playground-series-s3e1
```

本場未提交至 Kaggle。若要手動提交本次產生的 submission 檔案:

```bash
export KAGGLE_API_TOKEN=$(python3 -c "import json; print(json.load(open('/home/tjyen/.kaggle/kaggle.json'))['key'])")
uv run kaggle competitions submit -c playground-series-s3e1 \
  -f competitions/playground-series-s3e1/submissions/sub_lgb_xgb_cat_blend_0.55877_20260703_182947.csv \
  -m "engineered features blend, OOF 0.558768"
```
