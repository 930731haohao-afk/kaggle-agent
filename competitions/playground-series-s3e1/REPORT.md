# 競賽分析報告:playground-series-s3e1

> 產生方式:kaggle-report skill(數字來自 facts.json,敘述由 agent 撰寫)
> 素材等級:full | 產生日期:2026-07-03(Phase B 自我改進迭代後更新)

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
= 8);最終採用模型(best,experiment_id=7)使用 26 個特徵(best.n_features = 26,較先前
版本報告記載的 24 個新增 2 個:`knn_mean_dist_10`、`coastal_dist`,見第 7 節)。

**特別規則**(competition.special_rules):
- 不允許外部資料(external_data_allowed: false)
- 不允許預訓練模型(pretrained_models_allowed: false)
- 不允許存取網路(internet_access_allowed: false)
- 每日提交上限:5 次(daily_submission_limit: 5)

## 3. 模型規格

最佳實驗(best, experiment_id=7)為「5-way blend on 26 features (+knn_mean_dist_10
+coastal_dist)」——LGB、XGB、CAT 三個原始基模型,加上兩個以 Optuna 調參之 LightGBM 衍生
成員(`LGB_TUNED`:Optuna fold-0-proxy 調參版;`LGB_TUNED_SEED2024`:同超參、僅
random_state 改為 seed bagging 版)。各 base model 的 OOF 分數(best.base_models):

| 模型 | OOF RMSE |
|------|----------|
| LGB | 0.56073 |
| XGB | 0.56133 |
| CAT | 0.56106 |
| LGB_TUNED | 0.55887 |
| LGB_TUNED_SEED2024 | 0.55881 |

**Ensemble 權重**(best.ensemble):LGB=0.2、XGB=0.1、CAT=0.25、LGB_TUNED=0.25、
LGB_TUNED_SEED2024=0.2,ensemble score = 0.557088。

**選型理由**:五個成員彼此 OOF 分數相近(0.5588–0.5613 區間),顯示模型間存在互補的預測
誤差,適合以 OOF 網格搜尋(grid search)尋找凸組合權重做加權平均;搜尋結果給予兩個
Optuna 調參衍生的 LGB 成員合計 0.45 權重(0.25+0.2),高於三個原始基模型各自的權重,反映
調參後的模型品質較高,但三個原始基模型權重皆非零(未被完全淘汰),代表 ensemble 多樣性
仍有貢獻——這與 knowledge/experience.md 記載的「調參後的單模應加入 pool 而非替換原成員」
經驗一致(見第 7 節)。

> 註:best 是以 OOF score 最小者選出(本場 metric 為 rmse,minimize),與是否已提交至 Kaggle
> 無關——本場未提交至 Kaggle(見第 6 節),因此沒有 leaderboard 分數可與 best 對照。

## 4. 訓練規格

**CV 方案**(best.cv):

| scheme | n_splits | shuffle | seed |
|--------|----------|---------|------|
| KFold | 5 | true | 42 |

**為何用此 CV**:目標為連續值且無自然分層依據,資料列彼此獨立(每列為一個普查區塊,無
時間或群組結構),故標準 KFold 為合理選擇,無需 Stratified/Group/Time-based 切分。此 CV
方案自 experiment_id=2 起全程未變,確保跨實驗分數可直接比較(knowledge/experience.md「CV
設計」節的鐵律)。

**Objective 與關鍵超參**(best.base_models[].params,僅 LGB_TUNED / LGB_TUNED_SEED2024
於 facts.json 中記錄了 params;LGB/XGB/CAT 三個原始基模型在本筆實驗紀錄中僅存分數,其
超參與 experiment_id=2 相同,見該筆紀錄):

| 模型 | 關鍵超參 |
|------|----------|
| LGB_TUNED(Optuna fold-0-proxy 調參,50 trials) | learning_rate≈0.0116, num_leaves=121, max_depth=10, min_child_samples=82, subsample≈0.652, colsample_bytree≈0.549, reg_alpha≈0.546, reg_lambda≈0.058, n_estimators=2000, random_state=42 |
| LGB_TUNED_SEED2024(同超參,seed bagging) | 同上,唯 random_state=2024 |

> facts.json 本筆(experiment_id=7)的 base_models 未附 LGB/XGB/CAT 的 params 欄位;其
> 超參與 experiment_id=2 紀錄的 LGB(num_leaves=63 等)、XGB(max_depth=7 等)、
> CAT(depth=8 等)完全相同,僅訓練資料的特徵集從 24 擴充為 26 欄——無紀錄之處在此明確
> 標註,不臆測。

## 5. 推論程序

**後處理步驟**(best.postprocess):`clip_to_train_target_range`——將預測值裁切至訓練集
目標欄位的觀察範圍內(涵蓋 EDA 已知的目標欄位 top-code 上限,細節見 STATUS.md EDA 節,
數值未記入 facts.json)。

**Submission 格式**:id 欄位為 `id`,目標欄位為 `MedHouseVal`(competition.id_column /
target_column)。

**Submission 檔名**(best.submission):`sub_round4_geofeat_5way_blend_0.55709_20260703_212811.csv`

## 6. 評估指標

**指標定義**:RMSE(Root Mean Squared Error)= 預測誤差平方之平均值的平方根,誤差單位與
目標欄位相同(房價中位數)。

**分數總表**:

| 實驗 | 說明 | OOF RMSE |
|------|------|----------|
| experiment_id=1 | baseline: generic LGB+XGB+CAT blend | 0.56166 |
| experiment_id=2 | LGB+XGB+CAT weight-searched blend(24 個工程化特徵) | 0.558768 |
| experiment_id=3 | LGB 單模型探針:+smoothed geo_cluster_50 target-encoding(未採納) | 0.561017 |
| experiment_id=4 | +Optuna 調參 LGB,4-way blend | 0.557977 |
| experiment_id=5 | +seed-bagged 調參 LGB,5-way blend | 0.557859 |
| experiment_id=6 | LGB 單模型探針:+knn_mean_dist_10 +coastal_dist(採納) | 0.560444 |
| experiment_id=7(best) | 5-way blend,26 個特徵(含 KNN/海岸距離) | 0.557088 |

facts.json 的 `leaderboard` 欄位為 null,且 `missing` 列表包含 `"leaderboard"`——**無紀錄**。
本場並未提交至 Kaggle,因此無 Public/Private LB 分數可供比對,亦無法計算 CV↔LB gap。

best(experiment_id=7)相對 exp-2 blend 與 exp-1 baseline 的改善幅度:

```
exp2 -> exp7:  0.558768 - 0.557088 = 0.00168   (絕對改善)
               0.00168 / 0.558768 = 0.0030066...
               0.0030066 * 100 ≈ 0.301%        (相對改善百分比)

exp1 -> exp7:  0.56166 - 0.557088 = 0.004572   (絕對改善)
               0.004572 / 0.56166 = 0.0081386...
               0.0081386 * 100 ≈ 0.814%        (相對改善百分比)
```

## 7. 實驗軌跡

| experiment_id | timestamp | score | source_format |
|---------------|-----------|-------|----------------|
| 1 | 2026-07-03T12:02:12 | 0.56166 | generic_batch |
| 2 | 2026-07-03T18:29:47 | 0.558768 | v2 |
| 3 | 2026-07-03T18:31:02 | 0.561017 | v2 |
| 4 | 2026-07-03T21:24:30 | 0.557977 | v2 |
| 5 | 2026-07-03T21:25:21 | 0.557859 | v2 |
| 6 | 2026-07-03T21:26:27 | 0.560444 | v2 |
| 7 | 2026-07-03T21:28:11 | 0.557088 | v2 |

**突破點 1(exp 1→2)**:分數躍升發生在 experiment_id=2——由 baseline 的 8 個原始特徵改為
24 個工程化特徵(新增 households、bedroom_ratio、rooms_per_person、log 轉換、與主要城市的
距離、geo_cluster 等),並以 OOF 網格搜尋權重取代固定權重,使分數由 0.56166 降至 0.558768。

**Phase B 自我改進迭代(本次更新,4 輪)**:

- **Round 1(exp 4)**:對最強單模 LGB 執行 Optuna TPE 超參搜尋,採用 knowledge/experience.md
  記載的「fold-0 代理目標」調參法(避免完整 5-fold × 50 trials 逾時),再以全 5-fold 驗證
  得 LGB_TUNED 單模 OOF(優於原 LGB 的 0.56109)。依經驗庫建議「調參後的單模應加入 pool
  而非替換」,將其加為第 4 個 blend 成員,4-way 權重搜尋後 blend 降至 0.557977(較 exp 2
  的 0.558768 之改善見下方程式)。
- **Round 2(exp 5)**:對 Round 1 的調參超參執行 seed bagging(random_state 由 42 改為
  2024,其餘超參不變),新增第 5 個 blend 成員,5-way 權重搜尋後 blend 降至
  0.557859——符合經驗庫「seed bagging 是調參之後最便宜的殘餘增益」的預期。
- **Round 3(exp 6,探針)**:延續「地理座標資料」既有經驗(exp 3 曾證實粗粒度地理 target
  encoding 對此資料無增益),改測試兩個非 target-encoding 的原始幾何特徵:
  `knn_mean_dist_10`(對 train+test 合併座標做 KNN 平均鄰距,作為區塊密度代理)與
  `coastal_dist`(至加州海岸線錨點的最近距離)。以與 exp 3 相同的單模型探針法驗證,LGB
  單模 OOF 由 0.56109 降至 0.560444,判定為真實增益(高於既定雜訊門檻,見下方程式)而非
  雜訊,予以採納;探針腳本輸出顯示 `coastal_dist` 與目標的相關係數為新特徵中最強者之一
  (未記入 facts.json,細節見 `scripts/geo_knn_probe.py` 執行紀錄)。
- **Round 4(exp 7,best)**:在 Round 3 採納的 26 特徵集上,重新訓練 Round 2 的完整 5 成員
  pool(LGB、XGB、CAT、LGB_TUNED、LGB_TUNED_SEED2024),重新做 5-way 權重搜尋,blend 降至
  0.557088(較 Round 2 之改善見下方程式)。

各輪改善幅度(皆由 facts.json 中已列出的 score 相減而得,逐輪皆為改善,無需觸發「連續 2
輪未改善」停止準則;迭代在第 4 輪後停止,落在 protocol 建議的 2–4 輪區間上限,且訓練總
耗時遠低於 30 分鐘預算):

```
Round1: exp2 0.558768 - exp4 0.557977 = 0.000791   (改善)
Round2: exp4 0.557977 - exp5 0.557859 = 0.000118   (改善)
Round3 探針: exp2-LGB 0.56109 - exp6 0.560444 = 0.000646
            門檻(noise-level cutoff 採用) 0.0005 → 0.000646 > 0.0005,判定為真實增益
Round4: exp5 0.557859 - exp7 0.557088 = 0.000771   (改善)
```

**experiment_id=3 反思(exp 2 之後、Phase B 之前)**:假設「以較細緻的 50-cluster 平滑
target encoding 取代 experiment 2 中粗略的 25-cluster geo_cluster id,應能進一步改善
分數」,因 EDA 階段的快速 LGB 模型顯示 Longitude/Latitude 為最重要的兩個原始特徵。實驗
結果顯示 LGB 單模型 OOF 從 0.56109 變為 0.56102(delta -0.00007,屬雜訊等級,未帶來實質
改善,見 experiments.json experiment_id=3 之 notes),故未被採納。此結果與 Round 3 的對照
凸顯了關鍵區別:粗粒度地理**target encoding**無效,但非 target-encoding 的**幾何距離
特徵**(KNN 密度、海岸距離)仍能提供樹模型難以自行從原始座標切分推導出的訊號。

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

# Phase B iteration round 1: Optuna fold-0-proxy tuning of LGB, add to pool (4-way blend)
uv run python3 competitions/playground-series-s3e1/scripts/tune_lgb_optuna.py

# Phase B iteration round 2: seed bagging of the tuned LGB (5-way blend)
uv run python3 competitions/playground-series-s3e1/scripts/seed_bag_round2.py

# Phase B iteration round 3: KNN-density + coastal-distance geo feature probe
# (writes train_processed_v2.csv / test_processed_v2.csv if adopted)
uv run python3 competitions/playground-series-s3e1/scripts/geo_knn_probe.py

# Phase B iteration round 4: full 5-member pool retrain on the 26-feature set (best, exp 7)
uv run python3 competitions/playground-series-s3e1/scripts/round4_full_retrain.py

# Report generation
uv run python3 .claude/skills/kaggle-report/assets/collect.py playground-series-s3e1
```

本場未提交至 Kaggle。若要手動提交本次產生的最佳 submission 檔案:

```bash
export KAGGLE_API_TOKEN=$(python3 -c "import json; print(json.load(open('/home/tjyen/.kaggle/kaggle.json'))['key'])")
uv run kaggle competitions submit -c playground-series-s3e1 \
  -f competitions/playground-series-s3e1/submissions/sub_round4_geofeat_5way_blend_0.55709_20260703_212811.csv \
  -m "5-way blend (Optuna-tuned LGB + seed bag + KNN/coastal geo features), OOF 0.557088"
```
