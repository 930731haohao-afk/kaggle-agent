# 競賽分析報告:playground-series-s3e1

> 產生方式:kaggle-report skill(數字來自 facts.json,敘述由 agent 撰寫)
> 素材等級:full | 產生日期:2026-07-06
> 本報告所有數字皆出自 facts.json,經 verify_report.py 驗證。

## 1. 競賽目的

**What**:依加州人口普查區塊(block group)的 8 個統計特徵(收入中位數、屋齡、平均房間數、
平均臥室數、人口、平均入住人數、緯度、經度)預測該區塊的房屋價值中位數(`MedHouseVal`),
為連續數值輸出的迴歸問題,即經典 California Housing 資料集的 Kaggle Playground 版本。

**Why**:評估指標為 **RMSE(minimize)**。房價中位數為連續實數,RMSE 以原始單位呈現誤差
幅度並對大誤差施以平方懲罰,能引導模型優先壓制嚴重錯估,適合房價這類不允許離譜誤差的
迴歸任務。

| 項目 | 值 |
|------|-----|
| 競賽 | playground-series-s3e1 |
| 問題型別 | regression |
| 評估指標 | rmse(minimize) |
| 目標欄位 | MedHouseVal |
| 素材等級 | full |

## 2. 使用工具與環境

| 工具 | 用途 |
|------|------|
| LightGBM / XGBoost / CatBoost | 三個梯度提升樹基礎模型,構成 exp1–exp7 blend 主體與 exp8 7-way blend 的核心成員 |
| Optuna | exp4 對 LGB 做 TPE 超參搜尋(fold-0 proxy,詳見訓練規格節) |
| 自建樹搜尋 harness | Phase D-4 搜尋成員/權重組合空間,於 node #14 找到 exp8 之 7-way blend |
| 5-fold CV 框架(scikit-learn) | KFold(shuffle,seed 42)5 折交叉驗證;另供 KMeans geo_cluster 與 KNN 距離特徵計算 |
| uv | Python 套件與虛擬環境管理,所有腳本皆以 `uv run` 執行 |

本場工具鏈組合邏輯:Claude Code(LLM)負責決策——提出特徵工程假設(地理距離、比例、log
轉換)、判讀探針結果決定採納與否、決定何時停止迭代;Auto-ML 工具(Optuna、樹搜尋 harness)
負責系統化執行超參搜尋與 blend 組合空間探索,兩者分工互補。

## 3. 流程(how):五大元件

### 3.1 資料規格

| 項目 | 值 |
|------|-----|
| train 列數 | 37,137 |
| test 列數 | 24,759 |
| 原始欄位數 | 8 |
| 特別規則摘要 | 禁外部資料/禁預訓練模型/禁網路存取;每日提交上限 5 次 |

8 個原始特徵皆為數值型、無類別欄位;train/test 皆無缺失值、train 無重複列,facts.eda 亦無
高共線特徵對紀錄。目標 `MedHouseVal` 右偏(skew 0.97,mean 2.08、median 1.808),且在
5.00001 處被 top-code(佔 train 4.92%)——此為資料本身的誤差天花板,非建模可修復。

`MedInc` 為最強線性預測子(Pearson 0.70);緯度/經度單看相關性弱,但 EDA 快速 LGB 的重要度
排名將其列為前二,顯示地理訊號為非線性,此觀察驅動了後續的地理特徵工程。train/test 分佈
幾乎一致(平均值差異最大者為 Population 的 1.12%),無 covariate shift 疑慮。

### 3.2 實驗總表

> **語意澄清(本報告全文適用)**:本場指標為連續 RMSE,無取整議題,「決策分數」即(原始)
> OOF RMSE,兩欄恆相等。exp3、exp6 為 **LGB 單模探針**,其分數與 exp2 之 LGB 成員
> (0.56109)相比,不與 blend 分數相比;其餘各列為 blend 分數。

| exp | 階段 | 模型/成員 | 特徵數 | 原始 OOF | 決策分數 | 採納 |
|-----|------|-----------|--------|----------|----------|------|
| 1 | Baseline(通用批次) | LGB/XGB/CAT(.2/.2/.6) | 8 | 0.56166 | 0.56166 | 是,後由 exp2 取代 |
| 2 | Phase A(手刻管線) | LGB/XGB/CAT(.45/.15/.40) | 24 | 0.558768 | 0.558768 | 是,Phase A 最佳 |
| 3 | 迭代探針(geo TE) | LGB 單模(+geo_te) | 25 | 0.561017 | 0.561017 | 否,雜訊等級 |
| 4 | Phase B round1 | +LGB_TUNED(4-way) | 24 | 0.557977 | 0.557977 | 是 |
| 5 | Phase B round2 | +LGB_TUNED_SEED2024(5-way) | 24 | 0.557859 | 0.557859 | 是 |
| 6 | Phase B round3 探針 | LGB 單模(+2 個 geo 特徵) | 26 | 0.560444 | 0.560444 | 是,特徵獲採納 |
| 7 | Phase B round4 | 5-way blend(26 特徵) | 26 | 0.557088 | 0.557088 | 是,線性迭代最佳 |
| 8 | Phase D-4 樹搜尋(best) | 7-way blend(node #14) | 26 | **0.556329** | **0.556329** | 是,本場最佳(OOF-only) |

> **誠實但書**:exp8 為樹搜尋之 OOF-only 結果——未產生 test 預測、未提交 Kaggle,
> facts.json 本筆無 submission 欄位;其特徵數 26 出自 notes(與 exp7 同一特徵集),
> 非結構化欄位。

**best 成員表(exp8,7-way blend)**

| 成員 | 權重 | solo 分數 | 備註 |
|------|------|-----------|------|
| LGBORIG | 0.150 | 0.560732 | — |
| XGBORIG | 0.036 | 0.561331 | — |
| CATORIG | 0.249 | 0.561063 | — |
| SEEDBAG | 0.186 | 0.558812 | 即 LGB_TUNED_SEED2024,root 之 seed-bag 同胞 |
| REGNUDGE | 0.050 | 0.558208 | 正則化微調之 tuned-LGB;全樹最佳 solo |
| XGB_deep | 0.041 | 無紀錄 | 刻意多樣性成員 |
| CEILING | 0.289 | 0.561227 | top-code 感知兩階段 hybrid;solo 最弱但權重最大,單筆最大增益 |

選型理由:三樹模型 solo 分數相近、誤差互補,適合 OOF 權重搜尋的凸組合;Phase B 依經驗庫
「調參後單模加入 pool 而非替換」原則逐步擴充成員。exp8 由 harness v2 之 clip-aware
dirichlet(k=800)+coordinate-ascent 權重搜尋找到(22 個評估節點,wall 190.3s),CEILING
成員印證「blend 貢獻 ≠ solo 分數」。

### 3.3 訓練規格表

| exp | CV 方案 | folds | seed |
|-----|---------|-------|------|
| 1 | 5fold | 5 | 無紀錄 |
| 2 | KFold(shuffle) | 5 | 42 |
| 3 | KFold(shuffle) | 5 | 42 |
| 4 | KFold(shuffle) | 5 | 42 |
| 5 | KFold(shuffle) | 5 | 42 |
| 6 | KFold(shuffle) | 5 | 42 |
| 7 | KFold(shuffle) | 5 | 42 |
| 8 | KFold(shuffle) | 5 | 42 |

目標為連續值、各列為獨立普查區塊(無群組/時間結構),且 train/test 分佈近乎一致,標準
KFold 即為教科書式正解;自 exp2 起 CV 全程固定,確保跨實驗分數可直接比較。

Objective 一律為 RMSE。exp4 之 LGB_TUNED 以 Optuna TPE fold-0 proxy 調參(50 trials,搜尋
74.3s、全 CV 復驗 13.2s),關鍵超參:learning_rate≈0.0116、num_leaves=121、max_depth=10、
min_child_samples=82;seed-bag 成員僅將 random_state 由 42 改為 2024。exp8 新增成員
(REGNUDGE、XGB_deep、CEILING)之 params 無紀錄。

### 3.4 推論表

| exp | 後處理 | submission 檔 | 已提交 |
|-----|--------|---------------|--------|
| 1 | 無後處理紀錄 | sub_generic_0.56166_20260703_120212.csv | 否 |
| 2 | clip_to_train_target_range | sub_lgb_xgb_cat_blend_0.55877_20260703_182947.csv | 否 |
| 3 | 無後處理紀錄 | 無紀錄(單模探針) | 否 |
| 4 | clip_to_train_target_range | sub_lgb_xgb_cat_lgbtuned_blend_0.55798_20260703_212430.csv | 否 |
| 5 | clip_to_train_target_range | sub_5way_seedbag_blend_0.55786_20260703_212521.csv | 否 |
| 6 | 無後處理紀錄 | 無紀錄(單模探針) | 否 |
| 7 | clip_to_train_target_range | sub_round4_geofeat_5way_blend_0.55709_20260703_212811.csv | 否 |
| 8 | clip(於 metric_fn 內對 OOF 裁切評分) | 無紀錄(OOF-only,未產生 test 預測) | 否 |

`id_column = id`、`target_column = MedHouseVal`。exp8 與 exp2–exp7 的差異在裁切時點:線性
各輪僅於 submission 階段裁切,exp8 於權重搜尋的 metric_fn 內部即以裁切後 OOF 評分。本場為
無人值守批次執行,所有 submission 檔皆未上傳 Kaggle。

### 3.5 評估指標 / 排行榜

指標定義:RMSE = 預測誤差平方平均之平方根,單位與 `MedHouseVal` 相同。

本場無 LB 紀錄(未提交)——facts.json 之 `leaderboard` 為 null、missing 含 "leaderboard",
故無排行榜表,亦無 CV↔LB gap 可計。全場分數的唯一錨點為同一固定 CV 下的 OOF RMSE,最佳為
exp8 之 **0.556329**(各實驗與各成員分數見 3.2 節)。

## 4. 實驗軌跡

| exp | 時間 | 決策分數 | 階段 | 一句話摘要 |
|-----|------|----------|------|------------|
| 1 | 2026-07-03T12:02:12 | 0.56166 | Baseline(通用批次) | 8 原始特徵三模型 blend,固定權重基線 |
| 2 | 2026-07-03T18:29:47 | 0.558768 | Phase A | 24 工程化特徵+OOF 權重搜尋,全場最大單筆躍升 |
| 3 | 2026-07-03T18:31:02 | 0.561017 | 迭代探針 | geo target-encoding 探針,增益屬雜訊未採納 |
| 4 | 2026-07-03T21:24:30 | 0.557977 | Phase B round1 | Optuna 調參 LGB 加入 pool,4-way blend 改善 |
| 5 | 2026-07-03T21:25:21 | 0.557859 | Phase B round2 | seed-bag 調參 LGB 為第 5 成員,續改善 |
| 6 | 2026-07-03T21:26:27 | 0.560444 | Phase B round3 探針 | KNN 鄰距+海岸距離特徵,LGB 探針過門檻採納 |
| 7 | 2026-07-03T21:28:11 | 0.557088 | Phase B round4 | 5 成員 pool 於 26 特徵全量重訓,線性迭代最佳 |
| 8 | 2026-07-04T11:59:43 | **0.556329** | Phase D-4 樹搜尋 | 7-way clip-aware blend(node #14),本場最佳 |

- **突破點 1(exp1→exp2)**:特徵工程(比例、log 轉換、城市距離、geo_cluster)加上 OOF
  權重搜尋,0.56166→0.558768,為全場最大單筆增益。
- **突破點 2(exp3 vs exp6)**:粗粒度地理 target-encoding 無效(exp3,雜訊等級),但非
  target-encoding 的幾何特徵(KNN 密度、海岸距離)帶來真實增益(exp6)——同一「地理」
  槓桿、不同機制,結局相反。
- **突破點 3(exp7→exp8)**:樹搜尋以 clip-aware 評分與 CEILING hybrid 成員把分數推至
  0.556329;CEILING solo 最弱卻拿最大權重,是全樹單筆最大增益來源。

facts.unparsed 為空陣列,無法解析之紀錄:無。

## 5. 效能對照:四層消融

| 層級 | 配置 | 分數 | 相對改善 |
|------|------|------|----------|
| tier1 | 基線(Claude Code 直接執行,未引入 skill;exp1 通用批次 blend) | 0.56166 | —(基線) |
| tier2 | + kaggle-agent skill 六階段流程(exp2 Phase A 手刻管線) | 0.558768 | 見下方算式 |
| tier3 | + self-improvement 線性迭代(exp7 Phase B 四輪終點) | 0.557088 | 見下方算式 |
| tier4 | + 樹搜尋(exp8,node #14 7-way blend) | **0.556329** | 見下方算式 |

```
tier1→tier2: 0.56166 − 0.558768 = 0.002892,相對改善 0.002892 / 0.56166 = 0.5149%
tier2→tier3: 0.558768 − 0.557088 = 0.001680,相對改善 0.001680 / 0.558768 = 0.3007%
tier3→tier4: 0.557088 − 0.556329 = 0.000759,相對改善 0.000759 / 0.557088 = 0.1362%
```

本場由導入報告功能後之版本執行,分數自 tier2 起未低於前一層——符合計畫書目標三(效能不退步)。

## 6. 總結

本場資料乾淨(無缺失、無重複列)且全為數值特徵,建模的兩個結構性事實在 EDA 即已確立:
目標在 5.00001 處被 top-code(佔 train 4.92%),構成誤差天花板;地理訊號強但非線性,
單變量相關性看不出來。這兩點分別決定了後段的裁切後處理與前段的地理特徵工程方向。

關鍵決策有二。其一是自 exp2 起固定 KFold(5 折、shuffle、seed 42)不再變動,使全場八個
實驗的 OOF 分數可直接比較;其二是探針紀律——新想法先以 LGB 單模探針驗證,過雜訊門檻才
併入 pool(exp6 採納、exp3 否決),避免把雜訊當進步。

增益來源逐層遞減且各有歸屬:tier1→tier2 靠特徵工程與權重搜尋(0.56166→0.558768),是
最大單筆增益;tier2→tier3 靠 Phase B 四輪線性迭代(調參、seed bagging、幾何特徵)推進至
0.557088;tier3→tier4 靠樹搜尋的兩個本場專屬槓桿——clip-aware 評分與 top-code 感知的
CEILING 成員——收於 0.556329。

結果的可信度建立在同一固定 CV 的 OOF 之上,且逐層皆為正向改善;但須誠實註記:本場全程
未提交 Kaggle,無 LB 外部驗證,且 exp8 為 OOF-only 搜尋產物、未產生 test 預測,目前可直接
提交的最佳檔案仍是 exp7 的 submission。

**重現本實驗的最短路徑**:見第 7 節。

## 7. 重現指令

```bash
cd /home/tjyen/ai_agents/kaggle

# Stage 1: EDA
uv run python3 competitions/playground-series-s3e1/scripts/eda.py

# Stage 2: Feature engineering (writes train_processed.csv / test_processed.csv)
uv run python3 competitions/playground-series-s3e1/scripts/features.py

# Stage 3: Modeling — exp2 (LGB/XGB/CatBoost, 5-fold KFold, OOF weight-searched blend)
uv run python3 competitions/playground-series-s3e1/scripts/train.py

# exp3 (optional, informational only — probe not adopted):
uv run python3 competitions/playground-series-s3e1/scripts/tune_geo_te.py

# Phase B round1 — exp4: Optuna fold-0-proxy tuning of LGB, add to pool (4-way blend)
uv run python3 competitions/playground-series-s3e1/scripts/tune_lgb_optuna.py

# Phase B round2 — exp5: seed bagging of the tuned LGB (5-way blend)
uv run python3 competitions/playground-series-s3e1/scripts/seed_bag_round2.py

# Phase B round3 — exp6: KNN-density + coastal-distance geo feature probe
# (writes train_processed_v2.csv / test_processed_v2.csv if adopted)
uv run python3 competitions/playground-series-s3e1/scripts/geo_knn_probe.py

# Phase B round4 — exp7: full 5-member pool retrain on the 26-feature set
uv run python3 competitions/playground-series-s3e1/scripts/round4_full_retrain.py

# Phase D-4 tree search — exp8 (resumable; tree state in experiments_tree.json; OOF-only)
uv run python3 tree_search/run_s3e1.py

# Report generation
uv run python3 .claude/skills/kaggle-report/assets/collect.py playground-series-s3e1
```

本場未提交至 Kaggle。若要手動提交目前可提交的最佳 submission 檔(exp7):

```bash
export KAGGLE_API_TOKEN=$(python3 -c "import json; print(json.load(open('/home/tjyen/.kaggle/kaggle.json'))['key'])")
uv run kaggle competitions submit -c playground-series-s3e1 \
  -f competitions/playground-series-s3e1/submissions/sub_round4_geofeat_5way_blend_0.55709_20260703_212811.csv \
  -m "5-way blend (Optuna-tuned LGB + seed bag + KNN/coastal geo features), OOF 0.557088"
```

執行目錄為專案根目錄 `/home/tjyen/ai_agents/kaggle`;所有 Python 執行皆透過 `uv run`。
