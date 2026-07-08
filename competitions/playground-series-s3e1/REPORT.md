# 競賽分析報告:playground-series-s3e1

> 本報告由自動化報告流程產生:所有數字直接取自實驗原始紀錄並經自動一致性驗證,敘述由 AI 彙整。
> 紀錄完整度:完整 | 產生日期:2026-07-06

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

## 2. 本場工具、專業術語與採用策略

> 全案共同的研究設計、實驗流程、工具鏈、專業術語與五階段定義,見《前言》
> (docs/PREFACE.pdf);本節僅列本場特有的部分。

### 2.1 本場工具

全數為共同工具鏈(見《前言》第 3 節),無本場特有工具;樹搜尋工具本場使用第 2 版。
交叉驗證採標準 KFold(shuffle)5 折(設計理由見第 3.3 節)。

### 2.2 本場專業術語

| 專業術語 | 定義 |
|------|-----------|
| clip(裁切) | 將預測值限制在訓練集目標值的範圍內,避免離譜的外插預測 |
| top-code | 資料集把高於某上限的目標值一律記為該上限值,形成分數的天花板 |

### 2.3 採用策略

各階段與子階段的定義見《前言》第 6 節(編號跨場同義);主階段 1–4 本場皆有執行,
各子階段採用情形如下。

**階段 1｜無 skill 基線**

- 1.1 單模型基線 — 使用
- 1.2 三模型 blend — 使用

**階段 2｜kaggle-agent skill**

- 2.1 EDA 驅動 CV 設計 — 使用(併入 第 2 次實驗,無獨立分數)
- 2.2 EDA 驅動特徵工程 — 使用(併入 第 2 次實驗,無獨立分數)
- 2.3 指標感知後處理 — 使用(clip 裁切,併入 第 2 次實驗)
- 2.4 場內反思回退 — 使用(第 3 次實驗 先以單模快速試地理目標編碼,判定屬雜訊未採納)

**階段 3｜+線性自我迭代**

- 3.1 經驗庫先驗 — 本場未使用(經驗庫晚於本場建立)
- 3.2 Optuna 超參搜尋 — 使用
- 3.3 調參入池 — 使用(含於 3.2 之作法)
- 3.4 seed bagging — 使用
- 3.5 結構化去噪 — 本場未使用

**階段 4｜+樹搜尋**

- 4.1 單模型節點樹 — 本場未使用
- 4.2 ensemble 節點樹 — 使用
- 4.3 先驗注入+去重 — 使用(v2 內建)
- 4.4 邊界推進 mutation — 本場未使用
- 4.5 預算相位機 — 本場未使用

**階段 5｜+外部想法注入** — 全案已實測:注入機制建置完成,但注入候選未勝過階段4(階段5=階段4,見總結報告第 7 節)

## 3. 實驗方法

### 3.1 資料規格

| 項目 | 值 |
|------|-----|
| train 列數 | 37,137 |
| test 列數 | 24,759 |
| 原始欄位數 | 8 |
| 特別規則摘要 | 禁外部資料/禁預訓練模型/禁網路存取;每日提交上限 5 次 |

8 個原始特徵皆為數值型、無類別欄位;train/test 皆無缺失值、train 無重複列,亦無
高共線特徵對。目標 `MedHouseVal` 右偏(skew 0.97,mean 2.08、median 1.808),且在
5.00001 處被 top-code(佔 train 4.92%)——此為資料本身的誤差天花板,非建模可修復。

`MedInc` 為最強線性預測子(Pearson 0.70);緯度/經度單看相關性弱,但 EDA 快速 LGB 的重要度
排名將其列為前二,顯示地理訊號為非線性,此觀察驅動了後續的地理特徵工程。train/test 分佈
幾乎一致(平均值差異最大者為 Population 的 1.12%),無 covariate shift 疑慮。

### 3.2 實驗總表

> **語意澄清(本報告全文適用)**:本場指標為連續 RMSE,無取整議題,「決策分數」即(原始)
> OOF RMSE,兩欄恆相等。第 3、6 次實驗 為 **LGB 單模快速試驗**(新想法先只訓練單一 LGB
> 看分數,過門檻才納入完整流程),其分數與 第 2 次實驗 之 LGB 成員
> (0.56109)相比,不與 blend 分數相比;其餘各列為 blend 分數。

| 實驗編號 | 階段 | 模型/成員 | 特徵數 | 原始 OOF | 決策分數 | 是否採納 |
|-----|------|-----------|--------|----------|----------|------|
| 1 | 1.2 | LGB/XGB/CAT(.2/.2/.6) | 8 | 0.56166 | 0.56166 | 是,後由 第 2 次實驗 取代 |
| 2 | **2** | LGB/XGB/CAT(.45/.15/.40) | 24 | 0.558768 | 0.558768 | 是,階段 2 最佳 |
| 3 | 2.4 | LGB 單模(+geo_te) | 25 | 0.561017 | 0.561017 | 否,雜訊等級 |
| 4 | 3.2 | +LGB_TUNED(4-way) | 24 | 0.557977 | 0.557977 | 是 |
| 5 | 3.4 | +LGB_TUNED_SEED2024(5-way) | 24 | 0.557859 | 0.557859 | 是 |
| 6 | **3** | LGB 單模(+2 個 geo 特徵) | 26 | 0.560444 | 0.560444 | 是,特徵獲採納 |
| 7 | **3** | 5-way blend(26 特徵) | 26 | 0.557088 | 0.557088 | 是,線性迭代最佳 |
| 8 | 4.2 | 7-way blend(node #14) | 26 | **0.556329** | **0.556329** | 是,本場最佳(僅 OOF 分數) |
| 9 | **5** | 外部注入快評候選(EXT-09(stacking);沿用階段4成員 OOF) | 26 | 0.556369 | 0.556369 | 否(未勝過階段4,見 §5) |

> **補充說明**:第 8 次實驗 為樹搜尋結果,只算出 OOF 分數——未產生 test 預測、未提交 Kaggle,
> 該筆實驗紀錄無 submission 欄位;其特徵數 26 出自紀錄的文字備註(與 第 7 次實驗 同一特徵集),
> 非正式數值欄位。

**最佳解成員表(第 8 次實驗,7-way blend)**

| 成員 | 權重 | solo 分數 | 備註 |
|------|------|-----------|------|
| LGBORIG | 0.150 | 0.560732 | — |
| XGBORIG | 0.036 | 0.561331 | — |
| CATORIG | 0.249 | 0.561063 | — |
| SEEDBAG | 0.186 | 0.558812 | 即 LGB_TUNED_SEED2024,root 之 seed-bag 同胞 |
| REGNUDGE | 0.050 | 0.558208 | 正則化微調之 tuned-LGB;全樹最佳 solo |
| XGB_deep | 0.041 | 無紀錄 | 刻意多樣性成員 |
| CEILING | 0.289 | 0.561227 | top-code 感知的兩階段組合成員;solo 最弱但權重最大,單筆最大增益 |

選型理由:三樹模型 solo 分數相近、誤差互補,適合 OOF 權重搜尋的凸組合;階段 3 的線性
迭代依「調參後單模加入池而非替換」原則逐步擴充成員。第 8 次實驗 由樹搜尋工具(第 2 版)的
權重搜尋找到(評分時先做 clip 裁切再計分;22 個評估節點,耗時 190.3 秒),CEILING 成員
印證「blend 貢獻 ≠ solo 分數」。

### 3.3 訓練規格表

| 實驗編號 | CV 方案 | folds | seed |
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
KFold 即為教科書式正解;自 第 2 次實驗 起 CV 全程固定,確保跨實驗分數可直接比較。

Objective 一律為 RMSE。第 4 次實驗 之 LGB_TUNED 以 Optuna(TPE)調參——每組參數先只在第 0 折
評分以省時,勝出者再以完整 5 折復驗(50
trials,搜尋耗時 74.3 秒、全 CV 復驗 13.2 秒),關鍵超參:learning_rate≈0.0116、
num_leaves=121、max_depth=10、min_child_samples=82;seed-bag 成員僅將 random_state 由
42 改為 2024。第 8 次實驗 新增成員(REGNUDGE、XGB_deep、CEILING)之超參無紀錄。

### 3.4 推論表

| 實驗編號 | 後處理 | submission 檔 | 是否已提交 |
|-----|--------|---------------|--------|
| 1 | 無後處理紀錄 | sub_generic_0.56166_20260703_120212.csv | 否 |
| 2 | clip_to_train_target_range | sub_lgb_xgb_cat_blend_0.55877_20260703_182947.csv | 否 |
| 3 | 無後處理紀錄 | 無紀錄(單模快速試驗) | 否 |
| 4 | clip_to_train_target_range | sub_lgb_xgb_cat_lgbtuned_blend_0.55798_20260703_212430.csv | 否 |
| 5 | clip_to_train_target_range | sub_5way_seedbag_blend_0.55786_20260703_212521.csv | 否 |
| 6 | 無後處理紀錄 | 無紀錄(單模快速試驗) | 否 |
| 7 | clip_to_train_target_range | sub_round4_geofeat_5way_blend_0.55709_20260703_212811.csv | 否 |
| 8 | clip(於評分函式內對 OOF 裁切評分) | 無紀錄(僅 OOF 分數,未產生 test 預測) | 否 |

`id_column = id`、`target_column = MedHouseVal`。第 8 次實驗 與 第 2–7 次實驗 的差異在裁切時點:線性
各輪僅於 submission 階段裁切,第 8 次實驗 於權重搜尋的評分函式內部即以裁切後 OOF 評分。本場為
無人值守批次執行,所有 submission 檔皆未上傳 Kaggle。

### 3.5 評估指標 / 排行榜

指標定義:RMSE = 預測誤差平方平均之平方根,單位與 `MedHouseVal` 相同。

本場未提交 Kaggle,無排行榜紀錄,故無排行榜表,亦無 CV↔LB gap 可計。全場分數的唯一比較
基準為同一固定 CV 下的 OOF RMSE,最佳為 第 8 次實驗 之 **0.556329**(各實驗與各成員分數見第 3.2 節)。

## 4. 實驗軌跡

| 實驗編號 | 時間 | 決策分數 | 階段 | 摘要 |
|-----|------|----------|------|------------|
| 1 | 2026-07-03T12:02:12 | 0.56166 | 1.2 | 8 原始特徵三模型 blend,固定權重基線 |
| 2 | 2026-07-03T18:29:47 | 0.558768 | **2** | 24 工程化特徵+OOF 權重搜尋,全場最大單筆躍升 |
| 3 | 2026-07-03T18:31:02 | 0.561017 | 2.4 | 單模快速試 geo target-encoding,增益屬雜訊未採納 |
| 4 | 2026-07-03T21:24:30 | 0.557977 | 3.2 | Optuna 調參 LGB 加入池,4-way blend 改善 |
| 5 | 2026-07-03T21:25:21 | 0.557859 | 3.4 | seed-bag 調參 LGB 為第 5 成員,續改善 |
| 6 | 2026-07-03T21:26:27 | 0.560444 | **3** | KNN 鄰距+海岸距離特徵,LGB 單模快速試驗過門檻採納 |
| 7 | 2026-07-03T21:28:11 | 0.557088 | **3** | 5 成員池於 26 特徵全量重訓,線性迭代最佳 |
| 8 | 2026-07-04T11:59:43 | **0.556329** | 4.2 | 7-way blend(先裁切再評分,node #14),本場最佳 |
| 9 | 2026-07-08T11:46:18 | 0.556369 | 5 | 外部注入快評:EXT-09(stacking) 候選 OOF 未勝過階段4,不採納 |

- **轉折 1(第 1 次實驗→第 2 次實驗)**:特徵工程(比例、log 轉換、城市距離、geo_cluster)加上 OOF
  權重搜尋,0.56166→0.558768,為全場最大單筆增益。
- **轉折 2(第 3 次實驗 vs 第 6 次實驗)**:粗粒度地理 target-encoding 無效(第 3 次實驗,雜訊等級),但非
  target-encoding 的幾何特徵(KNN 密度、海岸距離)帶來真實增益(第 6 次實驗)——同一「地理」
  槓桿、不同機制,結局相反。
- **轉折 3(第 7 次實驗→第 8 次實驗)**:樹搜尋以「先裁切再評分」與 top-code 感知的 CEILING 成員把分數
  推至 0.556329;CEILING solo 最弱卻拿最大權重,是全樹單筆最大增益來源。

## 5. 效能對照(逐階段)

| 階段 | 配置 | 分數 | 相對改善 |
|------|------|------|----------|
| **1** | 基線(Claude Code 直接執行,未引入 skill;第 1 次實驗 通用批次 blend) | 0.56166 | —(基線) |
| **2** | + kaggle-agent skill 六階段流程(第 2 次實驗 手刻管線) | 0.558768 | 0.51% |
| **3** | + self-improvement 線性迭代(第 7 次實驗 四輪終點) | 0.557088 | 0.30% |
| **4** | + 樹搜尋(第 8 次實驗,node #14 7-way blend) | **0.556329** | 0.14% |

**相對改善計算式**(每階段對前一階段)

```
階段1→2: 0.56166 − 0.558768 = 0.002892,相對改善 0.002892 / 0.56166 = 0.5149%
階段2→3: 0.558768 − 0.557088 = 0.001680,相對改善 0.001680 / 0.558768 = 0.3007%
階段3→4: 0.557088 − 0.556329 = 0.000759,相對改善 0.000759 / 0.557088 = 0.1362%
```

本場由導入報告功能後之版本執行,分數自階段 2 起未低於前一階段——符合計畫書目標三(效能不退步)。

**子階段分數表**(定義見第 2.3 節;僅列本場有既有紀錄者)

| 子階段 | 分數 | 出處 |
|--------|------|------|
| 1.1(最佳單模:CAT) | 0.56391 | 第 1 次實驗 |
| 1.2(三模權重 blend) | 0.56166 | 第 1 次實驗 |
| 2.4(場內反思:單模快速試 geo_cluster_50 平滑目標編碼,判定雜訊未採納) | 0.561017 | 第 3 次實驗 |
| 3.2(Optuna 入池) | 0.557977 | 第 4 次實驗 |
| 3.4(seed bagging) | 0.557859 | 第 5 次實驗 |
| 4.2(樹搜尋 7-way,先裁切再評分) | **0.556329** | 第 8 次實驗 |

**階段 5｜+外部想法注入(快評)**

在階段4收斂點注入外部想法候選(本場來源:EXT-09(stacking)),以快取 OOF 評估其能否勝過階段4:

```
階段5 注入候選 OOF rmse = 0.556369
階段4(committed)      = 0.556329
delta = -0.000040  →  未勝過階段4,搜尋不採納  →  階段5 = 階段4(外部注入無增益)
```

（階段5 為全案「停滯時注入外部想法」機制的快評結果,非完整重跑;機制與 15 場跨場
結論見《總結報告》第 7 節與 docs/phase_j_j3_findings。）

## 6. 總結

本場資料乾淨(無缺失、無重複列)且全為數值特徵,建模的兩個結構性事實在 EDA 即已確立:
目標在 5.00001 處被 top-code(佔 train 4.92%),構成誤差天花板;地理訊號強但非線性,
單變量相關性看不出來。這兩點分別決定了後段的裁切後處理與前段的地理特徵工程方向。

關鍵決策有二。其一是自 第 2 次實驗 起固定 KFold(5 折、shuffle、seed 42)不再變動,使全場八個
實驗的 OOF 分數可直接比較;其二是先試後納的紀律——新想法一律先只訓練單一 LGB 快速驗證,
過雜訊門檻才併入池(第 6 次實驗 採納、第 3 次實驗 否決),避免把雜訊當進步。

增益來源逐階遞減且各有歸屬:階段1→2 靠特徵工程與權重搜尋(0.56166→0.558768),是
最大單筆增益;階段2→3 靠線性迭代四輪(調參、seed bagging、幾何特徵)推進至
0.557088;階段3→4 靠樹搜尋的兩個本場專屬作法——「先裁切再評分」與 top-code 感知的
CEILING 成員——收於 0.556329。

結果的可信度建立在同一固定 CV 的 OOF 之上,且逐階皆為正向改善;但須如實註記:本場全程
未提交 Kaggle,無 LB 外部驗證,且 第 8 次實驗 為只算出 OOF 分數的搜尋產物、未產生 test 預測,
目前可直接提交的最佳檔案仍是 第 7 次實驗 的 submission。

**重現本實驗的最短路徑**:見第 7 節。

## 7. 重現指令

```bash
cd /home/tjyen/ai_agents/kaggle

# 階段 2:EDA
uv run python3 competitions/playground-series-s3e1/scripts/eda.py

# 階段 2:特徵工程(輸出 train_processed.csv / test_processed.csv)
uv run python3 competitions/playground-series-s3e1/scripts/features.py

# 階段 2 / 第 2 次實驗:建模(LGB/XGB/CatBoost,5-fold KFold,OOF 權重搜尋 blend)
uv run python3 competitions/playground-series-s3e1/scripts/train.py

# 第 3 次實驗(2.4,僅供參考——單模快速試驗未採納):
uv run python3 competitions/playground-series-s3e1/scripts/tune_geo_te.py

# 3.2 / 第 4 次實驗:Optuna 調參 LGB(僅以第 0 折評分挑參),加入池(4-way blend)
uv run python3 competitions/playground-series-s3e1/scripts/tune_lgb_optuna.py

# 3.4 / 第 5 次實驗:調參 LGB 的 seed bagging(5-way blend)
uv run python3 competitions/playground-series-s3e1/scripts/seed_bag_round2.py

# 階段 3 / 第 6 次實驗:KNN 密度 + 海岸距離地理特徵(單模快速試驗)
# (若採納會輸出 train_processed_v2.csv / test_processed_v2.csv)
uv run python3 competitions/playground-series-s3e1/scripts/geo_knn_probe.py

# 階段 3 / 第 7 次實驗:5 成員池於 26 特徵全量重訓
uv run python3 competitions/playground-series-s3e1/scripts/round4_full_retrain.py

# 4.2 / 第 8 次實驗:樹搜尋(可中斷續跑;只算 OOF 分數、不產生 submission)
uv run python3 tree_search/run_s3e1.py

# 報告產生(本檔)
uv run python3 .claude/skills/kaggle-report/assets/collect.py playground-series-s3e1
```

本場未提交至 Kaggle。若要手動提交目前可提交的最佳 submission 檔(第 7 次實驗):

```bash
export KAGGLE_API_TOKEN=$(python3 -c "import json; print(json.load(open('/home/tjyen/.kaggle/kaggle.json'))['key'])")
uv run kaggle competitions submit -c playground-series-s3e1 \
  -f competitions/playground-series-s3e1/submissions/sub_round4_geofeat_5way_blend_0.55709_20260703_212811.csv \
  -m "5-way blend (Optuna-tuned LGB + seed bag + KNN/coastal geo features), OOF 0.557088"
```

執行目錄為專案根目錄 `/home/tjyen/ai_agents/kaggle`;所有 Python 執行皆透過 `uv run`。
