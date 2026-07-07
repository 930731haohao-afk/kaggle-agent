# 競賽分析報告:playground-series-s3e5

> 本報告由自動化報告流程產生:所有數字直接取自實驗原始紀錄並經自動一致性驗證,敘述由 AI 彙整。
> 紀錄完整度:完整 | 產生日期:2026-07-06

## 1. 競賽目的

**What**:依葡萄酒的 11 項理化量測值(酸度結構、殘糖、氯化物、游離/總二氧化硫、密度、pH、
硫酸鹽、酒精濃度)預測品質評分 `quality`,標籤為 3 至 8 的**序數(ordinal)整數等級**。

**Why**:評估指標為 **quadratic_weighted_kappa(QWK,maximize)**。品質等級之間有順序關係
——把 5 分誤判為 6 分遠輕於誤判為 8 分;QWK 以「預測與真實等級距離的平方」加權懲罰誤差,
比準確率或無序多分類損失更貼合「越接近真實等級越好」的評分邏輯,是序數目標的合理指標。
指標特性也直接決定了本場的建模路線:迴歸頭 + 指標特化的切點後處理。

| 項目 | 值 |
|------|-----|
| 競賽 | playground-series-s3e5 |
| 問題型別 | classification |
| 評估指標 | quadratic_weighted_kappa(maximize) |
| 目標欄位 | quality |

## 2. 本場工具、術語與採用策略

> 全案共同的研究設計、實驗流程、工具鏈、術語與五階段定義,見《前言》
> (docs/PREFACE.pdf);本節僅列本場特有的部分。

### 2.1 本場工具

本場特有工具:**OptimizedRounder** 切點後處理(定義見第 2.2 節),為本場決定性的一步。
其餘全數為共同工具鏈(見《前言》第 3 節);樹搜尋工具本場第 1 版與第 2 版皆有使用。
交叉驗證採依 `quality` 分層的 StratifiedKFold(設計理由見第 3.3 節)。

### 2.2 本場術語

| 術語 | 定義 |
|------|-----------|
| OptimizedRounder | 在 OOF 上搜尋最佳切點,把連續預測值離散化為等級的後處理方法 |
| post-rounder QWK | 先經切點離散化、再計算的 QWK 分數;本場所有決策一律看此分數 |
| 巢狀切點診斷 | 用部分折擬合切點、留出折驗證,檢驗切點是否對全 OOF 過度擬合 |
| 迴歸頭 / 多分類頭 | 模型的輸出形式:迴歸頭輸出連續數值,多分類頭輸出各等級的機率 |

### 2.3 採用策略

各階段與子階段的定義見《前言》第 6 節(編號跨場同義);下表列本場使用情形。

| 階段 | 簡稱 | 本場是否使用 |
|------|------|------|
| **1** | 無 skill 基線 | 使用 |
| 1.1 | 單模型基線 | 使用 |
| 1.2 | 三模型 blend | 使用 |
| **2** | kaggle-agent skill | 使用 |
| 2.1 | EDA 驅動 CV 設計 | 使用(併入 exp2/3,無獨立分數) |
| 2.2 | EDA 驅動特徵工程 | 使用(併入 exp2/3,無獨立分數) |
| 2.3 | 指標感知後處理 | 使用(exp2→exp3,四捨五入改 OptimizedRounder) |
| 2.4 | 場內反思回退 | 本場未使用 |
| **3** | +線性自我迭代 | 使用 |
| 3.1 | 經驗庫先驗 | 本場未使用(經驗庫晚於本場建立) |
| 3.2 | Optuna 超參搜尋 | 使用 |
| 3.3 | 調參入池 | 使用(含於 3.2 之作法) |
| 3.4 | seed bagging | 使用 |
| 3.5 | 結構化去噪 | 本場未使用 |
| **4** | +樹搜尋 | 使用 |
| 4.1 | 單模型節點樹 | 使用(工具第 1 版之最佳節點) |
| 4.2 | ensemble 節點樹 | 使用 |
| 4.3 | 先驗注入+去重 | 使用(v2 內建) |
| 4.4 | 邊界推進 mutation | 使用(LGBBOUND 成員) |
| 4.5 | 預算相位機 | 本場未使用 |
| **5** | +外部想法注入 | 本場未執行(全案規劃中) |
| 5.1 | 外部想法庫先驗 | 本場未執行 |
| 5.2 | 重組 mutation | 本場未執行 |

## 3. 實驗方法

### 3.1 資料規格

| 項目 | 值 |
|------|-----|
| train 列數 | 2,056 |
| test 列數 | 1,372 |
| 原始欄位數 | 11 |
| 特別規則摘要 | 禁外部資料/禁預訓練模型/禁網路存取;每日提交上限 5 次 |

11 個原始欄位皆為數值型,無類別欄位。目標 `quality` 為整數(mean 5.720817、median 6.0、
skew 0.266307),共 6 個等級且嚴重不平衡:quality 3 僅 12 列、quality 8 僅 39 列,quality 5
(839 列)與 quality 6(778 列)為絕對大宗;非 log 轉換候選。

資料品質乾淨:train/test 皆無缺失值、無重複列、無高共線特徵對(`citric acid` 有 211 個零值,
屬合理的理化量測)。train/test 特徵分佈幾乎一致(最大平均差 2.078843%,citric acid),無
共變數偏移。EDA 對單一特徵的訊號排序:`alcohol`(Spearman 0.504246)與 `sulphates`
(0.456985)最強。EDA 之驗證建議即為「整數目標 → 分層 K-fold」,見第 3.3 節。

exp1(通用批次)直接用 11 個原始欄位;exp2–11 經特徵工程展開為 21 個特徵(SO2 比值、
酸度比值、酒精×硫酸鹽/密度交互作用、糖/酒精比等),階段 3 的線性迭代期間特徵集固定不變。

### 3.2 實驗總表

> **語意澄清(本報告全文適用)**:本場為離散化指標,所有實驗的「決策分數」一律為**後處理後
> 的 QWK**(exp2 為四捨五入,其餘為 OptimizedRounder 切點);原始迴歸 OOF 分數未被記錄、
> 也從未作為決策依據(避免「原始分數進步但離散化後不進步」的陷阱),故下表「原始 OOF」欄
> 一律標「—」。

| exp | 階段 | 模型/成員 | 特徵數 | 原始 OOF | 決策分數 | 是否採納 |
|-----|------|-----------|--------|----------|----------|------|
| 1 | 1.2 | LGB/XGB/CAT(.1/.4/.5) | 11 | — | 0.47871 | 是,初始基線 |
| 2 | 2.3 | LGB/XGB/CAT(.2/.2/.6) | 21 | — | 0.47191 | 否,取整方式對照組 |
| 3 | 2.3 | 同 exp2 成員與權重 | 21 | — | 0.52687 | 是,階段 2 最佳 |
| 4 | **3** | 3-way(更細的權重搜尋) | 21 | — | 0.52986 | 是 |
| 5 | 3.2 | 4-way blend(+LGB_tuned) | 21 | — | 0.56293 | 是 |
| 6 | 3.4 | 5-way blend(seed-bag LGB_tuned) | 21 | — | 0.56293 | 否,無增益 |
| 7 | **3** | exp5/6 之 5-way blend(巢狀切點診斷) | 21 | — | 0.54649 | —(診斷) |
| 8 | 3.2 | 6-way blend(+CAT_tuned) | 21 | — | 0.56769 | 是,線性迭代最佳 |
| 9 | 3.4 | 7-way blend(seed-bag CAT_tuned) | 21 | — | 0.56716 | 否,退步 |
| 10 | **3** | exp8 之 6-way blend(巢狀切點診斷) | 21 | — | 0.56393 | —(診斷) |
| 11 | **3** | 7-way blend(多分類期望值解碼頭) | 21 | — | 0.56769 | 否,新成員權重歸零 |
| 12 | 4.2 | 3-way blend(node #17) | 無紀錄 | — | **0.57066** | 是,OOF-only |

線性迭代選型邏輯:以三個梯度提升樹迴歸模型為基礎,依序做(a)更細粒度的權重搜尋
(exp4)、(b)Optuna 直接以 post-rounder QWK 為目標調校 LGB 與 CAT,調校版**加入而非取代**
集成池(exp5、exp8);seed-bagging(exp6、exp9)與多分類期望值解碼頭(exp11)皆未通過
決策判準。exp9、exp11 連兩輪無改善後依協定停止線性迭代。

**最佳解成員表(exp12,3-way blend,node #17)**

| 成員 | 權重 | solo 分數 | 備註 |
|------|------|-----------|------|
| LGB_tuned(root) | 0.151 | 無紀錄 | 即線性迭代之 Optuna 調校 LGB 成員 |
| CAT_tuned | 0.551 | 無紀錄 | 即線性迭代之 Optuna 調校 CAT 成員,權重最高 |
| LGBBOUND | 0.297 | 0.55784 | 邊界推進 LGB(max_depth 3→2):solo 比 root 差 0.005,blend 貢獻 +0.0026 |

exp12 由樹搜尋工具(第 2 版,22 個評估節點、1 次回溯、耗時 793.7 秒)找到,勝出的兩個
原因:(1)LGBBOUND——線性迭代的 Optuna 調參有超參數停在搜尋範圍的邊緣,把範圍再往外推
一步的成員 solo 較弱但夠不一樣;(2)足額的權重搜尋預算(每節點 800 次權重抽樣)——僅
200 次的粗搜在同一組成員上只找到 0.56601,離散指標上縮減權重搜尋預算會讓勝場悄悄變回平手。

> **補充說明**:exp12 為 **OOF-only 搜尋結果**——未產生 test 預測、無 submission 檔、未提交
> Kaggle;最佳解紀錄依 OOF 分數選出,與是否提交無關。其權重與切點皆直接對全 OOF 搜尋,
> 未對此結果重跑巢狀切點診斷(exp7、exp10 的診斷模式)。

### 3.3 訓練規格表

| exp | CV 方案 | folds | seed |
|-----|---------|-------|------|
| 1 | 5fold | 5 | 無紀錄 |
| 2–12 | 5fold(StratifiedKFold on quality) | 5 | 42 |

`quality` 為 6 級且嚴重不平衡的序數目標(quality 3 僅 12 列),一般隨機 K-fold 會使稀有
等級在部分折中掛零、QWK 不穩定,故 exp2 起依標籤分層抽樣,且 folds/seed 全程固定(含
exp12 樹搜尋),各實驗分數可直接比較;此亦與 EDA 的驗證建議一致。

各成員之損失函數與最終超參數未完整記錄,標「無紀錄」;調參設定僅見於實驗紀錄的文字備註
(Optuna TPE、40 trials、600 秒逾時、完整 5-fold CV,目標函數為該 trial 自身 OOF 的
post-rounder QWK)。

### 3.4 推論表

| exp | 後處理 | submission 檔 | 是否已提交 |
|-----|--------|---------------|--------|
| 1 | 無後處理紀錄 | sub_generic_0.47871_20260703_120341.csv | 否 |
| 2 | round_to_nearest_int + clip[3,8] | 無紀錄 | 否 |
| 3 | OptimizedRounder(cutpoints=[3.642, 4.594, 5.657, 6.196, 7.386])+ clip[3,8] | 無紀錄 | 否 |
| 4–6, 8, 9, 11 | OptimizedRounder(各實驗切點各異)+ clip[3,8] | 無紀錄 | 否 |
| 7, 10 | OptimizedRounder(nested:4 折擬合、留出折套用)+ clip[3,8],診斷用 | —(診斷) | 否 |
| 12 | OptimizedRounder 切點與 blend 權重在每個節點的評分函式內聯合搜尋 | —(OOF-only,未產生) | 否 |

`id_column = Id`、`target_column = quality`。線性迭代最佳 exp8 之切點為
`[3.574, 4.586, 5.609, 6.174, 7.628]`;exp12 之具體切點無正式數值紀錄,標「無紀錄」。
本場執行環境未設定 Kaggle 憑證,所有實驗皆未提交排行榜。

### 3.5 評估指標 / 排行榜

指標定義:QWK = 觀察一致性對隨機期望一致性之修正,以「預測與真實等級距離的平方」加權,
越接近 1 代表預測排序與真實等級越一致。

| 項目 | 分數 |
|------|------|
| exp1(通用批次基線) | 0.47871 |
| exp3(階段 2 最佳) | 0.52687 |
| exp8(線性迭代最佳) | 0.56769 |
| exp12(樹搜尋,本場最佳解紀錄) | **0.57066** |
| Public LB / Private LB | 無紀錄 |

本場未提交 Kaggle,無排行榜紀錄,無法計算 CV↔LB gap,不作推測性比較。各階段增益與切點
過擬診斷之衍生算式:

```
線性迭代增益(exp8 − exp3):0.56769 − 0.52687 = 0.04082
樹搜尋增益(exp12 − exp8):0.57066 − 0.56769 = 0.00297
巢狀切點診斷 gap(全 OOF 切點 − 巢狀切點):
  5-way(exp7):0.56293 − 0.54649 = 0.01644
  6-way(exp10):0.56769 − 0.56393 = 0.00376(實驗紀錄的文字備註記為 +0.00377,四捨五入位數差異)
```

巢狀切點診斷顯示全 OOF 切點擬合的過擬風險隨集成池成熟而縮小(0.01644 → 0.00377),
全 OOF 切點在本場屬可接受的標準做法;但 exp12 未重跑此診斷,見第 3.2 節補充說明。

## 4. 實驗軌跡

| exp | 時間 | 決策分數 | 階段 | 摘要 |
|-----|------|----------|------|------------|
| 1 | 2026-07-03T12:03:41 | 0.47871 | 1.2 | 11 特徵三模型 blend,初始基線 |
| 2 | 2026-07-03T18:48:21 | 0.47191 | 2.3 | 21 特徵,四捨五入後處理(取整方式對照組) |
| 3 | 2026-07-03T18:48:21 | 0.52687 | 2.3 | 同一 blend 改用 OptimizedRounder,階段 2 最佳 |
| 4 | 2026-07-03T23:08:22 | 0.52986 | **3** | 改用更細粒度的權重搜尋 |
| 5 | 2026-07-03T23:09:50 | 0.56293 | 3.2 | 加入 Optuna 直接 QWK 調校之 LGB_tuned |
| 6 | 2026-07-03T23:10:42 | 0.56293 | 3.4 | seed-bag LGB_tuned,無增益 |
| 7 | 2026-07-03T23:11:20 | 0.54649 | **3** | 巢狀切點檢驗(5-way),非模型變更 |
| 8 | 2026-07-03T23:13:44 | 0.56769 | 3.2 | 加入 CAT_tuned,線性迭代最佳 |
| 9 | 2026-07-03T23:15:20 | 0.56716 | 3.4 | seed-bag CAT_tuned,退步 |
| 10 | 2026-07-03T23:15:32 | 0.56393 | **3** | 巢狀切點檢驗(6-way),非模型變更 |
| 11 | 2026-07-03T23:16:51 | 0.56769 | **3** | 多分類期望值解碼頭權重歸零,連兩輪無改善停止 |
| 12 | 2026-07-04T11:59:43 | **0.57066** | 4.2 | node #17 3-way blend,本場最佳 |

- **轉折 1(exp2→3)**:同一組成員與權重,後處理由四捨五入改為 OptimizedRounder 切點,
  0.47191 → 0.52687(同一 blend 上取整方式帶來 +0.05496 增益)——本場單一最大增益來源。
- **轉折 2(exp4→5、exp5→8)**:把 Optuna 目標函數直接設為 post-rounder QWK 調校 LGB
  (0.52986 → 0.56293)、再同法調校 CAT(0.56293 → 0.56769),兩次調參佔線性迭代大部分增益。
- **轉折 3(exp11→12)**:樹搜尋工具(第 2 版)以邊界推進成員 LGBBOUND + 足額權重搜尋預算,
  0.56769 → 0.57066(勝線性迭代 +0.00297、勝第 1 版樹搜尋之 0.56766 達 +0.00300,屬結構性
  勝出而非切點噪音量級)。

## 5. 效能對照(逐階段)

| 階段 | 配置 | 分數 | 相對改善 |
|------|------|------|----------|
| **1** | 基線(Claude Code 直接執行,未引入 skill;exp1 通用批次) | 0.47871 | —(基線) |
| **2** | + kaggle-agent skill 六階段流程(exp3,OptimizedRounder) | 0.52687 | 見下方算式 |
| **3** | + self-improvement 線性迭代(exp8,6-way blend + Optuna 直接 QWK 調參) | 0.56769 | 見下方算式 |
| **4** | + 樹搜尋(exp12,node #17,3-way blend) | **0.57066** | 見下方算式 |

```
階段1→2: 0.52687 − 0.47871 = 0.04816,相對改善 0.04816 / 0.47871 = 10.0604%
階段2→3: 0.56769 − 0.52687 = 0.04082,相對改善 0.04082 / 0.52687 = 7.7476%
階段3→4: 0.57066 − 0.56769 = 0.00297,相對改善 0.00297 / 0.56769 = 0.5232%
```

本場由導入報告功能後之版本執行,分數自階段 2 起未低於前一階段——符合計畫書目標三(效能不退步)。

**子階段分數表**(定義見第 2.3 節;僅列本場有既有紀錄者)

| 子階段 | 分數 | 出處 |
|--------|------|------|
| 1.1(最佳單模:XGB) | 0.46995 | exp1 |
| 1.2(三模權重 blend) | 0.47871 | exp1 |
| 2.3(指標感知後處理前:naive round,對照組) | 0.47191 | exp2 |
| 2.3(指標感知後處理後:OptimizedRounder) | 0.52687 | exp3 |
| 3.2(Optuna 直接優化 post-rounder QWK 調參 LGB,加入池) | 0.56293 | exp5 |
| 3.4(seed bagging,LGB_tuned seed 2024,無增益) | 0.56293 | exp6 |
| 3.2(Optuna 直接優化 post-rounder QWK 調參 CatBoost,加入池,線性迭代最佳) | 0.56769 | exp8 |
| 3.4(seed bagging,CAT_tuned seed 2024,退步未採用) | 0.56716 | exp9 |
| 4.1(樹搜尋工具第 1 版,node #11,首版樹搜尋最佳) | 0.56766 | exp12 紀錄的文字備註引用 |
| 4.2(樹搜尋工具第 2 版,node #17,邊界推進成員入池,本場最佳) | **0.57066** | exp12 |

## 6. 總結

本場資料乾淨(無缺失、無重複、無共線問題),真正的難點在目標本身:`quality` 是 6 級、嚴重
不平衡的序數標籤,而 QWK 是離散化指標。第一個關鍵決策是採迴歸頭而非多分類頭——讓大宗等級
的連續訊號幫助定位資料稀少的極端等級;exp11 的多分類期望值解碼頭成員權重被搜尋歸零,反向
印證了這個選擇。

第二個關鍵決策是把「後處理」當一級公民:同一組模型與權重,後處理由四捨五入換成
OptimizedRounder 切點即帶來本場單一最大增益(exp2→exp3);其後所有加入/捨棄決策一律只看
post-rounder QWK,並把同一原則貫徹到 Optuna 的目標函數(直接優化 post-rounder QWK)與樹
搜尋評估器(取整規則內建於每個節點)。

各階段增益來源清楚分層:階段1→2 來自 skill 管線的特徵工程與切點後處理(0.47871 →
0.52687),階段2→3 來自 Optuna 直接 QWK 調參的成員 LGB_tuned、CAT_tuned(→ 0.56769),
階段3→4 來自樹搜尋挖出的邊界推進成員 LGBBOUND(→ 0.57066)。本場為此批競賽中前兩階段
增幅最大的一場,顯示離散化指標下「指標特化後處理 + 直接優化最終指標」的複利效果。

結果可信度:CV 全程固定同一組 StratifiedKFold 折,分數跨實驗可直接比較;巢狀切點診斷
(exp7、exp10)顯示切點過擬風險隨集成池成熟而縮小。但須如實標注:本場所有實驗皆未提交
Kaggle(無憑證),最終 0.57066 為 OOF-only 結果且未重跑巢狀診斷,缺乏排行榜對照驗證。

**重現本實驗的最短路徑**:見第 7 節。

## 7. 重現指令

```bash
cd /home/tjyen/ai_agents/kaggle

# exp1:通用批次管線(11 特徵)
uv run python3 competitions/run_competition.py playground-series-s3e5

# exp2、3(2.3):手刻管線(EDA → 特徵工程 → 訓練/CV/集成 → naive-round vs OptimizedRounder)
uv run python3 competitions/playground-series-s3e5/scripts/eda.py
uv run python3 competitions/playground-series-s3e5/scripts/features.py
uv run python3 competitions/playground-series-s3e5/scripts/train.py

# exp4–11(階段 3 自我改進迭代,依序執行;每步寫入 experiments.json,
# 並快取 scripts/cache/*.npz 供後續步驟重用)
uv run python3 competitions/playground-series-s3e5/scripts/iterate.py base          # exp4
uv run python3 competitions/playground-series-s3e5/scripts/iterate.py tune_lgb      # Optuna 調校 LGB(寫入 cache)
uv run python3 competitions/playground-series-s3e5/scripts/iterate.py r1_pool       # exp5(3.2)
uv run python3 competitions/playground-series-s3e5/scripts/iterate.py seed_bag      # exp6(3.4)
uv run python3 competitions/playground-series-s3e5/scripts/iterate.py nested_cut    # exp7
uv run python3 competitions/playground-series-s3e5/scripts/iterate.py tune_cat      # Optuna 調校 CAT(寫入 cache)
uv run python3 competitions/playground-series-s3e5/scripts/iterate.py r2_pool       # exp8(3.2)
uv run python3 competitions/playground-series-s3e5/scripts/iterate.py seed_bag      # exp9(3.4)
uv run python3 competitions/playground-series-s3e5/scripts/iterate.py nested_cut    # exp10
uv run python3 competitions/playground-series-s3e5/scripts/iterate.py r3_multiclass # exp11
uv run python3 competitions/playground-series-s3e5/scripts/iterate.py submit        # 產出提交檔

# exp12(4.2):樹搜尋第 2 版(best;可中斷/續跑;樹狀態存 experiments_tree_v2.json,
# 與第 1 版的 experiments_tree.json 各自獨立;OOF-only,不產生提交檔)
uv run python3 tree_search/run_s3e5_v2.py

# 提交至 Kaggle(本場執行環境未設定憑證,以下指令供後續有憑證時使用)
export KAGGLE_API_TOKEN=$(cat ~/.kaggle/kaggle_api_token.txt | tr -d '[:space:]')
uv run kaggle competitions submit -c playground-series-s3e5 -f <submission.csv> -m "<msg>"
```

執行目錄為專案根目錄 `/home/tjyen/ai_agents/kaggle`;所有 Python 執行皆透過 `uv run` 以確保
套件環境一致。
