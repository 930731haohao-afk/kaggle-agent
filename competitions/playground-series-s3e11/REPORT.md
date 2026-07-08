# 競賽分析報告:playground-series-s3e11

> 本報告由自動化報告流程產生:所有數字直接取自實驗原始紀錄並經自動一致性驗證,敘述由 AI 彙整。
> 紀錄完整度:完整 | 產生日期:2026-07-06

## 1. 競賽目的

**What**:依門市與商品層級屬性(門市面積與五個附設設施旗標、商品重量/包裝、銷售額/銷量、
顧客家庭屬性等 15 個欄位)預測媒體行銷活動成本(`cost`),為連續數值之表格型迴歸問題;
資料集屬 Aygun et al.(Nature 2026)Kaggle Playground 基準(Season 3 Episode 11)。

**Why**:評估指標為 **RMSLE(minimize)**。成本類目標關心相對誤差而非絕對誤差:RMSLE 在
log1p 空間計算 RMSE,懲罰比例偏差並壓抑大值樣本對損失的支配,適合恆正的金額型目標。
此定義也直接決定訓練策略——對 log1p(cost) 以 RMSE objective 訓練即等同直接優化 RMSLE。

| 項目 | 值 |
|------|-----|
| 競賽 | playground-series-s3e11 |
| 問題型別 | regression |
| 評估指標 | rmsle(minimize) |
| 目標欄位 | cost |

## 2. 本場工具、專業術語與採用策略

> 全案共同的研究設計、實驗流程、工具鏈、專業術語與五階段定義,見《前言》
> (docs/PREFACE.pdf);本節僅列本場特有的部分。

### 2.1 本場工具

全數為共同工具鏈(見《前言》第 3 節),無本場特有工具;樹搜尋工具本場使用第 2 版
(重用線性迭代之 OOF 快取)。交叉驗證採標準 KFold(shuffle)5 折(設計理由見第 3.3 節)。

### 2.2 本場專業術語

| 專業術語 | 定義 |
|------|-----------|
| store profile(門市組合) | `store_sqft` 加五個設施旗標構成的欄位組合;同一組合大量重複出現,是本場最強的訊號來源 |

### 2.3 採用策略

各階段與子階段的定義見《前言》第 6 節(編號跨場同義);主階段 1–4 本場皆有執行,
各子階段採用情形如下。

**階段 1｜無 skill 基線**

- 1.1 單模型基線 — 使用
- 1.2 三模型 blend — 使用

**階段 2｜kaggle-agent skill**

- 2.1 EDA 驅動 CV 設計 — 使用(併入 第 2 次實驗,無獨立分數)
- 2.2 EDA 驅動特徵工程 — 使用(第 2 次實驗→第 3 次實驗,`store_te` 目標編碼)
- 2.3 指標感知後處理 — 使用(log1p 目標對齊 RMSLE,第 2 次實驗 起全部實驗)
- 2.4 場內反思回退 — 本場未使用

**階段 3｜+線性自我迭代**

- 3.1 經驗庫先驗 — 本場未使用(經驗庫晚於本場建立)
- 3.2 Optuna 超參搜尋 — 使用
- 3.3 調參入池 — 使用(含於 3.2 之作法,入池不替換)
- 3.4 seed bagging — 使用
- 3.5 結構化去噪 — 本場未使用

**階段 4｜+樹搜尋**

- 4.1 單模型節點樹 — 本場未使用
- 4.2 ensemble 節點樹 — 使用
- 4.3 先驗注入+去重 — 使用(v2 內建)
- 4.4 邊界推進 mutation — 使用(CAT depth 10→12 突破調參上界)
- 4.5 預算相位機 — 本場未使用

**階段 5｜+外部想法注入** — 全案已實測:注入機制建置完成,但注入候選未勝過階段4(階段5=階段4,見總結報告第 7 節)

## 3. 實驗方法

### 3.1 資料規格

| 項目 | 值 |
|------|-----|
| train 列數 | 360,336 |
| test 列數 | 240,224 |
| 原始欄位數 | 15(全數值) |
| 特別規則摘要 | 禁外部資料/禁預訓練模型/禁網路存取;每日提交上限 5 次 |

EDA 已執行:15 個特徵全為數值型,其中多數實為低基數「類別型偽裝」
欄位(五個 0/1 設施旗標、家庭屬性等);train 無缺失值、無重複列;`salad_bar` 與
`prepared_food` 相關 0.999839,為唯一高共線特徵對(近乎重複欄)。

目標 `cost` ∈ [50.79, 149.75],mean 99.614729、median 98.81、skew 0.019132——分布近乎
對稱,EDA 判定非 log 轉換候選;採 log1p 目標的理由是「RMSLE = log1p 空間之 RMSE,直接
優化競賽指標」,而非矯正偏態,預測經 expm1 後 clip ≥ 0。

低訊號資料集:單變量關聯最強者僅 `florist`(pearson -0.110414)。`store_sqft` 加五個設施
旗標構成大量重複出現的 store profile(門市組合),是最強特徵候選,需以 fold-safe 目標
編碼防洩漏;EDA 的驗證建議為「連續目標、各列獨立 → 標準 KFold」。

### 3.2 實驗總表

> **語意澄清(本報告全文適用)**:RMSLE 為連續迴歸指標,本場無取整/門檻類後處理,故各
> 實驗的「原始 OOF」即「決策分數」,兩欄同值;且本場為無人值守批次執行,全程未提交
> Kaggle,所有決策皆以本機 OOF 為準,下文不再重複解釋。

| 實驗編號 | 階段 | 模型/成員 | 特徵數 | 原始 OOF | 決策分數 | 是否採納 |
|-----|------|-----------|--------|----------|----------|------|
| 1 | 1.2 | LGB/XGB/CAT(0.7/0.0/0.3) | 15 | 0.29723 | 0.29723 | 基線參照 |
| 2 | **2** | LGB/XGB/CAT(0.2/0.0/0.8),log1p 目標 | 15 | 0.2971 | 0.2971 | 同特徵對照組 |
| 3 | 2.2 | LGB/XGB/CAT(0.2/0.0/0.8)+ store_te | 21 | 0.296143 | 0.296143 | 是,階段 2 最佳 |
| 4 | **3** | LGB/CAT_orig(0.2/0.8),移除 XGB | 21 | 0.296143 | 0.296143 | 是,分數不變 |
| 5 | 3.2 | + CAT_tuned(Optuna,0.9 權重) | 21 | 0.295781 | 0.295781 | 是 |
| 6 | 3.4 | + CAT_tuned seed=2024(4-way) | 21 | 0.295715 | 0.295715 | 是 |
| 7 | **3** | 4-way pool + 門市組合均值特徵 | 24 | 0.2962 | 0.2962 | 否,退步棄用 |
| 8 | 3.4 | + CAT_tuned seed=7(5-way) | 21 | 0.295648 | 0.295648 | 是,線性迭代最佳 |
| 9 | 4.2 | 7-way blend(depth-12 CAT 家族 + DEEPLGB) | 21(同 第 8 次實驗) | **0.29528** | **0.29528** | 是,本場最佳 |
| 10 | **5** | 外部注入快評候選(EXT-09(stacking);沿用階段4成員 OOF) | 21 | 0.298328 | 0.298328 | 否(未勝過階段4,見 §5) |

**最佳解成員表(第 9 次實驗,node #20,新成員家族出現後的純權重再搜尋)**

| 成員 | 權重 | solo 分數 | 備註 |
|------|------|-----------|------|
| CAT_D12_S7 | 0.2478 | 0.295461 | 調參 CAT depth 10→12(突破 Optuna 搜尋上界),全樹最佳 solo |
| CAT_D12_S3000 | 0.3187 | 0.295462 | depth-12 家族 seed 變體 |
| CAT_D12_S3001 | 0.239 | 0.295483 | depth-12 家族 seed 變體 |
| CAT_S7_D10 | 0.0091 | 0.295779 | 線性迭代之 depth-10 tuned CAT(seed 7),自快取重用 |
| CAT_S99 | 0.0031 | 0.295786 | 第 4 個 tuned-CAT seed,線性迭代未嘗試 |
| CAT_TUNED_ROOT | 0.0039 | 0.29579 | 線性迭代最強 solo(Optuna 調參 CAT,seed 42),快取重用 |
| DEEPLGB | 0.1783 | 0.295833 | 刻意多樣化之深 LGB(num_leaves 255);solo 平庸但權重第 3 大 |

選型脈絡:第 1–3 次實驗 三模型權重搜尋連兩輪將 XGB 權重歸 0,第 4 次實驗 移除 XGB 後 blend 分數不變,
驗證零權重裁決無代價;第 5 次實驗 起 Optuna 調參 CatBoost 成為主力,權重全數流向 tuned CAT 家族;
第 7 次實驗 在 store_te 之上再加門市組合均值特徵全面退步,棄用回退;第 8 次實驗 以第三個 seed 收在
0.295648,增益已縮至噪音級,依協定停止線性迭代。

> **補充說明**:最佳解紀錄(第 9 次實驗)以 OOF 分數最小選出,為**只算出 OOF 分數**的樹搜尋結果
> ——未產生 test 預測、無 submission 檔、未提交 Kaggle;其權重由權重搜尋直接對全 OOF 擬合
> (無巢狀驗證),0.0004 等級的增益帶有 OOF 權重過擬風險,方向性結論(深度突破 Optuna
> 上界後重開 blend)較第 4 位小數穩健。

### 3.3 訓練規格表

| 實驗編號 | CV 方案 | folds | seed |
|-----|---------|-------|------|
| 1 | 5fold | 5 | 無紀錄 |
| 2 | 5fold_kfold_shuffle | 5 | 42 |
| 3 | 5fold_kfold_shuffle | 5 | 42 |
| 4 | 5fold_kfold_shuffle | 5 | 42 |
| 5 | 5fold_kfold_shuffle | 5 | 42 |
| 6 | 5fold_kfold_shuffle | 5 | 42 |
| 7 | 5fold_kfold_shuffle | 5 | 42 |
| 8 | 5fold_kfold_shuffle | 5 | 42 |
| 9 | KFold(shuffle) | 5 | 42 |

目標為連續值、各列獨立(無時間/群組結構,見 EDA 的驗證建議),且門市組合大量重複、
隨機切分安全,故用 KFold(shuffle, seed=42);第 2–9 次實驗(含樹搜尋)沿用同一組固定折,
跨實驗分數可直接比較。

Objective 與關鍵超參:全程對 log1p(cost) 以 RMSE objective 訓練,預測 expm1 後
clip ≥ 0(第 2 次實驗 起)。第 5 次實驗 之 Optuna(TPE,每組參數先只在第 0 折評分以省時)最佳
CatBoost 參數:

```
Optuna:TPE 40 trials(319.3s,timeout guard 480s),僅以第 0 折評分挑參
CatBoost tuned:depth=10, learning_rate≈0.0824, l2_leaf_reg≈5.482,
               min_data_in_leaf=33, random_strength≈0.0916
第 9 次實驗 樹搜尋關鍵變體:CAT depth 10→12(Optuna 原搜尋空間上界為 10)、
                     DEEPLGB num_leaves=255
```

其餘 base model(手設 LGB/XGB/CAT)之超參無結構化紀錄,不臆測。

### 3.4 推論表

| 實驗編號 | 後處理 | submission 檔 | 是否已提交 |
|-----|--------|---------------|--------|
| 1 | 無後處理紀錄 | sub_generic_0.29723_20260703_121020.csv | 否 |
| 2 | expm1 + clip ≥ 0 | sub_base_0.29710_20260703_192507.csv | 否 |
| 3 | expm1 + clip ≥ 0 | sub_engineered_0.29614_20260703_192819.csv | 否 |
| 4 | expm1 + clip ≥ 0 | sub_r1_noxgb_0.29614_20260703_222353.csv | 否 |
| 5 | expm1 + clip ≥ 0 | sub_r2_cattuned_0.29578_20260703_223101.csv | 否 |
| 6 | expm1 + clip ≥ 0 | sub_r3_seedbag_0.29571_20260703_223204.csv | 否 |
| 7 | expm1 + clip ≥ 0 | sub_r4_deepstore_0.29620_20260703_223633.csv(棄用) | 否 |
| 8 | expm1 + clip ≥ 0 | sub_r5_seedbag3_0.29565_20260703_223829.csv | 否 |
| 9 | 無(僅 OOF 分數) | 無紀錄(未產生 test 預測) | 否 |

「後處理」欄之 expm1 + clip ≥ 0 為 log1p 目標之逆轉換與安全網(各實驗均有註記),
非指標特化後處理。欄位格式:id 欄 `id`、目標欄 `cost`。本場為無人值守批次執行,僅產生
本機 submission 檔、未觸碰 Kaggle 憑證,「是否已提交」一律為否。

### 3.5 評估指標 / 排行榜

指標定義:RMSLE = 對預測值與真值各取 log1p 後計算 RMSE,衡量比例(相對)誤差,越低越好。

| 項目 | OOF RMSLE |
|------|-----------|
| CAT_tuned(第 8 次實驗 成員,Optuna 調參,seed 42) | 0.29579 |
| CAT_tuned_seed2024(第 8 次實驗 成員) | 0.29591 |
| CAT_tuned_seed7(第 8 次實驗 成員) | 0.29578 |
| Ensemble(第 8 次實驗,線性迭代終點,5-way) | 0.295648 |
| Ensemble(第 9 次實驗,樹搜尋 best,7-way) | **0.29528** |
| Public / Private LB | 無紀錄(未提交) |

本場未提交 Kaggle,無排行榜紀錄,故無排行榜表,CV↔LB gap 無法計算。第 9 次實驗 相對 第 8 次實驗
之改善:

```
第 8 次實驗 − 第 9 次實驗:0.295648 − 0.29528 = 0.000368
```

## 4. 實驗軌跡

| 實驗編號 | 時間 | 決策分數 | 階段 | 摘要 |
|-----|------|----------|------|------------|
| 1 | 2026-07-03T12:10:20 | 0.29723 | 1.2 | 15 原始特徵三模型 blend,設定待超越基線 |
| 2 | 2026-07-03T19:25:07 | 0.2971 | **2** | 換 log1p 目標 + 調參 + 提前停止,同特徵對照組 |
| 3 | 2026-07-03T19:28:19 | 0.296143 | 2.2 | 21 特徵 + 門市組合 fold-safe 目標編碼,階段 2 最佳 |
| 4 | 2026-07-03T22:23:54 | 0.296143 | **3** | 移除連兩輪零權重之 XGB,分數不變 |
| 5 | 2026-07-03T22:31:01 | 0.295781 | 3.2 | Optuna 調參 CatBoost 入池,線性迭代主要躍升 |
| 6 | 2026-07-03T22:32:05 | 0.295715 | 3.4 | tuned CAT seed bagging(seed 2024,4-way) |
| 7 | 2026-07-03T22:36:33 | 0.2962 | **3** | 門市組合均值特徵全面退步,棄用 |
| 8 | 2026-07-03T22:38:30 | 0.295648 | 3.4 | 第三 seed(7)5-way blend,線性迭代最佳後停止 |
| 9 | 2026-07-04T12:17:25 | **0.29528** | 4.2 | 樹搜尋 node #20 之 7-way 再混合,本場最佳 |
| 10 | — | 0.298328 | 5 | 外部注入快評:EXT-09(stacking) 候選 OOF 未勝過階段4,不採納 |

- **突破點 1(第 2 次實驗→第 3 次實驗)**:`store_te`(門市組合的 fold-safe 目標編碼)一舉由 0.2971
  降至 0.296143,為階段 2 主要增益來源。
- **突破點 2(第 4 次實驗→第 5 次實驗)**:Optuna(每組參數先只在第 0 折評分)調參 CatBoost 以「入池不替換」
  加入,0.296143 → 0.295781,為線性迭代階段最大單筆增益。
- **突破點 3(第 8 次實驗→第 9 次實驗)**:樹搜尋把 CAT depth 推到 12——超出 Optuna 自身搜尋上界 10
  ——並在新 solo 家族出現後重開 blend 權重搜尋,0.295648 → 0.29528。

## 5. 效能對照(逐階段)

| 階段 | 配置 | 分數 | 相對改善 |
|------|------|------|----------|
| **1** | 基線(Claude Code 直接執行,未引入 skill;第 1 次實驗 通用批次三模型 blend) | 0.29723 | —(基線) |
| **2** | + kaggle-agent skill 六階段流程(第 3 次實驗,特徵工程版) | 0.296143 | 0.37% |
| **3** | + self-improvement 線性迭代(第 8 次實驗,5-way blend) | 0.295648 | 0.17% |
| **4** | + 樹搜尋(第 9 次實驗,node #20 之 7-way blend) | **0.29528** | 0.12% |

**相對改善計算式**(每階段對前一階段)

```
階段1→2: 0.29723 − 0.296143 = 0.001087,相對改善 0.001087 / 0.29723 = 0.3657%
階段2→3: 0.296143 − 0.295648 = 0.000495,相對改善 0.000495 / 0.296143 = 0.1671%
階段3→4: 0.295648 − 0.29528 = 0.000368,相對改善 0.000368 / 0.295648 = 0.1245%
```

本場由導入報告功能後之版本執行,分數自階段 2 起未低於前一階段——符合計畫書目標三(效能不退步)。

注:指標為 RMSLE(minimize),分數越低越好,各階段逐階下降、無同值階段。階段 4(第 9 次實驗)為
只算出 OOF 分數的樹搜尋結果,本場無任何 Kaggle 提交、無排行榜對照,各階段分數皆為同一組固定折之
本機 OOF。

**子階段分數表**(定義見第 2.3 節;僅列本場有既有紀錄者)

| 子階段 | 分數 | 出處 |
|--------|------|------|
| 1.1(最佳單模:LGB) | 0.29731 | 第 1 次實驗 |
| 1.2(三模權重 blend) | 0.29723 | 第 1 次實驗 |
| 2.2(特徵工程前:15 原始特徵,log1p+RMSE 目標對照組) | 0.2971 | 第 2 次實驗 |
| 2.2(特徵工程後:21 特徵 + `store_te` fold-safe 目標編碼) | 0.296143 | 第 3 次實驗 |
| 3.2(Optuna 調參 CatBoost——僅以第 0 折評分挑參,加入池) | 0.295781 | 第 5 次實驗 |
| 3.4(seed bagging,tuned CAT seed 2024,4-way) | 0.295715 | 第 6 次實驗 |
| 3.4(seed bagging 擴充,tuned CAT 第 3 個 seed,5-way,線性迭代終點) | 0.295648 | 第 8 次實驗 |
| 4.2(樹搜尋 node #20,CAT depth 推進至 12,7-way 再混合,本場最佳) | **0.29528** | 第 9 次實驗 |

**階段 5｜+外部想法注入(快評)**

在階段4收斂點注入外部想法候選(本場來源:EXT-09(stacking)),以快取 OOF 評估其能否勝過階段4:

```
階段5 注入候選 OOF rmsle = 0.298328
階段4(committed)      = 0.295280
delta = -0.003048  →  未勝過階段4,搜尋不採納  →  階段5 = 階段4(外部注入無增益)
```

（階段5 為全案「停滯時注入外部想法」機制的快評結果,非完整重跑;機制與 15 場跨場
結論見《總結報告》第 7 節與 docs/phase_j_j3_findings。）

## 6. 總結

本場資料量大(360,336 列)而訊號弱:15 個全數值特徵之單變量關聯皆極低(最強僅
`florist` pearson -0.110414),真正的結構藏在 store_sqft 加五個設施旗標構成的重複
store profile 中。目標近乎對稱、非 log 轉換候選,採 log1p 目標純粹是為了讓 RMSE
objective 直接等於競賽指標 RMSLE。

關鍵決策有四:以 fold-safe 目標編碼 `store_te` 榨取門市組合訊號(第 3 次實驗,階段 2 主要
增益);尊重零權重裁決移除 XGB(第 4 次實驗,分數不變、釋出預算);Optuna 調參 CatBoost 入池
加 seed bagging(第 5–8 次實驗);第 7 次實驗 在 store_te 之上疊門市組合均值特徵退步後果斷回退,
並在增益縮至噪音級時依協定停止線性迭代。

各階段增益中階段1→2 最大(特徵工程),階段2→3 次之(調參 + seed bagging),
階段3→4 由樹搜尋貢獻:最大單筆發現是把 CAT depth 推到 12——Optuna 的最優解原本就
落在其搜尋上界 10 上,上界本身即是下一個突變方向;而最終 0.29528 來自新 depth-12 家族
出現後重開的 blend 權重再搜尋,且 solo 平庸的 DEEPLGB 仍拿到 0.1783 權重,再次印證
blend 貢獻與 solo 分數脫鉤。

可信度方面須誠實:本場全程僅本機 OOF、未提交 Kaggle,無排行榜外部驗證;第 2–9 次實驗 使用同一
組固定折,分數排序可直接比較,但 第 9 次實驗 僅有 OOF 分數且權重直接對全 OOF 擬合,0.0004 等級
增益帶有過擬風險。方向性結論(深度突破調參上界、突破後重開 blend)是穩健的部分。

**重現本實驗的最短路徑**:見第 7 節。

## 7. 重現指令

```bash
cd /home/tjyen/ai_agents/kaggle

# 資料下載(需先設定 KAGGLE_API_TOKEN;data/ 已存在者可略過)
uv run kaggle competitions download -c playground-series-s3e11 \
  -p competitions/playground-series-s3e11/data

# 階段 2:EDA
uv run python3 competitions/playground-series-s3e11/scripts/eda.py

# 第 1 次實驗(階段 1):通用批次管線
uv run python3 competitions/run_competition.py playground-series-s3e11

# 第 2 次實驗(階段 2):skill base(log1p 目標,15 原始特徵,同特徵對照組)
uv run python3 competitions/playground-series-s3e11/scripts/train.py base

# 第 3 次實驗(2.2):skill engineered(21 特徵 + store_te,階段 2 最佳)
uv run python3 competitions/playground-series-s3e11/scripts/train.py engineered

# 階段 3(第 4–8 次實驗;checkpointed,cache 命中會跳過已訓練成員)
uv run python3 competitions/playground-series-s3e11/scripts/iterate2.py r1
uv run python3 competitions/playground-series-s3e11/scripts/iterate2.py tune
uv run python3 competitions/playground-series-s3e11/scripts/iterate2.py r2
uv run python3 competitions/playground-series-s3e11/scripts/iterate2.py r3
uv run python3 competitions/playground-series-s3e11/scripts/iterate2.py r4   # 退步,棄用
uv run python3 competitions/playground-series-s3e11/scripts/iterate2.py r5   # 第 8 次實驗,線性迭代最佳

# 第 9 次實驗(4.2):樹搜尋工具第 2 版(本場最佳;只算 OOF 分數,重用 scripts/cache/*.npz)
uv run python3 tree_search/run_s3e11.py

# 報告產生(本檔)
uv run python3 .claude/skills/kaggle-report/assets/collect.py playground-series-s3e11
uv run python3 .claude/skills/kaggle-report/assets/verify_report.py \
    competitions/playground-series-s3e11/REPORT.md competitions/playground-series-s3e11/facts.json
bash .claude/skills/kaggle-report/assets/md2pdf.sh competitions/playground-series-s3e11/REPORT.md \
    competitions/playground-series-s3e11/s3e11_REPORT.pdf

# 提交至 Kaggle(本場未執行;需先設定有效之 KAGGLE_API_TOKEN)
export KAGGLE_API_TOKEN=$(cat ~/.kaggle/kaggle_api_token.txt | tr -d '[:space:]')
uv run kaggle competitions submit -c playground-series-s3e11 \
    -f competitions/playground-series-s3e11/submissions/sub_r5_seedbag3_0.29565_20260703_223829.csv \
    -m "R5 5-way seed-bagged tuned CatBoost blend, OOF RMSLE 0.295648"
```

執行目錄為專案根目錄 `/home/tjyen/ai_agents/kaggle`;所有 Python 執行皆透過 `uv run` 以
確保套件環境一致。本場為無人值守批次執行,未提交至 Kaggle 排行榜、未觸碰 Kaggle 憑證。
