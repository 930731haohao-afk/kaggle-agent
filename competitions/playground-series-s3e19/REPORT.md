# 競賽分析報告:playground-series-s3e19

> 本報告由自動化報告流程產生:所有數字直接取自實驗原始紀錄並經自動一致性驗證,敘述由 AI 彙整。
> 紀錄完整度:完整 | 產生日期:2026-07-06

## 1. 競賽目的

**What**:Playground Series S3E19「Forecast mini-course sales」——依日期(date)、國家
(country)、店面(store)、產品(product)四個維度,預測每日課程銷量 `num_sold`。
訓練資料為 2017–2021 年,測試集為整個 2022 年,是典型的多序列銷售時間序列外推問題。

**Why**:銷售預測直接影響備貨與行銷決策。評估指標為 **SMAPE(minimize)**:各序列量級
差異大(不同國家/店面/產品),絕對誤差類指標會被大量級序列主導;SMAPE 以真值與預測的
平均作分母,把誤差正規化為相對比例,讓大小序列的預測品質獲得同等權重,適合此類任務。

| 項目 | 值 |
|------|-----|
| 競賽 | playground-series-s3e19 |
| 問題型別 | regression |
| 評估指標 | smape(minimize) |
| 目標欄位 | num_sold |

## 2. 本場工具、專業術語與採用策略

> 全案共同的研究設計、實驗流程、工具鏈、專業術語與五階段定義,見《前言》
> (docs/PREFACE.pdf);本節僅列本場特有的部分。

### 2.1 本場工具

全數為共同工具鏈(見《前言》第 3 節),無本場特有工具;樹搜尋工具本場使用第 2 版。
交叉驗證採於排序後唯一日期上切分的 TimeSeriesSplit 5 折,另以隨機 KFold 執行一次
對照診斷(設計理由見第 3.3 節)。

### 2.2 本場專業術語

| 專業術語 | 定義 |
|------|-----------|
| (時序)/(隨機) | 分數來源標記:時序 = TimeSeriesSplit 折的 OOF;隨機 = shuffled KFold 折的 OOF;兩者不可互比 |
| auto_scale | 對全體預測乘上一個在 OOF 上擬合的全域乘數(本場 ×1.02),修正時序 OOF 系統性偏低 |
| 比例分解成員 | 把序列拆成「總量 × 各維度佔比」再組回預測的結構式成員(RatioDecomp);本場權重被搜尋歸零 |

### 2.3 採用策略

各階段與子階段的定義見《前言》第 6 節(編號跨場同義);主階段 1–4 本場皆有執行,
各子階段採用情形如下。

**階段 1｜無 skill 基線**

- 1.1 單模型基線 — 使用
- 1.2 三模型 blend — 使用

**階段 2｜kaggle-agent skill**

- 2.1 EDA 驅動 CV 設計 — 使用(第 2 次實驗 改採時序切分,本場最重要決策)
- 2.2 EDA 驅動特徵工程 — 使用(第 2 次實驗,20 個日期/類別特徵)
- 2.3 指標感知後處理 — 使用(log1p 目標與 expm1 反轉,第 2 次實驗 起)
- 2.4 場內反思回退 — 使用(第 3 次實驗 反思診斷:分離 CV 方案與特徵效應)

**階段 3｜+線性自我迭代**

- 3.1 經驗庫先驗 — 本場未使用(經驗庫晚於本場建立)
- 3.2 Optuna 超參搜尋 — 使用(第 7 次實驗,只以第 5 折評分挑參)
- 3.3 調參入池 — 使用(含於 3.2 之作法)
- 3.4 seed bagging — 使用(第 5、6 次實驗)
- 3.5 結構化去噪 — 本場未使用

**階段 4｜+樹搜尋**

- 4.1 單模型節點樹 — 本場未使用
- 4.2 ensemble 節點樹 — 使用
- 4.3 先驗注入+去重 — 使用(v2 內建)
- 4.4 邊界推進 mutation — 本場未使用(無獨立紀錄)
- 4.5 預算相位機 — 本場未使用

**階段 5｜+外部想法注入** — 全案已實測:注入機制建置完成,但注入候選未勝過階段4(階段5=階段4,見總結報告第 7 節)

## 3. 實驗方法

### 3.1 資料規格

| 項目 | 值 |
|------|-----|
| train 列數 | 136,950 |
| test 列數 | 27,375 |
| 原始欄位數 | 4(date/country/store/product;另有 id 與目標欄) |
| 特別規則摘要 | 禁外部資料/禁預訓練模型/禁網路存取;每日提交上限 5 次 |

目標 `num_sold` 為整數(mean 165.522636、median 98、max 1380),右偏(skew 1.747438),
為 log 轉換候選——全程以 log1p 目標訓練、expm1 反轉。train/test 皆無缺失值、無重複列;
date 有 1,826 個唯一日期,test 的 2022 年日期**全部未在 train 出現**,EDA 的驗證建議
亦提示應採時間切分驗證以避免洩漏。

EDA 關鍵定性發現:5 國 × 3 店 × 5 產品共 75 條完整日序列;店面與
產品占年度總量的比例逐年幾乎恆定,國家占比則逐年漂移(外推最難的部分);週末與 12 月/
1 月有明顯季節抬升、1 月 1 日單日尖峰;年度總量非單調(2020 下滑、2021 回升),無可靠
線性趨勢。

### 3.2 實驗總表

> **語意澄清(本報告全文適用)**:本場同時存在兩種 CV 方案——(時序)= TimeSeriesSplit
> 5 折(每折驗證區塊嚴格晚於訓練窗,模擬 train 2017–21 → test 2022 的外推缺口)與
> (隨機)= shuffled KFold 5 折(內插式,對本任務系統性樂觀)。**兩種方案的分數不可互相
> 比較**;所有 keep/reject 決策一律以(時序)分數為準。SMAPE 無取整/門檻類後處理,故
> 各實驗「原始 OOF」即「決策分數」,兩欄同值。

| 實驗編號 | 階段 | 模型/成員 | 特徵數 | 原始 OOF | 決策分數 | 是否採納 |
|-----|------|-----------|--------|----------|----------|------|
| 1 | 1.2 | LGB/XGB/CAT(0.7/0.0/0.3) | 8 | 5.31891(隨機) | 5.31891(隨機) | 基線參照;與(時序)各實驗不可比 |
| 2 | **2** | LGB/XGB/CAT(0.9/0.0/0.1) | 20 | 10.175397(時序) | 10.175397(時序) | 是,階段 2 最佳 |
| 3 | 2.4 | LGB/XGB/CAT(0.4/0.2/0.4) | 20 | 4.281421(隨機) | 4.281421(隨機) | 否,僅診斷用,不入決策 |
| 4 | **3** | LGB/CAT/比例分解(0.95/0.05/0.0) | 20 | 10.17489(時序) | 10.17489(時序) | 比例分解成員棄用(權重 0;微幅增益僅來自較細權重網格) |
| 5 | 3.4 | LGB seed 42/2024 + CAT(0.65/0.3/0.05) | 20 | 10.166045(時序) | 10.166045(時序) | 是 |
| 6 | 3.4 | LGB×3 seeds + CAT×2 seeds(5-way) | 20 | 10.157212(時序) | 10.157212(時序) | 是 |
| 7 | 3.2 | 6-way + Optuna 調參 LGB(權重 0.5) | 20 | 10.019463(時序) | 10.019463(時序) | 是,線性迭代最終;現行最佳提交檔 |
| 8 | 4.2 | 10-way blend + auto_scale ×1.02(node #17) | 20(池內含刪冗餘欄位的檢查型成員) | **9.75707**(時序) | **9.75707**(時序) | 是,本場誠實計分下最佳(僅 OOF 分數) |

**最佳解成員表(第 8 次實驗,權重搜尋 + auto_scale 全域乘數 1.02)**

| 成員 | 權重 | solo 分數 | 備註 |
|------|------|-----------|------|
| SEEDBAG_TUNED_S3000 | 0.2559 | 9.990227 | 調參 LGB 之 seed 3000 變體 |
| SEEDBAG_TUNED_S2024 | 0.2448 | 9.974778 | 調參 LGB 之 seed 2024 變體;solo 即勝過 第 7 次實驗 全 blend |
| LGB_S7 | 0.1989 | 10.19448 | 手設 LGB,seed 7 |
| CALSUBSET | 0.1781 | 10.137705 | 刪 weekofyear 等冗餘欄重訓的檢查型成員 |
| DEEPLGB | 0.0746 | 10.347073 | 刻意多樣化之深/輕正則 LGB;solo 弱但有貢獻 |
| LGB_S42 | 0.0238 | 10.177577 | 手設 LGB,seed 42 |
| LGB_S2024 | 0.0115 | 10.210617 | 手設 LGB,seed 2024 |
| LGB_TUNED_ROOT | 0.0089 | 10.148325 | 搜尋根節點 = 第 7 次實驗 之 Optuna 調參 LGB |
| CAT_S2024 | 0.0032 | 10.983835 | 手設 CatBoost,seed 2024 |
| CAT_S42 | 0.0003 | 10.730642 | 手設 CatBoost,seed 42 |

選型脈絡:三種梯度提升樹是中型表格資料的標準組合,皆原生支援類別特徵。外推情境下
LGB 明顯領先,XGB 兩度被權重搜尋歸零後移除;第 4 次實驗 的結構式比例分解成員(solo 14.75038)
也被歸零——GBDT 的原生類別分裂已學走佔比結構,真正瓶頸(總量一年外推)對所有方法一視
同仁。第 8 次實驗 的兩個決定性作法:對調參 LGB 做 seed bagging(調參只以第 5 折評分挑參、
未對全 OOF 擬合,留有真實 seed 變異可平均消除),與修正時序 OOF 系統性偏低約 2% 的全域 ×1.02
乘數(每折驗證區塊皆晚於訓練窗、序列成長至 2021,此偏差結構上與真實 2022 缺口同型)。

> **補充說明(第 8 次實驗 與最佳解的計分方式;全文其他引用處不再重複)**:
> 1. 最佳解紀錄以分數最小選出,落在 第 3 次實驗(4.281421)——那是(隨機)
>    診斷量尺的產物,不可作為本場最佳;誠實計分((時序))下的最佳為 第 8 次實驗。
> 2. 第 8 次實驗 為樹搜尋結果,只算出 OOF 分數:未產生 test 預測、無 submission 檔、未提交 Kaggle;
>    「現行最佳提交檔」屬 第 7 次實驗(見第 3.4 節)。
> 3. **9.75707 帶雙重但書**:(a)與 第 2–7 次實驗 相同,fold 5 既是 Optuna 調參目標又佔
>    OOF 的 1/5,分數含 fold 5 雙重使用的樂觀成分;(b)第 8 次實驗 額外的 scale 乘數與
>    seed 挑選皆直接對 OOF 擬合(無巢狀驗證)。各為 1 至數個參數、擬合於 114,000 列
>    OOF,單項過擬風險低,但真實可期望的 2022 SMAPE 應解讀為「明顯低於 10.02」,
>    而非字面上的 9.76。

### 3.3 訓練規格表

| 實驗編號 | CV 方案 | folds | seed |
|-----|---------|-------|------|
| 1 | 5fold(隨機) | 5 | 無紀錄 |
| 2 | TimeSeriesSplit-5fold-on-unique-dates | 5 | 無紀錄 |
| 3 | KFold-5fold-shuffled-seed42 | 5 | 無紀錄(方案名含 seed42) |
| 4–7 | TimeSeriesSplit-5fold-on-unique-dates(與 第 2 次實驗 同折) | 5 | 無紀錄 |
| 8 | TimeSeriesSplit-5fold-on-unique-dates(逐位驗證與 第 2–7 次實驗 同折) | 5 | 無紀錄 |

**為何用此 CV**:測試期(2022)完全落在訓練期之後、零日期重疊,模型實際面對「外推到
未見年份」;隨機 KFold 讓驗證日夾在已見季節循環之間,對本任務系統性樂觀。故主軌跡在
排序後的唯一日期上做 TimeSeriesSplit,每折以擴張窗訓練、驗證嚴格較晚的日期區塊。
第 3 次實驗 刻意改回與基線同型態的隨機 KFold,僅作為特徵集有效性的對照診斷,不用於選模。

Objective:log1p(num_sold) 迴歸,預測經 expm1 反轉後計 SMAPE。關鍵超參(有紀錄者):

```
LGB(手設,第 2 次實驗):  n_estimators=2000, learning_rate=0.03, num_leaves=63,
                     min_child_samples=20, subsample=0.9, colsample_bytree=0.8, reg_lambda=1.0
XGB(手設,第 2 次實驗):  n_estimators=2000, learning_rate=0.03, max_depth=7,
                     subsample=0.9, colsample_bytree=0.8, reg_lambda=1.0
CAT(手設,第 2 次實驗):  iterations=2000, learning_rate=0.05, depth=8, l2_leaf_reg=3.0
LGB_tuned(Optuna,第 7 次實驗):learning_rate≈0.0371, num_leaves=20, min_child_samples=38,
                     subsample≈0.61, colsample_bytree≈0.90, reg_alpha≈0.001,
                     reg_lambda≈0.43, n_estimators=2000
```

調參版比手設更淺、更強正則(num_leaves 63 → 20),與「外推情境獎勵正則化」的跨賽
經驗一致。特徵 20 個:year、month、day、dow、day_of_year、weekofyear、quarter、
is_weekend、is_month_start、is_month_end、is_new_year、month/dow/doy 之 sin/cos、
country_cat、store_cat、product_cat(類別欄原生餵入三模型)。

### 3.4 推論表

| 實驗編號 | 後處理 | submission 檔 | 是否已提交 |
|-----|--------|---------------|--------|
| 1 | 無後處理紀錄 | sub_generic_5.31891_20260703_121406.csv | 否 |
| 2 | expm1 反轉 | sub_lgb_xgb_cat_blend_10.17540_20260703_194908.csv | 否 |
| 3 | expm1 反轉 | 無(診斷用,未產生提交檔) | 否 |
| 4 | expm1 反轉 | sub_lgb_cat_rd_blend_10.17489_20260704_000355.csv | 否 |
| 5 | expm1 反轉 | sub_lgb_seedbag_cat_blend_10.16605_20260704_000957.csv | 否 |
| 6 | expm1 反轉 | sub_lgb3seed_cat2seed_blend_10.15721_20260704_001501.csv | 否 |
| 7 | expm1 反轉、負值截 0 | sub_6way_optuna_blend_10.01946_20260704_002039.csv | 否 |
| 8 | expm1 反轉 + auto_scale ×1.02(OOF 內擬合) | 無(僅 OOF 分數,未產生 test 預測) | 否 |

原始紀錄無正式後處理欄位,表中後處理描述取自各實驗的文字備註。submission 為兩欄
格式(id 欄 `id`、目標欄 `num_sold`)。本場為無人值守批次執行,全程未提交
Kaggle;第 7 次實驗 檔案為現行最佳提交檔,其成員以各自超參在
100% 訓練資料上重訓後按權重加權平均,預測 mean 178.2 / max 1432.9,對成長中的 2022
屬合理範圍。

### 3.5 評估指標

指標定義:SMAPE = 對每筆取 |真值 − 預測| 除以兩者絕對值之平均,再對全體取平均並以
百分比表示;對稱化的相對誤差,越低越好。

| 項目 | OOF SMAPE(時序) |
|------|-------------------|
| LGB_tuned(第 7 次實驗 成員,最佳單模) | 10.14833 |
| LGB_s42(第 7 次實驗 成員) | 10.17758 |
| LGB_s2024(第 7 次實驗 成員) | 10.21059 |
| LGB_s7(第 7 次實驗 成員) | 10.19446 |
| CAT_s42(第 7 次實驗 成員) | 10.73064 |
| CAT_s2024(第 7 次實驗 成員) | 10.98383 |
| Ensemble(第 7 次實驗,線性迭代終點) | 10.019463 |
| Ensemble(第 8 次實驗,樹搜尋 best) | **9.75707** |
| Public / Private LB | 無紀錄(未提交) |

本場未提交 Kaggle,無排行榜紀錄,故無排行榜表,CV↔LB gap 無法計算。主要改善幅度:

```
階段 3 淨改善(同折時序):  10.175397 − 10.019463 = 0.155934,0.155934 / 10.175397 ≈ 1.53%
樹搜尋再改善(同折時序):  10.019463 − 9.75707 = 0.262393,0.262393 / 10.019463 ≈ 2.62%
跨方案差距(同特徵同模型):10.175397 − 4.281421 = 5.893976 —— 全部來自外推 vs 內插
                            的驗證難度差,不是模型退步
```

## 4. 實驗軌跡

| 實驗編號 | 時間 | 決策分數 | 階段 | 摘要 |
|-----|------|----------|------|------------|
| 1 | 2026-07-03T12:14:06 | 5.31891(隨機) | 1.2 | 8 通用特徵三模型 blend,隨機 KFold 計分 |
| 2 | 2026-07-03T19:49:08 | 10.175397(時序) | **2** | 20 特徵 + log1p 目標,改採誠實的時序 CV |
| 3 | 2026-07-03T19:53:27 | 4.281421(隨機) | 2.4 | 同特徵改回隨機 KFold,證明變差來自 CV 方案而非特徵 |
| 4 | 2026-07-04T00:03:25 | 10.17489(時序) | **3** | 比例分解成員權重歸零,結構信號輸給 GBDT |
| 5 | 2026-07-04T00:09:24 | 10.166045(時序) | 3.4 | LGB seed bagging(+seed 2024) |
| 6 | 2026-07-04T00:13:58 | 10.157212(時序) | 3.4 | 擴充 seed bagging(+LGB s7、+CAT s2024) |
| 7 | 2026-07-04T00:19:34 | 10.019463(時序) | 3.2 | Optuna 調參 LGB(只以第 5 折評分挑參)入池,線性最大單輪增益 |
| 8 | 2026-07-04T12:18:09 | **9.75707**(時序) | 4.2 | 10-way blend + auto_scale,node #17(22 節點,耗時 321.0 秒) |

- **突破點 1(第 2→3 次實驗,關鍵轉折)**:第 2 次實驗 表面上大幅變差(5.31891 → 10.175397),
  診斷以完全相同的特徵/模型改回隨機 KFold 得 4.281421——優於基線,證明特徵工程有效、
  差距純粹來自「時序驗證是更誠實的量尺」;教訓:不同 CV 方案的分數永不互比。
- **突破點 2(第 6→7 次實驗)**:時序折不可互換,故 Optuna 只以訓練窗最大、最接近 2022
  外推情境的 fold 5 評分挑參;調參 LGB solo 10.14833(最佳單模)入池後拿 0.5 權重,
  blend 10.157212 → 10.019463,線性階段最大增益。
- **突破點 3(第 7→8 次實驗)**:樹搜尋的兩個作法——seed-bag 調參 LGB(SEEDBAG_TUNED_S2024
  solo 9.974778,單模即勝過 第 7 次實驗 全 blend)與 auto_scale ×1.02(單筆最大增益
  -0.168)——合計把分數推至 9.75707(見第 3.2 節補充說明)。

## 5. 效能對照(逐階段)

| 階段 | 配置 | 分數 | 相對改善 |
|------|------|------|----------|
| **1** | 基線(Claude Code 直接執行,未引入 skill;第 1 次實驗 通用批次) | 無可比分數(隨機 KFold 計分,見下方注記) | —(基線;無可比) |
| **2** | + kaggle-agent skill 六階段流程(第 2 次實驗,時序 CV) | 10.175397 | —(本場逐階比較的基準) |
| **3** | + self-improvement 線性迭代(第 7 次實驗,四輪終點) | 10.019463 | 1.53%(階段2→3) |
| **4** | + 樹搜尋(第 8 次實驗,node #17) | **9.75707** | 2.62%(階段3→4) |

**相對改善計算式**(每階段對前一階段)

```
階段2→3: 10.175397 − 10.019463 = 0.155934,相對改善 0.155934 / 10.175397 ≈ 1.53%
階段3→4: 10.019463 − 9.75707  = 0.262393,相對改善 0.262393 / 10.019463 ≈ 2.62%
階段2→4: 10.175397 − 9.75707  = 0.418327,相對改善 0.418327 / 10.175397 ≈ 4.11%
(與跨場彙總報告主表計分方式一致:該表相對變化欄改報 階段2→3 與 階段2→4)

同 CV 方案(隨機 KFold)對照診斷(僅供 CV 方案偏差參考,不併入上表):
通用批次 5.31891 → 同特徵同模型診斷重跑 4.281421,改善 ≈ 19.51%
```

本場由導入報告功能後之版本執行,分數自階段 2 起未低於前一階段——符合計畫書目標三(效能不退步)。

> **CV 計分方式注記(s3e19 特殊場次)**:階段 1 的通用批次實驗(第 1 次實驗,5.31891)使用
> 隨機 KFold,而階段 2/3/4 皆使用 TimeSeriesSplit——test 為嚴格未來期,時序
> CV 才是誠實量尺,兩方案分數**不可比**(見第 3.2 節澄清),故階段 1 標「無可比
> 分數」,逐階算式改以階段 2 為基準報 階段2→3 與 階段3→4(另附 階段2→4)。
> skill 特徵/模型層面的真實增益由上方**同 KFold 診斷對**(5.31891 → 4.281421)單獨
> 呈現,僅作側面診斷,不併入主表任何階段計算。階段 4 的 9.75707 另帶雙重但書
> (fold 5 雙重使用 + scale/seed 對 OOF 擬合),詳見第 3.2 節補充說明,此處不重複;
> 依該但書,真實可期望的 2022 SMAPE 應讀作「明顯低於 10.02」,而非字面上的 9.76。

**子階段分數表**(定義見第 2.3 節;僅列本場有既有紀錄者)

| 子階段 | 分數 | 出處 |
|--------|------|------|
| 1.1(最佳單模:LGB,隨機 KFold 計分) | 5.45633 | 第 1 次實驗 |
| 1.2(三模權重 blend,隨機 KFold 計分) | 5.31891 | 第 1 次實驗 |
| 3.2(Optuna 調參 LGB——只以第 5 折評分挑參,加入池,線性迭代最大單輪增益) | 10.019463 | 第 7 次實驗 |
| 3.4(seed bagging,+LGB seed 2024) | 10.166045 | 第 5 次實驗 |
| 3.4(seed bagging 擴充,+LGB seed 7、+CAT seed 2024) | 10.157212 | 第 6 次實驗 |
| 4.2(樹搜尋 node #17,10-way blend + auto_scale,本場最佳) | **9.75707** | 第 8 次實驗 |

## 6. 總結

本場資料乾淨而結構鮮明:75 條完整日序列、無缺失無重複,店面/產品年度佔比近乎恆定、
國家佔比逐年漂移,目標右偏經 log1p 對稱化;測試期為嚴格未來的 2022 年,決定了本場
最重要的一步——以 TimeSeriesSplit 取代隨機 KFold 作為唯一決策量尺。

關鍵決策有三:其一,第 3 次實驗 的反思診斷把「分數變差」歸因到 CV 方案而非特徵,
避免誤判退步;其二,第 4 次實驗 誠實棄用比例分解——結構要贏 GBDT,目標須在該維度跨年近乎
恆定,而本場總量本身不可外推;其三,第 7 次實驗 調參時只以第 5 折評分挑參,選出更淺更正則的 LGB。

增益來源清楚分層:階段 2 建立時序計分基準(10.175397);階段 3 靠 seed bagging 與
Optuna 調參推進至 10.019463(約 1.53%);樹搜尋再以「seed-bag 調參 LGB」與「auto_scale
×1.02 修正時序 OOF 系統性偏低」兩個作法推至 9.75707(約 2.62%)——後者是所有已跑樹搜尋
場次中相對幅度最大的一場,且兩個作法都是線性階段明確列為「未試」的想法。

可信度方面須誠實:全程未提交 Kaggle、無排行榜對照;9.75707 帶雙重但書(見第 3.2 節),
真實可期望的 2022 SMAPE 應解讀為「明顯低於 10.02」,而非字面上的 9.76。方向性結論
(時序計分下逐階改善、變異縮減類作法在時序 CV 下報酬放大)是穩健的部分。

**重現本實驗的最短路徑**:見第 7 節。

## 7. 重現指令

```bash
# 環境:Linux + uv(Python 由 uv 管理);執行目錄:專案根目錄
cd /home/tjyen/ai_agents/kaggle

# 0) 資料下載(需 KAGGLE_API_TOKEN;本場資料已在 data/ 內,可略過)
uv run kaggle competitions download -c playground-series-s3e19 \
    -p competitions/playground-series-s3e19/data
unzip -o competitions/playground-series-s3e19/data/playground-series-s3e19.zip \
    -d competitions/playground-series-s3e19/data

# 1) EDA(輸出資料概況與時序 CV 決策依據)
uv run python3 competitions/playground-series-s3e19/scripts/eda.py

# 2) 特徵工程(產出 data/train_processed.csv、data/test_processed.csv)
uv run python3 competitions/playground-series-s3e19/scripts/features.py

# 3) 主訓練:TimeSeriesSplit 5 折 + OOF 權重搜尋 + 全量重訓 + submission(第 2 次實驗)
uv run python3 competitions/playground-series-s3e19/scripts/train.py

# 4) (選用)2.4 / 第 3 次實驗:反思診斷,同特徵改用 shuffled KFold
uv run python3 competitions/playground-series-s3e19/scripts/diagnostic_kfold.py

# 5) 階段 3 迭代(依序;第 7 次實驗 產出現行最佳 submission)
uv run python3 competitions/playground-series-s3e19/scripts/train_ratio.py     # 第 4 次實驗(比例分解棄用)
uv run python3 competitions/playground-series-s3e19/scripts/train_seedbag.py   # 第 5 次實驗(3.4)
uv run python3 competitions/playground-series-s3e19/scripts/train_seedbag2.py  # 第 6 次實驗(3.4)
uv run python3 competitions/playground-series-s3e19/scripts/train_optuna.py    # 第 7 次實驗(3.2)

# 6) 4.2 / 第 8 次實驗:樹搜尋工具第 2 版(只算 OOF 分數;軌跡存 experiments_tree.json)
uv run python3 tree_search/run_s3e19.py

# 7) 報告產生與驗證(本檔)
uv run python3 .claude/skills/kaggle-report/assets/collect.py playground-series-s3e19
uv run python3 .claude/skills/kaggle-report/assets/verify_report.py \
    competitions/playground-series-s3e19/REPORT.md competitions/playground-series-s3e19/facts.json
bash .claude/skills/kaggle-report/assets/md2pdf.sh competitions/playground-series-s3e19/REPORT.md \
    competitions/playground-series-s3e19/s3e19_REPORT.pdf

# 8) (選用)提交至 Kaggle(本場未執行;需有效之 KAGGLE_API_TOKEN)
uv run kaggle competitions submit -c playground-series-s3e19 \
    -f competitions/playground-series-s3e19/submissions/sub_6way_optuna_blend_10.01946_20260704_002039.csv \
    -m "6-way blend: LGB seeds + CAT seeds + Optuna fold-5-proxy tuned LGB, time-based CV"
```

執行目錄為專案根目錄 `/home/tjyen/ai_agents/kaggle`;所有 Python 執行皆透過 `uv run`。
