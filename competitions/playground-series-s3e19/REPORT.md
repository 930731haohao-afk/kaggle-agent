# 競賽分析報告:playground-series-s3e19

> 產生方式:kaggle-report skill(數字來自 facts.json,敘述由 agent 撰寫)
> 素材等級:full | 產生日期:2026-07-06
> 本報告所有數字皆出自 facts.json,經 verify_report.py 驗證。

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
| 素材等級 | full |

## 2. 使用工具與環境

| 工具 | 用途 |
|------|------|
| LightGBM / XGBoost / CatBoost | 三個梯度提升樹基模型;LGB 為全程主力,XGB 於 exp 4 起因權重歸零而移除 |
| Optuna | Phase B 超參搜尋(TPE,fold-5 代理目標):exp 7 調出更淺、更強正則的 LGB |
| 自建樹搜尋 harness(v2) | Phase D-5 搜尋 seed/特徵子集/混合權重組合空間,於 node #17 找到 exp 8 的 10-way blend |
| 時序 CV 框架(scikit-learn) | TimeSeriesSplit 5 折(於排序後的唯一日期上切分)為決策量尺;另以 shuffled KFold 執行一次對照診斷 |
| uv | Python 套件與虛擬環境管理,所有腳本皆以 `uv run` 執行 |

本場工具鏈組合邏輯:Claude Code(LLM)負責決策——選擇誠實的時序 CV、設計 Reflexion
診斷分離「特徵 vs CV 方案」效應、否決比例分解成員、決定調參代理折與何時停損;Auto-ML
工具(Optuna、樹搜尋 harness)負責系統化執行超參搜尋與組合空間探索,兩者分工互補。

## 3. 流程(how):五大元件

### 3.1 資料規格

| 項目 | 值 |
|------|-----|
| train 列數 | 136,950 |
| test 列數 | 27,375 |
| 原始欄位數 | 4(date/country/store/product;另有 id 與目標欄) |
| 特別規則摘要 | 禁外部資料/禁預訓練模型/禁網路存取;每日提交上限 5 次 |

目標 `num_sold` 為整數(mean 165.522636、median 98、max 1380),右偏(skew 1.747438),
為 log 轉換候選——全程以 log1p 目標訓練、expm1 反轉。train/test 皆無缺失值、無重複列;
date 有 1,826 個唯一日期,test 的 2022 年日期**全部未在 train 出現**,facts.eda 的
validation_hint 亦提示應採時間切分驗證以避免洩漏。

EDA(`scripts/eda.py`)關鍵定性發現:5 國 × 3 店 × 5 產品共 75 條完整日序列;店面與
產品占年度總量的比例逐年幾乎恆定,國家占比則逐年漂移(外推最難的部分);週末與 12 月/
1 月有明顯季節抬升、1 月 1 日單日尖峰;年度總量非單調(2020 下滑、2021 回升),無可靠
線性趨勢。

### 3.2 實驗總表

> **語意澄清(本報告全文適用)**:本場同時存在兩種 CV 方案——(時序)= TimeSeriesSplit
> 5 折(每折驗證區塊嚴格晚於訓練窗,模擬 train 2017–21 → test 2022 的外推缺口)與
> (隨機)= shuffled KFold 5 折(內插式,對本任務系統性樂觀)。**兩種方案的分數不可互相
> 比較**;所有 keep/reject 決策一律以(時序)分數為準。SMAPE 無取整/門檻類後處理,故
> 各實驗「原始 OOF」即「決策分數」,兩欄同值。

| exp | 階段 | 模型/成員 | 特徵數 | 原始 OOF | 決策分數 | 採納 |
|-----|------|-----------|--------|----------|----------|------|
| 1 | Baseline(通用批次) | LGB/XGB/CAT(0.7/0.0/0.3) | 8 | 5.31891(隨機) | 5.31891(隨機) | 基線參照;與(時序)各實驗不可比 |
| 2 | Phase A(skill 六階段) | LGB/XGB/CAT(0.9/0.0/0.1) | 20 | 10.175397(時序) | 10.175397(時序) | 是,Phase A 最佳 |
| 3 | Reflexion 診斷 | LGB/XGB/CAT(0.4/0.2/0.4) | 20 | 4.281421(隨機) | 4.281421(隨機) | 否,僅診斷用,不入決策 |
| 4 | Phase B R1 | LGB/CAT/RatioDecomp(0.95/0.05/0.0) | 20 | 10.17489(時序) | 10.17489(時序) | RD 成員棄用(權重 0;微幅增益僅來自較細權重網格) |
| 5 | Phase B R2 | LGB seed 42/2024 + CAT(0.65/0.3/0.05) | 20 | 10.166045(時序) | 10.166045(時序) | 是 |
| 6 | Phase B R3 | LGB×3 seeds + CAT×2 seeds(5-way) | 20 | 10.157212(時序) | 10.157212(時序) | 是 |
| 7 | Phase B R4(線性終點) | 6-way + Optuna 調參 LGB(權重 0.5) | 20 | 10.019463(時序) | 10.019463(時序) | 是,線性迭代最終;現行最佳提交檔 |
| 8 | Phase D-5 樹搜尋(best) | 10-way blend + auto_scale ×1.02(node #17) | 20(池內含特徵子集探針) | **9.75707**(時序) | **9.75707**(時序) | 是,本場誠實口徑最佳(OOF-only) |

**best 成員表(exp 8,dirichlet 權重搜尋 + auto_scale 全域乘數 1.02)**

| 成員 | 權重 | solo 分數 | 備註 |
|------|------|-----------|------|
| SEEDBAG_TUNED_S3000 | 0.2559 | 9.990227 | 調參 LGB 之 seed 3000 變體 |
| SEEDBAG_TUNED_S2024 | 0.2448 | 9.974778 | 調參 LGB 之 seed 2024 變體;solo 即勝過 exp 7 全 blend |
| LGB_S7 | 0.1989 | 10.19448 | 手設 LGB,seed 7 |
| CALSUBSET | 0.1781 | 10.137705 | 刪 weekofyear 等冗餘欄之特徵子集探針 |
| DEEPLGB | 0.0746 | 10.347073 | 刻意多樣化之深/輕正則 LGB;solo 弱但有貢獻 |
| LGB_S42 | 0.0238 | 10.177577 | 手設 LGB,seed 42 |
| LGB_S2024 | 0.0115 | 10.210617 | 手設 LGB,seed 2024 |
| LGB_TUNED_ROOT | 0.0089 | 10.148325 | 搜尋根節點 = exp 7 之 Optuna 調參 LGB |
| CAT_S2024 | 0.0032 | 10.983835 | 手設 CatBoost,seed 2024 |
| CAT_S42 | 0.0003 | 10.730642 | 手設 CatBoost,seed 42 |

選型脈絡:三種梯度提升樹是中型表格資料的標準組合,皆原生支援類別特徵。外推情境下
LGB 明顯領先,XGB 兩度被權重搜尋歸零後移除;exp 4 的結構式比例分解成員(solo 14.75038)
也被歸零——GBDT 的原生類別分裂已學走佔比結構,真正瓶頸(總量一年外推)對所有方法一視
同仁。exp 8 的兩個決定性槓桿:對調參 LGB 做 seed bagging(調參目標是 fold-5 代理而非
全 OOF,留有真實 seed 變異可平均消除),與修正時序 OOF 系統性偏低約 2% 的全域 ×1.02
乘數(每折驗證區塊皆晚於訓練窗、序列成長至 2021,此偏差結構上與真實 2022 缺口同型)。

> **誠實但書(exp 8 與 facts.best;全文其他引用處不再重複)**:
> 1. facts.json 的 `best` 欄位以分數最小選出,落在 exp 3(4.281421)——那是(隨機)
>    診斷量尺的產物,不可作為本場最佳;誠實口徑((時序))的最佳為 exp 8。
> 2. exp 8 為 OOF-only 樹搜尋結果:未產生 test 預測、無 submission 檔、未提交 Kaggle;
>    「現行最佳提交檔」屬 exp 7(見 3.4 節)。
> 3. **9.75707 帶雙重但書**:(a)與 exp 2–7 相同,fold 5 既是 Optuna 調參目標又佔
>    OOF 的 1/5,分數含 fold-5 double-dip 樂觀成分;(b)exp 8 額外的 scale 乘數與
>    seed 挑選皆直接對 OOF 擬合(無巢狀驗證)。各為 1 至數個參數、擬合於 114,000 列
>    OOF,單項過擬風險低,但真實可期望的 2022 SMAPE 應解讀為「明顯低於 10.02」,
>    而非字面上的 9.76。

### 3.3 訓練規格表

| exp | CV 方案 | folds | seed |
|-----|---------|-------|------|
| 1 | 5fold(隨機) | 5 | 無紀錄 |
| 2 | TimeSeriesSplit-5fold-on-unique-dates | 5 | 無紀錄 |
| 3 | KFold-5fold-shuffled-seed42 | 5 | 無紀錄(方案名含 seed42) |
| 4–7 | TimeSeriesSplit-5fold-on-unique-dates(與 exp 2 同折) | 5 | 無紀錄 |
| 8 | TimeSeriesSplit-5fold-on-unique-dates(逐位驗證與 exp 2–7 同折) | 5 | 無紀錄 |

**為何用此 CV**:測試期(2022)完全落在訓練期之後、零日期重疊,模型實際面對「外推到
未見年份」;隨機 KFold 讓驗證日夾在已見季節循環之間,對本任務系統性樂觀。故主軌跡在
排序後的唯一日期上做 TimeSeriesSplit,每折以擴張窗訓練、驗證嚴格較晚的日期區塊。
exp 3 刻意改回與基線同型態的隨機 KFold,僅作為特徵集有效性的對照診斷,不用於選模。

Objective:log1p(num_sold) 迴歸,預測經 expm1 反轉後計 SMAPE。關鍵超參(有紀錄者):

```
LGB(手設,exp 2):  n_estimators=2000, learning_rate=0.03, num_leaves=63,
                     min_child_samples=20, subsample=0.9, colsample_bytree=0.8, reg_lambda=1.0
XGB(手設,exp 2):  n_estimators=2000, learning_rate=0.03, max_depth=7,
                     subsample=0.9, colsample_bytree=0.8, reg_lambda=1.0
CAT(手設,exp 2):  iterations=2000, learning_rate=0.05, depth=8, l2_leaf_reg=3.0
LGB_tuned(Optuna,exp 7):learning_rate≈0.0371, num_leaves=20, min_child_samples=38,
                     subsample≈0.61, colsample_bytree≈0.90, reg_alpha≈0.001,
                     reg_lambda≈0.43, n_estimators=2000
```

調參版比手設更淺、更強正則(num_leaves 63 → 20),與「外推情境獎勵正則化」的跨賽
經驗一致。特徵 20 個:year、month、day、dow、day_of_year、weekofyear、quarter、
is_weekend、is_month_start、is_month_end、is_new_year、month/dow/doy 之 sin/cos、
country_cat、store_cat、product_cat(類別欄原生餵入三模型)。

### 3.4 推論表

| exp | 後處理 | submission 檔 | 已提交 |
|-----|--------|---------------|--------|
| 1 | 無後處理紀錄 | sub_generic_5.31891_20260703_121406.csv | 否 |
| 2 | expm1 反轉 | sub_lgb_xgb_cat_blend_10.17540_20260703_194908.csv | 否 |
| 3 | expm1 反轉 | 無(診斷用,未產生提交檔) | 否 |
| 4 | expm1 反轉 | sub_lgb_cat_rd_blend_10.17489_20260704_000355.csv | 否 |
| 5 | expm1 反轉 | sub_lgb_seedbag_cat_blend_10.16605_20260704_000957.csv | 否 |
| 6 | expm1 反轉 | sub_lgb3seed_cat2seed_blend_10.15721_20260704_001501.csv | 否 |
| 7 | expm1 反轉、負值截 0 | sub_6way_optuna_blend_10.01946_20260704_002039.csv | 否 |
| 8 | expm1 反轉 + auto_scale ×1.02(OOF 內擬合) | 無(OOF-only,未產生 test 預測) | 否 |

facts 各實驗無 postprocess 欄位,表中後處理描述取自各實驗 notes。submission 為兩欄
格式(id 欄 `id`、目標欄 `num_sold`)。本場為無人值守批次執行,全程未提交 Kaggle
(leaderboard 列於 facts.missing);exp 7 檔案為現行最佳提交檔,其成員以各自超參在
100% 訓練資料上重訓後按權重加權平均,預測 mean 178.2 / max 1432.9,對成長中的 2022
屬合理範圍。

### 3.5 評估指標

指標定義:SMAPE = 對每筆取 |真值 − 預測| 除以兩者絕對值之平均,再對全體取平均並以
百分比表示;對稱化的相對誤差,越低越好。

| 項目 | OOF SMAPE(時序) |
|------|-------------------|
| LGB_tuned(exp 7 成員,最佳單模) | 10.14833 |
| LGB_s42(exp 7 成員) | 10.17758 |
| LGB_s2024(exp 7 成員) | 10.21059 |
| LGB_s7(exp 7 成員) | 10.19446 |
| CAT_s42(exp 7 成員) | 10.73064 |
| CAT_s2024(exp 7 成員) | 10.98383 |
| Ensemble(exp 7,線性迭代終點) | 10.019463 |
| Ensemble(exp 8,樹搜尋 best) | **9.75707** |
| Public / Private LB | 無紀錄(未提交) |

本場無排行榜表(leaderboard 為空),CV↔LB gap 無法計算。主要改善幅度:

```
Phase B 淨改善(同折時序):  10.175397 − 10.019463 = 0.155934,0.155934 / 10.175397 ≈ 1.53%
樹搜尋再改善(同折時序):  10.019463 − 9.75707 = 0.262393,0.262393 / 10.019463 ≈ 2.62%
跨方案差距(同特徵同模型):10.175397 − 4.281421 = 5.893976 —— 全部來自外推 vs 內插
                            的驗證難度差,不是模型退步
```

## 4. 實驗軌跡

**實驗階段對照表**(內部代號使用前先定義,R-W8)

| 代號 | 白話名稱 | 對應層級 |
|------|----------|----------|
| Phase A | kaggle-agent skill 六階段首跑 | tier2 |
| Phase B | self-improvement 線性迭代 | tier3 |
| Phase D-5 | 樹搜尋執行(harness v2) | tier4 |

| exp | 時間 | 決策分數 | 階段 | 一句話摘要 |
|-----|------|----------|------|------------|
| 1 | 2026-07-03T12:14:06 | 5.31891(隨機) | Baseline(通用批次) | 8 通用特徵三模型 blend,隨機 KFold 口徑 |
| 2 | 2026-07-03T19:49:08 | 10.175397(時序) | Phase A | 20 特徵 + log1p 目標,改採誠實的時序 CV |
| 3 | 2026-07-03T19:53:27 | 4.281421(隨機) | Reflexion 診斷 | 同特徵改回隨機 KFold,證明變差來自 CV 方案而非特徵 |
| 4 | 2026-07-04T00:03:25 | 10.17489(時序) | Phase B R1 | 比例分解成員權重歸零,結構信號輸給 GBDT |
| 5 | 2026-07-04T00:09:24 | 10.166045(時序) | Phase B R2 | LGB seed bagging(+seed 2024) |
| 6 | 2026-07-04T00:13:58 | 10.157212(時序) | Phase B R3 | 擴充 seed bagging(+LGB s7、+CAT s2024) |
| 7 | 2026-07-04T00:19:34 | 10.019463(時序) | Phase B R4 | Optuna fold-5 代理調參 LGB 入池,線性最大單輪增益 |
| 8 | 2026-07-04T12:18:09 | **9.75707**(時序) | Phase D-5 樹搜尋 | 10-way blend + auto_scale,node #17(22 節點,wall 321.0s) |

- **突破點 1(exp 2→3,關鍵轉折)**:exp 2 表面上大幅變差(5.31891 → 10.175397),
  診斷以完全相同的特徵/模型改回隨機 KFold 得 4.281421——優於基線,證明特徵工程有效、
  差距純粹來自「時序驗證是更誠實的量尺」;教訓:不同 CV 方案的分數永不互比。
- **突破點 2(exp 6→7)**:時序折不可互換,故以訓練窗最大、最接近 2022 外推情境的
  fold 5 作 Optuna 代理目標;調參 LGB solo 10.14833(最佳單模)入池後拿 0.5 權重,
  blend 10.157212 → 10.019463,線性階段最大增益。
- **突破點 3(exp 7→8)**:樹搜尋兩槓桿——seed-bag 調參 LGB(SEEDBAG_TUNED_S2024
  solo 9.974778,單模即勝過 exp 7 全 blend)與 auto_scale ×1.02(單筆最大增益
  -0.168)——合計把分數推至 9.75707(但書見 3.2 節)。

無法解析之紀錄:無(facts.unparsed 為空陣列)。

## 5. 效能對照:四層消融

| 層級 | 配置 | 分數 | 相對改善 |
|------|------|------|----------|
| tier1 | 基線(Claude Code 直接執行,未引入 skill;exp 1 通用批次) | 無可比 generic(隨機 KFold 口徑,見下方 call-out) | —(基線;無可比) |
| tier2 | + kaggle-agent skill 六階段流程(exp 2,時序 CV) | 10.175397 | —(本場消融比較的錨點) |
| tier3 | + self-improvement 線性迭代(exp 7,Phase B 四輪終點) | 10.019463 | 見下方算式(tier2→tier3) |
| tier4 | + 樹搜尋(exp 8,harness v2 node #17) | **9.75707** | 見下方算式(tier3→tier4) |

```
tier2→tier3: 10.175397 − 10.019463 = 0.155934,相對改善 0.155934 / 10.175397 ≈ 1.53%
tier3→tier4: 10.019463 − 9.75707  = 0.262393,相對改善 0.262393 / 10.019463 ≈ 2.62%
tier2→tier4: 10.175397 − 9.75707  = 0.418327,相對改善 0.418327 / 10.175397 ≈ 4.11%
(與 benchmark_facts.json 主表口徑一致:該表相對變化欄改報 tier2→tier3 與 tier2→tier4)

同 CV 方案(隨機 KFold)側寫診斷對(僅供 CV-scheme 偏誤參考,不併入上表):
generic 5.31891 → 同特徵同模型診斷重跑 4.281421,改善 ≈ 19.51%
```

本場由導入報告功能後之版本執行,分數自 tier2 起未低於前一層——符合計畫書目標三(效能不退步)。

> **CV 口徑注記(s3e19 特殊場次)**:tier1 的 generic 批次實驗(exp 1,5.31891)使用
> 隨機 KFold,而 tier2/tier3/tier4 皆使用 TimeSeriesSplit——test 為嚴格未來期,時序
> CV 才是誠實量尺,兩方案分數**不可比**(見 3.2 節澄清),故 tier1 標「無可比
> generic」,消融算式改以 tier2 為錨點報 tier2→tier3 與 tier3→tier4(另附 tier2→tier4)。
> skill 特徵/模型層面的真實增益由上方**同 KFold 診斷對**(5.31891 → 4.281421)單獨
> 呈現,僅作側面診斷,不併入主表任何 tier 計算。tier4 的 9.75707 另帶雙重但書
> (fold-5 double-dip + scale/seed OOF 擬合),詳見 3.2 節 call-out,此處不重複;
> 依該但書,真實可期望的 2022 SMAPE 應讀作「明顯低於 10.02」,而非字面上的 9.76。

### 子刻度分解(選填)

依 §5 附錄子刻度分類法(僅作分類字典,不強制逐級測量),下表僅列本場 experiments.json
天然存在對應中間紀錄之子刻度;無對應紀錄之子刻度不列,分數逐字引用自 facts.json。

| 子刻度 | 優化辦法 | 分數 | 出處(exp# 或 tree node) |
|--------|----------|------|--------------------------|
| 1.a | 單模預設參數(tier1 三模型中最佳者:LGB,隨機 KFold 口徑) | 5.45633 | exp1 |
| 1.b | 多模+OOF 權重搜尋 blend(tier1 三模型混合,隨機 KFold 口徑) | 5.31891 | exp1 |
| 3.d | seed bagging(+LGB seed 2024) | 10.166045 | exp5 |
| 3.d | seed bagging 擴充(+LGB seed 7、+CAT seed 2024) | 10.157212 | exp6 |
| 3.b | Optuna fold-5 代理調參 LGB,加入 pool(線性迭代最大單輪增益) | 10.019463 | exp7 |
| 4.b | 樹搜尋 harness v2(node #17,10-way blend + auto_scale,本場最佳) | 9.75707 | exp8 |

## 6. 總結

本場資料乾淨而結構鮮明:75 條完整日序列、無缺失無重複,店面/產品年度佔比近乎恆定、
國家佔比逐年漂移,目標右偏經 log1p 對稱化;測試期為嚴格未來的 2022 年,決定了本場
最重要的一步——以 TimeSeriesSplit 取代隨機 KFold 作為唯一決策量尺。

關鍵決策有三:其一,exp 3 的 Reflexion 診斷把「分數變差」歸因到 CV 方案而非特徵,
避免誤判退步;其二,exp 4 誠實棄用比例分解——結構要贏 GBDT,目標須在該維度跨年近乎
恆定,而本場總量本身不可外推;其三,exp 7 以 fold-5 代理調參,選出更淺更正則的 LGB。

增益來源清楚分層:skill 首跑(Phase A)建立時序口徑基準(10.175397);線性迭代(Phase B)靠 seed bagging 與
Optuna 調參推進至 10.019463(約 1.53%);樹搜尋再以「seed-bag 調參 LGB」與「auto_scale
×1.02 修正時序 OOF 系統性偏低」兩槓桿推至 9.75707(約 2.62%)——後者是所有已跑樹搜尋
場次中相對幅度最大的一場,且兩個槓桿都是線性階段明確列為「未試」的想法。

可信度方面須誠實:全程未提交 Kaggle、無排行榜錨點;9.75707 帶雙重但書(見 3.2 節),
真實可期望的 2022 SMAPE 應解讀為「明顯低於 10.02」,而非字面上的 9.76。方向性結論
(時序口徑下逐層改善、變異縮減類槓桿在時序 CV 下報酬放大)是穩健的部分。

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

# 3) 主訓練:TimeSeriesSplit 5 折 + OOF 權重搜尋 + 全量重訓 + submission(exp 2)
uv run python3 competitions/playground-series-s3e19/scripts/train.py

# 4) (選用)Reflexion 診斷:同特徵改用 shuffled KFold(exp 3)
uv run python3 competitions/playground-series-s3e19/scripts/diagnostic_kfold.py

# 5) Phase B 迭代(依序;exp 7 產出現行最佳 submission)
uv run python3 competitions/playground-series-s3e19/scripts/train_ratio.py     # exp 4(RD 棄用)
uv run python3 competitions/playground-series-s3e19/scripts/train_seedbag.py   # exp 5
uv run python3 competitions/playground-series-s3e19/scripts/train_seedbag2.py  # exp 6
uv run python3 competitions/playground-series-s3e19/scripts/train_optuna.py    # exp 7

# 6) Phase D-5 樹搜尋 harness v2(exp 8,OOF-only;軌跡存 experiments_tree.json)
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
