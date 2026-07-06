# 競賽分析報告:playground-series-s3e19

> 產生方式:kaggle-report skill(數字來自 facts.json,敘述由 agent 撰寫)
> 素材等級:full | 產生日期:2026-07-03(2026-07-04 更新:Phase B 自我改進迭代,實驗 #4–#7;
> 2026-07-04 再更新:Phase G-1b 樹搜尋成果入帳,實驗 #8)

## 1. 競賽目的

**What(解什麼問題)**:本競賽為 Kaggle Playground Series Season 3 Episode 19(Aygun et al. Nature 2026 Kaggle Playground benchmark),任務是「Forecast mini-course sales」——依日期(date)、國家(country)、店面(store)、產品(product)四個維度,預測每日課程銷量 `num_sold`。訓練資料涵蓋 2017 至 2021 年,測試集為整個 2022 年,屬於典型的零售銷售時間序列外推問題。

**Why(為何重要、指標為何合理)**:銷售預測是零售與電商的核心營運問題,直接影響備貨與行銷決策。評估指標為 **SMAPE**(symmetric mean absolute percentage error,minimize):各序列的銷量規模差異大(不同國家/店面/產品的量級不同),絕對誤差類指標會被大量級序列主導;SMAPE 以「真值與預測值的平均」作分母,將誤差正規化為相對比例,使大小序列的預測品質獲得同等權重,對此類多序列、多量級的銷售預測是合理選擇。

| 項目 | 值 |
|------|-----|
| 競賽 | playground-series-s3e19 |
| URL | https://www.kaggle.com/competitions/playground-series-s3e19 |
| 問題型別 | regression |
| 評估指標 | smape(minimize) |
| 目標欄位 | num_sold |
| ID 欄位 | id |

## 2. 流程(how):五大元件

### 2.1 資料規格

- **檔案**:train.csv、test.csv、sample_submission.csv。
- **列數**:facts.json 無紀錄(依 STATUS.md 敘述脈絡:訓練集為 2017-01-01 至 2021-12-31 的每日資料,測試集為 2022 全年;精確列數此處不引數字)。
- **欄位**:原始特徵為 date 加三個類別欄(country、store、product),目標為整數 `num_sold`;無缺失值、無重複列。基線實驗(#1)使用 8 個通用特徵,本場工程後為 20 個特徵(清單見第 2.3 節)。
- **序列結構**(EDA `scripts/eda.py` 執行結果,定性描述):5 國 × 3 店 × 5 產品共 75 條完整日序列;店面與產品占年度總量的比例逐年幾乎恆定,國家占比則逐年緩慢漂移;週末(尤其週日)與 12 月/1 月有明顯季節性抬升,1 月 1 日有單日尖峰;目標右偏、log1p 後接近對稱;年度總量非單調(2020 年下滑、2021 年回升)。

**特別規則**:

| 規則 | 值 |
|------|-----|
| 外部資料 | 不允許 |
| 預訓練模型 | 不允許 |
| 網路存取 | 不允許 |
| 每日提交上限 | 5 |

### 2.2 模型規格

本場共八筆實驗;**facts.json 的 best(分數最佳)為實驗 #3(診斷用 KFold 重跑),但那是內插式 CV 的樂觀量尺,不可與時間序實驗直接比較**。在誠實的 TimeSeriesSplit 量尺下,**實驗 #8(Phase G-1b,樹搜尋 v2,blend SMAPE 9.75707)** 現為新的最佳,勝過先前的線性迭代終點實驗 #7(10.019463);但實驗 #8 是 OOF-only 的樹搜尋產物,**未產生提交檔、未提交至 Kaggle**,故「現行最佳提交」仍是實驗 #7 對應的 submission 檔(見第 2.4 節)。兩種 CV 方案(TimeSeriesSplit vs KFold)的分數不可跨方案比較(詳見第 2.3、3 節)。

#### 實驗 #2 — 提交所用(TimeSeriesSplit 5-fold)

| Base model | OOF SMAPE |
|------------|-----------|
| LightGBM | 10.17758 |
| XGBoost | 10.46 |
| CatBoost | 10.73064 |

Ensemble:OOF 加權平均網格搜尋(grid_step 0.1),最佳權重 **LGB 0.9 / XGB 0.0 / CAT 0.1**,blend SMAPE **10.175397**。

#### 實驗 #3 — facts.best(診斷用,shuffled KFold,未產生提交檔)

| Base model | OOF SMAPE |
|------------|-----------|
| LightGBM | 4.31618 |
| XGBoost | 4.33201 |
| CatBoost | 4.3239 |

Ensemble:同樣的權重網格搜尋,最佳權重 **LGB 0.4 / XGB 0.2 / CAT 0.4**,blend SMAPE **4.281421**。

**選型理由**:LGB/XGB/CatBoost 三種梯度提升樹是中型表格資料的標準組合,皆原生支援類別特徵並可用早停控制訓練成本。外推情境(實驗 #2)下 LGB 明顯領先、XGB 權重被搜到 0——與基線實驗 #1(權重 LGB 0.7 / XGB 0.0 / CAT 0.3)的型態一致;內插情境(實驗 #3)三模型幾乎同分,blend 分散權重。真實測試集(2022 年)為外推情境,故 Phase B 以降的所有決策皆以 TimeSeriesSplit 分數為準。

#### Phase B 自我改進迭代(實驗 #4–#7,皆 TimeSeriesSplit 5-fold)

**實驗 #4 — 比例分解成員(棄用)**:新增「每日總量調和線性迴歸 × country 佔比線性趨勢外推(歸一化)× 國內店/產品固定佔比」的結構式成員(RatioDecomp),與 LGB、CAT 三方權重搜尋(XGB 因兩度權重 0 而移除)。

| Base model | OOF SMAPE |
|------------|-----------|
| LightGBM | 10.17758 |
| CatBoost | 10.73064 |
| RatioDecomp | 14.75038 |

權重搜尋(grid_step 0.05)結果 **LGB 0.95 / CAT 0.05 / RD 0.0**,blend 10.17489——RD 權重歸零,增益僅來自更細的權重網格。結論:結構信號在此輸給 GBDT(詳見第 3 節)。

**實驗 #5 — seed bagging**:LGB 加第二 seed(2024)。LGB_s42 10.17758 / LGB_s2024 10.21062 / CAT_s42 10.73064;權重 **0.65 / 0.3 / 0.05**,blend **10.166045**。

**實驗 #6 — 擴充 seed bagging**:再加 LGB seed 7 與 CAT seed 2024。LGB_s7 10.19446 / CAT_s2024 10.98383;五方權重 **LGB_s42 0.4 / LGB_s2024 0.2 / LGB_s7 0.3 / CAT_s42 0.1 / CAT_s2024 0.0**,blend **10.157212**。

**實驗 #7 — Optuna fold-5 代理調參 LGB(線性迭代終點)**:以 TimeSeriesSplit 最後一折(訓練窗最大、最接近真實 2022 外推情境)為 Optuna 目標(TPE、40 trials),調參版 LGB solo OOF **10.14833**(最佳單模),依「加入池不替換」原則進六方權重搜尋:**LGB_s42 0.0 / LGB_s2024 0.1 / LGB_s7 0.4 / CAT_s42 0.0 / CAT_s2024 0.0 / LGB_tuned 0.5**,blend **10.019463**。誠實註記:fold 5 既是調參目標又佔 OOF 的 1/5,此分數含部分樂觀成分(方向性增益仍真實,未被調參的 folds 2–3 上調參版亦小勝原版)。

#### 2.2c 樹搜尋最佳(best under TimeSeriesSplit,實驗 #8)——本次更新(Phase G-1b)新增

Phase D-5(harness v2,2026-07-04)樹搜尋在實驗 #7 的相同 20-特徵集上另闢節點空間,於
`experiments_tree.json` 的 node #17(22 個評估節點:13 solo/9 blend,0 失敗,總 wall
321.0s;此節點於第 18/22 次評估找到,node wall_s=21.8s)找到 SMAPE 更低的 10-way blend
(best.base_models):

| 模型 | OOF SMAPE | 權重 |
|------|-----------|------|
| LGB_TUNED_ROOT | 10.148325 | 0.0089 |
| LGB_S42 | 10.177577 | 0.0238 |
| LGB_S2024 | 10.210617 | 0.0115 |
| LGB_S7 | 10.19448 | 0.1989 |
| CAT_S42 | 10.730642 | 0.0003 |
| CAT_S2024 | 10.983835 | 0.0032 |
| SEEDBAG_TUNED_S2024 | 9.974778 | 0.2448 |
| DEEPLGB | 10.347073 | 0.0746 |
| CALSUBSET | 10.137705 | 0.1781 |
| SEEDBAG_TUNED_S3000 | 9.990227 | 0.2559 |

**Ensemble**(best.ensemble):dirichlet 權重搜尋 + `auto_scale` 全域乘數(scale_used =
1.02,在 metric_fn 內部對 OOF 直接套用,而非僅在 submission 階段套用),score = 9.75707。
較實驗 #7 改善 -0.262393(約 -2.6% 相對改善,詳細計算見第 2.5 節程式區塊)——是目前為止所有已跑樹搜尋競賽中相對margin
最大的一場。

**兩個關鍵槓桿(best.notes)**:(1) 對 Optuna 調參後的 LGB 本身做 seed bagging
(SEEDBAG_TUNED_S2024,solo 9.974778)——STATUS.md 明確列為「未試」且原本擔心可能中性
(依 s3e5 對「直接調參目標」做 seed bagging 可能無效的前例),但因本場調參目標是 fold-5
代理(而非全 OOF),仍留有真實 seed 變異可供平均消除,結果單一 solo 就打敗整個線性 6-way
blend;(2) 全域 ×1.02 的 OOF-fitted 乘數(auto_scale),修正 TimeSeriesSplit OOF 系統性
偏低約 2%(每折驗證區塊皆晚於其訓練窗、序列持續成長至 2021 年)——本輪單筆最大增益
(-0.168,發生於 8-way 層的 node #15)。

> **誠實警語(逐字引用 STATUS.md〈Appendix: Phase D-5 tree-search v2 sweep — the
> TIME-SERIES case〉)**:「like the linear 10.01946, the 9.75707 carries fold-5
> double-dip optimism, PLUS the scale parameter and the seed selection are
> OOF-fitted. All are 1-to-few-parameter fits on 114k rows (low overfit risk
> individually), but the true expected 2022 SMAPE is best read as "meaningfully
> below 10.02", not literally 9.76.」完整節點鏈、TimeSeriesSplit 折重現驗證、與
> backtrack/dedup 記錄見 `competitions/playground-series-s3e19/STATUS.md` 同一 Appendix。
>
> **重要澄清**:best.notes 明確記載這是 **OOF-only 搜尋結果——未產生任何 test 預測,亦
> 未提交至 Kaggle**(facts.json 本筆無 submission 欄位)。本場的「現行最佳提交」仍是實驗
> #7 對應的 submission 檔(見第 2.4 節)。

### 2.3 訓練規格

| 實驗 | CV 方案 | n_splits | seed |
|------|---------|----------|------|
| #2 | TimeSeriesSplit-5fold-on-unique-dates | 5 | 無紀錄 |
| #3(診斷) | KFold-5fold-shuffled-seed42 | 5 | 無紀錄(方案字串內含 seed42) |
| #4–#7(Phase B) | TimeSeriesSplit-5fold-on-unique-dates(與 #2 同折) | 5 | 無紀錄 |
| #8(樹搜尋,Phase G-1b) | TimeSeriesSplit-5fold-on-unique-dates(與 #2/#4–#7 同折) | 5 | 無紀錄 |

**Objective 與關鍵超參**(實驗 #2 紀錄):目標經 log1p 轉換後以迴歸目標訓練,預測以 expm1 反轉後計算 SMAPE。

| 模型 | 關鍵超參 |
|------|----------|
| LightGBM | n_estimators 2000、learning_rate 0.03、num_leaves 63、min_child_samples 20、subsample 0.9、colsample_bytree 0.8、reg_lambda 1.0 |
| XGBoost | n_estimators 2000、learning_rate 0.03、max_depth 7、subsample 0.9、colsample_bytree 0.8、reg_lambda 1.0 |
| CatBoost | iterations 2000、learning_rate 0.05、depth 8、l2_leaf_reg 3.0 |

三模型皆對各 fold 的驗證區塊做早停。

**為何用此 CV**:測試期(2022)完全落在訓練期(2017–2021)之後、零重疊,模型實際面對的是「外推到未見年份」;隨機 KFold 會讓驗證日夾在已見的季節循環之間,對此任務系統性樂觀。因此主實驗(#2)採 TimeSeriesSplit 於排序後的唯一日期上切分,每折以擴張窗訓練、驗證嚴格較晚的日期區塊,忠實模擬 train→test 的時間缺口。實驗 #3 刻意改用與基線相同型態的隨機 KFold,只作為「特徵集是否有效」的對照診斷,不用於選模。

**特徵(20 個)**:year、month、day、dow、day_of_year、weekofyear、quarter、is_weekend、is_month_start、is_month_end、is_new_year、month_sin、month_cos、dow_sin、dow_cos、doy_sin、doy_cos、country_cat、store_cat、product_cat。

### 2.4 推論程序

- **最終模型(實驗 #7)**:六個成員(LGB seeds 42/2024/7、CAT seeds 42/2024、Optuna 調參版 LGB)以各自超參在 100% 訓練資料上重訓,按權重 0.0 / 0.1 / 0.4 / 0.0 / 0.0 / 0.5 加權平均。
- **後處理**:facts.json 的 postprocess 欄位無紀錄;依實驗 notes,預測值由 log1p 空間經 expm1 反轉回原始銷量尺度(訓練腳本另將負值截斷為 0)。
- **Submission(現行最佳)**:`sub_6way_optuna_blend_10.01946_20260704_002039.csv`,格式為兩欄(`id`, `num_sold`),與 sample_submission.csv 之形狀、id 順序逐一驗證通過。Phase B 前的提交檔為 `sub_lgb_xgb_cat_blend_10.17540_20260703_194908.csv`。本場為無人值守批次執行,**未**實際提交至 Kaggle。

### 2.5 評估指標

**定義**:SMAPE = 100% × 平均(|真值 − 預測| / ((|真值| + |預測|)/2)),對稱化的相對誤差,越低越好。

#### 分數總表

| 實驗 | CV 方案 | Blend SMAPE |
|------|---------|-------------|
| #1 基線 | 5fold | 5.31891 |
| #2 | TimeSeriesSplit 5-fold | 10.175397 |
| #3 診斷 | shuffled KFold 5-fold | 4.281421 |
| #4 +RatioDecomp(棄用) | TimeSeriesSplit 5-fold | 10.17489 |
| #5 seed bagging | TimeSeriesSplit 5-fold | 10.166045 |
| #6 擴充 seed bagging | TimeSeriesSplit 5-fold | 10.157212 |
| #7 +Optuna 調參 LGB(線性迭代終點) | TimeSeriesSplit 5-fold | 10.019463 |
| #8 樹搜尋 v2(node #17,best) | TimeSeriesSplit 5-fold | **9.75707** |

(#2/#3 的 per-model OOF 見第 2.2 節前段,#4–#7 的見第 2.2 節 Phase B 小節,#8 見第 2.2c 節;#1 的 per-model 分數為 LGB 5.45633 / XGB 7.41858 / CAT 6.09778。)

Public/Private LB:無紀錄(未提交,facts.missing 含 leaderboard),故無 CV↔LB gap 可計算。

同方案(隨機 KFold)下,工程後特徵相對基線的改善幅度:

```
5.31891 − 4.281421 = 1.037489
1.037489 / 5.31891 × 100 ≈ 19.5%(SMAPE 下降)
```

跨方案的差距(同特徵、同模型,僅 CV 方案不同):

```
10.175397 − 4.281421 = 5.893976
```

此差距全部來自「外推 vs 內插」的驗證難度差異,不是模型退步(見第 3 節)。

Phase B 淨改善(同一 TimeSeriesSplit 方案、同折):

```
10.175397 − 10.019463 = 0.155934
0.155934 / 10.175397 × 100 ≈ 1.53%(SMAPE 下降)
```

樹搜尋(exp #8)相對線性迭代終點(exp #7)的改善(同一 TimeSeriesSplit 方案、同折):

```
10.019463 − 9.75707 = 0.262393
0.262393 / 10.019463 × 100 ≈ 2.6188...%(SMAPE 下降,約 -2.6%,四捨五入至第一位小數)
```

## 3. 實驗軌跡

| id | timestamp | score (SMAPE) | source_format |
|----|-----------|---------------|---------------|
| 1 | 2026-07-03T12:14:06 | 5.31891 | generic_batch |
| 2 | 2026-07-03T19:49:08 | 10.175397 | v2 |
| 3 | 2026-07-03T19:53:27 | 4.281421 | v2 |
| 4 | 2026-07-04T00:03:25 | 10.17489 | v2 |
| 5 | 2026-07-04T00:09:24 | 10.166045 | v2 |
| 6 | 2026-07-04T00:13:58 | 10.157212 | v2 |
| 7 | 2026-07-04T00:19:34 | 10.019463 | v2 |
| 8 | 2026-07-04T12:18:09 | 9.75707 | v2 |

**軌跡敘述與突破點**:

- **#1 → #2**:加入 20 個日曆/週期/類別特徵並改採 log1p 目標,但同時把 CV 從隨機 5-fold 換成時間序 TimeSeriesSplit。分數表面上大幅變差(5.31891 → 10.175397),一度像是退步。
- **#2 → #3(Reflexion 診斷,本場的關鍵轉折)**:為釐清變差來自「特徵」還是「CV 方案」,以完全相同的 20 個特徵與三模型、改回與基線同型態的隨機 KFold 重測,得到 **4.281421**——優於基線的 5.31891。結論:特徵工程確實有效(同方案下最佳分數出現在實驗 #3),實驗 #2 的分數純粹反映時間序驗證是更誠實、更困難的量尺;預期實際 LB 落在兩個數字之間、偏向時間序估計。
- **#4(Phase B round 1,比例分解——最高期望值的假說,實測失敗)**:EDA 顯示店/產品年度佔比極穩、且僅用三個類別欄的加法 OLS 就能解釋 log 目標絕大部分變異,理應適合「總量 × 佔比」分解;但結構式成員 solo OOF 14.75038,權重搜尋歸零。追加 oracle 診斷(餵入真實每日總量)顯示佔比部分與 GBDT 相當而非更好——GBDT 的原生類別分裂已把佔比結構學走,真正瓶頸是「總量一年外推」(5 年序列含 2020 下滑、2021 回升,無可靠趨勢),而該瓶頸對所有方法一視同仁。與 s3e20(結構信號完勝 GBDT)對照,界定了該模式的邊界條件:結構要贏,目標須跨年近恆定。
- **#5、#6(round 2–3,seed bagging)**:轉用跨賽驗證過的低成本配方,兩輪皆有小幅實質增益(10.17489 → 10.166045 → 10.157212),與過往競賽「遞減但為正」的型態一致。
- **#7(round 4,Optuna fold-5 代理調參,本場最大單輪增益)**:TimeSeriesSplit 折不可互換,故以最後一折(訓練窗最大、最接近 2022 外推情境)取代慣用的 fold-0 作為調參代理目標;調出的 LGB 更淺更強正則(num_leaves 63 → 20),再次印證「外推情境獎勵正則化」。調參版加入池(不替換)後拿下 0.5 權重,blend 10.157212 → **10.019463**。註記:fold 5 兼作調參目標與 1/5 的 OOF,此數字含部分樂觀成分。
- **教訓**:不同 CV 方案的分數不可直接比較;v2 schema 的 `cv.strategy` 欄位讓每筆實驗的方案都可回溯,本場正是靠它避免誤判。Phase B 全程鎖定同一 TimeSeriesSplit 折,確保 keep/reject 決策在誠實量尺上進行。
- **#7 → #8(Phase G-1b,本次更新新增)**:實驗 #8 不是線性迭代的延續回合,而是 Phase D-5
  (2026-07-04)以 harness v2 執行的**樹搜尋(tree-search)**結果——來源 `experiments_tree.json`
  的 node #17(22 個評估節點/13 solo+9 blend,wall 321.0s)。樹搜尋在 exp #7 的相同
  TimeSeriesSplit 折與 20-特徵集上另闢節點空間,找到 SMAPE 更低的 10-way blend,分數由
  10.019463 降至 9.75707(相對改善見上方第 2.5 節程式區塊,約 -2.6%,本輪相對margin 為所有已跑
  樹搜尋競賽中最大)。決定性槓桿為(1)對 Optuna 調參 LGB 做 seed bagging(STATUS.md 原列
  為「未試」)、(2)修正 TimeSeriesSplit OOF 系統性偏低的全域 ×1.02 乘數(詳見第 2.2c 節)。
  **誠實 CV-only 警語**(逐字):「like the linear 10.01946, the 9.75707 carries fold-5
  double-dip optimism, PLUS the scale parameter and the seed selection are OOF-fitted.
  All are 1-to-few-parameter fits on 114k rows (low overfit risk individually), but the
  true expected 2022 SMAPE is best read as "meaningfully below 10.02", not literally
  9.76.」此結果為 OOF-only 搜尋產物,**未提交至 Kaggle**;不可與 exp #7 實際提交的
  submission 檔案混淆(見第 2.4 節)。完整節點鏈見
  `competitions/playground-series-s3e19/STATUS.md`〈Appendix: Phase D-5 tree-search v2
  sweep〉。

**無法解析之紀錄**:無(facts.unparsed 為空)。

## 4. 重現指令

```bash
# 環境:Linux + uv(Python 由 uv 管理);執行目錄:專案根目錄
cd /home/tjyen/ai_agents/kaggle

# 0) 資料下載(需 KAGGLE_API_TOKEN;本場資料已在 data/ 內,可略過)
export KAGGLE_API_TOKEN=$(cat ~/.kaggle/kaggle_api_token.txt | tr -d '[:space:]')
uv run kaggle competitions download -c playground-series-s3e19 \
    -p competitions/playground-series-s3e19/data
unzip -o competitions/playground-series-s3e19/data/playground-series-s3e19.zip \
    -d competitions/playground-series-s3e19/data

# 1) EDA(輸出資料概況與 CV 方案決策依據)
uv run python3 competitions/playground-series-s3e19/scripts/eda.py

# 2) 特徵工程(產出 data/train_processed.csv、data/test_processed.csv)
uv run python3 competitions/playground-series-s3e19/scripts/features.py

# 3) 主訓練:TimeSeriesSplit 5-fold CV + OOF 權重搜尋 + 全量重訓 + 產出 submission
#    (submission 寫入 competitions/playground-series-s3e19/submissions/,並記錄 experiments.json)
uv run python3 competitions/playground-series-s3e19/scripts/train.py

# 4) (選用)Reflexion 診斷:同特徵改用 shuffled KFold,對照基線
uv run python3 competitions/playground-series-s3e19/scripts/diagnostic_kfold.py

# 5) Phase B 迭代(依序;#7 產出現行最佳 submission)
uv run python3 competitions/playground-series-s3e19/scripts/train_ratio.py     # exp #4(RD 棄用)
uv run python3 competitions/playground-series-s3e19/scripts/train_seedbag.py   # exp #5
uv run python3 competitions/playground-series-s3e19/scripts/train_seedbag2.py  # exp #6
uv run python3 competitions/playground-series-s3e19/scripts/train_optuna.py    # exp #7(現行最佳提交對應版本)

# 5b) Phase D-5 樹搜尋 v2(best under TimeSeriesSplit,exp #8)— resumable;軌跡存於 experiments_tree.json
uv run python3 tree_search/run_s3e19.py

# 6) (選用)提交至 Kaggle
uv run kaggle competitions submit -c playground-series-s3e19 \
    -f competitions/playground-series-s3e19/submissions/sub_6way_optuna_blend_10.01946_20260704_002039.csv \
    -m "6-way blend: LGB seeds + CAT seeds + Optuna fold-5-proxy tuned LGB, time-based CV"
```
