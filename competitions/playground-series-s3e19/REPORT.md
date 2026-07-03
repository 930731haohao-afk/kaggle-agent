# 競賽分析報告:playground-series-s3e19

> 產生方式:kaggle-report skill(數字來自 facts.json,敘述由 agent 撰寫)
> 素材等級:full | 產生日期:2026-07-03

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

## 2. 資料規格

- **檔案**:train.csv、test.csv、sample_submission.csv。
- **列數**:facts.json 無紀錄(依 STATUS.md 敘述脈絡:訓練集為 2017-01-01 至 2021-12-31 的每日資料,測試集為 2022 全年;精確列數此處不引數字)。
- **欄位**:原始特徵為 date 加三個類別欄(country、store、product),目標為整數 `num_sold`;無缺失值、無重複列。基線實驗(#1)使用 8 個通用特徵,本場工程後為 20 個特徵(清單見第 4 節)。
- **序列結構**(EDA `scripts/eda.py` 執行結果,定性描述):5 國 × 3 店 × 5 產品共 75 條完整日序列;店面與產品占年度總量的比例逐年幾乎恆定,國家占比則逐年緩慢漂移;週末(尤其週日)與 12 月/1 月有明顯季節性抬升,1 月 1 日有單日尖峰;目標右偏、log1p 後接近對稱;年度總量非單調(2020 年下滑、2021 年回升)。

**特別規則**:

| 規則 | 值 |
|------|-----|
| 外部資料 | 不允許 |
| 預訓練模型 | 不允許 |
| 網路存取 | 不允許 |
| 每日提交上限 | 5 |

## 3. 模型規格

本場共三筆實驗;**facts.json 的 best(分數最佳)為實驗 #3(診斷用 KFold 重跑),而實際產生提交檔的是實驗 #2(時間序 CV)**。兩者特徵與模型完全相同,僅 CV 方案不同,分數不可跨方案直接比較(詳見第 4、7 節)。

### 實驗 #2 — 提交所用(TimeSeriesSplit 5-fold)

| Base model | OOF SMAPE |
|------------|-----------|
| LightGBM | 10.17758 |
| XGBoost | 10.46 |
| CatBoost | 10.73064 |

Ensemble:OOF 加權平均網格搜尋(grid_step 0.1),最佳權重 **LGB 0.9 / XGB 0.0 / CAT 0.1**,blend SMAPE **10.175397**。

### 實驗 #3 — facts.best(診斷用,shuffled KFold,未產生提交檔)

| Base model | OOF SMAPE |
|------------|-----------|
| LightGBM | 4.31618 |
| XGBoost | 4.33201 |
| CatBoost | 4.3239 |

Ensemble:同樣的權重網格搜尋,最佳權重 **LGB 0.4 / XGB 0.2 / CAT 0.4**,blend SMAPE **4.281421**。

**選型理由**:LGB/XGB/CatBoost 三種梯度提升樹是中型表格資料的標準組合,皆原生支援類別特徵並可用早停控制訓練成本。外推情境(實驗 #2)下 LGB 明顯領先、XGB 權重被搜到 0——與基線實驗 #1(權重 LGB 0.7 / XGB 0.0 / CAT 0.3)的型態一致;內插情境(實驗 #3)三模型幾乎同分,blend 分散權重。最終提交採用實驗 #2 的權重(0.9/0.0/0.1),因為真實測試集(2022 年)正是外推情境。

## 4. 訓練規格

| 實驗 | CV 方案 | n_splits | seed |
|------|---------|----------|------|
| #2(提交) | TimeSeriesSplit-5fold-on-unique-dates | 5 | 無紀錄 |
| #3(診斷) | KFold-5fold-shuffled-seed42 | 5 | 無紀錄(方案字串內含 seed42) |

**Objective 與關鍵超參**(實驗 #2 紀錄):目標經 log1p 轉換後以迴歸目標訓練,預測以 expm1 反轉後計算 SMAPE。

| 模型 | 關鍵超參 |
|------|----------|
| LightGBM | n_estimators 2000、learning_rate 0.03、num_leaves 63、min_child_samples 20、subsample 0.9、colsample_bytree 0.8、reg_lambda 1.0 |
| XGBoost | n_estimators 2000、learning_rate 0.03、max_depth 7、subsample 0.9、colsample_bytree 0.8、reg_lambda 1.0 |
| CatBoost | iterations 2000、learning_rate 0.05、depth 8、l2_leaf_reg 3.0 |

三模型皆對各 fold 的驗證區塊做早停。

**為何用此 CV**:測試期(2022)完全落在訓練期(2017–2021)之後、零重疊,模型實際面對的是「外推到未見年份」;隨機 KFold 會讓驗證日夾在已見的季節循環之間,對此任務系統性樂觀。因此主實驗(#2)採 TimeSeriesSplit 於排序後的唯一日期上切分,每折以擴張窗訓練、驗證嚴格較晚的日期區塊,忠實模擬 train→test 的時間缺口。實驗 #3 刻意改用與基線相同型態的隨機 KFold,只作為「特徵集是否有效」的對照診斷,不用於選模。

**特徵(20 個)**:year、month、day、dow、day_of_year、weekofyear、quarter、is_weekend、is_month_start、is_month_end、is_new_year、month_sin、month_cos、dow_sin、dow_cos、doy_sin、doy_cos、country_cat、store_cat、product_cat。

## 5. 推論程序

- **最終模型**:實驗 #2 的三個模型以相同超參在 100% 訓練資料上重訓,再按權重 LGB 0.9 / XGB 0.0 / CAT 0.1 加權平均。
- **後處理**:facts.json 的 postprocess 欄位無紀錄;依實驗 notes,預測值由 log1p 空間經 expm1 反轉回原始銷量尺度(訓練腳本另將負值截斷為 0)。
- **Submission**:`sub_lgb_xgb_cat_blend_10.17540_20260703_194908.csv`,格式為兩欄(`id`, `num_sold`),與 sample_submission.csv 之形狀、id 順序逐一驗證通過。本場為無人值守批次執行,**未**實際提交至 Kaggle。

## 6. 評估指標

**定義**:SMAPE = 100% × 平均(|真值 − 預測| / ((|真值| + |預測|)/2)),對稱化的相對誤差,越低越好。

### 分數總表

| 實驗 | CV 方案 | LGB | XGB | CAT | Blend |
|------|---------|------|------|------|-------|
| #1 基線 | 5fold | 5.45633 | 7.41858 | 6.09778 | **5.31891** |
| #2 提交 | TimeSeriesSplit 5-fold | 10.17758 | 10.46 | 10.73064 | **10.175397** |
| #3 診斷 | shuffled KFold 5-fold | 4.31618 | 4.33201 | 4.3239 | **4.281421** |

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

此差距全部來自「外推 vs 內插」的驗證難度差異,不是模型退步(見第 7 節)。

## 7. 實驗軌跡

| id | timestamp | score (SMAPE) | source_format |
|----|-----------|---------------|---------------|
| 1 | 2026-07-03T12:14:06 | 5.31891 | generic_batch |
| 2 | 2026-07-03T19:49:08 | 10.175397 | v2 |
| 3 | 2026-07-03T19:53:27 | 4.281421 | v2 |

**軌跡敘述與突破點**:

- **#1 → #2**:加入 20 個日曆/週期/類別特徵並改採 log1p 目標,但同時把 CV 從隨機 5-fold 換成時間序 TimeSeriesSplit。分數表面上大幅變差(5.31891 → 10.175397),一度像是退步。
- **#2 → #3(Reflexion 診斷,本場的關鍵轉折)**:為釐清變差來自「特徵」還是「CV 方案」,以完全相同的 20 個特徵與三模型、改回與基線同型態的隨機 KFold 重測,得到 **4.281421**——優於基線的 5.31891。結論:特徵工程確實有效(同方案下最佳分數出現在實驗 #3),實驗 #2 的分數純粹反映時間序驗證是更誠實、更困難的量尺;預期實際 LB 落在兩個數字之間、偏向時間序估計。
- **教訓**:不同 CV 方案的分數不可直接比較;v2 schema 的 `cv.strategy` 欄位讓每筆實驗的方案都可回溯,本場正是靠它避免誤判。

**無法解析之紀錄**:無(facts.unparsed 為空)。

## 8. 重現指令

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

# 5) (選用)提交至 Kaggle
uv run kaggle competitions submit -c playground-series-s3e19 \
    -f competitions/playground-series-s3e19/submissions/sub_lgb_xgb_cat_blend_10.17540_20260703_194908.csv \
    -m "LGB+XGB+CAT blend, time-based CV"
```
