# 競賽分析報告:playground-series-s3e11

> 產生方式:kaggle-report skill(數字來自 facts.json,敘述由 agent 撰寫)
> 素材等級:full | 產生日期:2026-07-03

## 1. 競賽目的

**What**:本場競賽(Kaggle Playground Series, Season 3 Episode 11)要求預測媒體行銷活動的
成本(`cost`)。輸入為門市與商品層級的屬性(門市面積、附設設施旗標、商品重量/包裝、
顧客家庭屬性等),輸出為連續數值的活動成本,是一個標準的表格型迴歸問題。此資料集亦為
Aygun et al.(Nature 2026)Kaggle Playground 基準之一(來源:`competition.notes`)。

**Why**:評估指標為 **RMSLE(Root Mean Squared Logarithmic Error,minimize)**。成本類目標
通常關心「相對誤差」而非絕對誤差——把一筆高成本活動預測差一截,與把一筆低成本活動預測差
同樣金額,對業務的意義不同;RMSLE 在對數空間計算平方誤差,懲罰的是比例偏差,並且天然
壓抑大數值樣本對損失的支配,是成本/金額類目標的合理選擇。實務上這也決定了訓練策略:
對 log1p(cost) 以 RMSE objective 訓練,即可直接優化 RMSLE(見節 4)。

| 項目 | 值 |
|------|-----|
| 競賽 | playground-series-s3e11 |
| 問題型別 | regression |
| 評估指標 | rmsle(minimize) |
| 目標欄位 | cost |

## 2. 資料規格

- 訓練/測試列數:facts.json 未記載,**無紀錄**(實際形狀見 `scripts/eda.py` 執行輸出)。
- 原始特徵 15 欄(來源:`experiments[0].n_features`、`experiments[1].features`):門市屬性
  (`store_sqft` 與 coffee_bar/video_store/salad_bar/prepared_food/florist 五個附設設施旗標)、
  商品屬性(`gross_weight`、`units_per_case`、`recyclable_package`、`low_fat`、銷售額/銷量)、
  顧客家庭屬性(`total_children`、`num_children_at_home`、`avg_cars_at home(approx).1`)。
  全部為數值型,但多數實為低基數「類別型偽裝」欄位(EDA 發現,見下)。
- 特別規則:不可使用外部資料、不可使用預訓練模型、不可存取網路;每日提交上限 5 次
  (來源:`competition.special_rules`)。
- **EDA 摘要**(`scripts/eda.py`,質性描述;數值細節見腳本輸出):目標 `cost` 分布近乎對稱、
  **並非**右偏,log1p 轉換後反而輕微左偏——採用 log 目標的理由是「直接優化 RMSLE」而非
  矯正偏態;無缺失值、無重複列、train/test id 不重疊、各欄位 train/test 平均一致(無明顯
  covariate shift);所有原始特徵與目標的單變量相關皆極弱(低訊號資料集);最強訊號來自
  `store_sqft` 加五個設施旗標組成的「store profile」組合——其組平均對目標的解釋力遠高於
  任何單一欄位,是特徵工程主攻方向(需 out-of-fold 編碼防洩漏)。

## 3. 模型規格

本場共三筆實驗,皆為 LightGBM + XGBoost + CatBoost 三模型加權集成,差別在特徵集與訓練方法:

**實驗 1(`generic_batch`,通用批次基線;15 原始特徵)— base models:**

| Model | OOF RMSLE |
|-------|-----------|
| LGB | 0.29731 |
| XGB | 0.29877 |
| CAT | 0.2977 |

Ensemble 權重(來源:`experiments[0].ensemble.weights`):LGB 0.7 / XGB 0.0 / CAT 0.3,
Ensemble 分數(來源:`experiments[0].ensemble.score`):**0.29723**。

**實驗 2(`v2`,skill 管線 base 模式;15 原始特徵 + log1p 目標 + early stopping)— base models:**

| Model | OOF RMSLE | time_s |
|-------|-----------|--------|
| LGB | 0.29765 | 94.4 |
| XGB | 0.29827 | 25.7 |
| CAT | 0.29715 | 49.2 |

Ensemble 權重(來源:`experiments[1].ensemble.weights`):LGB 0.2 / XGB 0.0 / CAT 0.8,
集成分數(來源:`experiments[1].score`):**0.2971**。

**實驗 3(`v2`,skill 管線 engineered 模式;21 特徵,facts.best)— base models:**

| Model | OOF RMSLE | time_s |
|-------|-----------|--------|
| LGB | 0.29661 | 70.4 |
| XGB | 0.29713 | 24.5 |
| CAT | 0.29618 | 58.3 |

Ensemble 權重(來源:`experiments[2].ensemble.weights`):LGB 0.2 / XGB 0.0 / CAT 0.8,
集成分數(來源:`experiments[2].score`,即 `facts.best.score`):**0.296143**。

**特徵差異**:實驗 3 在 15 個原始欄位之上加入 6 個工程特徵(來源:`experiments[2].features`,
n_features = 21):`amenity_count`(五個設施旗標加總)、`weight_per_case`、`sales_ratio`、
`children_away`、`cars_per_child`,以及關鍵的 **`store_te`** —— store profile(`store_sqft` +
五個設施旗標的組合鍵)的 K-fold target encoding,於每個 fold 內僅以訓練折資料計算組平均、
套用到驗證折,測試集端取五個 fold 編碼的平均(來源:`experiments[2].notes`),避免目標洩漏。

**選型理由**:三個梯度提升樹模型在表格資料上穩健且歸納偏誤互異,以 OOF 權重搜尋做加權集成
可降低單一模型方差。兩次 skill 管線 run 的權重搜尋皆收斂到 CatBoost 為主(0.8)、XGBoost
權重 0 的組合,顯示 CatBoost 在此低訊號資料集上最強、XGB 與其他兩者高度冗餘。

## 4. 訓練規格

| 實驗 | CV scheme | n_splits | seed |
|------|-----------|----------|------|
| 1 | 5fold | 5 | 無紀錄 |
| 2 | 5fold_kfold_shuffle | 5(自 strategy 名稱) | 42 |
| 3 | 5fold_kfold_shuffle | 5(自 strategy 名稱) | 42 |

(來源:`experiments[].cv`。實驗 1 的 seed 無對應欄位,寫「無紀錄」。)

**Objective**:實驗 2、3 對 **log1p(cost)** 以 RMSE objective 訓練(LGB `regression`、
XGB `reg:squarederror`、CAT `RMSE`),因為 RMSLE 即「log1p 空間的 RMSE」,如此 objective
與競賽指標完全一致;預測時經 expm1 逆轉換並 clip 至非負(來源:`experiments[1].notes`、
`experiments[2].notes`)。各模型均使用 early stopping。其餘超參數 facts.json 未記錄欄位,
無紀錄(可直接查 `scripts/train.py`)。

**為何用此 CV**:EDA 未發現時間或群組結構;store profile 每種組合重複數千列,隨機切分不會
讓任何 fold 缺少某個 profile,故採標準 5-fold KFold(shuffle, seed=42)。實驗 2 與 3 使用同一
組 fold 切分,分數可直接比較。target encoding 在 fold 迴圈內計算,與 CV 方案一致、無洩漏。

## 5. 推論程序

`facts.best`(實驗 3)之 `postprocess` 欄位未記錄 → **無後處理紀錄**;惟 `experiments[2].notes`
記載推論流程本身包含「expm1 逆轉換 + clip 至非負」,這是 log 目標訓練的必要配套步驟而非
額外後處理。測試集的 `store_te` 特徵取五個 fold 編碼平均、未見過的 profile 以訓練折全域
平均代入(fallback)。

Submission:`sub_engineered_0.29614_20260703_192819.csv`(來源:`experiments[2].submission`),
格式為兩欄 —— `id`(`competition.id_column`)與 `cost`(`competition.target_column`),
每列對應一筆測試樣本的成本預測值。

**注意**:本場為週末自主批次執行,三筆實驗皆**未提交** Kaggle 排行榜(`facts.missing` 含
`leaderboard`),無 Public/Private LB 分數。

## 6. 評估指標

**指標定義**:RMSLE = sqrt(mean((log1p(pred) − log1p(actual))²)),衡量對數空間的均方根誤差,
懲罰相對(比例)偏差。

| 項目 | OOF RMSLE |
|------|-----------|
| 實驗 1 Ensemble(generic 基線) | 0.29723 |
| 實驗 2 Ensemble(skill base) | 0.2971 |
| 實驗 3 Ensemble(skill engineered,**facts.best**) | **0.296143** |
| Public LB | 無紀錄(未提交) |
| Private LB | 無紀錄(未提交) |

**相對基線改善**(衍生算式,依 Hard Rule 置於 code block):

```
實驗 3 − 實驗 1:0.29723 − 0.296143 = 0.001087   (RMSLE 下降,~0.37% 相對改善)
實驗 3 − 實驗 2:0.29710 − 0.296143 = 0.000957   (特徵工程貢獻,佔改善絕大部分)
實驗 2 − 實驗 1:0.29723 − 0.29710  = 0.000130   (方法論差異:log1p 目標 + early stopping)
```

CV↔LB gap:無排行榜紀錄,無法計算。

## 7. 實驗軌跡

| experiment_id | timestamp | score | source_format |
|---------------|-----------|-------|----------------|
| 1 | 2026-07-03T12:10:20 | 0.29723 | generic_batch |
| 2 | 2026-07-03T19:25:07 | 0.2971 | v2 |
| 3 | 2026-07-03T19:28:19 | 0.296143 | v2 |

(來源:`facts.trajectory`。三筆實驗中最佳分數出現於第 3 筆。)

**突破點**:主要改善發生在第 2 → 第 3 筆之間。第 2 筆只是把通用基線的方法論換成「log1p
目標 + early stopping + 權重搜尋」,分數僅微幅改善;第 3 筆加入 6 個工程特徵——尤其是
store profile 的 K-fold target encoding(`store_te`)——帶來本場絕大部分的增益(見節 6 的
衍生算式)。這與 EDA 的判讀一致:單欄位訊號極弱,但 store profile 組合的組平均是資料中
最強的可用結構;在低訊號 playground 資料集上,能把這類「組合層級」訊號餵給模型的特徵
工程,比模型/超參數調整更有價值。

`facts.unparsed` 為空陣列,無法解析之紀錄:無。

## 8. 重現指令

```bash
cd /home/tjyen/ai_agents/kaggle

# 0) 環境:依 pyproject.toml 以 uv 管理;資料已在 competitions/playground-series-s3e11/data/
#    (train.csv / test.csv / sample_submission.csv;如需重新下載:)
# export KAGGLE_API_TOKEN=$(cat ~/.kaggle/kaggle_api_token.txt | tr -d '[:space:]')
# uv run kaggle competitions download -c playground-series-s3e11 -p competitions/playground-series-s3e11/data

# 1) EDA(全量資料、僅快速統計)
uv run python3 competitions/playground-series-s3e11/scripts/eda.py

# 2) 實驗 2:skill base 模式(15 原始特徵,方法論對照)
uv run python3 competitions/playground-series-s3e11/scripts/train.py base

# 3) 實驗 3:skill engineered 模式(21 特徵,含 store_te target encoding;最佳)
uv run python3 competitions/playground-series-s3e11/scripts/train.py engineered
#    → submissions/sub_engineered_*.csv,並自動以 log_experiment_v2 寫入 experiments.json

# 4) (未執行)提交排行榜:
# uv run kaggle competitions submit -c playground-series-s3e11 \
#     -f competitions/playground-series-s3e11/submissions/sub_engineered_0.29614_20260703_192819.csv \
#     -m "engineered blend"
```

執行目錄為專案根目錄 `/home/tjyen/ai_agents/kaggle`;所有 Python 執行皆透過 `uv run`
以確保套件環境一致。特徵工程程式碼在 `scripts/features.py`(被 `train.py` 匯入),
target encoding 因需 fold 內計算,實作於 `train.py`。
