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

本場共八筆實驗:前三筆(Phase A)為 LightGBM + XGBoost + CatBoost 三模型加權集成,差別在
特徵集與訓練方法;後五筆(Phase B 自我改進迭代 R1–R5)在最佳特徵集上調整模型池組成
(刪除 XGBoost、Optuna 調參 CatBoost、seed bagging、特徵消融)。

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

**實驗 3(`v2`,skill 管線 engineered 模式;21 特徵,Phase A 最佳)— base models:**

| Model | OOF RMSLE | time_s |
|-------|-----------|--------|
| LGB | 0.29661 | 70.4 |
| XGB | 0.29713 | 24.5 |
| CAT | 0.29618 | 58.3 |

Ensemble 權重(來源:`experiments[2].ensemble.weights`):LGB 0.2 / XGB 0.0 / CAT 0.8,
集成分數(來源:`experiments[2].score`):**0.296143**。

**特徵差異**:實驗 3 在 15 個原始欄位之上加入 6 個工程特徵(來源:`experiments[2].features`,
n_features = 21):`amenity_count`(五個設施旗標加總)、`weight_per_case`、`sales_ratio`、
`children_away`、`cars_per_child`,以及關鍵的 **`store_te`** —— store profile(`store_sqft` +
五個設施旗標的組合鍵)的 K-fold target encoding,於每個 fold 內僅以訓練折資料計算組平均、
套用到驗證折,測試集端取五個 fold 編碼的平均(來源:`experiments[2].notes`),避免目標洩漏。

**選型理由**:三個梯度提升樹模型在表格資料上穩健且歸納偏誤互異,以 OOF 權重搜尋做加權集成
可降低單一模型方差。兩次 skill 管線 run 的權重搜尋皆收斂到 CatBoost 為主(0.8)、XGBoost
權重 0 的組合,顯示 CatBoost 在此低訊號資料集上最強、XGB 與其他兩者高度冗餘。

**Phase B 自我改進迭代(實驗 4–8,同一 21 特徵集,除實驗 7)**:

| 實驗 | 改動(每輪一項) | Ensemble 分數 |
|------|------------------|----------------|
| 4(R1) | 刪除 XGBoost(前兩輪權重皆 0),LGB+CAT 雙模型池 | 0.296143 |
| 5(R2) | Optuna(TPE 40 trials,fold-0 proxy)調參 CatBoost,調參版**加入**池(不替換) | 0.295781 |
| 6(R3) | 調參 CatBoost 以 random_seed=2024 重訓,加為第 4 成員(seed bagging) | 0.295715 |
| 7(R4) | 加 3 個 per-store_combo 特徵均值(24 特徵)——**退步,棄用** | 0.2962 |
| 8(R5) | 調參 CatBoost 第三個 seed(7),5-way 池 | **0.295648** |

(分數來源:`experiments[3..7].score`;facts.best = 實驗 8。)

實驗 5 的 Optuna 最佳參數(來源:`experiments[4].notes`,原值照錄):depth=10、
learning_rate=0.08243442179862394、l2_leaf_reg=5.4822780685788235、min_data_in_leaf=33、
random_strength=0.09160286047373326;搜尋耗時 319.3s(40 trials,timeout guard 480s)。

**實驗 8(facts.best,R5 5-way 池)— base models 與權重:**

| Member | OOF RMSLE | time_s | 權重 |
|--------|-----------|--------|------|
| LGB | 0.29661 | 75.5 | 0.0 |
| CAT_orig | 0.29618 | 58.2 | 0.0 |
| CAT_tuned | 0.29579 | 31.4 | 0.4 |
| CAT_tuned_seed2024 | 0.29591 | 34.0 | 0.2 |
| CAT_tuned_seed7 | 0.29578 | 31.9 | 0.4 |

(來源:`experiments[7].base_models`、`experiments[7].ensemble.weights`。)
權重搜尋把全部權重給了調參 CatBoost 家族——調參後單模(0.29579)已勝 Phase A 三模型
blend(0.296143),LGB 與原參數 CatBoost 淪為冗餘。

## 4. 訓練規格

| 實驗 | CV scheme | n_splits | seed |
|------|-----------|----------|------|
| 1 | 5fold | 5 | 無紀錄 |
| 2–8 | 5fold_kfold_shuffle | 5(自 strategy 名稱) | 42 |

(來源:`experiments[].cv`。實驗 1 的 seed 無對應欄位,寫「無紀錄」。實驗 2–8 全部使用
同一 fold 切分,分數可直接比較。)

**Objective**:實驗 2、3 對 **log1p(cost)** 以 RMSE objective 訓練(LGB `regression`、
XGB `reg:squarederror`、CAT `RMSE`),因為 RMSLE 即「log1p 空間的 RMSE」,如此 objective
與競賽指標完全一致;預測時經 expm1 逆轉換並 clip 至非負(來源:`experiments[1].notes`、
`experiments[2].notes`)。各模型均使用 early stopping。其餘超參數 facts.json 未記錄欄位,
無紀錄(可直接查 `scripts/train.py`)。

**為何用此 CV**:EDA 未發現時間或群組結構;store profile 每種組合重複數千列,隨機切分不會
讓任何 fold 缺少某個 profile,故採標準 5-fold KFold(shuffle, seed=42)。實驗 2 與 3 使用同一
組 fold 切分,分數可直接比較。target encoding 在 fold 迴圈內計算,與 CV 方案一致、無洩漏。

## 5. 推論程序

`facts.best`(實驗 8)之 `postprocess` 欄位未記錄 → **無後處理紀錄**;惟各實驗 notes
記載推論流程本身包含「expm1 逆轉換 + clip 至非負」,這是 log 目標訓練的必要配套步驟而非
額外後處理。測試集的 `store_te` 特徵取五個 fold 編碼平均、未見過的 profile 以訓練折全域
平均代入(fallback)。最終預測為 5 個成員測試預測(各自已是 5-fold 平均)在 log 空間的
加權和,再做 expm1 + clip。

Submission:`sub_r5_seedbag3_0.29565_20260703_223829.csv`(來源:`experiments[7].submission`,
即 `facts.best.submission`),格式為兩欄 —— `id`(`competition.id_column`)與 `cost`
(`competition.target_column`),每列對應一筆測試樣本的成本預測值。

**注意**:本場為週末自主批次執行,八筆實驗皆**未提交** Kaggle 排行榜(`facts.missing` 含
`leaderboard`),無 Public/Private LB 分數。

## 6. 評估指標

**指標定義**:RMSLE = sqrt(mean((log1p(pred) − log1p(actual))²)),衡量對數空間的均方根誤差,
懲罰相對(比例)偏差。

| 項目 | OOF RMSLE |
|------|-----------|
| 實驗 1 Ensemble(generic 基線) | 0.29723 |
| 實驗 2 Ensemble(skill base) | 0.2971 |
| 實驗 3 Ensemble(skill engineered,Phase A 最佳) | 0.296143 |
| 實驗 5 Ensemble(R2 Optuna CatBoost) | 0.295781 |
| 實驗 6 Ensemble(R3 seed bagging) | 0.295715 |
| 實驗 8 Ensemble(R5 seed bagging ×3,**facts.best**) | **0.295648** |
| Public LB | 無紀錄(未提交) |
| Private LB | 無紀錄(未提交) |

**相對基線改善**(衍生算式,依 Hard Rule 置於 code block):

```
Phase A:
實驗 3 − 實驗 1:0.29723 − 0.296143 = 0.001087   (RMSLE 下降,~0.37% 相對改善)
實驗 3 − 實驗 2:0.29710 − 0.296143 = 0.000957   (特徵工程貢獻,佔改善絕大部分)
實驗 2 − 實驗 1:0.29723 − 0.29710  = 0.000130   (方法論差異:log1p 目標 + early stopping)

Phase B(自我改進迭代):
實驗 5 − 實驗 3:0.296143 − 0.295781 = 0.000362  (Optuna 調參 CatBoost,Phase B 最大單項)
實驗 6 − 實驗 5:0.295781 − 0.295715 = 0.000066  (seed bagging 第 2 個 seed)
實驗 8 − 實驗 6:0.295715 − 0.295648 = 0.000067  (seed bagging 第 3 個 seed)
實驗 7 − 實驗 6:0.2962   − 0.295715 = 0.000485  (退步:per-combo 特徵均值,棄用)
實驗 8 − 實驗 3:0.296143 − 0.295648 = 0.000495  (Phase B 總增益,~0.17% 相對改善)
```

CV↔LB gap:無排行榜紀錄,無法計算。

## 7. 實驗軌跡

| experiment_id | timestamp | score | source_format |
|---------------|-----------|-------|----------------|
| 1 | 2026-07-03T12:10:20 | 0.29723 | generic_batch |
| 2 | 2026-07-03T19:25:07 | 0.2971 | v2 |
| 3 | 2026-07-03T19:28:19 | 0.296143 | v2 |
| 4 | 2026-07-03T22:23:54 | 0.296143 | v2 |
| 5 | 2026-07-03T22:31:01 | 0.295781 | v2 |
| 6 | 2026-07-03T22:32:05 | 0.295715 | v2 |
| 7 | 2026-07-03T22:36:33 | 0.2962 | v2 |
| 8 | 2026-07-03T22:38:30 | 0.295648 | v2 |

(來源:`facts.trajectory`。最佳分數出現於第 8 筆,即 `facts.best`。)

**突破點(Phase A)**:主要改善發生在第 2 → 第 3 筆之間。第 2 筆只是把通用基線的方法論換成
「log1p 目標 + early stopping + 權重搜尋」,分數僅微幅改善;第 3 筆加入 6 個工程特徵——尤其是
store profile 的 K-fold target encoding(`store_te`)——帶來 Phase A 絕大部分的增益(見節 6 的
衍生算式)。這與 EDA 的判讀一致:單欄位訊號極弱,但 store profile 組合的組平均是資料中
最強的可用結構;在低訊號 playground 資料集上,能把這類「組合層級」訊號餵給模型的特徵
工程,比模型/超參數調整更有價值。

**突破點(Phase B)**:第 4 筆(R1)驗證刪除兩輪零權重的 XGBoost 不損分數(與第 3 筆分數
完全相同);第 5 筆(R2)Optuna fold-0 proxy 調參 CatBoost 是 Phase B 最大單項增益,且調參版
是「加入」模型池而非替換原成員(跨競賽驗證過的配方);第 6、8 筆(R3/R5)兩次 seed bagging
各貢獻噪音級以上的小增益;第 7 筆(R4)在 store_te 之上再加 per-combo 特徵均值反而全面退步
——樹模型已能從 store_te 與原始欄位取得該組合的訊號,額外的組彙總只添冗餘——故棄用,
最終最佳(第 8 筆)回到 21 特徵集。迭代依「連續退步/增益縮至噪音級即停」原則於 R5 後收手。

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

# 3) 實驗 3:skill engineered 模式(21 特徵,含 store_te target encoding;Phase A 最佳)
uv run python3 competitions/playground-series-s3e11/scripts/train.py engineered
#    → submissions/sub_engineered_*.csv,並自動以 log_experiment_v2 寫入 experiments.json

# 4) Phase B 自我改進迭代(實驗 4–8;checkpointed,cache 命中會跳過已訓練成員)
uv run python3 competitions/playground-series-s3e11/scripts/iterate2.py r1    # 實驗 4
uv run python3 competitions/playground-series-s3e11/scripts/iterate2.py tune  # Optuna(參數存 scripts/cache/)
uv run python3 competitions/playground-series-s3e11/scripts/iterate2.py r2    # 實驗 5
uv run python3 competitions/playground-series-s3e11/scripts/iterate2.py r3    # 實驗 6
uv run python3 competitions/playground-series-s3e11/scripts/iterate2.py r4    # 實驗 7(退步,棄用)
uv run python3 competitions/playground-series-s3e11/scripts/iterate2.py r5    # 實驗 8(最佳)

# 5) (未執行)提交排行榜:
# uv run kaggle competitions submit -c playground-series-s3e11 \
#     -f competitions/playground-series-s3e11/submissions/sub_r5_seedbag3_0.29565_20260703_223829.csv \
#     -m "r5 seed-bagged tuned CatBoost blend"
```

執行目錄為專案根目錄 `/home/tjyen/ai_agents/kaggle`;所有 Python 執行皆透過 `uv run`
以確保套件環境一致。特徵工程程式碼在 `scripts/features.py`(被 `train.py` 匯入),
target encoding 因需 fold 內計算,實作於 `train.py`。
