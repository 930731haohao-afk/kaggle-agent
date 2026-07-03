# 競賽分析報告:playground-series-s3e9

> 產生方式:kaggle-report skill(數字來自 facts.json,敘述由 agent 撰寫)
> 素材等級:full | 產生日期:2026-07-03

## 1. 競賽目的

**What**:本競賽要解決的問題是「Predict concrete compressive strength」——根據混凝土配方
中各成分的用量(水泥、爐渣、飛灰、水、減水劑、粗細骨材)與養護天數,預測混凝土的抗壓強度
(`Strength`)。這是一個 regression 任務。

**Why**:評估指標為 rmse,以 minimize 為優化方向。RMSE 對誤差取平方後開根號,會放大對「大幅
偏離」預測的懲罰,這對混凝土抗壓強度預測是合理的選擇——工程上嚴重低估或高估強度的後果(結構
安全風險、材料浪費)遠比小誤差嚴重,因此比起對離群誤差不敏感的 MAE,RMSE 更能反映此任務中
「避免大誤差」優先於「平均誤差小」的實務考量。

| 項目 | 值 |
|------|-----|
| 競賽 | playground-series-s3e9 |
| URL | https://www.kaggle.com/competitions/playground-series-s3e9 |
| 問題型別 | regression |
| 評估指標 | rmse(minimize) |
| 目標欄位 | Strength |
| ID 欄位 | id |

## 2. 資料規格

facts.json 未記錄列數/欄位數(row/column counts),故列數與欄位數寫「無紀錄」。可從
`experiments[].n_features` 得知特徵數量的演變:baseline(experiment_id 1)使用 8 個原始特徵,
最佳實驗(experiment_id 2)使用 22 個特徵(8 原始 + 14 工程特徵)。

**特別規則**(來源:`competition.special_rules`):
- 外部資料:不允許(`external_data_allowed: false`)
- 預訓練模型:不允許(`pretrained_models_allowed: false`)
- 網路存取:不允許(`internet_access_allowed: false`)
- 每日提交上限:5 次(`daily_submission_limit: 5`)

`competition.notes` 記載:「Aygun et al. Nature 2026 Kaggle Playground benchmark
(Season 3 Episode 9).」

本場素材等級為 `full`(非 baseline-only),故 EDA 與特徵工程階段皆已執行(細節見
STATUS.md,惟本節數字僅取自 facts.json)。

## 3. 模型規格

最佳實驗(experiment_id 2)使用的模型清單與各自 OOF 分數(來源:`best.base_models`):

| Model | RMSE | 訓練時間(s) |
|-------|------|--------------|
| LGB | 12.11061 | 6.7 |
| XGB | 12.12086 | 7.7 |
| CAT | 12.07459 | 2.7 |

**Ensemble 權重**(來源:`best.ensemble.weights`,方法 `oof_weight_search_grid0.05`):

| Model | 權重 |
|-------|------|
| LGB | 0.15 |
| XGB | 0.0 |
| CAT | 0.85 |

Ensemble 後最終分數(來源:`best.score`):**12.073474**。

**選型理由**(敘述,參考 STATUS.md 脈絡):三個模型皆為梯度提升樹,適合小型表格資料。
CatBoost 的 ordered boosting 正則化機制在本資料集上單模型表現最好(12.07459),LGB 加入
15% 權重後仍能小幅拉低整體 RMSE,XGB 權重被搜尋為 0——即使在本輪已對 LGB/XGB 加入較強正則
化(見第 4 節超參數),XGB 仍未被納入最終 blend。這與 baseline(experiment_id 1)呈現的模式
一致:baseline 的 blend_weights 中 LGB=0.0、XGB=0.0、CAT=1.0,CatBoost 拿下 100% 權重。

## 4. 訓練規格

**CV 方案**(來源:`best.cv`):

| 欄位 | 值 |
|------|-----|
| strategy | 5fold_stratified_strength_decile |
| seed | 42 |
| n_splits | 無紀錄(best.cv 未記錄此欄位) |

**Objective 與關鍵超參**:facts.json 的 `best.base_models` 未提供結構化的 `params` 欄位,
但 `best.notes`(逐字節錄)記載:

> "Regularized LGB(num_leaves=15,depth5,L1=2/L2=4)/XGB(depth4,L1=2/L2=4) + CatBoost(depth6,l2=6)
> with 8 engineered features (log_age, water/binder ratios) vs baseline exp#1 (generic untuned
> blend, RMSE 12.54287, CatBoost 100% weight)."

**為何用此 CV**:strategy 名稱顯示此為「以 Strength 十分位數分層」的 5-fold
StratifiedKFold(`5fold_stratified_strength_decile`)。相較於單純 KFold,對迴歸目標分層
可讓每個 fold 的目標分布更一致,在資料量較小時能降低 fold 間變異——這與 baseline
(experiment_id 1)僅記錄 `cv.scheme = "5fold"`(未分層)形成對比。

## 5. 推論程序

**後處理**:`best.postprocess` 未記錄 → 無後處理紀錄。

**Submission 格式**(來源:`best.submission`、`competition.id_column`/`target_column`):

| 欄位 | 值 |
|------|-----|
| Submission 檔名 | sub_blend_12.07347_20260703_191007.csv |
| ID 欄位 | id |
| 目標欄位 | Strength |

## 6. 評估指標

**指標定義**:RMSE(Root Mean Squared Error)是預測值與真實值差異平方之平均值,再開根號,
數值與目標欄位同單位,對大誤差的懲罰重於 MAE。

**分數總表**(來源:`best.*`、baseline 實驗):

| 項目 | RMSE |
|------|------|
| Baseline blend(experiment_id 1) | 12.54287 |
| LGB(experiment_id 2) | 12.11061 |
| XGB(experiment_id 2) | 12.12086 |
| CAT(experiment_id 2) | 12.07459 |
| **最佳 Ensemble(experiment_id 2)** | **12.073474** |

`facts.missing` 記錄 `["leaderboard"]`,即本場 **未提交至 Kaggle,無 Public/Private LB
分數**,故無法計算 CV↔LB gap(僅當 leaderboard 存在時才計算,依規格第 6 節)。

以下為 baseline 與最佳 CV 分數的改善量(衍生計算,兩數字皆逐字取自 facts.json,置於
code block 中):

```
12.54287 (baseline blend_score, experiment_id 1)
- 12.073474 (best score, experiment_id 2)
= 0.469396

相對改善: 0.469396 / 12.54287 * 100 = 3.742333...%
```

## 7. 實驗軌跡

**逐實驗分數表**(來源:`trajectory`):

| experiment_id | timestamp | score | source_format |
|---------------|-----------|-------|----------------|
| 1 | 2026-07-03T12:06:32 | 12.54287 | generic_batch |
| 2 | 2026-07-03T19:10:07 | 12.073474 | v2 |
| 3 | 2026-07-03T19:10:47 | 12.094826 | v2 |
| 4 | 2026-07-03T19:11:24 | 12.073474 | v2 |

**突破點敘述**:分數躍升發生在 experiment_id 2(timestamp 2026-07-03T19:10:07),score 由
baseline 的 12.54287 降至 12.073474。依 `experiments[2].notes`(逐字節錄同第 4 節),此次
變動同時改了兩件事:(a) 對 LGB/XGB 加入明確正則化超參數(num_leaves=15/depth5/L1=2/L2=4
與 depth4/L1=2/L2=4),(b) 從 8 個原始特徵擴充為 22 個工程特徵(含 log_age、
water/binder 比值等)。experiment_id 3 是同一份 notes 描述的延伸嘗試(於 features 清單中
多加入 3 個 interaction 特徵,`n_features` 從 22 增至 25),但 score 反而變差
(12.094826,劣於 experiment_id 2 的 12.073474),隨後 experiment_id 4 以與
experiment_id 2 相同的 22 特徵清單重新執行,score 完全重現為 12.073474。

`facts.unparsed` 為空陣列,無「無法解析之紀錄」。

## 8. 重現指令

以下命令序列參考 STATUS.md 的 Reproduce 節,並依 `competitions/playground-series-s3e9/scripts/`
實際檔名組成。執行目錄為專案根目錄,需先安裝 `uv`。

```bash
cd /home/tjyen/ai_agents/kaggle

# 1. EDA
uv run python3 competitions/playground-series-s3e9/scripts/eda.py

# 2. 特徵工程 + 訓練 + 5-fold CV + OOF weight-search blend + 產生 submission
#    (features.py 由 train.py 匯入呼叫,無需獨立執行步驟)
uv run python3 competitions/playground-series-s3e9/scripts/train.py

# 3. 提交至 Kaggle(本次執行未提交;需要有效 KAGGLE_API_TOKEN 憑證)
export KAGGLE_API_TOKEN=$(cat ~/.kaggle/kaggle_api_token.txt | tr -d '[:space:]')
uv run kaggle competitions submit -c playground-series-s3e9 \
  -f competitions/playground-series-s3e9/submissions/sub_blend_12.07347_20260703_191007.csv \
  -m "<msg>"
```
