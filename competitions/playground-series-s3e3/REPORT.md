# 競賽分析報告:playground-series-s3e3

> 產生方式:kaggle-report skill(數字來自 facts.json,敘述由 agent 撰寫)
> 素材等級:full | 產生日期:2026-07-03

## 1. 競賽目的

**What**:本場競賽要求依員工的人事資料(如年齡、職級、月薪、任職年資、滿意度評分、是否加班等)
預測該員工是否會離職(`Attrition`,二元標籤)。這是一個二元分類問題。

**Why**:評估指標為 **ROC-AUC(maximize)**。目標欄位嚴重不平衡(離職樣本僅占少數),ROC-AUC
以「排序能力」衡量模型能否將實際離職者排在較高風險分數,不受分類門檻與類別比例影響,較準確率
等指標更適合此類不平衡二元分類問題。

| 項目 | 值 |
|------|-----|
| 競賽 | playground-series-s3e3 |
| 問題型別 | classification |
| 評估指標 | roc_auc(maximize) |
| 目標欄位 | Attrition |

## 2. 資料規格

- 依 `competition.notes`,本場為 Aygün et al. Nature 2026 Kaggle Playground benchmark(Season 3
  Episode 3)。原始欄位數依實驗紀錄:實驗 1(`generic_batch`)使用 33 個特徵(來源:
  `experiments[0].n_features`);實驗 2、3(`v2`)經特徵工程後為 48 個特徵(來源:
  `experiments[1].n_features`、`experiments[2].n_features`)。
- 特別規則:不可使用外部資料、不可使用預訓練模型、不可存取網路;每日提交上限 5 次(來源:
  `competition.special_rules`)。
- `material_level` 為 `full`,EDA 已完整執行(細節見 STATUS.md,本節數字仍僅列 facts.json 所載
  之特徵數與規則)。

## 3. 模型規格

本場有三筆實驗紀錄:

**實驗 1(`generic_batch`,通用批次管線)— base models:**

| Model | OOF ROC-AUC |
|-------|-------------|
| LGB | 0.81362 |
| XGB | 0.80511 |
| CAT | 0.80874 |

Ensemble 權重(來源:`experiments[0].ensemble.weights`):LGB 0.7 / XGB 0.0 / CAT 0.3,
Ensemble 分數(來源:`experiments[0].score`):**0.81624**。

**實驗 2(`v2`,特徵工程 + 類別不平衡加權)— base models:**

| Model | OOF ROC-AUC |
|-------|-------------|
| LGB | 0.819008 |
| XGB | 0.79196 |
| CAT | 0.776632 |

Ensemble 權重(來源:`experiments[1].ensemble.weights`):LGB 1.0 / XGB 0.0 / CAT 0.0
(grid_weight_search),Ensemble 分數(來源:`experiments[1].score`):**0.819008**。

**實驗 3(`v2`,特徵工程 + 移除不平衡加權)— base models,`facts.best` 所指之最佳實驗:**

| Model | OOF ROC-AUC |
|-------|-------------|
| LGB | 0.832925 |
| XGB | 0.807634 |
| CAT | 0.762684 |

Ensemble 權重(來源:`experiments[2].ensemble.weights`):LGB 1.0 / XGB 0.0 / CAT 0.0
(grid_weight_search),Ensemble 分數(來源:`experiments[2].score` 與 `best.score`):
**0.832925**。

**選型理由**:三個梯度提升樹模型(LightGBM、XGBoost、CatBoost)在表格型資料上普遍穩健,以權重
搜尋(grid_weight_search)決定集成比例。三筆實驗中,權重搜尋皆將 CatBoost 之權重收斂為 0,
LightGBM 單獨已是三個實驗中表現最好的 base model,故最終 ensemble 分數等同 LGB 單模型分數。
實驗 3 相較實驗 2 移除了類別不平衡加權(`scale_pos_weight`/`class_weights`)並調整 LightGBM
正則化,OOF 分數由 0.819008 升至 0.832925(細節見節 7)。

## 4. 訓練規格

| 實驗 | CV scheme | n_splits | seed |
|------|-----------|----------|------|
| 1 | 5fold | 5 | 無紀錄 |
| 2 | StratifiedKFold | 5 | 42 |
| 3 | StratifiedKFold | 5 | 42 |

(來源:`experiments[].cv`;實驗 1 之 `cv` 未記錄 seed 欄位,故寫「無紀錄」。)

各 base model 之 objective/超參數在 `experiments[].base_models[]` 中未記錄對應欄位,故此項為
「無紀錄」。

**為何用此 CV**:目標欄位 `Attrition` 為不平衡二元分類(少數類別為離職樣本),若採用一般隨機
K-fold 切分,各 fold 之正負樣本比例可能不一致,導致驗證 AUC 分數不穩定。實驗 2、3 因此改用
StratifiedKFold(依 `Attrition` 分層抽樣),確保每個 fold 的類別比例與整體一致,使交叉驗證分數
更能反映模型的真實泛化能力。

## 5. 推論程序

`facts.best`(實驗 3)之 `postprocess` 欄位未記錄任何後處理步驟,故此欄寫「無後處理紀錄」。
其 submission 檔名為 `sub_blend_0.83292_20260703_183913.csv`(來源:`experiments[2].submission`
與 `best.submission`),對應 `id_column = id`、`target_column = Attrition`。

需特別說明:`facts.json` 之 `leaderboard` 欄位為 null,`missing` 陣列列出 `leaderboard`
——本場三筆實驗皆**未提交至 Kaggle 排行榜**(依指示本次為僅產生 submission 檔案之無人值守批次
執行,未觸碰 Kaggle 憑證或執行提交)。

## 6. 評估指標

**指標定義**:ROC-AUC(Receiver Operating Characteristic - Area Under Curve)= 模型將正樣本
排序高於負樣本的機率,數值愈高代表排序能力愈好(下界為隨機猜測、上界為完美排序),不受決策
門檻影響。

| 項目 | 分數 |
|------|------|
| 實驗 1 Ensemble(OOF ROC-AUC) | 0.81624 |
| 實驗 2 Ensemble(OOF ROC-AUC) | 0.819008 |
| 實驗 3 Ensemble(OOF ROC-AUC,facts.best) | 0.832925 |
| Public LB | 無紀錄 |
| Private LB | 無紀錄 |

(來源:`experiments[].score`、`facts.leaderboard`。`facts.leaderboard` 為 null,故 Public/
Private LB 兩欄皆寫「無紀錄」,無法計算 CV↔LB gap。)

## 7. 實驗軌跡

| experiment_id | timestamp | score | source_format |
|---------------|-----------|-------|----------------|
| 1 | 2026-07-03T12:02:55 | 0.81624 | generic_batch |
| 2 | 2026-07-03T18:37:57 | 0.819008 | v2 |
| 3 | 2026-07-03T18:39:13 | 0.832925 | v2 |

（來源：`facts.trajectory`）

**突破點**:分數躍升發生在第 3 筆實驗(`v2`,experiment_id 3),OOF ROC-AUC 由第 2 筆的
0.819008 升至 0.832925。依 `experiments[2].notes` 記載,第 2 筆實驗中 XGB/CAT 之 OOF AUC
(0.79196、0.776632)皆低於實驗 1 通用管線中對應模型的分數(0.80511、0.80874),推測是
`scale_pos_weight`(XGB)/`class_weights`(CatBoost)等類別不平衡加權手法扭曲了排序而非提升
排序能力(ROC-AUC 為排序型指標,對決策門檻不敏感)。第 3 筆實驗移除上述加權,並將 LightGBM
之 `num_leaves` 等正則化參數調整得更保守,使 LGB OOF 由 0.819008 升至 0.832925,XGB 亦由
0.79196 升至 0.807634;CatBoost 之 OOF(0.762684)仍未回升,但因權重搜尋將其權重收斂為 0,
未影響最終 ensemble 分數。

以下算式呈現實驗 3 相對於第 1 筆通用批次基準(0.81624)之絕對提升:

```
實驗 3 OOF ROC-AUC − 實驗 1 OOF ROC-AUC = 0.832925 − 0.81624 = 0.016685
```

`facts.unparsed` 為空陣列,無法解析之紀錄:無。

## 8. 重現指令

```bash
cd /home/tjyen/ai_agents/kaggle

# Stage 1: EDA
uv run python3 competitions/playground-series-s3e3/scripts/eda.py

# Stage 3: 建模(實驗 2,含特徵工程 + 類別不平衡加權)
uv run python3 competitions/playground-series-s3e3/scripts/train.py

# Stage 3 自我改進迭代(實驗 3,最佳結果:移除不平衡加權)
uv run python3 competitions/playground-series-s3e3/scripts/train_v2.py
```

執行目錄為專案根目錄 `/home/tjyen/ai_agents/kaggle`;所有 Python 執行皆透過 `uv run` 以確保套件
環境一致。本次為無人值守批次執行,未提交至 Kaggle 排行榜、未觸碰 Kaggle 憑證。
