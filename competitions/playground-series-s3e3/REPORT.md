# 競賽分析報告:playground-series-s3e3

> 產生方式:kaggle-report skill(數字來自 facts.json,敘述由 agent 撰寫)
> 素材等級:full | 產生日期:2026-07-04(Phase G-1a 樹搜尋成果入帳後更新)

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

## 2. 流程(how):五大元件

### 2.1 資料規格

- 依 `competition.notes`,本場為 Aygün et al. Nature 2026 Kaggle Playground benchmark(Season 3
  Episode 3)。原始欄位數依實驗紀錄:實驗 1(`generic_batch`)使用 33 個特徵(來源:
  `experiments[0].n_features`);實驗 2、3、4、5、7(`v2`)經特徵工程後為 48 個特徵(來源:
  `experiments[1].n_features` 等);實驗 6(CatBoost 原生類別重試)改用 40 個特徵(保留類別欄
  原始字串而非 label/freq 編碼,來源:`experiments[5].n_features`)。
- 特別規則:不可使用外部資料、不可使用預訓練模型、不可存取網路;每日提交上限 5 次(來源:
  `competition.special_rules`)。
- `material_level` 為 `full`,EDA 已完整執行(細節見 STATUS.md,本節數字仍僅列 facts.json 所載
  之特徵數與規則)。

### 2.2 模型規格

本場累計 8 筆實驗紀錄,**`facts.best` 現為實驗 8——一筆樹搜尋(tree-search)結果**,而非
線性迭代終點(實驗 7)。實驗 1–7(線性迭代)先完整說明如下,實驗 8 見本節末的 2.2a 小節。

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

**實驗 3(`v2`,特徵工程 + 移除不平衡加權)— base models:**

| Model | OOF ROC-AUC |
|-------|-------------|
| LGB | 0.832925 |
| XGB | 0.807634 |
| CAT | 0.762684 |

Ensemble 權重(來源:`experiments[2].ensemble.weights`):LGB 1.0 / XGB 0.0 / CAT 0.0
(grid_weight_search),Ensemble 分數(來源:`experiments[2].score`):**0.832925**。

**實驗 4(Phase B round 1,LGB Optuna 全 5-fold CV AUC 直接調參,單模)— base models:**

| Model | OOF ROC-AUC |
|-------|-------------|
| LGB_tuned | 0.837305 |

Ensemble(來源:`experiments[3].ensemble`):single_model,LGB_tuned 權重 1.0,分數
(來源:`experiments[3].score`):**0.837305**。

**實驗 5(Phase B round 2,調參版加入池 + seed bagging,5-way blend)— base models:**

| Model | OOF ROC-AUC |
|-------|-------------|
| LGB_orig | 0.832925 |
| XGB | 0.807634 |
| CAT | 0.762684 |
| LGB_tuned | 0.837305 |
| LGB_tuned_seed2024 | 0.83345 |

Ensemble 權重(來源:`experiments[4].ensemble.weights`):LGB_orig 0.3 / XGB 0.0 / CAT 0.0 /
LGB_tuned 0.7 / LGB_tuned_seed2024 0.0(grid_weight_search),Ensemble 分數(來源:
`experiments[4].score`):**0.837776**。

**實驗 6(Phase B round 3,CatBoost 原生類別特徵重試,單模診斷)— base models:**

| Model | OOF ROC-AUC |
|-------|-------------|
| CAT_native_best | 0.814259 |

Ensemble(來源:`experiments[5].ensemble`):single_model,CAT_native_best 權重 1.0,分數
(來源:`experiments[5].score`):**0.814259**。

**實驗 7(Phase B round 4,加入 CAT_native 之 6-way blend,rank-average)— base models,
`facts.best` 所指之最佳實驗:**

| Model | OOF ROC-AUC |
|-------|-------------|
| LGB_orig | 0.832925 |
| XGB | 0.807634 |
| CAT_orig | 0.762684 |
| LGB_tuned | 0.837305 |
| LGB_tuned_seed2024 | 0.83345 |
| CAT_native | 0.814259 |

Ensemble 權重(來源:`experiments[6].ensemble.weights`):LGB_orig 0.3 / XGB 0.0 /
CAT_orig 0.0 / LGB_tuned 0.7 / LGB_tuned_seed2024 0.0 / CAT_native 0.0
(rank_average_same_weights),Ensemble 分數(來源:`experiments[6].score` 與 `best.score`):
**0.83814**。

**選型理由**:三個梯度提升樹模型(LightGBM、XGBoost、CatBoost)在表格型資料上普遍穩健,以權重
搜尋(grid_weight_search)決定集成比例。實驗 1–3、5、7 之權重搜尋皆將 CatBoost(無論是否採用
原生類別特徵)之權重收斂為 0。Phase B 迭代(實驗 4–7)聚焦 LightGBM:對其做 Optuna 全 5-fold
CV ROC-AUC 直接優化(實驗 4),將調參版加入而非取代原池、再做同參數 seed bagging(實驗 5),
使 blend 由 0.832925 升至 0.837776。實驗 6 針對 STATUS.md 的既有未解問題(CatBoost 原生類別
處理未試)做診斷性重試,將 CatBoost 之弱點從 0.762684 大幅拉升至 0.814259,但仍低於其餘成員,
權重搜尋在實驗 7 的 6-way pool 中依然給予 CAT_native 權重 0——確認「CatBoost 在此小樣本資料
上結構性偏弱」為穩定結論而非單純缺乏原生類別支援。實驗 7 最終改採 rank-average(而非機率空間
權重混合)在相同權重下取得線性迭代最終分數 0.83814。

#### 2.2a 樹搜尋最佳(best, experiment_id=8)——本次更新新增

Phase E-5「規模實驗」(scaling experiment,2026-07-04)以 80-節點預算跑了一棵**與 Phase D-2
的 22-節點 v2 掃描不同的全新樹**(來源:`experiments_tree_scale.json`,D-2 的
`experiments_tree.json` 未被觸碰),於 node #55(`KITCHENBLEND` lineage,explore phase)
找到本場目前最佳 ROC-AUC:

**Ensemble**(best.ensemble):method = "KITCHENBLEND: dirichlet(k=800)+coordinate-ascent
blend over every solo node evaluated so far in the 80-node scale run (explore phase)",
score = **0.845051**。facts.json 本筆未附結構化 `base_models` 陣列(此 lineage 是「目前
為止所有 solo 節點」的 kitchen-sink 混合,成員數隨搜尋進度變動,非固定小池,故未逐一列出
——無紀錄,不臆測;完整曲線見 `docs/scaling_experiment.md`)。

**選型理由/來源說明**(best.notes):此為 Phase E-5 樹搜尋結果,不是線性迭代第 8 輪,也不是
D-2 v2 掃描的延伸。所有 3 次 eval-40 之後的全域最佳刷新皆來自同一條 KITCHENBLEND lineage
(混合當時已評估過的所有 solo 節點,包含個別已被 exploit 階段各模型佇列淘汰的 CatBoost
變體與邊界推進模型);其餘 5 條 explore lineage 皆未突破 exploit 階段的 0.843524 天花板。
完整曲線表、跨賽事校準、與 Stage-4 預算建議見 `docs/scaling_experiment.md`。

> **重要澄清**:best.notes 明確記載這是 **OOF-only 搜尋結果——未產生任何 test 預測,亦
> 未提交至 Kaggle**(facts.json 本筆無 submission 欄位)。facts.best 是以 OOF score
> 最大者選出(本場 metric 為 roc_auc,maximize),與是否已提交至 Kaggle 無關;本場所有
> 實驗(含實驗 8)皆未提交至 Kaggle(見第 2.4/2.5 節)。

### 2.3 訓練規格

| 實驗 | CV scheme | n_splits | seed |
|------|-----------|----------|------|
| 1 | 5fold | 5 | 無紀錄 |
| 2 | StratifiedKFold | 5 | 42 |
| 3 | StratifiedKFold | 5 | 42 |
| 4 | StratifiedKFold | 5 | 42 |
| 5 | StratifiedKFold | 5 | 42 |
| 6 | StratifiedKFold | 5 | 42 |
| 7 | StratifiedKFold | 5 | 42 |
| 8(best,樹搜尋) | StratifiedKFold | 5 | 42 |

(來源:`experiments[].cv`;實驗 1 之 `cv` 未記錄 seed 欄位,故寫「無紀錄」。實驗 2–8 之
StratifiedKFold(n_splits=5, seed=42)為同一組固定折,跨實驗可直接比較分數——樹搜尋
(實驗 8)沿用與線性迭代完全相同的 CV 折。)

各 base model 之 objective/超參數在 `experiments[].base_models[]` 中未記錄對應欄位,故此項為
「無紀錄」(實驗 4 之 Optuna 最佳超參記於 `experiments[3].notes` 文字說明,非結構化欄位,依
Hard Rule 1 不作為表格數字引用)。

**為何用此 CV**:目標欄位 `Attrition` 為不平衡二元分類(少數類別為離職樣本),若採用一般隨機
K-fold 切分,各 fold 之正負樣本比例可能不一致,導致驗證 AUC 分數不穩定。實驗 2 起因此改用
StratifiedKFold(依 `Attrition` 分層抽樣),確保每個 fold 的類別比例與整體一致,使交叉驗證分數
更能反映模型的真實泛化能力;Phase B 迭代(實驗 4–7)延續同一 CV 方案以維持跨實驗可比性。

### 2.4 推論程序

`facts.best`(實驗 8,樹搜尋)之 `postprocess` 欄位未記錄任何後處理步驟,故此欄寫「無後處理
紀錄」。**Submission 檔名(best.submission)：無紀錄**——實驗 8 是樹搜尋(OOF-only)結果,
facts.json 本筆未附 submission 欄位,未產生 test 預測、未提交 Kaggle。

線性迭代終點(實驗 7)之 `postprocess` 欄位同樣未記錄任何後處理步驟(其 ensemble 方法為
rank-average,屬於集成合併而非後處理);其 submission 檔名為
`sub_blend_0.83814_20260703_233303.csv`(來源:`experiments[6].submission`),對應
`id_column = id`、`target_column = Attrition`——供參考,該檔案對應 OOF 0.83814,非目前
best 的 0.845051。

需特別說明:`facts.json` 之 `leaderboard` 欄位為 null,`missing` 陣列列出 `leaderboard`
——本場所有實驗(含實驗 8)皆**未提交至 Kaggle 排行榜**(依指示本次為僅產生 submission
檔案之無人值守批次執行,未觸碰 Kaggle 憑證或執行提交)。

### 2.5 評估指標

**指標定義**:ROC-AUC(Receiver Operating Characteristic - Area Under Curve)= 模型將正樣本
排序高於負樣本的機率,數值愈高代表排序能力愈好(下界為隨機猜測、上界為完美排序),不受決策
門檻影響。

| 項目 | 分數 |
|------|------|
| 實驗 1 Ensemble(OOF ROC-AUC) | 0.81624 |
| 實驗 2 Ensemble(OOF ROC-AUC) | 0.819008 |
| 實驗 3 Ensemble(OOF ROC-AUC) | 0.832925 |
| 實驗 4 單模(OOF ROC-AUC) | 0.837305 |
| 實驗 5 Ensemble(OOF ROC-AUC) | 0.837776 |
| 實驗 6 單模診斷(OOF ROC-AUC) | 0.814259 |
| 實驗 7 Ensemble(OOF ROC-AUC,線性迭代終點) | 0.83814 |
| 實驗 8 樹搜尋 Ensemble(OOF ROC-AUC,facts.best) | 0.845051 |
| Public LB | 無紀錄 |
| Private LB | 無紀錄 |

(來源:`experiments[].score`、`facts.leaderboard`。`facts.leaderboard` 為 null,故 Public/
Private LB 兩欄皆寫「無紀錄」,無法計算 CV↔LB gap。)

實驗 7 相對於本場第一輪自我改進迭代前最佳結果(實驗 3,0.832925)之絕對提升,以及實驗 8
(樹搜尋 best)相對實驗 7 與實驗 3 之絕對提升:

```
實驗 7 OOF ROC-AUC − 實驗 3 OOF ROC-AUC = 0.83814 − 0.832925 = 0.005215

實驗 8 OOF ROC-AUC − 實驗 7 OOF ROC-AUC = 0.845051 − 0.83814 = 0.006911
實驗 8 OOF ROC-AUC − 實驗 3 OOF ROC-AUC = 0.845051 − 0.832925 = 0.012126
```

## 3. 實驗軌跡

| experiment_id | timestamp | score | source_format |
|---------------|-----------|-------|----------------|
| 1 | 2026-07-03T12:02:55 | 0.81624 | generic_batch |
| 2 | 2026-07-03T18:37:57 | 0.819008 | v2 |
| 3 | 2026-07-03T18:39:13 | 0.832925 | v2 |
| 4 | 2026-07-03T23:29:51 | 0.837305 | v2 |
| 5 | 2026-07-03T23:30:44 | 0.837776 | v2 |
| 6 | 2026-07-03T23:31:25 | 0.814259 | v2 |
| 7 | 2026-07-03T23:33:03 | 0.83814 | v2 |
| 8 | 2026-07-04T11:59:43 | 0.845051 | v2 |

（來源：`facts.trajectory`）

**突破點**:第一次躍升發生在實驗 3(`v2`,experiment_id 3),OOF ROC-AUC 由實驗 2 的
0.819008 升至 0.832925(移除類別不平衡加權 + 加強 LightGBM 正則化,詳見 STATUS.md 既有
Reflexion 記錄)。本輪 Phase B 自我改進迭代(experiment_id 4–7)在此基礎上再度躍升:實驗 4
以 Optuna(TPE,50 trials,timeout 600s,實際 111.3s)直接對完整 5-fold StratifiedKFold OOF
ROC-AUC 做目標函式優化(而非先調代理指標),LGB 單模由 0.832925 升至 0.837305(來源:
`experiments[3].notes`);實驗 5 將此調參版加入(而非取代)實驗 3 的原池,並對同一組超參數做
seed=2024 的第二次訓練(seed bagging),5-way 權重搜尋 blend 再升至 0.837776(來源:
`experiments[4].notes`);實驗 6 為診斷性重試,針對 STATUS.md 標記的未解問題(CatBoost 原生
類別特徵處理)單獨測試,OOF 由原本編碼版的 0.762684 大幅提升至 0.814259,但仍未達到具競爭力
水準;實驗 7 將此 CatBoost 原生版加入實驗 5 的 5-way pool 成 6-way,權重搜尋仍將其權重收斂為
0(與 CatBoost 之前兩次結果一致),但改用 rank-average(而非機率空間權重混合)在相同權重組合
下取得線性迭代終點分數 0.83814(來源:`experiments[6].notes`)。

**突破點 2(exp 7→8,本次更新新增)**:實驗 8 不是線性迭代的延續回合,而是 Phase E-5
(2026-07-04)以「規模實驗」(80-節點預算)跑出的**樹搜尋(tree-search)**結果——來源
`experiments_tree_scale.json` 的 node #55(`KITCHENBLEND` lineage,explore phase)。此樹
與 Phase D-2 的 22-節點 v2 掃描樹(`experiments_tree.json`)是各自獨立的兩棵樹。分數由實驗 7
的 0.83814 升至 0.845051(見上方程式,絕對提升 0.006911)。**誠實 CV-only 警語**:此結果為
OOF-only 搜尋產物——tree_search harness 未產生任何 test 預測檔,facts.json 本筆亦無
submission 欄位,**未提交至 Kaggle**;不可與實驗 7 實際提交的 submission 檔案混淆(見第 2.4
節)。完整曲線表(每 evals 5/10/…/80 的 best-so-far AUC)、遲來的 backtrack 是否值得、與
跨賽事校準見 `docs/scaling_experiment.md`。

`facts.unparsed` 為空陣列,無法解析之紀錄:無。

## 4. 重現指令

```bash
cd /home/tjyen/ai_agents/kaggle

# Stage 1: EDA
uv run python3 competitions/playground-series-s3e3/scripts/eda.py

# Stage 3: 建模(實驗 2,含特徵工程 + 類別不平衡加權)
uv run python3 competitions/playground-series-s3e3/scripts/train.py

# Stage 3 自我改進迭代(實驗 3,舊最佳:移除不平衡加權)
uv run python3 competitions/playground-series-s3e3/scripts/train_v2.py

# Phase B 自我改進迭代 round 1(實驗 4:Optuna 全 5-fold CV AUC 直接調參 LGB)
uv run python3 competitions/playground-series-s3e3/scripts/iterate_round1_optuna_lgb.py

# Phase B round 2(實驗 5:調參版加入池 + seed bagging,5-way blend)
uv run python3 competitions/playground-series-s3e3/scripts/iterate_round2_pool_seedbag.py

# Phase B round 3(實驗 6:CatBoost 原生類別特徵重試,診斷用)
uv run python3 competitions/playground-series-s3e3/scripts/iterate_round3_catboost_native.py

# Phase B round 4(實驗 7,線性迭代終點:加入 CAT_native 之 6-way blend,rank-average)
uv run python3 competitions/playground-series-s3e3/scripts/iterate_round4_add_catnative_blend.py

# Phase E-5 樹搜尋規模實驗(實驗 8,本場目前 best;可中斷/續跑;樹狀態存
# experiments_tree_scale.json,與 D-2 的 experiments_tree.json 各自獨立)
uv run python3 tree_search/run_s3e3_scale.py
```

執行目錄為專案根目錄 `/home/tjyen/ai_agents/kaggle`;所有 Python 執行皆透過 `uv run` 以確保套件
環境一致。本次為無人值守批次執行,未提交至 Kaggle 排行榜、未觸碰 Kaggle 憑證。
