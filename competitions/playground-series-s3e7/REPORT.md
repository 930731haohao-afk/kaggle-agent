# 競賽分析報告:playground-series-s3e7

> 產生方式:kaggle-report skill(數字來自 facts.json,敘述由 agent 撰寫)
> 素材等級:full | 產生日期:2026-07-03

## 1. 競賽目的

**What**:本競賽要解決的問題是 Predict hotel reservation cancellation——根據訂房紀錄的
各項欄位,預測該筆訂房最終是否會被取消(目標欄位 `booking_status`,二元分類)。

**Why**:評估指標為 roc_auc,以 maximize 方向優化。ROC-AUC 是一個與分類門檻無關的排序型
指標,直接衡量模型能否把「會取消」的訂房排在「不會取消」的訂房之前,不需要事先選定機率門檻,
也不要求機率經過校準,因此很適合本題這種二元分類、且正負類別存在一定不平衡的場景。

| 項目 | 值 |
|------|-----|
| 競賽 | playground-series-s3e7 |
| 問題型別 | classification |
| 評估指標 | roc_auc(maximize) |
| 目標欄位 | booking_status |

## 2. 資料規格

facts.json 僅記錄各實驗使用的特徵數,未記錄訓練/測試集的實際列數與欄位型別分佈,故列數/欄位型別
概述為**無紀錄**。可回溯的資料規模資訊來自 `experiments[].n_features`:

| 實驗 | 特徵數 |
|------|--------|
| 通用 baseline blend(exp 1) | 17 |
| Iter1:17 原始 + 工程特徵(exp 2) | 31 |
| Iter2/最終:17 原始 + 精簡工程特徵(exp 3) | 25 |

素材等級為 **full**(本場已執行 EDA 與特徵工程,並非僅有 baseline)。

特別規則(來源:`competition.special_rules`):不允許外部資料(external_data_allowed:
false)、不允許預訓練模型(pretrained_models_allowed: false)、不允許存取網路
(internet_access_allowed: false)、每日提交上限 5 次(daily_submission_limit: 5)。

## 3. 模型規格

最終選用模型(best,依 OOF score 選出的 exp 3)由三個基模型加權混合而成
(來源:`best.base_models`):

| 模型 | OOF ROC-AUC | 訓練時間(秒) |
|------|-------------|----------------|
| LGB | 0.898824 | 30.3 |
| XGB | 0.898765 | 31.2 |
| CAT | 0.896709 | 47.1 |

Ensemble 權重與分數(來源:`best.ensemble`):LGB 0.5 / XGB 0.4 / CAT 0.1,混合後
OOF ROC-AUC = **0.899395**。

**選型理由**:三個模型的單模型 OOF 分數彼此接近(0.8967–0.8988),屬於多樣但實力相當的
樹模型組合,權重搜尋給 CatBoost 較低權重(0.1)反映其單模略弱;LGB 與 XGB 權重相近(0.5/0.4)
反映兩者表現幾乎並列。此組合較單一最佳模型(LGB, 0.898824)進一步提升到 0.899395。

## 4. 訓練規格

CV 方案(來源:`best.cv`):

| scheme | n_splits | seed | strategy |
|--------|----------|------|----------|
| 5fold | 5 | 42 | StratifiedKFold(booking_status) |

**為何用此 CV**:目標欄位為二元分類且存在類別不平衡,直接對 `booking_status` 做
StratifiedKFold 可確保每一折的正負類別比例一致,避免因某一折類別分佈偏移而使 CV 分數
不穩定或不可信。

各基模型關鍵超參(來源:`best.base_models[].params`,皆有紀錄):

| 模型 | 關鍵超參 |
|------|----------|
| LGB | objective=binary, metric=auc, n_estimators=3000, learning_rate=0.03, num_leaves=63, min_child_samples=30, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=1.0, random_state=42, n_jobs=-1, verbose=-1 |
| XGB | objective=binary:logistic, n_estimators=3000, learning_rate=0.03, max_depth=6, min_child_weight=5, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=1.0, random_state=42, n_jobs=-1, eval_metric=auc, early_stopping_rounds=150 |
| CAT | loss_function=Logloss, eval_metric=AUC, iterations=4000, learning_rate=0.03, depth=7, l2_leaf_reg=5.0, random_seed=42, thread_count=-1, verbose=False |

## 5. 推論程序

**後處理步驟**:facts.json 之 `best` 未記錄 `postprocess` 欄位 → **無後處理紀錄**(ROC-AUC
為排序型指標,不需要機率門檻轉換)。

**Submission 檔名與格式**(來源:`best.submission`、`competition.id_column`/`target_column`):
檔名 `sub_blend_0.89939_20260703_190223.csv`,欄位為 id 欄 `id` 與目標欄 `booking_status`
(輸出為取消機率,而非硬分類標籤,符合 ROC-AUC 評分需求)。

## 6. 評估指標

**指標定義**:ROC-AUC 衡量模型將正類(取消)排在隨機一筆負類(未取消)之前的機率,數值介於
0.5(隨機)到 1(完美排序)之間,越高越好(maximize)。

分數總表(來源:`best.*`、`leaderboard`):

| 項目 | ROC-AUC |
|------|---------|
| LGB(base model) | 0.898824 |
| XGB(base model) | 0.898765 |
| CAT(base model) | 0.896709 |
| Ensemble(最終) | 0.899395 |
| Public LB | 無紀錄 |
| Private LB | 無紀錄 |

**CV↔LB gap**:facts.json 之 `leaderboard` 為空(`missing` 清單包含 `leaderboard`)——本場
僅完成本機 CV,尚未提交至 Kaggle 排行榜,故無法計算 CV↔LB gap。

## 7. 實驗軌跡

逐實驗分數表(來源:`trajectory`):

| experiment_id | timestamp | score | source_format |
|---------------|-----------|-------|----------------|
| 1 | 2026-07-03T12:05:42 | 0.89882 | generic_batch |
| 2 | 2026-07-03T18:59:38 | 0.89788 | v2 |
| 3 | 2026-07-03T19:02:23 | 0.899395 | v2 |

**突破點敘述**:分數躍升發生在 experiment 3。experiment 2(17 原始 + 14 工程特徵,含月份/日期
的週期性編碼與價格交乘項)分數為 0.89788,低於通用 baseline 的 0.89882,呈現退步訊號
(來源:exp 2 之 `notes`:"Delta vs generic baseline 0.89882: -0.00094")。據此反思,
experiment 3 將工程特徵精簡為 8 個(移除週期性月份/日期編碼、price_per_night、
special×price 交乘項),分數回升至 0.899395,較 baseline 進步
(來源:exp 3 之 `notes`:"Delta vs generic baseline 0.89882: +0.00057")。以下為以
facts.json 內 `best.score`(0.899395)與 exp 1 之 `score`(0.89882)重新計算之衍生差值,
僅用於驗證上述 notes 描述之方向一致,不作為新事實:

```
0.899395 (exp3 最終 ensemble score) − 0.89882 (exp1 baseline score) = 0.000575
0.89788  (exp2 ensemble score)      − 0.89882 (exp1 baseline score) = -0.00094
```

`unparsed`:facts.json 之 `unparsed` 清單為空,無法解析之紀錄:無。

## 8. 重現指令

```bash
cd /home/tjyen/ai_agents/kaggle

# Stage 1: EDA
uv run python3 competitions/playground-series-s3e7/scripts/eda.py

# Stage 2-3: feature engineering + modeling (LGB/XGB/CAT, 5-fold StratifiedKFold,
# OOF weight-searched blend, writes submissions/ + logs experiments.json)
uv run python3 competitions/playground-series-s3e7/scripts/train.py

# Report generation (this file)
uv run python3 .claude/skills/kaggle-report/assets/collect.py playground-series-s3e7
uv run python3 .claude/skills/kaggle-report/assets/verify_report.py \
    competitions/playground-series-s3e7/REPORT.md competitions/playground-series-s3e7/facts.json
bash .claude/skills/kaggle-report/assets/md2pdf.sh competitions/playground-series-s3e7/REPORT.md

# Not submitted to Kaggle in this run (no credentials available in this environment).
# To submit the final submission file to the leaderboard:
export KAGGLE_API_TOKEN=$(python3 -c "import json; print(json.load(open('/home/tjyen/.kaggle/kaggle.json'))['key'])")
uv run kaggle competitions submit -c playground-series-s3e7 \
    -f competitions/playground-series-s3e7/submissions/sub_blend_0.89939_20260703_190223.csv \
    -m "LGB+XGB+CAT weight-searched blend, trimmed features"
```
