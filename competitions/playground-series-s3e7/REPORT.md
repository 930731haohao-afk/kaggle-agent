# 競賽分析報告:playground-series-s3e7

> 產生方式:kaggle-report skill(數字來自 facts.json,敘述由 agent 撰寫)
> 素材等級:full | 產生日期:2026-07-03(Phase B 自我改進迭代後更新)

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
| Iter2:17 原始 + 精簡工程特徵(exp 3) | 25 |
| Phase B 三輪(exp 4–6,特徵集凍結不變) | 25 |

素材等級為 **full**(本場已執行 EDA 與特徵工程,並非僅有 baseline)。

特別規則(來源:`competition.special_rules`):不允許外部資料(external_data_allowed:
false)、不允許預訓練模型(pretrained_models_allowed: false)、不允許存取網路
(internet_access_allowed: false)、每日提交上限 5 次(daily_submission_limit: 5)。

## 3. 模型規格

最終選用模型(best,依 OOF score 選出的 exp 6)由三個基模型加權混合而成
(來源:`best.base_models`):

| 模型 | OOF ROC-AUC | 訓練時間(秒) |
|------|-------------|----------------|
| LGB(Optuna 調參,Phase B R1) | 0.899215 | 25.7 |
| XGB(手設參數,保留多樣性) | 0.898765 | 27.7 |
| CAT(手設參數) | 0.896709 | 46.0 |

Ensemble 權重與分數(來源:`best.ensemble`):LGB 0.54 / XGB 0.4 / CAT 0.06,
混合方法 `prob_0.01grid`(機率混合、0.01 步長權重網格搜尋),混合後
OOF ROC-AUC = **0.899893**。

**選型理由**:Phase B Round 1 以 Optuna(50 trials、TPE、fold-0 代理目標)只調 LGB,
LGB 單模由 0.898824 升至 0.899215,成為最強單模。Round 2 以同法調 XGB 時,XGB 單模雖由
0.898765 微升至 0.89886,但混合分數反而退步(exp 5 之 0.899722)——調參後的 XGB 收斂到
與調參 LGB 相似的淺樹結構,喪失 ensemble 多樣性,故保留手設 XGB。CatBoost 單模最弱
(0.896709),權重搜尋僅給 0.06。

## 4. 訓練規格

CV 方案(來源:`best.cv`,六個實驗全程固定同一方案,分數可直接比較):

| scheme | n_splits | seed | strategy |
|--------|----------|------|----------|
| 5fold | 5 | 42 | StratifiedKFold(booking_status) |

**為何用此 CV**:目標欄位為二元分類且存在類別不平衡,直接對 `booking_status` 做
StratifiedKFold 可確保每一折的正負類別比例一致,避免因某一折類別分佈偏移而使 CV 分數
不穩定或不可信。

各基模型關鍵超參(來源:`best.base_models[].params`,皆有紀錄;LGB 為 Optuna 調參結果,
數值四捨五入至 3–4 位小數):

| 模型 | 關鍵超參 |
|------|----------|
| LGB(Optuna) | objective=binary, metric=auc, n_estimators=3000, learning_rate≈0.068, num_leaves=178, max_depth=3, min_child_samples=47, subsample≈0.889, colsample_bytree≈0.532, reg_alpha≈2.14, reg_lambda≈0.0115, random_state=42, n_jobs=-1, verbose=-1 |
| XGB | objective=binary:logistic, n_estimators=3000, learning_rate=0.03, max_depth=6, min_child_weight=5, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=1.0, random_state=42, n_jobs=-1, eval_metric=auc, early_stopping_rounds=150 |
| CAT | loss_function=Logloss, eval_metric=AUC, iterations=4000, learning_rate=0.03, depth=7, l2_leaf_reg=5.0, random_seed=42, thread_count=-1, verbose=False |

值得注意:Optuna 找到的 LGB 最佳解是「淺而強正則」(max_depth=3、較高學習率、
reg_alpha≈2.14),num_leaves=178 在 depth=3 之下實際不起作用;這與跨競賽經驗
「小/中型資料獎勵正則化而非容量」一致。

## 5. 推論程序

**後處理步驟**:facts.json 之 `best` 未記錄 `postprocess` 欄位 → **無後處理紀錄**(ROC-AUC
為排序型指標,不需要機率門檻轉換)。

**Submission 檔名與格式**(來源:`best.submission`、`competition.id_column`/`target_column`):
檔名 `sub_blend_0.89989_20260703_205132.csv`,欄位為 id 欄 `id` 與目標欄 `booking_status`
(輸出為取消機率,而非硬分類標籤,符合 ROC-AUC 評分需求)。

## 6. 評估指標

**指標定義**:ROC-AUC 衡量模型將正類(取消)排在隨機一筆負類(未取消)之前的機率,數值介於
0.5(隨機)到 1(完美排序)之間,越高越好(maximize)。

分數總表(來源:`best.*`、`leaderboard`):

| 項目 | ROC-AUC |
|------|---------|
| LGB(base model,Optuna 調參) | 0.899215 |
| XGB(base model) | 0.898765 |
| CAT(base model) | 0.896709 |
| Ensemble(最終,exp 6) | 0.899893 |
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
| 4 | 2026-07-03T20:39:32 | 0.899891 | v2 |
| 5 | 2026-07-03T20:48:03 | 0.899722 | v2 |
| 6 | 2026-07-03T20:51:32 | 0.899893 | v2 |

**突破點敘述**:本場有兩個階段的躍升。

*Phase A(exp 1–3)*:experiment 2(17 原始 + 14 工程特徵,含月份/日期的週期性編碼與
價格交乘項)分數為 0.89788,低於通用 baseline 的 0.89882,呈現退步訊號(來源:exp 2 之
`notes`:"Delta vs generic baseline 0.89882: -0.00094")。據此反思,experiment 3 將
工程特徵精簡為 8 個(移除週期性月份/日期編碼、price_per_night、special×price 交乘項),
分數回升至 0.899395(來源:exp 3 之 `notes`:"+0.00057")。

*Phase B(exp 4–6,自我改進迭代;特徵集與 CV 凍結,每輪只改一件事)*:
- exp 4(Round 1,**主要躍升**):Optuna 調 LGB(50 trials、TPE、fold-0 代理目標),
  LGB 單模 0.898824 → 0.899215,混合 0.899395 → 0.899891(來源:exp 4 之 `notes`:
  "+0.000496")。
- exp 5(Round 2,退步、棄用):同法調 XGB,單模微升至 0.89886,但混合退至 0.899722
  (來源:exp 5 之 `notes`:"-0.000169")——調參使 XGB 與 LGB 樹結構趨同、多樣性下降。
- exp 6(Round 3,最終):凍結 exp 4 基模型、僅改混合層(0.01 步長權重網格),達 0.899893
  (來源:exp 6 之 `notes`:"+0.000002",噪音等級;rank-average 混合 0.899884 更差)。
  連同 Round 2 視為兩輪無實質改進,依停止準則收手。

以下為以 facts.json 內各實驗 `score` 重新計算之衍生差值,僅用於驗證上述 notes 描述之方向
一致,不作為新事實:

```
0.899893 (exp6 最終)  − 0.89882  (exp1 baseline)     = +0.001073
0.899893 (exp6 最終)  − 0.899395 (exp3 Phase A 最佳)  = +0.000498
0.899891 (exp4)       − 0.899395 (exp3)              = +0.000496
0.899722 (exp5)       − 0.899891 (exp4)              = −0.000169
0.899893 (exp6)       − 0.899891 (exp4)              = +0.000002
0.89788  (exp2)       − 0.89882  (exp1)              = −0.00094
```

`unparsed`:facts.json 之 `unparsed` 清單為空,無法解析之紀錄:無。

## 8. 重現指令

```bash
cd /home/tjyen/ai_agents/kaggle

# Stage 1: EDA
uv run python3 competitions/playground-series-s3e7/scripts/eda.py

# Stage 2-3 (Phase A): feature engineering + modeling (LGB/XGB/CAT, 5-fold
# StratifiedKFold, OOF weight-searched blend) → exp 3
uv run python3 competitions/playground-series-s3e7/scripts/train.py

# Phase B Round 1: Optuna-tuned LGB (50 trials, fold-0 proxy) + blend → exp 4
uv run python3 competitions/playground-series-s3e7/scripts/optuna_lgb.py

# Phase B Round 3: ensemble refinement (fine 0.01 grid + rank blend) → exp 6 (final)
uv run python3 competitions/playground-series-s3e7/scripts/blend_refine.py

# Report generation (this file)
uv run python3 .claude/skills/kaggle-report/assets/collect.py playground-series-s3e7
uv run python3 .claude/skills/kaggle-report/assets/verify_report.py \
    competitions/playground-series-s3e7/REPORT.md competitions/playground-series-s3e7/facts.json
bash .claude/skills/kaggle-report/assets/md2pdf.sh competitions/playground-series-s3e7/REPORT.md

# Not submitted to Kaggle in this run (no credentials available in this environment).
# To submit the final submission file to the leaderboard:
export KAGGLE_API_TOKEN=$(python3 -c "import json; print(json.load(open('/home/tjyen/.kaggle/kaggle.json'))['key'])")
uv run kaggle competitions submit -c playground-series-s3e7 \
    -f competitions/playground-series-s3e7/submissions/sub_blend_0.89989_20260703_205132.csv \
    -m "Optuna-tuned LGB + XGB + CAT blend (0.54/0.40/0.06), OOF 0.899893"
```
