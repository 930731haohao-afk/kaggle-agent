# 競賽分析報告:playground-series-s3e7

> 產生方式:kaggle-report skill(數字來自 facts.json,敘述由 agent 撰寫)
> 素材等級:full | 產生日期:2026-07-04(Phase G-1a 樹搜尋成果入帳後更新)

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

**facts.json 目前的 best 是 experiment_id=7——一筆樹搜尋(tree-search)結果**(見 3b 節),
而非本節原本描述的線性迭代最終回合(exp 6)。兩者皆完整說明。

### 3a. 線性迭代最佳(experiment_id=6,3 輪 Phase B 迭代之終點)

由三個基模型加權混合而成(來源:`experiments[5].base_models`,即 exp 6):

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

### 3b. 樹搜尋最佳(best, experiment_id=7)——本次更新新增

Phase F-2(harness v3 驗證跑,2026-07-04)在 `experiments_tree_v3.json`(全新樹,D-3 的
22-節點 v2 掃描樹 `experiments_tree.json` 未被觸碰)的 node #48 找到本場目前最佳 ROC-AUC:
rank-space 38-member Dirichlet(k=800)+coordinate-ascent blend(權重搜尋將 29/38 個成員
歸零,9 個非零權重存活者,來源:`best.base_models`):

| 成員 | 權重 | 備註 |
|------|------|------|
| SEEDBAG | 0.358 | — |
| XGBDIV family(4 個節點合計) | 0.459 | — |
| EXPL_BOUND2 | 0.114 | solo 0.897541(boundary-push 衍生 depth-2 LGB);solo 較弱但第 3 大權重 |
| root | 0.038 | — |
| FEATPRUNE-child | 0.018 | — |
| XGBHAND-child | 0.013 | — |

**Ensemble**(best.ensemble):method = "harness_v3 mandatory explore-burst kitchen-sink
blend: rank-space dirichlet(k=800)+coordinate-ascent over the FULL 38-member solo pool",
score = **0.900455**。

**選型理由/來源說明**(best.notes):此為 Phase F-2 樹搜尋結果,不是線性迭代第 7 輪,也不是
D-3 的 22-節點 v2 掃描延伸。全部超越 v2 重現高原(0.900054)的增益皆來自 harness v3 的
**強制 explore-burst 機制**(kitchen-sink mega-blend),於 eval 39(所有第一代 lineage
三振 plateau 後)自動注入。誠實風險註記(來源:best.notes):權重直接對全 OOF 擬合、無巢狀
驗證,相對 v2 的 ~0.0002 增益帶有 OOF 權重過擬風險——「burst mega-blend 優於手工成長 blend」
的方向性結論才是穩健的部分,第 4 位小數不是。

> **重要澄清**:best.notes 明確記載這是 **OOF-only 搜尋結果——未產生任何 test 預測,亦
> 未提交至 Kaggle**(facts.json 本筆無 submission 欄位)。facts.best 是以 OOF score 最大者
> 選出(本場 metric 為 roc_auc,maximize),與是否已提交至 Kaggle 無關;本場所有實驗(含
> 實驗 7)皆未提交至 Kaggle(見第 5/6 節)。

## 4. 訓練規格

CV 方案(來源:`best.cv`,experiment_id=7 為現在的 best;7 個實驗全程固定同一方案,分數可
直接比較):

| scheme | n_splits | seed | strategy |
|--------|----------|------|----------|
| 5fold | 5 | 42 | StratifiedKFold(booking_status) |

**為何用此 CV**:目標欄位為二元分類且存在類別不平衡,直接對 `booking_status` 做
StratifiedKFold 可確保每一折的正負類別比例一致,避免因某一折類別分佈偏移而使 CV 分數
不穩定或不可信。樹搜尋(experiment_id=7)沿用完全相同的 CV 折。

各基模型關鍵超參:facts.json 本筆 best(experiment_id=7,樹搜尋)的 base_models 僅附
score/weight/note,**未附 params 欄位——無紀錄**,不臆測。以下為線性迭代 exp 6 記錄的
參考超參(來源:`experiments[5].base_models[].params`,皆有紀錄;LGB 為 Optuna 調參結果,
數值四捨五入至 3–4 位小數):

| 模型 | 關鍵超參(exp6 紀錄) |
|------|----------|
| LGB(Optuna) | objective=binary, metric=auc, n_estimators=3000, learning_rate≈0.068, num_leaves=178, max_depth=3, min_child_samples=47, subsample≈0.889, colsample_bytree≈0.532, reg_alpha≈2.14, reg_lambda≈0.0115, random_state=42, n_jobs=-1, verbose=-1 |
| XGB | objective=binary:logistic, n_estimators=3000, learning_rate=0.03, max_depth=6, min_child_weight=5, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=1.0, random_state=42, n_jobs=-1, eval_metric=auc, early_stopping_rounds=150 |
| CAT | loss_function=Logloss, eval_metric=AUC, iterations=4000, learning_rate=0.03, depth=7, l2_leaf_reg=5.0, random_seed=42, thread_count=-1, verbose=False |

值得注意:Optuna 找到的 LGB 最佳解是「淺而強正則」(max_depth=3、較高學習率、
reg_alpha≈2.14),num_leaves=178 在 depth=3 之下實際不起作用;這與跨競賽經驗
「小/中型資料獎勵正則化而非容量」一致。

## 5. 推論程序

**後處理步驟**:facts.json 之 `best`(experiment_id=7,樹搜尋)未記錄 `postprocess` 欄位 →
**無後處理紀錄**(ROC-AUC 為排序型指標,不需要機率門檻轉換)。

**Submission 檔名(best.submission):無紀錄**——experiment_id=7 是樹搜尋(OOF-only)結果,
facts.json 本筆未附 submission 欄位,未產生 test 預測、未提交 Kaggle。欄位格式仍為
`competition.id_column`/`target_column`:id 欄 `id`、目標欄 `booking_status`。

線性迭代終點(exp 6)有提交檔案供參考:`sub_blend_0.89989_20260703_205132.csv`(來源:
`experiments[5].submission`),對應 OOF 0.899893,非目前 best 的 0.900455。

## 6. 評估指標

**指標定義**:ROC-AUC 衡量模型將正類(取消)排在隨機一筆負類(未取消)之前的機率,數值介於
0.5(隨機)到 1(完美排序)之間,越高越好(maximize)。

分數總表(來源:`best.*`、`leaderboard`):

| 項目 | ROC-AUC |
|------|---------|
| LGB(base model,Optuna 調參) | 0.899215 |
| XGB(base model) | 0.898765 |
| CAT(base model) | 0.896709 |
| Ensemble(線性迭代終點,exp 6) | 0.899893 |
| 樹搜尋 Ensemble(exp 7,facts.best) | **0.900455** |
| Public LB | 無紀錄 |
| Private LB | 無紀錄 |

**CV↔LB gap**:facts.json 之 `leaderboard` 為空(`missing` 清單包含 `leaderboard`)——本場
僅完成本機 CV,尚未提交至 Kaggle 排行榜,故無法計算 CV↔LB gap。

exp7(樹搜尋 best)相對 exp6(線性迭代終點)之改善:

```
exp7 - exp6:  0.900455 - 0.899893 = 0.000562   (絕對改善)
```

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
| 7 | 2026-07-04T11:59:43 | 0.900455 | v2 |

**突破點敘述**:本場有三個階段的躍升。

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
0.900455 (exp7 樹搜尋) − 0.899893 (exp6 線性迭代終點) = +0.000562
```

*Phase F-2 樹搜尋(exp 7,本次更新新增,facts.best)*:experiment_id=7 不是線性迭代的
延續回合,而是 Phase F-2(2026-07-04)以 harness v3 執行的**樹搜尋(tree-search)**結果——
來源 `experiments_tree_v3.json`(全新樹,D-3 的 22-節點 v2 掃描樹 `experiments_tree.json`
未被觸碰)的 node #48(60-節點預算,硬上限;最佳解出現於 eval 46,wall 1593s)。分數由
exp 6 的 0.899893 升至 0.900455(見上方程式,絕對改善 +0.000562)。全部超越 v2 重現高原
(0.900054)的增益皆來自 harness v3 的強制 explore-burst 機制(kitchen-sink mega-blend)。
**誠實 CV-only 警語**:此結果為 OOF-only 搜尋產物——tree_search harness 未產生任何 test
預測檔,facts.json 本筆亦無 submission 欄位,**未提交至 Kaggle**;不可與 exp 6 實際提交的
submission 檔案混淆(見第 5 節)。完整節點鏈、policy 行為誠實記錄(phase machine/burst
payoff/boundary-push/dedup/reopen-blend)、與操作面問題(3 次重啟)見
`competitions/playground-series-s3e7/STATUS.md`〈Appendix: Phase F-2 harness v3
validation run〉。

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

# Phase B Round 3: ensemble refinement (fine 0.01 grid + rank blend) → exp 6 (linear-iteration end)
uv run python3 competitions/playground-series-s3e7/scripts/blend_refine.py

# Phase F-2 tree-search v3 validation run (exp 7, current best; resumable; tree state
# in experiments_tree_v3.json, independent of the v2-sweep experiments_tree.json)
uv run python3 tree_search/run_s3e7_v3.py

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
