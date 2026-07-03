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
最佳實驗(experiment_id 8)使用 22 個特徵(8 原始 + 14 工程特徵,與 experiment_id 2 相同
特徵清單)。

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

最佳實驗(experiment_id 8)是 7 成員加權 blend:「7-way blend: Round-3 pool + CAT_seed2 +
LGB_tuned_seed3 (seed-bag expansion)」。7 個成員 = experiment_id 7 的 5 成員 pool(orig
LGB / orig XGB / orig CAT / Optuna 調參 LGB / 調參 LGB 第 2 seed)再加 2 個 seed-bag 成員。
各成員單模 OOF 分數(來源:`experiments[]` 中 experiment_id 6/7/8 的 `base_models`):

| Model | RMSE | 訓練時間(s) | 來源實驗 |
|-------|------|--------------|----------|
| LGB(orig,正則化) | 12.11061 | 6.6 | experiment_id 6 |
| XGB(orig,正則化) | 12.12086 | 6.6 | experiment_id 6 |
| CAT(orig) | 12.07459 | 2.7 | experiment_id 6 |
| LGB_tuned(Optuna) | 12.12074 | 371.6 | experiment_id 6 |
| LGB_tuned_seed2(seed 1042) | 12.10843 | 4.4 | experiment_id 7 |
| CAT_seed2(seed 1042) | 12.07852 | 2.7 | experiment_id 8 |
| LGB_tuned_seed3(seed 2042) | 12.0977 | 6.0 | experiment_id 8 |

**Ensemble 權重**(來源:`best.ensemble.weights`,方法
`oof_weight_search(dirichlet8000+coord_descent_fine)`):

| Model | 權重 |
|-------|------|
| LGB | 0.0 |
| XGB | 0.0 |
| CAT | 0.478 |
| LGB_tuned | 0.0 |
| LGB_tuned_seed2 | 0.127 |
| CAT_seed2 | 0.265 |
| LGB_tuned_seed3 | 0.129 |

Ensemble 後最終分數(來源:`best.score`):**12.070034**。

**選型理由**(敘述,參考 STATUS.md 脈絡):七個成員皆為梯度提升樹,適合小型表格資料。
CatBoost 的 ordered boosting 正則化機制在本資料集上單模型最強(12.07459),其原 seed 與
第 2 seed 的合計權重(衍生計算,置於 code block):

```
CAT + CAT_seed2 權重合計: 0.478 + 0.265 = 0.743
LGB_tuned_seed2 + LGB_tuned_seed3 權重合計: 0.127 + 0.129 = 0.256
```

值得注意的是:調參 LGB 的「原 seed」(LGB_tuned,單模 12.12074)與 orig LGB/XGB 權重皆為
0——本輪增益完全來自 seed bagging(以不同 random seed 重訓同超參模型再平均)的變異數
縮減,而非調參本身。

## 4. 訓練規格

**CV 方案**(來源:`best.cv`):

| 欄位 | 值 |
|------|-----|
| strategy | 5fold_stratified_strength_decile |
| seed | 42 |
| n_splits | 無紀錄(best.cv 未記錄此欄位) |

**Objective 與關鍵超參**:facts.json 的 `best.base_models` 未提供結構化的 `params` 欄位。
orig 三模型超參記載於 experiment_id 2 的 notes(逐字節錄):

> "Regularized LGB(num_leaves=15,depth5,L1=2/L2=4)/XGB(depth4,L1=2/L2=4) + CatBoost(depth6,l2=6)
> with 8 engineered features (log_age, water/binder ratios) vs baseline exp#1 (generic untuned
> blend, RMSE 12.54287, CatBoost 100% weight)."

Optuna 調參 LGB 的超參記載於 experiment_id 6 的 `base_models[3].note`(逐字節錄):

> "Optuna 60-trial full-5fold-CV objective tuned LGB; params={'learning_rate':
> 0.019795655587585677, 'num_leaves': 11, 'max_depth': 3, 'min_child_samples': 31,
> 'subsample': 0.6018504151952752, 'colsample_bytree': 0.5006835428888052,
> 'reg_alpha': 0.1120861378640153, 'reg_lambda': 0.3020804107290036, 'n_estimators': 3000}"

seed-bag 成員與原成員超參完全相同,僅更換 random seed(1042 / 2042;見 experiment_id 7/8
的 `base_models[].note`)。

**為何用此 CV**:strategy 名稱顯示此為「以 Strength 十分位數分層」的 5-fold
StratifiedKFold(`5fold_stratified_strength_decile`)。相較於單純 KFold,對迴歸目標分層
可讓每個 fold 的目標分布更一致,在資料量較小時能降低 fold 間變異——這與 baseline
(experiment_id 1)僅記錄 `cv.scheme = "5fold"`(未分層)形成對比。

## 5. 推論程序

**後處理**:`best.postprocess` 未記錄 → 無後處理紀錄。

**Submission 格式**(來源:`best.submission`、`competition.id_column`/`target_column`):

| 欄位 | 值 |
|------|-----|
| Submission 檔名 | sub_blend_12.07003_20260703_235120.csv |
| ID 欄位 | id |
| 目標欄位 | Strength |

## 6. 評估指標

**指標定義**:RMSE(Root Mean Squared Error)是預測值與真實值差異平方之平均值,再開根號,
數值與目標欄位同單位,對大誤差的懲罰重於 MAE。

**分數總表**(來源:`best.*`、各實驗紀錄):

| 項目 | RMSE |
|------|------|
| Baseline blend(experiment_id 1) | 12.54287 |
| 3-way blend(experiment_id 2,前一輪最佳) | 12.073474 |
| Phase B Round 1:dup-smoothing(experiment_id 5) | 12.081216 |
| Phase B Round 2:+Optuna LGB(experiment_id 6) | 12.073474 |
| Phase B Round 3:+seed-bag 調參 LGB(experiment_id 7) | 12.071429 |
| **最佳 Ensemble(experiment_id 8,7-way seed-bag)** | **12.070034** |

`facts.missing` 記錄 `["leaderboard"]`,即本場 **未提交至 Kaggle,無 Public/Private LB
分數**,故無法計算 CV↔LB gap(僅當 leaderboard 存在時才計算,依規格第 6 節)。

以下為 baseline / 前一輪最佳與最佳 CV 分數的改善量(衍生計算,數字皆逐字取自
facts.json,置於 code block 中):

```
12.54287 (baseline blend_score, experiment_id 1)
- 12.070034 (best score, experiment_id 8)
= 0.472836

相對改善: 0.472836 / 12.54287 * 100 = 3.769758...%

Phase B 迭代增益:
12.073474 (experiment_id 2/4, 前一輪最佳)
- 12.070034 (experiment_id 8)
= 0.003440
```

## 7. 實驗軌跡

**逐實驗分數表**(來源:`trajectory`):

| experiment_id | timestamp | score | source_format |
|---------------|-----------|-------|----------------|
| 1 | 2026-07-03T12:06:32 | 12.54287 | generic_batch |
| 2 | 2026-07-03T19:10:07 | 12.073474 | v2 |
| 3 | 2026-07-03T19:10:47 | 12.094826 | v2 |
| 4 | 2026-07-03T19:11:24 | 12.073474 | v2 |
| 5 | 2026-07-03T23:41:35 | 12.081216 | v2 |
| 6 | 2026-07-03T23:49:50 | 12.073474 | v2 |
| 7 | 2026-07-03T23:49:55 | 12.071429 | v2 |
| 8 | 2026-07-03T23:51:20 | 12.070034 | v2 |

**突破點敘述**:主要分數躍升發生在 experiment_id 2(timestamp 2026-07-03T19:10:07),
score 由 baseline 的 12.54287 降至 12.073474。依 `experiments[2].notes`(逐字節錄同第 4
節),此次變動同時改了兩件事:(a) 對 LGB/XGB 加入明確正則化超參數
(num_leaves=15/depth5/L1=2/L2=4 與 depth4/L1=2/L2=4),(b) 從 8 個原始特徵擴充為 22 個
工程特徵(含 log_age、water/binder 比值等)。experiment_id 3 是延伸嘗試(加入 3 個
interaction 特徵,`n_features` 從 22 增至 25),score 反而變差(12.094826),隨後
experiment_id 4 以相同 22 特徵重新執行,score 完全重現為 12.073474。

**Phase B 自我改進迭代(experiment_id 5–8)**:每輪僅改一件事,fold/seed/特徵不變:
- Round 1(experiment_id 5):fold-safe 重複列群組目標平滑(訓練目標改為訓練 fold 內同
  配方群組的平均 Strength;驗證目標與指標不動)→ 12.081216,劣於 12.073474,棄用。依
  notes:GBDT 平方損失本就隱含以群組均值擬合重複列,顯式平滑無新資訊。
- Round 2(experiment_id 6):Optuna(TPE 60 trials,完整 5-fold CV 為目標函式,實際
  371.6s)調參 LGB 加入 pool 為第 4 成員 → 12.073474(持平;調參 LGB 單模 12.12074 劣於
  orig LGB 12.11061,權重搜尋給 0)。
- Round 3(experiment_id 7):調參 LGB 以第 2 個 seed(1042)重訓、加為第 5 成員 →
  **12.071429**(改善;權重 CAT 0.773 / LGB_tuned_seed2 0.227)。
- Round 4(experiment_id 8):seed-bag 權重最大的 CatBoost(seed 1042)+ 第 3 個調參 LGB
  seed(2042),7-way 權重搜尋 → **12.070034**(最終最佳)。

整個 Phase B 的增益(第 6 節 code block 中的 12.073474 − 12.070034 衍生計算)完全來自
seed bagging,調參與去噪(dup-smoothing)本身皆未直接貢獻——與 STATUS.md 的標籤噪音
天花板診斷一致:分數已貼近噪音上限時,對獨立 seed 的模型取平均是僅存的「免費」改善方向。

`facts.unparsed` 為空陣列,無「無法解析之紀錄」。

## 8. 重現指令

以下命令序列參考 STATUS.md 的 Reproduce 節,並依 `competitions/playground-series-s3e9/scripts/`
實際檔名組成。執行目錄為專案根目錄,需先安裝 `uv`。

```bash
cd /home/tjyen/ai_agents/kaggle

# 1. EDA
uv run python3 competitions/playground-series-s3e9/scripts/eda.py

# 2. 特徵工程 + 訓練 + 5-fold CV + OOF weight-search blend(3-way,exp #2)
#    (features.py 由 train.py 匯入呼叫,無需獨立執行步驟)
uv run python3 competitions/playground-series-s3e9/scripts/train.py

# 3. Phase B 自我改進迭代(依序執行;round 4 讀取 round 2/3 產生的 npz checkpoint)
uv run python3 competitions/playground-series-s3e9/scripts/train_dup_smooth.py      # round 1(負面結果)
uv run python3 competitions/playground-series-s3e9/scripts/train_optuna_pool.py     # rounds 2+3
uv run python3 competitions/playground-series-s3e9/scripts/train_round4_seedbag.py  # round 4 → 最佳 submission

# 4. 提交至 Kaggle(本次執行未提交;需要有效 KAGGLE_API_TOKEN 憑證)
export KAGGLE_API_TOKEN=$(cat ~/.kaggle/kaggle_api_token.txt | tr -d '[:space:]')
uv run kaggle competitions submit -c playground-series-s3e9 \
  -f competitions/playground-series-s3e9/submissions/sub_blend_12.07003_20260703_235120.csv \
  -m "<msg>"
```
