# 競賽分析報告:playground-series-s3e9

> 產生方式:kaggle-report skill(數字來自 facts.json,敘述由 agent 撰寫)
> 素材等級:full | 產生日期:2026-07-06
> 本報告所有數字皆出自 facts.json,經 verify_report.py 驗證。

## 1. 競賽目的

**What**:依混凝土配方中八種成分用量(水泥、爐渣、飛灰、水、減水劑、粗骨材、細骨材)與
養護天數(`AgeInDays`)預測混凝土抗壓強度(`Strength`),為連續數值輸出的迴歸問題,即
經典 Concrete Compressive Strength 資料集的 Kaggle Playground 版本。

**Why**:評估指標為 **RMSE(minimize)**。抗壓強度為工程安全量,嚴重錯估的後果(結構
風險、材料浪費)遠重於小誤差;RMSE 以原始單位呈現誤差並對大誤差施以平方懲罰,能引導
模型優先壓制離譜預測,適合此任務。

| 項目 | 值 |
|------|-----|
| 競賽 | playground-series-s3e9 |
| 問題型別 | regression |
| 評估指標 | rmse(minimize) |
| 目標欄位 | Strength |
| 素材等級 | full |

## 2. 使用工具與環境

| 工具 | 用途 |
|------|------|
| LightGBM / XGBoost / CatBoost | 三個梯度提升樹基模型,構成 exp1–8 加權 blend 主體;seed-bag 變體為 exp7/exp8 的新增成員 |
| Optuna | exp6 對 LGB 做 TPE 超參搜尋(60 trials,完整 5-fold CV 為目標函式,371.6s) |
| 自建樹搜尋 harness(v1/v2) | Phase C-2a 單模節點空間搜尋、Phase E-1 ensemble-default 節點空間「復仇戰」;結果為 OOF-only,不進 experiments.json |
| 5-fold CV 框架(scikit-learn) | 以 `Strength` 十分位分層的 StratifiedKFold,5 折交叉驗證 |
| uv | Python 套件與虛擬環境管理,所有腳本皆以 `uv run` 執行 |

本場工具鏈組合邏輯:Claude Code(LLM)負責決策——診斷重複配方列的標籤噪音結構、提出並
否決假設(交互特徵、重複群組平滑)、決定每輪只改一件事與何時停損;Auto-ML 工具(Optuna、
樹搜尋 harness)負責系統化執行超參搜尋與成員/權重組合空間探索,兩者分工互補。

## 3. 流程(how):五大元件

### 3.1 資料規格

| 項目 | 值 |
|------|-----|
| train 列數 | 5,407 |
| test 列數 | 3,605 |
| 原始欄位數 | 8 |
| 特別規則摘要 | 禁外部資料/禁預訓練模型/禁網路存取;每日提交上限 5 次 |

8 個特徵全為數值型、無類別欄位;train/test 皆無缺失值。目標 `Strength` 近似對稱
(mean 35.45、median 33.95、skew 0.38),非 log 轉換候選。`AgeInDays` 為最強關聯特徵
(pearson 0.334、spearman 0.604)且右偏(skew 2.75),支持 log 養護時間類特徵;爐渣、
飛灰、減水劑為零膨脹欄位(零值列數 3,166 / 3,927 / 3,143),屬配方選用而非缺失。

> **語意澄清(重複列口徑)**:facts.eda 記錄 train 重複列 2,401——此為「特徵完全相同、
> 但量測 `Strength` 不同」的非首見重複列數(佔比 44.4%,exp5 notes 之 dup_frac 0.4441);
> STATUS.md 的「約 56% 列屬於某重複群組」是同一結構的含首列口徑,兩者不矛盾。此為量測/
> 重抽樣噪音而非洩漏,構成本場 CV 分數的不可約噪音上限,並懲罰高容量模型。

train/test 分佈接近(平均值差異最大者為 `AgeInDays` 的 -5.02% 與 `BlastFurnaceSlag` 的
-4.79%),無 covariate shift 疑慮;facts.eda 無高共線特徵對紀錄。

### 3.2 實驗總表

> **語意澄清(本報告全文適用)**:本場指標為連續 RMSE,無取整/門檻類後處理,「原始 OOF」
> 即「決策分數」,兩欄恆相等;且全部決策皆以本機 OOF 為準(未提交 Kaggle,見 3.4 節)。

| exp | 階段 | 模型/成員 | 特徵數 | 原始 OOF | 決策分數 | 採納 |
|-----|------|-----------|--------|----------|----------|------|
| 1 | Baseline(通用批次) | LGB/XGB/CAT(0/0/1.0) | 8 | 12.54287 | 12.54287 | 基線參照 |
| 2 | Phase A(正則化+特徵工程) | LGB/XGB/CAT(0.15/0/0.85) | 22 | 12.073474 | 12.073474 | 是,Phase A 最佳 |
| 3 | 迭代(交互特徵探索) | 同 exp2 成員,+3 交互特徵 | 25 | 12.094826 | 12.094826 | 否,變差後還原 |
| 4 | 還原確認重跑 | 同 exp2 | 22 | 12.073474 | 12.073474 | 是,逐位元重現 exp2 |
| 5 | Phase B R1(dup 群組平滑) | LGB/XGB/CAT(0.05/0.1/0.85) | 22 | 12.081216 | 12.081216 | 否,變差後還原 |
| 6 | Phase B R2(+Optuna LGB) | 4-way(調參 LGB 權重 0) | 22 | 12.073474 | 12.073474 | 持平,成員保留 |
| 7 | Phase B R3(seed-bag 調參 LGB) | 5-way(CAT 0.773/seed2 0.227) | 22 | 12.071429 | 12.071429 | 是 |
| 8 | Phase B R4(seed-bag 擴充) | 7-way seed-bag blend | 22 | **12.070034** | **12.070034** | 是,本場最佳 |

**best 成員表(exp8,7-way blend;權重搜尋 dirichlet8000+coord_descent_fine)**

| 成員 | 權重 | solo 分數 | 備註 |
|------|------|-----------|------|
| LGB(orig,正則化) | 0.0 | 12.11061 | 權重歸零 |
| XGB(orig,正則化) | 0.0 | 12.12086 | 權重歸零 |
| CAT(orig) | 0.478 | 12.07459 | 最強單模,權重最大 |
| LGB_tuned(Optuna) | 0.0 | 12.12074 | 調參原 seed 權重歸零 |
| LGB_tuned_seed2 | 0.127 | 12.10843 | 調參 LGB 之 seed 1042 變體 |
| CAT_seed2 | 0.265 | 12.07852 | CAT 之 seed 1042 變體 |
| LGB_tuned_seed3 | 0.129 | 12.0977 | 調參 LGB 之 seed 2042 變體 |

選型脈絡:小型噪音資料獎勵正則化而非容量——exp1 未正則化時 LGB(13.2055)/XGB
(12.98156)過擬合嚴重、權重全數歸零由 CAT 獨拿;exp2 對兩者施加淺深度+強 L1/L2 後
LGB 才取回 0.15 權重。線性迭代(Phase B)四輪中,去噪(exp5)與調參(exp6)皆未改善,全部增益
來自 seed bagging 的變異數縮減:調參 LGB 原 seed 權重為 0,其 seed 變體卻合計拿下約
四分之一權重,CAT 的 seed-bag(exp8)則貢獻單輪最大增益。

### 3.3 訓練規格表

| exp | CV 方案 | folds | seed |
|-----|---------|-------|------|
| 1 | 5fold | 5 | 無紀錄 |
| 2–8 | 5fold_stratified_strength_decile(StratifiedKFold on `Strength` 十分位) | 5 | 42 |

目標為連續值且資料量小、噪音大,對 `Strength` 十分位分層可讓每折目標分佈一致、降低折間
變異;exp2 起 CV 全程固定,八個實驗分數可直接比較,樹搜尋兩輪亦沿用完全相同的折。

Objective 一律為 RMSE。關鍵超參(節錄自 exp2/exp6 notes):

```
LGB(手設正則化): num_leaves=15, max_depth=5, reg_alpha(L1)=2, reg_lambda(L2)=4
XGB(手設正則化): max_depth=4, reg_alpha(L1)=2, reg_lambda(L2)=4
CAT(手設):      depth=6, l2_leaf_reg=6
LGB_tuned(Optuna 60-trial): learning_rate≈0.0198, num_leaves=11, max_depth=3,
    min_child_samples=31, subsample≈0.602, colsample_bytree≈0.501,
    reg_alpha≈0.112, reg_lambda≈0.302, n_estimators=3000
seed-bag 成員超參與原成員完全相同,僅換 random seed(1042 / 2042)
```

### 3.4 推論表

| exp | 後處理 | submission 檔 | 已提交 |
|-----|--------|---------------|--------|
| 1 | 無後處理紀錄 | sub_generic_12.54287_20260703_120632.csv | 否 |
| 2 | 無後處理紀錄 | sub_blend_12.07347_20260703_191007.csv | 否 |
| 3 | 無後處理紀錄 | sub_blend_12.09483_20260703_191047.csv | 否 |
| 4 | 無後處理紀錄 | sub_blend_12.07347_20260703_191124.csv | 否 |
| 5 | 無後處理紀錄 | 無紀錄 | 否 |
| 6 | 無後處理紀錄 | 無紀錄 | 否 |
| 7 | 無後處理紀錄 | 無紀錄 | 否 |
| 8 | 無後處理紀錄 | sub_blend_12.07003_20260703_235120.csv | 否 |

`id_column = id`、`target_column = Strength`。本場無任何後處理紀錄(迴歸原值直接輸出)。
所有 submission 檔皆為本機產出、未上傳 Kaggle(本次執行無 API 憑證);exp3 之劣化
submission 檔記錄後已自 submissions/ 刪除(STATUS.md),僅保留最佳與基線檔案。

### 3.5 評估指標 / 排行榜

指標定義:RMSE = 預測誤差平方平均之平方根,單位與 `Strength` 相同,對大誤差懲罰重於 MAE。

| 項目 | OOF RMSE |
|------|----------|
| LGB(exp2 成員,正則化) | 12.11061 |
| XGB(exp2 成員,正則化) | 12.12086 |
| CAT(exp2 成員) | 12.07459 |
| Ensemble(exp2,Phase A 最佳) | 12.073474 |
| Ensemble(exp8,本場最佳) | **12.070034** |
| Public / Private LB | 無紀錄(未提交) |

本場無排行榜表(leaderboard 為 null、列於 facts.missing),CV↔LB gap 無法計算。全場
唯一分數錨點為同一固定 CV 下的 OOF RMSE。整體與 Phase B 改善量:

```
exp1 → exp8:12.54287 − 12.070034 = 0.472836,相對改善 0.472836 / 12.54287 = 3.7698%
Phase B 增益:12.073474(exp2/4)− 12.070034(exp8)= 0.003440
```

## 4. 實驗軌跡

**實驗階段對照表**(內部代號使用前先定義,R-W8)

| 代號 | 白話名稱 | 對應層級 |
|------|----------|----------|
| Phase A | kaggle-agent skill 六階段首跑 | tier2 |
| Phase B | self-improvement 線性迭代 | tier3 |
| Phase C-2a | 樹搜尋原型(harness v1,單模節點空間搜尋) | tier4 相關(OOF-only,未入 experiments.json) |
| Phase E-1 | 樹搜尋執行(harness v2,ensemble-default 復仇戰) | tier4 相關(OOF-only,未入 experiments.json) |

| exp | 時間 | 決策分數 | 階段 | 一句話摘要 |
|-----|------|----------|------|------------|
| 1 | 2026-07-03T12:06:32 | 12.54287 | Baseline(通用批次) | 8 原始特徵未調參 blend,LGB/XGB 過擬合權重歸零 |
| 2 | 2026-07-03T19:10:07 | 12.073474 | Phase A | 正則化+22 工程特徵,全場最大單筆躍升 |
| 3 | 2026-07-03T19:10:47 | 12.094826 | 迭代探索 | 加入 3 個交互特徵反而變差,還原 |
| 4 | 2026-07-03T19:11:24 | 12.073474 | 還原確認 | 還原後重跑,逐位元重現 exp2,確認決定論 |
| 5 | 2026-07-03T23:41:35 | 12.081216 | Phase B R1 | 重複群組目標平滑無效,還原 |
| 6 | 2026-07-03T23:49:50 | 12.073474 | Phase B R2 | Optuna 調參 LGB 入 pool,權重 0 持平 |
| 7 | 2026-07-03T23:49:55 | 12.071429 | Phase B R3 | seed-bag 調參 LGB 為第 5 成員,改善 |
| 8 | 2026-07-03T23:51:20 | **12.070034** | Phase B R4 | seed-bag CAT + 第 3 個調參 LGB seed,7-way 本場最佳 |

- **突破點 1(exp1→exp2)**:同時施加明確正則化(淺深度+L1/L2)與 22 個工程特徵
  (log_age、water/binder 比值等),12.54287→12.073474,佔全場改善的絕大部分。
- **突破點 2(exp3/exp5 的負面結果)**:交互特徵(樹模型本可自行學得,徒增過擬合面)與
  重複群組平滑(平方損失本就隱含擬合群組均值)先後失敗,確立「標籤噪音上限」診斷——
  正確的診斷不保證顯式修正有效。
- **突破點 3(exp6→exp7→exp8)**:Phase B 全部增益來自 seed bagging(−0.00204、
  −0.00140,報酬遞減);在噪音上限處,對獨立 seed 模型取平均是僅存的「免費」方向。

facts.unparsed 為空陣列,無法解析之紀錄:無。

## 5. 效能對照:四層消融

| 層級 | 配置 | 分數 | 相對改善 |
|------|------|------|----------|
| tier1 | 基線(Claude Code 直接執行,未引入 skill;exp1 通用批次 blend) | 12.54287 | —(基線) |
| tier2 | + kaggle-agent skill 六階段流程(exp2 Phase A) | 12.073474 | 見下方算式 |
| tier3 | + self-improvement 線性迭代(exp8,Phase B 四輪終點) | **12.070034** | 見下方算式 |
| tier4 | + 樹搜尋(無樹搜尋節點超越線性最佳,與 tier3 同為 exp8) | **12.070034** | 0%(樹搜尋追平,未超越) |

```
tier1→tier2: 12.54287 − 12.073474 = 0.469396,相對改善 0.469396 / 12.54287 = 3.7423%
tier2→tier3: 12.073474 − 12.070034 = 0.003440,相對改善 0.003440 / 12.073474 = 0.0285%
tier3→tier4: 12.070034 − 12.070034 = 0,相對改善 0%(追平,依協定誠實標記)
```

本場由導入報告功能後之版本執行,分數自 tier2 起未低於前一層——符合計畫書目標三(效能不退步)。

> **誠實但書(tier4 口徑)**:tier4 之 12.070034 即 exp8 本身——docs/benchmark_facts.json
> 註記「no tree-search entry beat the linear best — tie with tier3」。實際樹搜尋跑了兩輪
> (STATUS.md,OOF-only、不產生 submission):v1(Phase C-2a,單模節點空間,20 節點)
> 最佳僅 12.07459(即已知的 CAT 單模),輸給線性最佳;v2(Phase E-1「復仇戰」,
> ensemble-default 節點空間,26 節點)以 13-way blend(node #13)在第 14 個評估節點
> 精確追平 12.070034(至小數第 6 位),其後所有變異僅追平或變差。兩條獨立路徑收斂於
> 同一分數,與第 3.1 節的標籤噪音上限診斷一致——本場為噪音上限場次。

### 子刻度分解(選填)

依 §5 附錄子刻度分類法(僅作分類字典,不強制逐級測量),下表僅列本場 experiments.json
天然存在對應中間紀錄之子刻度;無對應紀錄之子刻度不列,分數逐字引用自 facts.json。

| 子刻度 | 優化辦法 | 分數 | 出處(exp# 或 tree node) |
|--------|----------|------|--------------------------|
| 1.a | 單模預設參數(tier1 三模型中最佳者:CAT,亦為 blend 全部權重) | 12.54287 | exp1 |
| 1.b | 多模+OOF 權重搜尋 blend(權重收斂為 100% CAT) | 12.54287 | exp1 |
| 2.d | 場內 reflexion(+3 交互特徵,反而變差,已還原) | 12.094826 | exp3 |
| 3.e | 結構化去噪嘗試:重複列群組目標平滑(無改善,已還原) | 12.081216 | exp5 |
| 3.b | Optuna 全 5-fold CV 調參 LGB,加入 pool(權重搜尋給 0,持平) | 12.073474 | exp6 |
| 3.d | seed bagging(調參 LGB 之 seed 1042 版,第 5 成員) | 12.071429 | exp7 |
| 3.d | seed bagging 擴充(+CAT seed 1042、+調參 LGB 第 3 個 seed,7-way,線性迭代最佳) | 12.070034 | exp8 |

## 6. 總結

本場資料小而噪:5,407 列、8 個數值配方特徵,`AgeInDays` 為對數線性主訊號;最關鍵的
結構性事實是 2,401 列非首見重複配方列帶有不同量測 `Strength`——不可約標籤噪音直接
劃定 CV 分數上限,並使本場全程獎勵正則化而懲罰容量。

關鍵決策有三:其一,exp2 起固定 `Strength` 十分位分層的 5-fold CV(seed 42),使八個
實驗與兩輪樹搜尋分數全部可直接比較;其二,exp1 的權重歸零異常被正確歸因為過擬合,
以「正則化+特徵工程」一次修正,成為全場最大躍升;其三,反思紀律——交互特徵(exp3)
與重複群組平滑(exp5)變差即還原,exp4 並以逐位元重現確認管線決定論。

增益來源逐層遞減且歸屬清楚:tier1→tier2 靠正則化與特徵工程(12.54287→12.073474);
tier2→tier3 靠 Phase B 線性迭代,且全部來自 seed bagging(12.070034),調參與去噪
皆未直接貢獻;tier3→tier4 為 0%——樹搜尋 v1 因單模節點空間結構性搆不到 blend 而落敗,
v2 擴充 ensemble-default 節點空間後精確追平線性最佳,但 26 節點內無任何變異超越之。

可信度方面須誠實:本場全程未提交 Kaggle,無 LB 外部錨點,所有結論建立在同一固定 CV 的
OOF 之上。但「線性迭代與 v2 樹搜尋兩條獨立方法收斂於同一分數、且去噪/調參/交互特徵
全數失敗」共同支持噪音上限解讀:12.070034 很可能已貼近本資料可達的下限,而非搜尋不足。

**重現本實驗的最短路徑**:見第 7 節。

## 7. 重現指令

```bash
cd /home/tjyen/ai_agents/kaggle

# Stage 1:EDA
uv run python3 competitions/playground-series-s3e9/scripts/eda.py

# Phase A:特徵工程+建模(LGB/XGB/CAT,5-fold 十分位分層 CV,OOF 權重搜尋 blend)→ exp2
#(features.py 由 train.py 匯入呼叫,無需獨立執行)
uv run python3 competitions/playground-series-s3e9/scripts/train.py

# Phase B 自我改進迭代(依序執行;round 4 讀取 round 2/3 產生的 npz checkpoint)
uv run python3 competitions/playground-series-s3e9/scripts/train_dup_smooth.py      # R1 → exp5(負面結果)
uv run python3 competitions/playground-series-s3e9/scripts/train_optuna_pool.py     # R2+R3 → exp6/exp7
uv run python3 competitions/playground-series-s3e9/scripts/train_round4_seedbag.py  # R4 → exp8 最佳 + submission

# 樹搜尋(OOF-only,不產生 submission;樹狀態各自持久化,可中斷續跑)
uv run python3 tree_search/run_s3e9.py       # v1 Phase C-2a → experiments_tree.json
uv run python3 tree_search/run_s3e9_v2.py    # v2 Phase E-1 → experiments_tree_v2.json

# 報告產生(本檔)
uv run python3 .claude/skills/kaggle-report/assets/collect.py playground-series-s3e9
uv run python3 .claude/skills/kaggle-report/assets/verify_report.py \
    competitions/playground-series-s3e9/REPORT.md competitions/playground-series-s3e9/facts.json
bash .claude/skills/kaggle-report/assets/md2pdf.sh competitions/playground-series-s3e9/REPORT.md \
    competitions/playground-series-s3e9/s3e9_REPORT.pdf

# 提交至 Kaggle(本場未執行;需先設定有效之 KAGGLE_API_TOKEN)
export KAGGLE_API_TOKEN=$(cat ~/.kaggle/kaggle_api_token.txt | tr -d '[:space:]')
uv run kaggle competitions submit -c playground-series-s3e9 \
  -f competitions/playground-series-s3e9/submissions/sub_blend_12.07003_20260703_235120.csv \
  -m "7-way seed-bag blend, OOF RMSE 12.070034"
```

執行目錄為專案根目錄 `/home/tjyen/ai_agents/kaggle`;所有 Python 執行皆透過 `uv run`。
