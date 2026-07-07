# 競賽分析報告:playground-series-s3e11

> 產生方式:kaggle-report skill(數字來自 facts.json,敘述由 agent 撰寫)
> 素材等級:full | 產生日期:2026-07-06
> 本報告所有數字皆出自 facts.json,經 verify_report.py 驗證。

## 1. 競賽目的

**What**:依門市與商品層級屬性(門市面積與五個附設設施旗標、商品重量/包裝、銷售額/銷量、
顧客家庭屬性等 15 個欄位)預測媒體行銷活動成本(`cost`),為連續數值之表格型迴歸問題;
資料集屬 Aygun et al.(Nature 2026)Kaggle Playground 基準(Season 3 Episode 11)。

**Why**:評估指標為 **RMSLE(minimize)**。成本類目標關心相對誤差而非絕對誤差:RMSLE 在
log1p 空間計算 RMSE,懲罰比例偏差並壓抑大值樣本對損失的支配,適合恆正的金額型目標。
此定義也直接決定訓練策略——對 log1p(cost) 以 RMSE objective 訓練即等同直接優化 RMSLE。

| 項目 | 值 |
|------|-----|
| 競賽 | playground-series-s3e11 |
| 問題型別 | regression |
| 評估指標 | rmsle(minimize) |
| 目標欄位 | cost |
| 素材等級 | full |

## 2. 使用工具與環境

| 工具 | 用途 |
|------|------|
| LightGBM / XGBoost / CatBoost | 梯度提升樹基模型:exp1–3 三模型 blend;XGB 於 exp4 起因連兩輪權重 0 被移出池;CatBoost 家族為 Phase B 與樹搜尋主力 |
| Optuna | Phase B 超參搜尋(TPE、fold-0 proxy):exp5 調 CatBoost,調參版以「入池不替換」方式加入 |
| 自建樹搜尋 harness(v2) | exp9:結構化搜尋模型/超參/blend 組合空間,重用線性迭代之 OOF 快取 |
| 5-fold CV 框架(scikit-learn) | KFold(shuffle, seed=42)5 折交叉驗證,exp2 起全程固定同一組折 |
| uv | Python 套件與虛擬環境管理,所有腳本皆以 `uv run` 執行 |

本場工具鏈組合邏輯:Claude Code(LLM)負責決策——log1p 目標的選定、store profile 目標編碼
的設計、尊重零權重裁決移除 XGB、R4 退步後回退、增益縮至噪音級時停損;Auto-ML 工具
(Optuna、樹搜尋 harness)負責系統化執行超參搜尋與組合空間探索,兩者分工互補。

## 3. 流程(how):五大元件

### 3.1 資料規格

| 項目 | 值 |
|------|-----|
| train 列數 | 360,336 |
| test 列數 | 240,224 |
| 原始欄位數 | 15(全數值) |
| 特別規則摘要 | 禁外部資料/禁預訓練模型/禁網路存取;每日提交上限 5 次 |

EDA 已執行(`eda_summary.py`):15 個特徵全為數值型,其中多數實為低基數「類別型偽裝」
欄位(五個 0/1 設施旗標、家庭屬性等);train 無缺失值、無重複列;`salad_bar` 與
`prepared_food` 相關 0.999839,為唯一高共線特徵對(近乎重複欄)。

目標 `cost` ∈ [50.79, 149.75],mean 99.614729、median 98.81、skew 0.019132——分布近乎
對稱,eda 判定非 log 轉換候選;採 log1p 目標的理由是「RMSLE = log1p 空間之 RMSE,直接
優化競賽指標」,而非矯正偏態,預測經 expm1 後 clip ≥ 0。

低訊號資料集:單變量關聯最強者僅 `florist`(pearson -0.110414)。`store_sqft` 加五個設施
旗標構成大量重複出現的「store profile」,是最強特徵候選,需以 fold 內 target encoding
防洩漏;eda 之 validation_hint 為「連續 i.i.d. 目標 → 標準 KFold」。

### 3.2 實驗總表

> **語意澄清(本報告全文適用)**:RMSLE 為連續迴歸指標,本場無取整/門檻類後處理,故各
> 實驗的「原始 OOF」即「決策分數」,兩欄同值;且本場為無人值守批次執行,全程未提交
> Kaggle(leaderboard 列於 facts.missing),所有決策皆以本機 OOF 為準,下文不再重複解釋。

| exp | 階段 | 模型/成員 | 特徵數 | 原始 OOF | 決策分數 | 採納 |
|-----|------|-----------|--------|----------|----------|------|
| 1 | Baseline(通用批次) | LGB/XGB/CAT(0.7/0.0/0.3) | 15 | 0.29723 | 0.29723 | 基線參照 |
| 2 | Phase A base | LGB/XGB/CAT(0.2/0.0/0.8),log1p 目標 | 15 | 0.2971 | 0.2971 | 方法學對照 |
| 3 | Phase A engineered | LGB/XGB/CAT(0.2/0.0/0.8)+ store_te | 21 | 0.296143 | 0.296143 | 是,Phase A 最佳 |
| 4 | Phase B R1 | LGB/CAT_orig(0.2/0.8),移除 XGB | 21 | 0.296143 | 0.296143 | 是,分數不變 |
| 5 | Phase B R2 | + CAT_tuned(Optuna,0.9 權重) | 21 | 0.295781 | 0.295781 | 是 |
| 6 | Phase B R3 | + CAT_tuned seed=2024(4-way) | 21 | 0.295715 | 0.295715 | 是 |
| 7 | Phase B R4 | 4-way pool + per-combo 均值特徵 | 24 | 0.2962 | 0.2962 | 否,退步棄用 |
| 8 | Phase B R5(線性終點) | + CAT_tuned seed=7(5-way) | 21 | 0.295648 | 0.295648 | 是,線性迭代最佳 |
| 9 | Phase D-6 樹搜尋(best) | 7-way blend(depth-12 CAT 家族 + DEEPLGB) | 21(同 exp8) | **0.29528** | **0.29528** | 是,本場最佳 |

**best 成員表(exp9,harness v2 node #20,phase-2 純權重再搜尋)**

| 成員 | 權重 | solo 分數 | 備註 |
|------|------|-----------|------|
| CAT_D12_S7 | 0.2478 | 0.295461 | 調參 CAT depth 10→12(突破 Optuna 搜尋上界),全樹最佳 solo |
| CAT_D12_S3000 | 0.3187 | 0.295462 | depth-12 家族 seed 變體 |
| CAT_D12_S3001 | 0.239 | 0.295483 | depth-12 家族 seed 變體 |
| CAT_S7_D10 | 0.0091 | 0.295779 | 線性迭代之 depth-10 tuned CAT(seed 7),自 npz 快取重用 |
| CAT_S99 | 0.0031 | 0.295786 | 第 4 個 tuned-CAT seed,線性迭代未嘗試 |
| CAT_TUNED_ROOT | 0.0039 | 0.29579 | 線性迭代最強 solo(Optuna 調參 CAT,seed 42),快取重用 |
| DEEPLGB | 0.1783 | 0.295833 | 刻意多樣化之深 LGB(num_leaves 255);solo 平庸但權重第 3 大 |

選型脈絡:exp1–3 三模型權重搜尋連兩輪將 XGB 權重歸 0,exp4 移除 XGB 後 blend 分數不變,
驗證零權重裁決無代價;exp5 起 Optuna 調參 CatBoost 成為主力,權重全數流向 tuned CAT 家族;
exp7 在 store_te 之上再加 per-combo 均值特徵全面退步,棄用回退;exp8 以第三個 seed 收在
0.295648,增益已縮至噪音級,依協定停止線性迭代。

> **誠實但書**:facts.best(exp9)以 OOF 分數最小選出,為 **OOF-only** 樹搜尋結果——未產生
> test 預測、無 submission 檔、未提交 Kaggle;其權重由 dirichlet 搜尋直接對全 OOF 擬合
> (無巢狀驗證),0.0004 等級的增益帶有 OOF 權重過擬風險,方向性結論(深度突破 Optuna
> 上界後重開 blend)較第 4 位小數穩健。

### 3.3 訓練規格表

| exp | CV 方案 | folds | seed |
|-----|---------|-------|------|
| 1 | 5fold | 5 | 無紀錄 |
| 2 | 5fold_kfold_shuffle | 5 | 42 |
| 3 | 5fold_kfold_shuffle | 5 | 42 |
| 4 | 5fold_kfold_shuffle | 5 | 42 |
| 5 | 5fold_kfold_shuffle | 5 | 42 |
| 6 | 5fold_kfold_shuffle | 5 | 42 |
| 7 | 5fold_kfold_shuffle | 5 | 42 |
| 8 | 5fold_kfold_shuffle | 5 | 42 |
| 9 | KFold(shuffle) | 5 | 42 |

目標為連續 i.i.d.、無時間/群組結構(eda validation_hint),且 store profile 大量重複、
隨機切分安全,故用 KFold(shuffle, seed=42);exp2–9(含樹搜尋)沿用同一組固定折,
跨實驗分數可直接比較。

Objective 與關鍵超參:全程對 log1p(cost) 以 RMSE objective 訓練,預測 expm1 後
clip ≥ 0(exp2 起,記於 notes)。exp5 之 Optuna(TPE,fold-0 proxy)最佳 CatBoost 參數:

```
Optuna:TPE 40 trials(319.3s,timeout guard 480s),fold-0 proxy
CatBoost tuned:depth=10, learning_rate≈0.0824, l2_leaf_reg≈5.482,
               min_data_in_leaf=33, random_strength≈0.0916
exp9 樹搜尋關鍵變體:CAT depth 10→12(Optuna 原搜尋空間上界為 10)、
                     DEEPLGB num_leaves=255
```

其餘 base model(手設 LGB/XGB/CAT)之超參無結構化紀錄,不臆測。

### 3.4 推論表

| exp | 後處理 | submission 檔 | 已提交 |
|-----|--------|---------------|--------|
| 1 | 無後處理紀錄 | sub_generic_0.29723_20260703_121020.csv | 否 |
| 2 | expm1 + clip ≥ 0 | sub_base_0.29710_20260703_192507.csv | 否 |
| 3 | expm1 + clip ≥ 0 | sub_engineered_0.29614_20260703_192819.csv | 否 |
| 4 | expm1 + clip ≥ 0 | sub_r1_noxgb_0.29614_20260703_222353.csv | 否 |
| 5 | expm1 + clip ≥ 0 | sub_r2_cattuned_0.29578_20260703_223101.csv | 否 |
| 6 | expm1 + clip ≥ 0 | sub_r3_seedbag_0.29571_20260703_223204.csv | 否 |
| 7 | expm1 + clip ≥ 0 | sub_r4_deepstore_0.29620_20260703_223633.csv(棄用) | 否 |
| 8 | expm1 + clip ≥ 0 | sub_r5_seedbag3_0.29565_20260703_223829.csv | 否 |
| 9 | 無(OOF-only) | 無紀錄(未產生 test 預測) | 否 |

「後處理」欄之 expm1 + clip ≥ 0 為 log1p 目標之逆轉換與安全網(記於各實驗 notes),
非指標特化後處理。欄位格式:id 欄 `id`、目標欄 `cost`。本場為無人值守批次執行,僅產生
本機 submission 檔、未觸碰 Kaggle 憑證,「已提交」一律為否。

### 3.5 評估指標 / 排行榜

指標定義:RMSLE = 對預測值與真值各取 log1p 後計算 RMSE,衡量比例(相對)誤差,越低越好。

| 項目 | OOF RMSLE |
|------|-----------|
| CAT_tuned(exp8 成員,Optuna 調參,seed 42) | 0.29579 |
| CAT_tuned_seed2024(exp8 成員) | 0.29591 |
| CAT_tuned_seed7(exp8 成員) | 0.29578 |
| Ensemble(exp8,線性迭代終點,5-way) | 0.295648 |
| Ensemble(exp9,樹搜尋 best,7-way) | **0.29528** |
| Public / Private LB | 無紀錄(未提交) |

`facts.leaderboard` 為 null(missing 列出 leaderboard):本場無排行榜表,CV↔LB gap 無法
計算。exp9 相對 exp8 之改善:

```
exp8 − exp9:0.295648 − 0.29528 = 0.000368
```

## 4. 實驗軌跡

**實驗階段對照表**(內部代號使用前先定義,R-W8)

| 代號 | 白話名稱 | 對應層級 |
|------|----------|----------|
| Phase A | kaggle-agent skill 六階段首跑 | tier2 |
| Phase B | self-improvement 線性迭代 | tier3 |
| Phase D-6 | 樹搜尋執行(harness v2) | tier4 |

| exp | 時間 | 決策分數 | 階段 | 一句話摘要 |
|-----|------|----------|------|------------|
| 1 | 2026-07-03T12:10:20 | 0.29723 | Baseline(通用批次) | 15 原始特徵三模型 blend,設定待超越基線 |
| 2 | 2026-07-03T19:25:07 | 0.2971 | Phase A base | 換 log1p 目標 + 調參 + early stop,方法學對照 |
| 3 | 2026-07-03T19:28:19 | 0.296143 | Phase A engineered | 21 特徵 + store_combo K-fold 目標編碼,Phase A 最佳 |
| 4 | 2026-07-03T22:23:54 | 0.296143 | Phase B R1 | 移除連兩輪零權重之 XGB,分數不變 |
| 5 | 2026-07-03T22:31:01 | 0.295781 | Phase B R2 | Optuna 調參 CatBoost 入池,Phase B 主要躍升 |
| 6 | 2026-07-03T22:32:05 | 0.295715 | Phase B R3 | tuned CAT seed bagging(seed 2024,4-way) |
| 7 | 2026-07-03T22:36:33 | 0.2962 | Phase B R4 | per-combo 均值特徵全面退步,棄用 |
| 8 | 2026-07-03T22:38:30 | 0.295648 | Phase B R5 | 第三 seed(7)5-way blend,線性迭代最佳後停止 |
| 9 | 2026-07-04T12:17:25 | **0.29528** | Phase D-6 樹搜尋 | harness v2 node #20 之 7-way 再混合,本場最佳 |

- **突破點 1(exp2→exp3)**:`store_te`(store_sqft + 五設施旗標之 combo 的 fold 內目標
  編碼)一舉由 0.2971 降至 0.296143,為 skill 首跑(Phase A)主要增益來源。
- **突破點 2(exp4→exp5)**:Optuna fold-0 proxy 調參 CatBoost 以「入池不替換」加入,
  0.296143 → 0.295781,為線性迭代階段最大單筆增益。
- **突破點 3(exp8→exp9)**:樹搜尋把 CAT depth 推到 12——超出 Optuna 自身搜尋上界 10
  ——並在新 solo 家族出現後重開 blend 權重搜尋,0.295648 → 0.29528。

無法解析之紀錄:無(facts.unparsed 為空陣列)。

## 5. 效能對照:四層消融

| 層級 | 配置 | 分數 | 相對改善 |
|------|------|------|----------|
| tier1 | 基線(Claude Code 直接執行,未引入 skill;exp1 通用批次三模型 blend) | 0.29723 | —(基線) |
| tier2 | + kaggle-agent skill 六階段流程(exp3,Phase A engineered) | 0.296143 | 見下方算式 |
| tier3 | + self-improvement 線性迭代(exp8,Phase B R5 之 5-way blend) | 0.295648 | 見下方算式 |
| tier4 | + 樹搜尋(exp9,harness v2 node #20 之 7-way blend) | **0.29528** | 見下方算式 |

```
tier1→tier2: 0.29723 − 0.296143 = 0.001087,相對改善 0.001087 / 0.29723 = 0.3657%
tier2→tier3: 0.296143 − 0.295648 = 0.000495,相對改善 0.000495 / 0.296143 = 0.1671%
tier3→tier4: 0.295648 − 0.29528 = 0.000368,相對改善 0.000368 / 0.295648 = 0.1245%
```

本場由導入報告功能後之版本執行,分數自 tier2 起未低於前一層——符合計畫書目標三(效能不退步)。

注:指標為 RMSLE(minimize),分數越低越好,四層逐層下降、無同值層。tier4(exp9)為
OOF-only 樹搜尋結果,本場無任何 Kaggle 提交、無排行榜錨點,四層分數皆為同一組固定折之
本機 OOF。

## 6. 總結

本場資料量大(360,336 列)而訊號弱:15 個全數值特徵之單變量關聯皆極低(最強僅
`florist` pearson -0.110414),真正的結構藏在 store_sqft 加五個設施旗標構成的重複
store profile 中。目標近乎對稱、非 log 轉換候選,採 log1p 目標純粹是為了讓 RMSE
objective 直接等於競賽指標 RMSLE。

關鍵決策有四:以 fold 內目標編碼 `store_te` 榨取 store profile 訊號(exp3,Phase A 主要
增益);尊重零權重裁決移除 XGB(exp4,分數不變、釋出預算);Optuna 調參 CatBoost 入池
加 seed bagging(exp5–8);exp7 在 store_te 之上疊 per-combo 均值特徵退步後果斷回退,
並在增益縮至噪音級時依協定停止線性迭代。

各層增益中 tier1→tier2 最大(特徵工程),tier2→tier3 次之(調參 + seed bagging),
tier3→tier4 由樹搜尋貢獻:最大單筆發現是把 CAT depth 推到 12——Optuna 的最優解原本就
落在其搜尋上界 10 上,上界本身即是下一個突變方向;而最終 0.29528 來自新 depth-12 家族
出現後重開的 blend 權重再搜尋,且 solo 平庸的 DEEPLGB 仍拿到 0.1783 權重,再次印證
blend 貢獻與 solo 分數脫鉤。

可信度方面須誠實:本場全程僅本機 OOF、未提交 Kaggle,無排行榜外部驗證;exp2–9 使用同一
組固定折,分數排序可直接比較,但 exp9 為 OOF-only 且權重直接對全 OOF 擬合,0.0004 等級
增益帶有過擬風險。方向性結論(深度突破調參上界、突破後重開 blend)是穩健的部分。

**重現本實驗的最短路徑**:見第 7 節。

## 7. 重現指令

```bash
cd /home/tjyen/ai_agents/kaggle

# 資料下載(需先設定 KAGGLE_API_TOKEN;data/ 已存在者可略過)
uv run kaggle competitions download -c playground-series-s3e11 \
  -p competitions/playground-series-s3e11/data

# Stage 1:EDA
uv run python3 competitions/playground-series-s3e11/scripts/eda.py

# exp1(對照組):通用批次管線
uv run python3 competitions/run_competition.py playground-series-s3e11

# exp2:skill base(log1p 目標,15 原始特徵,方法學對照)
uv run python3 competitions/playground-series-s3e11/scripts/train.py base

# exp3:skill engineered(21 特徵 + store_te,Phase A 最佳)
uv run python3 competitions/playground-series-s3e11/scripts/train.py engineered

# Phase B(exp4–exp8;checkpointed,cache 命中會跳過已訓練成員)
uv run python3 competitions/playground-series-s3e11/scripts/iterate2.py r1
uv run python3 competitions/playground-series-s3e11/scripts/iterate2.py tune
uv run python3 competitions/playground-series-s3e11/scripts/iterate2.py r2
uv run python3 competitions/playground-series-s3e11/scripts/iterate2.py r3
uv run python3 competitions/playground-series-s3e11/scripts/iterate2.py r4   # 退步,棄用
uv run python3 competitions/playground-series-s3e11/scripts/iterate2.py r5   # exp8,線性迭代最佳

# exp9:樹搜尋 harness v2(本場最佳;OOF-only,重用 scripts/cache/*.npz)
uv run python3 tree_search/run_s3e11.py

# 報告產生(本檔)
uv run python3 .claude/skills/kaggle-report/assets/collect.py playground-series-s3e11
uv run python3 .claude/skills/kaggle-report/assets/verify_report.py \
    competitions/playground-series-s3e11/REPORT.md competitions/playground-series-s3e11/facts.json
bash .claude/skills/kaggle-report/assets/md2pdf.sh competitions/playground-series-s3e11/REPORT.md \
    competitions/playground-series-s3e11/s3e11_REPORT.pdf

# 提交至 Kaggle(本場未執行;需先設定有效之 KAGGLE_API_TOKEN)
export KAGGLE_API_TOKEN=$(cat ~/.kaggle/kaggle_api_token.txt | tr -d '[:space:]')
uv run kaggle competitions submit -c playground-series-s3e11 \
    -f competitions/playground-series-s3e11/submissions/sub_r5_seedbag3_0.29565_20260703_223829.csv \
    -m "R5 5-way seed-bagged tuned CatBoost blend, OOF RMSLE 0.295648"
```

執行目錄為專案根目錄 `/home/tjyen/ai_agents/kaggle`;所有 Python 執行皆透過 `uv run` 以
確保套件環境一致。本場為無人值守批次執行,未提交至 Kaggle 排行榜、未觸碰 Kaggle 憑證。
