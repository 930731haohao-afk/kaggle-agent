# 競賽分析報告:playground-series-s3e7

> 產生方式:kaggle-report skill(數字來自 facts.json,敘述由 agent 撰寫)
> 素材等級:full | 產生日期:2026-07-06
> 本報告所有數字皆出自 facts.json,經 verify_report.py 驗證。

## 1. 競賽目的

**What**:依訂房紀錄的各項欄位(提前預訂天數、房價、住客組成、市場區隔、歷史取消紀錄等)
預測該筆訂房最終是否被取消(`booking_status`,1 = 取消),為二元分類問題。

**Why**:評估指標為 **ROC-AUC(maximize)**。取消與未取消存在輕度不平衡(約四成取消),
ROC-AUC 是與分類門檻無關的排序型指標,直接衡量模型能否把「會取消」的訂房排在「不會取消」
之前,不需選定機率門檻、也不要求機率校準,適合此類輕度不平衡的二元分類。

| 項目 | 值 |
|------|-----|
| 競賽 | playground-series-s3e7 |
| 問題型別 | classification |
| 評估指標 | roc_auc(maximize) |
| 目標欄位 | booking_status |
| 素材等級 | full |

## 2. 使用工具與環境

| 工具 | 用途 |
|------|------|
| LightGBM / XGBoost / CatBoost | 三個梯度提升樹基模型,構成 exp1–6 加權 blend 主體與 exp7 樹搜尋 solo pool 的核心 |
| Optuna | Phase B 超參搜尋(TPE、fold-0 proxy 目標):exp4 調 LGB、exp5 調 XGB |
| 自建樹搜尋 harness(v3) | Phase F-2 搜尋模型/超參/集成組合空間,於 node #48 找到 exp7 的 mega-blend |
| 5-fold CV 框架(scikit-learn) | 直接對 `booking_status` 分層的 StratifiedKFold,5 折交叉驗證 |
| uv | Python 套件與虛擬環境管理,所有腳本皆以 `uv run` 執行 |

本場工具鏈組合邏輯:Claude Code(LLM)負責決策——特徵取捨(反思後刪噪音特徵)、每輪只改
一件事的迭代設計、以多樣性理由否決調參結果、與何時停損;Auto-ML 工具(Optuna、樹搜尋
harness)負責系統化執行超參搜尋與組合空間探索,兩者分工互補。

## 3. 流程(how):五大元件

### 3.1 資料規格

| 項目 | 值 |
|------|-----|
| train 列數 | 42,100 |
| test 列數 | 28,068 |
| 原始欄位數 | 17 |
| 特別規則摘要 | 禁外部資料/禁預訓練模型/禁網路存取;每日提交上限 5 次 |

17 個特徵全為數值型(餐型、房型、市場區隔等類別欄位已預先 label-encode);train/test 皆
無缺失值,亦無高共線特徵對。目標 `booking_status` 為 0/1 整數,mean 0.392019(約四成
取消),非 log 轉換候選。單一最強關聯特徵為 `lead_time`(pearson 0.374865),
`no_of_special_requests` 呈保護性負相關(pearson -0.220278)。

> **誠實但書**:facts.eda 記錄 train 重複列 562;STATUS.md 敘述則為「無重複列/重複 id」,
> 兩者口徑不同(前者以特徵欄位計、不含 id 欄)。本報告以 facts.eda 之數字為準。

### 3.2 實驗總表

> **語意澄清(本報告全文適用)**:ROC-AUC 為排序型指標,本場無取整/門檻類後處理,故各
> 實驗的「原始 OOF」即「決策分數」,兩欄同值;且本場全部決策皆以本機 OOF 為準(未提交
> Kaggle,見 3.4 節)。

| exp | 階段 | 模型/成員 | 特徵數 | 原始 OOF | 決策分數 | 採納 |
|-----|------|-----------|--------|----------|----------|------|
| 1 | Baseline(通用批次) | LGB/XGB/CAT(0.1/0.6/0.3) | 17 | 0.89882 | 0.89882 | 基線參照 |
| 2 | Phase A iter1 | LGB/XGB/CAT(0.45/0.45/0.1) | 31 | 0.89788 | 0.89788 | 否,低於基線 |
| 3 | Phase A iter2(反思精簡) | LGB/XGB/CAT(0.5/0.4/0.1) | 25 | 0.899395 | 0.899395 | 是,Phase A 最佳 |
| 4 | Phase B R1 | LGB(Optuna)/XGB/CAT(0.55/0.4/0.05) | 25 | 0.899891 | 0.899891 | 是 |
| 5 | Phase B R2 | LGB(tuned)/XGB(Optuna)/CAT(0.5/0.35/0.15) | 25 | 0.899722 | 0.899722 | 否,多樣性受損 |
| 6 | Phase B R3(線性終點) | exp4 基模型 + prob_0.01grid(0.54/0.4/0.06) | 25 | 0.899893 | 0.899893 | 是,線性迭代最終 |
| 7 | Phase F-2 樹搜尋(best) | 38 成員 mega-blend(node #48,9 個非零權重) | 無紀錄 | **0.900455** | **0.900455** | 是,本場最佳 |

**best 成員表(exp7,rank-space mega-blend;權重搜尋將 29/38 個成員歸零)**

| 成員 | 權重 | solo 分數 | 備註 |
|------|------|-----------|------|
| SEEDBAG | 0.358 | 無紀錄 | 調參 LGB 之 seed-bagging 變體 |
| XGBDIV family(4 節點) | 0.459 | 無紀錄 | 刻意多樣化之深 XGB 家族,合計權重最大 |
| EXPL_BOUND2 | 0.114 | 0.897541 | boundary-push 衍生 depth-2 LGB;solo 較弱但第 3 大權重 |
| root | 0.038 | 0.899215 | 搜尋根節點 = exp4 之 Optuna 調參 LGB(同模型紀錄) |
| FEATPRUNE-child | 0.018 | 無紀錄 | 特徵修剪變體 |
| XGBHAND-child | 0.013 | 無紀錄 | 手設 XGB 之子節點 |

選型脈絡:exp4 以 Optuna 只調 LGB,單模 0.898824 升至 0.899215 成最強單模,blend
+0.000496;exp5 同法調 XGB,單模微升至 0.89886 但 blend 退步 -0.000169——調參後 XGB
與調參 LGB 樹形趨同、喪失多樣性,故保留手設 XGB。exp6 僅精修混合層(prob_0.01grid),
+0.000002 屬噪音級,連同 exp5 計兩輪無實質改進,依協定停止線性迭代。

exp7 由 harness v3 的強制 explore-burst 機制注入 kitchen-sink mega-blend(rank-space
Dirichlet(k=800)+coordinate-ascent,對全部 38 個 solo pool 成員做權重搜尋)而得;超越
v2 重現高原(0.900054)的增益全數來自此機制,而非任何單模突破。

> **誠實但書**:facts.best(exp7)以 OOF 分數最大選出,為 OOF-only 樹搜尋結果——未產生
> test 預測、無 submission 檔、未提交 Kaggle;其權重直接對全 OOF 擬合(無巢狀驗證),
> 0.0002 等級的增益帶有 OOF 權重過擬風險,方向性結論(burst mega-blend 優於手工成長
> blend)較第 4 位小數穩健。

### 3.3 訓練規格表

| exp | CV 方案 | folds | seed |
|-----|---------|-------|------|
| 1 | 5fold | 5 | 無紀錄 |
| 2 | 5fold(StratifiedKFold on booking_status) | 5 | 42 |
| 3 | 5fold(StratifiedKFold on booking_status) | 5 | 42 |
| 4 | 5fold(StratifiedKFold on booking_status) | 5 | 42 |
| 5 | 5fold(StratifiedKFold on booking_status) | 5 | 42 |
| 6 | 5fold(StratifiedKFold on booking_status) | 5 | 42 |
| 7 | 5fold(StratifiedKFold on booking_status) | 5 | 42 |

目標為二元且輕度不平衡,直接對 `booking_status` 分層即可讓每折正負比例一致,無需分箱;
exp2–7 固定同一組折,7 個實驗分數可直接比較,樹搜尋(exp7)亦沿用完全相同的折。

Objective 與關鍵超參(有紀錄者,節錄自 exp6 之 base_models params;LGB 為 Optuna 調參
結果,「淺而強正則」——與跨競賽經驗「中小型資料獎勵正則化而非容量」一致):

```
LGB(Optuna):objective=binary, n_estimators=3000, learning_rate≈0.068, max_depth=3,
             num_leaves=178(depth=3 下實際不起作用), min_child_samples=47,
             subsample≈0.889, colsample_bytree≈0.532, reg_alpha≈2.14, reg_lambda≈0.0115
XGB(手設): objective=binary:logistic, n_estimators=3000, learning_rate=0.03, max_depth=6,
             min_child_weight=5, subsample=0.8, colsample_bytree=0.8
CAT(手設): loss_function=Logloss, iterations=4000, learning_rate=0.03, depth=7,
             l2_leaf_reg=5.0
```

exp7(樹搜尋)的 base_models 僅附權重與部分 solo 分數,未附 params 欄位——無紀錄,不臆測。

### 3.4 推論表

| exp | 後處理 | submission 檔 | 已提交 |
|-----|--------|---------------|--------|
| 1 | 無後處理紀錄 | sub_generic_0.89882_20260703_120542.csv | 否 |
| 2 | 無後處理紀錄 | sub_blend_0.89788_20260703_185938.csv | 否 |
| 3 | 無後處理紀錄 | sub_blend_0.89939_20260703_190223.csv | 否 |
| 4 | 無後處理紀錄 | sub_blend_0.89989_20260703_203932.csv | 否 |
| 5 | 無後處理紀錄 | 無紀錄 | 否 |
| 6 | 無後處理紀錄 | sub_blend_0.89989_20260703_205132.csv | 否 |
| 7 | 無後處理紀錄 | 無紀錄(OOF-only,未產生 test 預測) | 否 |

ROC-AUC 為排序型指標,無門檻轉換或機率校準之後處理需求,facts 各實驗皆無 postprocess
欄位。表中 submission 檔為本機產出之預測檔;本場全程未提交 Kaggle(leaderboard 列於
facts.missing),故「已提交」一律為否。欄位格式:id 欄 `id`、目標欄 `booking_status`。

### 3.5 評估指標

指標定義:ROC-AUC = 隨機抽一筆正類(取消)與一筆負類(未取消),模型將正類排在前面的
機率;0.5 為隨機、1 為完美排序,越高越好。

| 項目 | OOF ROC-AUC |
|------|-------------|
| LGB(exp6 成員,Optuna 調參) | 0.899215 |
| XGB(exp6 成員,手設) | 0.898765 |
| CAT(exp6 成員,手設) | 0.896709 |
| Ensemble(exp6,線性迭代終點) | 0.899893 |
| Ensemble(exp7,樹搜尋 best) | **0.900455** |
| Public / Private LB | 無紀錄(未提交) |

本場無排行榜表(leaderboard 為空,列於 facts.missing),CV↔LB gap 無法計算。exp7 相對
exp6 之改善:

```
exp7 − exp6:0.900455 − 0.899893 = 0.000562
```

## 4. 實驗軌跡

**實驗階段對照表**(內部代號使用前先定義,R-W8)

| 代號 | 白話名稱 | 對應層級 |
|------|----------|----------|
| Phase A | kaggle-agent skill 六階段首跑 | tier2 |
| Phase B | self-improvement 線性迭代 | tier3 |
| Phase F-2 | 樹搜尋執行(harness v3) | tier4 |

| exp | 時間 | 決策分數 | 階段 | 一句話摘要 |
|-----|------|----------|------|------------|
| 1 | 2026-07-03T12:05:42 | 0.89882 | Baseline(通用批次) | 17 原始特徵三模型 blend,設定待超越基線 |
| 2 | 2026-07-03T18:59:38 | 0.89788 | Phase A iter1 | 加入 14 個工程特徵反而低於基線 |
| 3 | 2026-07-03T19:02:23 | 0.899395 | Phase A iter2 | 反思後精簡為 8 個工程特徵,回升越過基線 |
| 4 | 2026-07-03T20:39:32 | 0.899891 | Phase B R1 | Optuna 調參 LGB,線性階段主要躍升 |
| 5 | 2026-07-03T20:47:55 | 0.899722 | Phase B R2 | Optuna 調參 XGB 使多樣性受損,棄用 |
| 6 | 2026-07-03T20:51:32 | 0.899893 | Phase B R3 | 僅精修混合層,噪音級改善後依協定停止 |
| 7 | 2026-07-04T11:59:43 | **0.900455** | Phase F-2 樹搜尋 | harness v3 explore-burst mega-blend,本場最佳 |

- **突破點 1(exp2→exp3)**:iter1 的週期性月份/日期編碼與價格交乘項是噪音而非訊號,
  反思後刪除,分數由低於基線的 0.89788 回升至 0.899395,是一次「假設→驗證」式修正。
- **突破點 2(exp3→exp4)**:Optuna(fold-0 proxy)找到淺而強正則的 LGB 配置,單模
  0.898824 → 0.899215,blend +0.000496,為線性迭代階段最大單筆增益。
- **突破點 3(exp6→exp7)**:harness v3 在所有第一代 lineage 三振 plateau 後自動注入
  explore-burst 的 38 成員 mega-blend,由線性終點 0.899893 升至 0.900455(+0.000562)。

無法解析之紀錄:無(facts.unparsed 為空陣列)。

## 5. 效能對照:四層消融

| 層級 | 配置 | 分數 | 相對改善 |
|------|------|------|----------|
| tier1 | 基線(Claude Code 直接執行,未引入 skill;exp1 通用批次三模型 blend) | 0.89882 | —(基線) |
| tier2 | + kaggle-agent skill 六階段流程(exp3,Phase A 精簡特徵三模型 blend) | 0.899395 | 見下方算式 |
| tier3 | + self-improvement 線性迭代(exp6,Phase B 三輪之終點) | 0.899893 | 見下方算式 |
| tier4 | + 樹搜尋(exp7,node #48 mega-blend) | **0.900455** | 見下方算式 |

```
tier1→tier2: 0.899395 − 0.89882 = 0.000575,相對改善 0.000575 / 0.89882 = 0.0640%
tier2→tier3: 0.899893 − 0.899395 = 0.000498,相對改善 0.000498 / 0.899395 = 0.0554%
tier3→tier4: 0.900455 − 0.899893 = 0.000562,相對改善 0.000562 / 0.899893 = 0.0625%
```

本場由導入報告功能後之版本執行,分數自 tier2 起未低於前一層——符合計畫書目標三(效能不退步)。

### 子刻度分解(選填)

依 §5 附錄子刻度分類法(僅作分類字典,不強制逐級測量),下表僅列本場 experiments.json
天然存在對應中間紀錄之子刻度;無對應紀錄之子刻度不列,分數逐字引用自 facts.json。

| 子刻度 | 優化辦法 | 分數 | 出處(exp# 或 tree node) |
|--------|----------|------|--------------------------|
| 1.a | 單模預設參數(tier1 三模型中最佳者:XGB) | 0.89821 | exp1 |
| 1.b | 多模+OOF 權重搜尋 blend(tier1 三模型混合) | 0.89882 | exp1 |
| 2.b | iter1:17 原始 + 14 工程特徵(反而低於基線) | 0.89788 | exp2 |
| 2.b | iter2:精簡為 17 原始 + 8 工程特徵(回升越過基線) | 0.899395 | exp3 |
| 3.b | Optuna fold-0 代理調參 LGB,加入 pool(線性階段最大單筆增益) | 0.899891 | exp4 |
| 3.b | Optuna fold-0 代理調參 XGB(多樣性受損,blend 退步未採用) | 0.899722 | exp5 |
| 4.b | 樹搜尋 harness v2(較早 22-節點掃描之最佳) | 0.900242 | exp7 notes 引用 |
| 4.e | 樹搜尋 harness v3(node #48,explore-burst mega-blend,本場最佳) | 0.900455 | exp7 |

## 6. 總結

本場資料乾淨:17 個特徵全為已數值編碼、無缺失值、無高共線對,目標約四成取消的輕度不平衡,
直接對 `booking_status` 做 StratifiedKFold 即得穩定 CV;`lead_time` 是單一最強訊號。
exp2 起全程固定同一組折,使所有實驗分數可直接比較,是後續每一步小增益仍可辨識的前提。

關鍵決策有三:其一,exp2 堆特徵反而低於基線後,以反思刪除噪音特徵而非繼續堆疊,exp3 隨即
越過基線;其二,exp5 揭示「單模變強、blend 變差」的多樣性教訓,保留手設 XGB 而否決調參
結果;其三,exp6 改善僅噪音級,依「連兩輪無實質改進即停」協定誠實收手,把剩餘預算讓給
結構化搜尋。

各層增益皆為正但幅度極小,是本批次中口徑最緊的一場:skill 流程、線性迭代與樹搜尋各貢獻
一小步,沒有單一大躍升;最後一層的 0.900455 全數來自 harness v3 強制 explore-burst 注入
的 kitchen-sink mega-blend,而非任何單模突破——blend 貢獻與 solo 分數脫鉤(EXPL_BOUND2
solo 僅 0.897541 卻拿第 3 大權重)是本場最具遷移價值的觀察。

可信度方面須誠實:本場全程僅本機 OOF、未提交 Kaggle,無排行榜錨點;exp7 權重直接對全
OOF 擬合、無巢狀驗證,0.0002 等級增益帶有 OOF 權重過擬風險。方向性結論(burst
mega-blend 優於手工成長 blend)是穩健的部分,第 4 位小數不是。

**重現本實驗的最短路徑**:見第 7 節。

## 7. 重現指令

```bash
cd /home/tjyen/ai_agents/kaggle

# Stage 1:EDA
uv run python3 competitions/playground-series-s3e7/scripts/eda.py

# Phase A:特徵工程 + 建模(LGB/XGB/CAT,5-fold StratifiedKFold,OOF 權重搜尋 blend)→ exp3
uv run python3 competitions/playground-series-s3e7/scripts/train.py

# Phase B R1:Optuna 調參 LGB(50 trials,fold-0 proxy)+ blend → exp4
uv run python3 competitions/playground-series-s3e7/scripts/optuna_lgb.py

# Phase B R2:Optuna 調參 XGB(未採納)→ exp5
uv run python3 competitions/playground-series-s3e7/scripts/optuna_xgb.py

# Phase B R3:混合層精修(0.01 網格 + rank blend)→ exp6(線性迭代終點)
uv run python3 competitions/playground-series-s3e7/scripts/blend_refine.py

# Phase F-2:樹搜尋 harness v3(exp7,本場最佳;可中斷續跑,樹狀態存 experiments_tree_v3.json)
uv run python3 tree_search/run_s3e7_v3.py

# 報告產生(本檔)
uv run python3 .claude/skills/kaggle-report/assets/collect.py playground-series-s3e7
uv run python3 .claude/skills/kaggle-report/assets/verify_report.py \
    competitions/playground-series-s3e7/REPORT.md competitions/playground-series-s3e7/facts.json
bash .claude/skills/kaggle-report/assets/md2pdf.sh competitions/playground-series-s3e7/REPORT.md \
    competitions/playground-series-s3e7/s3e7_REPORT.pdf

# 提交至 Kaggle(本場未執行;需先設定有效之 KAGGLE_API_TOKEN)
export KAGGLE_API_TOKEN=$(python3 -c "import json; print(json.load(open('/home/tjyen/.kaggle/kaggle.json'))['key'])")
uv run kaggle competitions submit -c playground-series-s3e7 \
    -f competitions/playground-series-s3e7/submissions/sub_blend_0.89989_20260703_205132.csv \
    -m "Optuna-tuned LGB + XGB + CAT blend (0.54/0.40/0.06), OOF 0.899893"
```

執行目錄為專案根目錄 `/home/tjyen/ai_agents/kaggle`;所有 Python 執行皆透過 `uv run`。
