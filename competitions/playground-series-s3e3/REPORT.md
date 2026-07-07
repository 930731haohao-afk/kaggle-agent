# 競賽分析報告:playground-series-s3e3

> 產生方式:kaggle-report skill(數字來自 facts.json,敘述由 agent 撰寫)
> 素材等級:full | 產生日期:2026-07-06
> 本報告所有數字皆出自 facts.json,經 verify_report.py 驗證。

## 1. 競賽目的

**What**:依員工人事資料(年齡、職級、月薪、任職年資、各項滿意度評分、是否加班等 33 個
原始欄位)預測其是否離職(`Attrition`),為二元分類問題。

**Why**:評估指標為 **ROC-AUC(maximize)**。離職樣本僅 200 筆、留任 1477 筆,類別嚴重
不平衡;ROC-AUC 衡量「將實際離職者排在較高風險分數」的排序能力,不受分類門檻與類別比例
影響,較準確率等門檻敏感指標更適合此類不平衡二元分類問題。

| 項目 | 值 |
|------|-----|
| 競賽 | playground-series-s3e3 |
| 問題型別 | classification |
| 評估指標 | roc_auc(maximize) |
| 目標欄位 | Attrition |
| 素材等級 | full |

## 2. 使用工具與環境

| 工具 | 用途 |
|------|------|
| LightGBM / XGBoost / CatBoost | 梯度提升樹基礎模型:exp1/2/3/5/7 之 blend 成員,exp4(LGB 調參)與 exp6(CatBoost 原生類別)之 solo 模型 |
| Optuna | exp4 對 LGB 做超參搜尋,目標函式為完整 5-fold CV 之 OOF ROC-AUC |
| 自建樹搜尋 harness | exp8 規模實驗:結構化搜尋特徵/模型/超參與 blend 組合空間 |
| 5-fold CV 框架(scikit-learn) | 依 `Attrition` 分層抽樣的 StratifiedKFold,5 折交叉驗證 |
| uv | Python 套件與虛擬環境管理,所有腳本皆以 `uv run` 執行 |

本場工具鏈組合邏輯:Claude Code(LLM)負責決策——選型、特徵工程、診斷不平衡加權為何傷害
排序指標、判斷何時停損(線性迭代增益趨緩後停止,即 Phase B);Auto-ML 工具(Optuna、樹搜尋
harness)負責系統化執行超參搜尋與組合空間探索,兩者分工互補。

## 3. 流程(how):五大元件

### 3.1 資料規格

| 項目 | 值 |
|------|-----|
| train 列數 | 1677 |
| test 列數 | 1119 |
| 原始欄位數 | 33(25 個數值 + 8 個類別) |
| 特別規則摘要 | 禁外部資料/禁預訓練模型/禁網路存取;每日提交上限 5 次 |

目標 `Attrition` 為 0/1 整數標籤:離職(1)200 筆、留任(0)1477 筆,平均值 0.119261、
偏度 2.351659,類別嚴重不平衡。EDA 已執行(`eda_summary.py`):train/test 皆無缺失值、
無重複列、無 test 未見類別;`EmployeeCount`、`StandardHours` 標準差為 0、`Over18` 僅單一
取值,三欄為常數,特徵工程時剔除。

eda 的高共線特徵對清單為空(未達自動門檻),但 STATUS.md 記錄年資類特徵彼此相關
0.75–0.79、`JobLevel`↔`MonthlyIncome` r=0.91,屬中高度冗餘。特徵數上,exp1 使用 33 個
原始欄位,exp2–5/7 經特徵工程展開為 48 個,exp6 改用 40 個(類別欄保留原始字串供
CatBoost 原生處理)。

### 3.2 實驗總表

> **語意澄清(本報告全文適用)**:本場指標 ROC-AUC 為連續排序指標,無取整等後處理,故各
> 實驗的「原始 OOF」即「決策分數」。另,`facts.best` 以 OOF 分數最大者選出:best 為 exp8
> 之樹搜尋結果,屬 **OOF-only**——未產生 test 預測;本場所有實驗皆未提交 Kaggle 排行榜,
> 下文不再重複解釋。

| exp | 階段 | 模型/成員 | 特徵數 | 原始 OOF | 決策分數 | 採納 |
|-----|------|-----------|--------|----------|----------|------|
| 1 | Baseline(通用批次) | LGB+XGB+CAT blend | 33 | 0.81624 | 0.81624 | 基線 |
| 2 | Phase A(特徵工程+不平衡加權) | LGB+XGB+CAT blend | 48 | 0.819008 | 0.819008 | 否,旋被 exp3 取代 |
| 3 | Phase A(Reflexion:移除加權) | LGB+XGB+CAT blend | 48 | 0.832925 | 0.832925 | 是,Phase A 冠軍 |
| 4 | Phase B round1 | LGB_tuned(Optuna,solo) | 48 | 0.837305 | 0.837305 | 是,入池 |
| 5 | Phase B round2 | 5-way blend(+seed bag) | 48 | 0.837776 | 0.837776 | 是 |
| 6 | Phase B round3(診斷) | CAT_native(solo) | 40 | 0.814259 | 0.814259 | 否,診斷用 |
| 7 | Phase B round4 | 6-way rank-average blend | 48 | 0.83814 | 0.83814 | 是,線性迭代冠軍 |
| 8 | Phase E-5 樹搜尋(best) | KITCHENBLEND(全 solo 池混合) | 無紀錄 | **0.845051** | **0.845051** | 是,本場最佳 |

**best 成員表**:exp8 之 KITCHENBLEND 為「當時已評估之所有 solo 節點」的 kitchen-sink
混合(dirichlet(k=800)+coordinate-ascent),成員隨搜尋進度變動,facts.json 未附結構化
`base_models` 清單——成員/權重/solo 分數無紀錄,依規範不逐一列出。線性迭代冠軍 exp7 之
權重搜尋結果為 LGB_orig 0.3、LGB_tuned 0.7,其餘成員(XGB/CAT_orig/LGB_tuned_seed2024/
CAT_native)權重皆為 0,以 rank-average 合併。

選型理由:三個梯度提升樹在表格資料上穩健,以權重搜尋決定集成比例;各輪權重搜尋一致將
CatBoost 權重收斂為 0(exp6 原生類別處理將其 solo 由 0.762684 拉升至 0.814259,仍為最弱
成員),確認其在此小樣本資料上結構性偏弱。Phase B 聚焦 LightGBM:Optuna 直接以完整 5-fold
CV AUC 為目標(exp4)、調參版入池加 seed bagging(exp5)、終以 rank-average 收尾(exp7)。

### 3.3 訓練規格表

| exp | CV 方案 | folds | seed |
|-----|---------|-------|------|
| 1 | 5fold | 5 | 無紀錄 |
| 2 | StratifiedKFold | 5 | 42 |
| 3 | StratifiedKFold | 5 | 42 |
| 4 | StratifiedKFold | 5 | 42 |
| 5 | StratifiedKFold | 5 | 42 |
| 6 | StratifiedKFold | 5 | 42 |
| 7 | StratifiedKFold | 5 | 42 |
| 8 | StratifiedKFold | 5 | 42 |

類別不平衡下,一般隨機 K-fold 會使各折正負比例不穩、驗證 AUC 波動;exp2 起改用依
`Attrition` 分層的 StratifiedKFold(seed=42),且 exp2–8(含樹搜尋)沿用同一組固定折,
跨實驗分數可直接比較。exp4 之 Optuna 設定(TPE、50 trials、實際 111.3s)與最佳超參
(num_leaves=3、max_depth=4 等)僅記於 notes 文字;其餘 base model 之 objective/超參無結構化紀錄。

### 3.4 推論表

| exp | 後處理 | submission 檔 | 已提交 |
|-----|--------|---------------|--------|
| 1 | 無後處理紀錄 | sub_generic_0.81624_20260703_120255.csv | 否 |
| 2 | 無後處理紀錄 | sub_blend_0.81901_20260703_183757.csv | 否 |
| 3 | 無後處理紀錄 | sub_blend_0.83292_20260703_183913.csv | 否 |
| 4 | 無後處理紀錄 | 無紀錄 | 否 |
| 5 | 無後處理紀錄 | sub_blend_0.83778_20260703_233044.csv | 否 |
| 6 | 無後處理紀錄 | 無紀錄 | 否 |
| 7 | 無後處理紀錄 | sub_blend_0.83814_20260703_233303.csv | 否 |
| 8 | 無後處理紀錄 | 無紀錄(OOF-only) | 否 |

`id_column = id`、`target_column = Attrition`。本場為無人值守批次執行,僅產生 submission
檔案、未觸碰 Kaggle 憑證;exp8 為樹搜尋 OOF-only 結果,未產生 test 預測。

### 3.5 評估指標 / 排行榜

指標定義:ROC-AUC = 模型將正樣本(離職)排序高於負樣本(留任)之機率,愈高愈好,不受決策
門檻影響。各實驗決策分數已列於實驗總表,本場最佳為 exp8 之 **0.845051**(OOF)。

`facts.leaderboard` 為 null(`missing` 列出 leaderboard):Public/Private LB 皆無紀錄,
故本場無排行榜表,亦無法計算 CV↔LB gap。

## 4. 實驗軌跡

**實驗階段對照表**(內部代號使用前先定義,R-W8)

| 代號 | 白話名稱 | 對應層級 |
|------|----------|----------|
| Phase A | kaggle-agent skill 六階段首跑 | tier2 |
| Phase B | self-improvement 線性迭代 | tier3 |
| Phase E-5 | 樹搜尋執行(harness) | tier4 |

| exp | 時間 | 決策分數 | 階段 | 一句話摘要 |
|-----|------|----------|------|------------|
| 1 | 2026-07-03T12:02:55 | 0.81624 | Baseline(通用批次) | 33 原始特徵三模型 blend 基線 |
| 2 | 2026-07-03T18:37:57 | 0.819008 | Phase A | 48 特徵 + 不平衡加權,XGB/CAT 反而退步 |
| 3 | 2026-07-03T18:39:13 | 0.832925 | Phase A(Reflexion) | 移除加權 + 加強正則化,Phase A 冠軍 |
| 4 | 2026-07-03T23:29:51 | 0.837305 | Phase B round1 | Optuna 直接優化 5-fold CV AUC,LGB solo 新高 |
| 5 | 2026-07-03T23:30:44 | 0.837776 | Phase B round2 | 調參版入池 + seed bag,5-way blend |
| 6 | 2026-07-03T23:31:25 | 0.814259 | Phase B round3(診斷) | CatBoost 原生類別重試,大幅回升仍最弱 |
| 7 | 2026-07-03T23:33:03 | 0.83814 | Phase B round4 | 6-way rank-average,線性迭代冠軍 |
| 8 | 2026-07-04T11:59:43 | **0.845051** | Phase E-5 樹搜尋(best) | KITCHENBLEND node #55,本場最佳(OOF-only) |

- **突破點 1(exp2→exp3)**:診斷出 scale_pos_weight/class_weights 對排序指標無益反害
  (AUC 非門檻敏感),移除加權並加強 LGB 正則化,0.819008 → 0.832925,為單筆最大躍升。
- **突破點 2(exp3→exp7)**:Phase B 四輪各改一事(Optuna 直接 CV-AUC 調參、入池 + seed
  bag、CatBoost 原生類別診斷、rank-average),0.832925 → 0.83814,增益遞減後依協定停止。
- **突破點 3(exp7→exp8)**:樹搜尋規模實驗(80-節點預算)之 KITCHENBLEND lineage 於
  node #55 混合當時全部 solo 節點(含個別已被淘汰的 CatBoost 變體),0.83814 → 0.845051。

`facts.unparsed` 為空陣列,無法解析之紀錄:無。

## 5. 效能對照:四層消融

| 層級 | 配置 | 分數 | 相對改善 |
|------|------|------|----------|
| tier1 | 基線(Claude Code 直接執行,未引入 skill;exp1 通用批次 3 模型 blend) | 0.81624 | —(基線) |
| tier2 | + kaggle-agent skill 六階段流程(exp3,Phase A 冠軍) | 0.832925 | 見下方算式 |
| tier3 | + self-improvement 線性迭代(exp7,6-way rank-average blend) | 0.83814 | 見下方算式 |
| tier4 | + 樹搜尋(exp8,KITCHENBLEND node #55) | **0.845051** | 見下方算式 |

```
tier1→tier2: 0.832925 − 0.81624 = 0.016685,相對改善 0.016685 / 0.81624 = 2.0441%
tier2→tier3: 0.83814 − 0.832925 = 0.005215,相對改善 0.005215 / 0.832925 = 0.6261%
tier3→tier4: 0.845051 − 0.83814 = 0.006911,相對改善 0.006911 / 0.83814 = 0.8246%
```

本場由導入報告功能後之版本執行,分數自 tier2 起未低於前一層——符合計畫書目標三(效能不退步)。

### 子刻度分解(選填)

依 §5 附錄子刻度分類法(僅作分類字典,不強制逐級測量),下表僅列本場 experiments.json
天然存在對應中間紀錄之子刻度;無對應紀錄之子刻度不列,分數逐字引用自 facts.json。

| 子刻度 | 優化辦法 | 分數 | 出處(exp# 或 tree node) |
|--------|----------|------|--------------------------|
| 1.a | 單模預設參數(tier1 三模型中最佳者:LGB) | 0.81362 | exp1 |
| 1.b | 多模+OOF 權重搜尋 blend(tier1 三模型混合) | 0.81624 | exp1 |
| 2.d | 場內 reflexion 前:48 特徵 + 類別不平衡加權(XGB/CAT 反而退步) | 0.819008 | exp2 |
| 2.d | 場內 reflexion 後:移除加權 + 加強 LGB 正則化 | 0.832925 | exp3 |
| 3.b | Optuna 直接優化完整 5-fold CV AUC(LGB solo) | 0.837305 | exp4 |
| 3.d | 調參版入池 + seed bagging(5-way blend) | 0.837776 | exp5 |
| 4.b | 樹搜尋 harness v2(較早 22-節點掃描之最佳) | 0.841442 | exp8 notes 引用 |
| 4.b | 樹搜尋 harness v2 擴大規模版(80-節點,KITCHENBLEND node #55,本場最佳) | 0.845051 | exp8 |

## 6. 總結

本場資料小(1677 列)且嚴重不平衡(離職 200 筆對留任 1477 筆),無缺失、無重複,品質乾淨。
小樣本決定了兩個貫穿全程的原則:自 exp2 起固定 StratifiedKFold(seed=42)的同一組折以確保
跨實驗可比,以及對模型施加較強正則化,避免在 48 個部分相關的工程特徵上過擬合。

關鍵決策是 exp3 的 Reflexion 迭代:辨識出不平衡加權對 ROC-AUC 這種排序指標是壞交易,移除
加權並收緊 LGB 正則化後,分數由 0.819008 躍升至 0.832925,為全場最大單筆增益。其後 Phase B
每輪只改一件事——Optuna 直接以完整 CV AUC 為目標、調參版入池加 seed bag、CatBoost 原生
類別診斷、rank-average 合併——使線性迭代收在 0.83814。

各層增益來源分佈不均:tier1→tier2(skill 流程之特徵工程 + Reflexion)貢獻最大;
tier2→tier3(Phase B 四輪)增益遞減,依協定誠實停止;tier3→tier4 之樹搜尋以 KITCHENBLEND
混合當時全部 solo 節點突破線性天花板,收在 0.845051。CatBoost 個別結構性偏弱、但其變體在
大池混合中仍可能貢獻多樣性,是本場最有趣的一組正反證據。

可信度方面,exp2–8 全部使用同一組固定 CV 折,分數排序可直接比較;但本場無任何 Kaggle
提交,所有分數皆為 OOF,缺乏排行榜外部驗證是本報告最大的誠實但書。exp8 更是 OOF-only,
未產生 test 預測,若要採用需先重建預測並提交驗證。

**重現本實驗的最短路徑**:見第 7 節。

## 7. 重現指令

```bash
cd /home/tjyen/ai_agents/kaggle

# 資料下載(需先設定 KAGGLE_API_TOKEN;data/ 已存在者可略過)
uv run kaggle competitions download -c playground-series-s3e3 \
  -p competitions/playground-series-s3e3/data

# Stage 1: EDA
uv run python3 competitions/playground-series-s3e3/scripts/eda.py

# exp1(對照組):通用批次管線
uv run python3 competitions/run_competition.py playground-series-s3e3

# exp2:特徵工程 + 不平衡加權(v1)
uv run python3 competitions/playground-series-s3e3/scripts/train.py

# exp3:Reflexion 迭代,移除加權(Phase A 冠軍)
uv run python3 competitions/playground-series-s3e3/scripts/train_v2.py

# exp4:Phase B round1,Optuna 直接優化 5-fold CV AUC
uv run python3 competitions/playground-series-s3e3/scripts/iterate_round1_optuna_lgb.py

# exp5:Phase B round2,調參版入池 + seed bagging
uv run python3 competitions/playground-series-s3e3/scripts/iterate_round2_pool_seedbag.py

# exp6:Phase B round3,CatBoost 原生類別診斷
uv run python3 competitions/playground-series-s3e3/scripts/iterate_round3_catboost_native.py

# exp7:Phase B round4,6-way rank-average(線性迭代冠軍,產出 submission)
uv run python3 competitions/playground-series-s3e3/scripts/iterate_round4_add_catnative_blend.py

# exp8:Phase E-5 樹搜尋規模實驗(best;OOF-only;可中斷/續跑,
# 樹狀態存 experiments_tree_scale.json)
uv run python3 tree_search/run_s3e3_scale.py

# (選用)提交至 Kaggle——本場實際未執行此步,列出僅供後續採用
# uv run kaggle competitions submit -c playground-series-s3e3 \
#   -f competitions/playground-series-s3e3/submissions/<submission.csv> -m "<msg>"
```

執行目錄為專案根目錄 `/home/tjyen/ai_agents/kaggle`;所有 Python 執行皆透過 `uv run` 以
確保套件環境一致。本場為無人值守批次執行,未提交至 Kaggle 排行榜、未觸碰 Kaggle 憑證。
