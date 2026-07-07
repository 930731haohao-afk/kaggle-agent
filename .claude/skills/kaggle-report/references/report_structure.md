# 報告結構規格(7 節)

對齊計畫書定義:計畫書分兩部分——(甲)競賽目的(what/why)、(乙)流程(how)五大元件
(資料規格/模型規格/訓練規格/推論程序/評估指標)。本報告的節次結構在(甲)(乙)之外,
額外加入三節報告自身的附加價值(工具盤點、四層消融、敘述性總結)與既有的軌跡/重現
資訊,合計 7 節:

1. 競賽目的(what/why)= 計畫書(甲)
2. 使用工具與環境 = 報告額外附加,對照計畫書目標一(報告自動生成)所需的工具脈絡
3. 流程(how):五大元件(3.1–3.5)= 計畫書(乙)
4. 實驗軌跡 = 報告額外附加
5. 效能對照:四層消融 = 報告額外附加,對照計畫書目標三(效能不退步)
6. 總結(敘述性)= 報告額外附加
7. 重現指令 = 報告額外附加

每節列出:必填內容 + facts.json 對應來源。「無紀錄」處理見 SKILL.md Hard Rule 4。

## 1. 競賽目的(what / why)
- What:競賽要解決的問題(一段)。來源:competition.description、competition.name
- Why:為何重要/評估指標為何合理(一段)。由 competition.evaluation_metric 與
  problem_type 反推(如 MAE → 對離群值穩健的絕對誤差)
- 基本資訊表:名稱、URL、問題型別、指標、優化方向。來源:competition.*

## 2. 使用工具與環境(NEW)
- 固定表格「工具 | 用途」:只列出本場**實際用到**者,從以下候選中挑選(不足額不必湊):
  LightGBM/XGBoost/CatBoost 梯度提升樹、Optuna 超參搜尋、自建樹搜尋 harness(v2/v3)、
  OptimizedRounder 等指標特化後處理、uv 環境管理、5-fold CV 框架(scikit-learn)。
  來源:best.base_models[].name、best.postprocess、best.cv、experiments[].tuning.method、
  notes 內對 harness/工具版本的敘述。
- 版本號等**未進入 facts.json 的數字**(如 harness v2/v3、Optuna 版本):一律放 fenced
  code block(verify_report.py 對 code block 豁免),或乾脆不寫版本號。
- 1 段(≤4 行)簡述本場工具鏈組合邏輯:LLM(Claude Code)負責決策(選型、特徵工程、何時
  停損),Auto-ML 工具負責執行(超參搜尋、樹搜尋、集成)。

## 3. 流程(how):五大元件
五個子節(3.1–3.5)合起來對應計畫書(乙)的五大元件;子節本身無額外必填內容,五大元件的
必填內容分列如下。

### 3.1 資料規格
- 列數/欄位數/型別概述。來源:eda.n_rows_train/n_rows_test/n_features/numeric_features/
  categorical_features(若 eda 存在);否則 competition.notes、experiments[].n_features
- 目標分佈:mean/median/skew、是否整數、log 轉換候選。來源:eda.target.*
- 資料品質:缺失、重複、離群(zeros)、高共線特徵對。來源:eda.missing_train/
  duplicate_rows_train/numeric_stats/high_collinearity_pairs
- 特別規則(外部資料、每日提交上限)。來源:competition.special_rules
- facts.eda 為 null(missing 含 "eda_summary")或 material_level == "baseline-only"
  → 註明「本場未執行 EDA,以下僅資料基本形狀」;有 facts.eda 者不得再宣稱未執行 EDA

### 3.2 模型規格
- 使用的模型清單與各自分數表。來源:best.base_models(name/score/time_s)
- Ensemble 權重與分數。來源:best.ensemble.weights、best.ensemble.score
- 選型理由(敘述;可參考 STATUS.md 脈絡)
- 注意:facts.best 以 OOF score 最小/最大選出,不考慮 CV scheme 是否一致、也不代表已提交;若 best 與 leaderboard 所屬實驗不同,報告必須如 s3e16 範例明確區分兩者

### 3.3 訓練規格
- CV 方案表:scheme、n_splits、seed。來源:best.cv(seed 無紀錄則寫「無紀錄」)
- Objective 與關鍵超參(有紀錄才寫)。來源:best.base_models[].params
- 為何用此 CV(敘述;可引用 eda.validation_hint 與 status.sections 的 CV 決策脈絡)

### 3.4 推論程序
- 後處理步驟。來源:best.postprocess(無則寫「無後處理紀錄」)
- Submission 檔名與格式。來源:best.submission、competition.id_column/target_column

### 3.5 評估指標
- 指標定義(一句)+ 分數總表:各 base model、ensemble、(若有)Public/Private LB。
  來源:best.*、leaderboard
- CV↔LB gap:僅當 leaderboard 存在;差值以內嵌算式呈現並置於 fenced code block 內(Hard Rule 1;verify_report.py 對 code block 豁免,衍生數字一律走此模式)

## 4. 實驗軌跡
- 若本節(或全報告)正文會出現內部階段代號(Phase A/B/C…、round N、Batch N 等),節首
  必須先放「實驗階段對照表」(R-W8);表 schema 與裁剪規則見寫作規範。
- 逐實驗分數表(id、timestamp、score、source_format)。來源:trajectory
- 突破點敘述:分數躍升發生在哪筆、當時改了什麼(參考 experiments[].notes、STATUS.md)
- unparsed 非空 → 原文列出並註明「無法解析之紀錄」

## 5. 效能對照:四層消融(NEW)
- 固定表格「層級 | 配置 | 分數 | 相對改善」,固定四列(tier1–tier4),對照
  `docs/benchmark_facts.json` 該場之對應 row:
  - tier1:基線(Claude Code 直接執行,未引入 skill;即 generic batch `run_competition.py`)
  - tier2:+ kaggle-agent skill 六階段流程(EDA→特徵工程→建模→評估)
  - tier3:+ self-improvement 線性迭代(Phase B 經驗庫驅動的逐項改進)
  - tier4:+ 樹搜尋(harness v2/v3,結構化搜尋特徵/模型/超參組合空間)
  - 「相對改善」欄:每列相對於**前一層**(非 tier1 累計)的改善百分比,對應消融
    (ablation)邏輯——tier1→tier2 隔離 skill 流程的價值、tier2→tier3 隔離自我改進迭代
    的價值、tier3→tier4 隔離樹搜尋的價值。tier1 列無前層,寫「—(基線)」。
  - 分數欄:直接抄 `docs/benchmark_facts.json` 對應 tierN.value(與該場 facts.json 的
    experiment_id 對應,可交叉核對)。
  - 相對改善數字若非 facts.json/benchmark_facts.json 既有值(逐層增量比例通常需要現算)
    → 算式放 fenced code block(Hard Rule 1)。
  - 若某層與前層數值相同(如 s3e16 tier2=tier3):相對改善誠實寫「0%(迭代未帶來改善,
    依協定停止)」,不得省略或美化。
- 表後固定一句目標三聲明(逐字):「本場由導入報告功能後之版本執行,分數自 tier2 起未
  低於前一層——符合計畫書目標三(效能不退步)」。
- 特殊場次口徑注記(照 `docs/benchmark_summary.md` 慣例,僅相關場次需要):
  - s3e19:tier1 與 tier2/3/4 的 CV 方案不同(KFold vs TimeSeriesSplit),不可比,需
    另行註記改用 tier2→tier3/tier4 比較。
  - s3e16:tier2–tier4 皆為「取整(rounded)」口徑,非 collect.py 一般 adapter 輸出的
    raw blend_oof_mae;需一句話帶過(呼應第 3.5 節已作的 raw/rounded 澄清,不重複解釋)。
  - s3e20:tier1 為同 CV 方案下的 proxy 基線,非嚴格 generic-batch 對照。

## 6. 總結(NEW,敘述性)
- 3–5 段(每段 ≤4 行),完整故事線:資料特性 → 關鍵決策 → 各階段增益來源 → 最終結果與
  可信度。
- 段落內容一律引用前面章節**已出現**的數字或質性描述,禁止出現 facts.json 沒有的新數字。
- 末段加「重現本實驗的最短路徑」小段(≤3 行),指向第 7 節,不重複列出指令本身。

## 7. 重現指令
- 從資料下載到 submission 的完整命令序列(bash code block)
- 來源:STATUS.md 的 Reproduce 節(有則沿用)或依 competitions/<name>/scripts/ 實際檔名組出
- 註明執行目錄與 uv 需求

## 寫作與排版規範(強制)

以下規則對所有報告一律適用,違反者視為未通過 rubric(見 rubric.md R9)。

- **R-W1 來源不進正文**:報告開頭 metadata 區固定一句「本報告所有數字皆出自 facts.json,
  經 verify_report.py 驗證」;正文禁止逐項標「(來源:…)」。僅在少數必要處(如取自 notes
  的數字)以簡短括號註記。
- **R-W2 語意澄清只講一次**:如 raw/rounded 這類跨節概念,在首次出現處用一個 blockquote
  call-out 講清楚,其後一律用「(原始)」「(取整)」二字標記,不再重複解釋。
- **R-W3 無版本沿革**:報告永遠只呈現當前狀態;不寫「本次更新」「先前版本」——版本歷史交給
  git 管理。
- **R-W4 段落上限 4 行**:超過即改條列;每節先表格後敘述,敘述只寫表格讀不出來的 why。
- **R-W5 實驗一律稱 exp N**:表格中最佳值以粗體標示。
- **R-W6 固定表格 schema**(逐節列出,欄名一字不差):
  - §1 基本資訊表:`項目 | 值`(競賽/問題型別/評估指標(方向)/目標欄位/素材等級)
  - §2 工具表:`工具 | 用途`
  - §3.1 資料規格表:`項目 | 值`(train 列數/test 列數/原始欄位數/特別規則摘要)
  - §3.2 實驗總表:`exp | 階段 | 模型/成員 | 特徵數 | 原始 OOF | 決策分數 | 採納`
    (所有實驗一張表;成員細節只有 best 才展開成第二張「best 成員表」:
    `成員 | 權重 | solo 分數 | 備註`)
  - §3.3 訓練規格表:`exp | CV 方案 | folds | seed`
  - §3.4 推論表:`exp | 後處理 | submission 檔 | 已提交`
  - §3.5 排行榜表:`提交 | 日期 | Public | Private`(僅有 LB 的場次);gap/改善算式一律
    fenced code block
  - §4 階段對照表(僅代號出現時需要):`代號 | 白話名稱 | 對應層級`(R-W8)
  - §4 軌跡表:`exp | 時間 | 決策分數 | 階段 | 一句話摘要`;之後至多 3 條「突破點」bullet
    (各 ≤3 行)
  - §5 四層消融表:`層級 | 配置 | 分數 | 相對改善`(固定 tier1–tier4 四列)
- **R-W7 blockquote 僅用於 call-out**(語意澄清/誠實但書),不作一般引文。
- **R-W8 階段代號先定義**:內部階段代號(Phase A/B/C…、round N、Batch N 等)在報告中
  出現之前,必須先以固定的「實驗階段對照表」定義;此表放在第 4 節(實驗軌跡)開頭,
  schema 為 `代號 | 白話名稱 | 對應層級`(欄名一字不差),內容依該場實際出現的代號裁剪
  (未出現者不列)。常用全集(供裁剪時對照,勿整份照抄):
  - Phase A = kaggle-agent skill 六階段首跑(tier2)
  - Phase B = self-improvement 線性迭代(tier3)
  - Phase C–F = 樹搜尋原型與 harness 演進(v1→v3)
  - Phase E-*、G = 樹搜尋執行與成果入帳(tier4)
  - Phase J = 外部想法注入(tier5,規劃中)

  正文優先用白話(「線性迭代第 1 輪」優於「Phase B round 1」);代號僅保留在需要對應
  原始紀錄(experiments.json notes、STATUS.md)之處。
