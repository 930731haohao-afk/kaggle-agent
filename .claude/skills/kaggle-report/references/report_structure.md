# 報告結構規格(4 節)

對齊計畫書定義:計畫書分兩部分——(甲)競賽目的(what/why)、(乙)流程(how)五大元件
(資料規格/模型規格/訓練規格/推論程序/評估指標)。本報告的節次結構直接映射計畫書結構:
第 1 節 = (甲);第 2 節(五個子節 2.1–2.5)= (乙);第 3、4 節為報告額外附加的軌跡與重現
資訊,不屬計畫書兩部分之一。

每節列出:必填內容 + facts.json 對應來源。「無紀錄」處理見 SKILL.md Hard Rule 4。

## 1. 競賽目的(what / why)
- What:競賽要解決的問題(一段)。來源:competition.description、competition.name
- Why:為何重要/評估指標為何合理(一段)。由 competition.evaluation_metric 與
  problem_type 反推(如 MAE → 對離群值穩健的絕對誤差)
- 基本資訊表:名稱、URL、問題型別、指標、優化方向。來源:competition.*

## 2. 流程(how):五大元件
五個子節(2.1–2.5)合起來對應計畫書(乙)的五大元件;子節本身無額外必填內容,五大元件的
必填內容分列如下。

### 2.1 資料規格
- 列數/欄位數/型別概述。來源:competition.notes(若記載)、experiments[].n_features
- 特別規則(外部資料、每日提交上限)。來源:competition.special_rules
- material_level == "baseline-only" → 註明「本場未執行 EDA,以下僅資料基本形狀」

### 2.2 模型規格
- 使用的模型清單與各自分數表。來源:best.base_models(name/score/time_s)
- Ensemble 權重與分數。來源:best.ensemble.weights、best.ensemble.score
- 選型理由(敘述;可參考 STATUS.md 脈絡)
- 注意:facts.best 以 OOF score 最小/最大選出,不考慮 CV scheme 是否一致、也不代表已提交;若 best 與 leaderboard 所屬實驗不同,報告必須如 s3e16 範例明確區分兩者

### 2.3 訓練規格
- CV 方案表:scheme、n_splits、seed。來源:best.cv(seed 無紀錄則寫「無紀錄」)
- Objective 與關鍵超參(有紀錄才寫)。來源:best.base_models[].params
- 為何用此 CV(敘述)

### 2.4 推論程序
- 後處理步驟。來源:best.postprocess(無則寫「無後處理紀錄」)
- Submission 檔名與格式。來源:best.submission、competition.id_column/target_column

### 2.5 評估指標
- 指標定義(一句)+ 分數總表:各 base model、ensemble、(若有)Public/Private LB。
  來源:best.*、leaderboard
- CV↔LB gap:僅當 leaderboard 存在;差值以內嵌算式呈現並置於 fenced code block 內(Hard Rule 1;verify_report.py 對 code block 豁免,衍生數字一律走此模式)

## 3. 實驗軌跡
- 逐實驗分數表(id、timestamp、score、source_format)。來源:trajectory
- 突破點敘述:分數躍升發生在哪筆、當時改了什麼(參考 experiments[].notes、STATUS.md)
- unparsed 非空 → 原文列出並註明「無法解析之紀錄」

## 4. 重現指令
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
  - §2.1 資料規格表:`項目 | 值`(train 列數/test 列數/原始欄位數/特別規則摘要)
  - §2.2 實驗總表:`exp | 階段 | 模型/成員 | 特徵數 | 原始 OOF | 決策分數 | 採納`
    (所有實驗一張表;成員細節只有 best 才展開成第二張「best 成員表」:
    `成員 | 權重 | solo 分數 | 備註`)
  - §2.3 訓練規格表:`exp | CV 方案 | folds | seed`
  - §2.4 推論表:`exp | 後處理 | submission 檔 | 已提交`
  - §2.5 排行榜表:`提交 | 日期 | Public | Private`(僅有 LB 的場次);gap/改善算式一律
    fenced code block
  - §3 軌跡表:`exp | 時間 | 決策分數 | 階段 | 一句話摘要`;之後至多 3 條「突破點」bullet
    (各 ≤3 行)
- **R-W7 blockquote 僅用於 call-out**(語意澄清/誠實但書),不作一般引文。
