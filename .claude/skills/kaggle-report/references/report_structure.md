# 報告結構規格(8 節)

每節列出:必填內容 + facts.json 對應來源。「無紀錄」處理見 SKILL.md Hard Rule 4。

## 1. 競賽目的
- What:競賽要解決的問題(一段)。來源:competition.description、competition.name
- Why:為何重要/評估指標為何合理(一段)。由 competition.evaluation_metric 與
  problem_type 反推(如 MAE → 對離群值穩健的絕對誤差)
- 基本資訊表:名稱、URL、問題型別、指標、優化方向。來源:competition.*

## 2. 資料規格
- 列數/欄位數/型別概述。來源:competition.notes(若記載)、experiments[].n_features
- 特別規則(外部資料、每日提交上限)。來源:competition.special_rules
- material_level == "baseline-only" → 註明「本場未執行 EDA,以下僅資料基本形狀」

## 3. 模型規格
- 使用的模型清單與各自分數表。來源:best.base_models(name/score/time_s)
- Ensemble 權重與分數。來源:best.ensemble.weights、best.ensemble.score
- 選型理由(敘述;可參考 STATUS.md 脈絡)

## 4. 訓練規格
- CV 方案表:scheme、n_splits、seed。來源:best.cv(seed 無紀錄則寫「無紀錄」)
- Objective 與關鍵超參(有紀錄才寫)。來源:best.base_models[].params
- 為何用此 CV(敘述)

## 5. 推論程序
- 後處理步驟。來源:best.postprocess(無則寫「無後處理紀錄」)
- Submission 檔名與格式。來源:best.submission、competition.id_column/target_column

## 6. 評估指標
- 指標定義(一句)+ 分數總表:各 base model、ensemble、(若有)Public/Private LB。
  來源:best.*、leaderboard
- CV↔LB gap:僅當 leaderboard 存在;差值以內嵌算式呈現(Hard Rule 1)

## 7. 實驗軌跡
- 逐實驗分數表(id、timestamp、score、source_format)。來源:trajectory
- 突破點敘述:分數躍升發生在哪筆、當時改了什麼(參考 experiments[].notes、STATUS.md)
- unparsed 非空 → 原文列出並註明「無法解析之紀錄」

## 8. 重現指令
- 從資料下載到 submission 的完整命令序列(bash code block)
- 來源:STATUS.md 的 Reproduce 節(有則沿用)或依 competitions/<name>/scripts/ 實際檔名組出
- 註明執行目錄與 uv 需求
