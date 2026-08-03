# Submission Checklist / 提交檢查清單

<!--
 註解: 這是 SKILL.md 第三層(on-demand)的資源。只有在驗證出問題、或遇到不熟的
       競賽型別時才需要讀。2026-08-03 audit 之前這個檔案並不存在,SKILL.md 卻指向它。
 Note: Third-layer, on-demand resource for SKILL.md step 2. Before the 2026-08-03 audit
       SKILL.md pointed here but the file did not exist.
-->

`scripts/validate_submission.py` 自動檢查下面標記 **[auto]** 的項目;標 **[manual]** 的要由
agent 自己確認(通常需要競賽頁面或 config.yaml 的資訊)。
Items marked **[auto]** are enforced by `scripts/validate_submission.py`; **[manual]**
items need the agent's own judgement (competition page / config.yaml).

## 兩層判定 / The two tiers

驗證器把問題分成兩層,退出碼不同:
The validator separates failures into two tiers with distinct exit codes:

| Tier | Exit | 意義 / Meaning | 可否放行 / Waivable |
|---|---|---|---|
| STRUCTURAL | 1 | 檔案本身壞了 — Kaggle 會拒收,或對到錯的列 | 絕不 / never |
| SUSPICIOUS | 2 | CSV 合法,但幾乎確定是 bug(常數、全 0、超出範圍、機率/標籤搞反) | 只能用 `--allow-suspicious "理由"` 明確放行 |
| PASS | 0 | 通過 | — |

SUSPICIOUS 之所以「可放行但要說理由」:常數預測在極少數競賽是**合法且正確**的
(退化的 baseline、單一類別的 metric)。但它同時也是最常見的 bug 症狀,所以預設擋下,
要放行就必須寫下理由 —— 不允許無聲通過。
Constant predictions are legal on a handful of competitions, which is why this tier can
be waived; they are also the single most common symptom of a broken pipeline, which is
why the waiver must be written down rather than assumed.

## STRUCTURAL — 一定要過 / must pass

- **[auto] 列數** 與 sample_submission 完全相同(= 測試集列數)。
- **[auto] 欄位名稱與順序** 與 sample_submission 完全相同,含大小寫。
- **[auto] ID 無重複**。
- **[auto] ID 對齊** — 不只是集合相同,**順序也要相同**。順序錯掉的檔案在多數競賽仍會被
  接受並照 ID 對齊評分,但只要 ID 有型別轉換(`1` vs `1.0` vs `"1"`)就會整份錯位。
- **[auto] 無 NaN**。
- **[auto] 無 ±inf** — `isnull()` 對 inf 回傳 False,所以只檢查 NaN 的 gate 會放它過去。
  inf 通常來自未做 clip 的 log/除法逆轉換。
- **[manual] 檔案編碼與行尾** — UTF-8、無 BOM、無尾端空白列(`to_csv(..., index=False)` 即可)。
- **[manual] 沒有多餘的 index 欄** — 忘了 `index=False` 會多一欄,欄位檢查會抓到。

## SUSPICIOUS — 預設擋下 / blocks by default

- **[auto] 常數預測**(只有一個相異值)。最常見成因:模型沒 fit、預測到錯的欄、
  或整個 pipeline 回傳了 fill value。
- **[auto] 全 0**。佔位用的 `np.zeros(len(test))` 從來沒被填進去。
- **[auto] 超出範圍**。機率必須落在 [0, 1];其他情況以訓練目標的範圍 ±50% span 為界,
  超出通常代表尺度錯誤(log 空間沒還原、target transform 沒逆轉換、對到錯的欄)。
- **[auto] 機率 / 硬標籤搞反**(兩個方向都會靜默丟掉分數):
  - metric 吃硬標籤(accuracy / F1 / QWK)卻交了 0.73 → 需要先 threshold 或 argmax。
  - metric 吃機率(AUC / logloss)卻交了 0/1 → `predict_proba` 被寫成 `predict`,
    AUC 會塌到接近 0.5 的階梯值。
- **[manual] 分佈明顯偏離訓練目標**。驗證器只印出 `pred mean` vs `train target mean`,
  不擋;差一個數量級以上時要自己回頭查。

## 各問題型別 / Per-problem-type rules

### Binary classification
| Metric | 交什麼 / Submit | 常見錯誤 / Common failure |
|---|---|---|
| AUC / ROC-AUC | 正類機率(可以不校準,只看排序) | 交 0/1 硬標籤,排序資訊全失 |
| LogLoss | 校準過的機率,建議 clip 到 `[1e-15, 1-1e-15]` | 交 0 或 1 → 無限大損失 |
| Accuracy / F1 | 硬標籤,threshold 從 OOF 上調 | 交機率;或在測試集上調 threshold(洩漏) |

`--metric auc --expect probability` 會讓驗證器啟用 [0,1] 與「不可全是 0/1」的檢查。

### Multiclass classification
- 每類一欄的機率:欄位順序**必須**與 sample_submission 一致,且每列加總 ≈ 1
  (**[manual]** — 驗證器目前只檢查最後一欄,多欄機率要自己加驗 row-sum)。
- 單欄硬標籤:值必須落在訓練標籤集合內(`--train/--target` 才能檢查)。
- 標籤型別:sample 用字串類別名就交字串,別交 0/1/2 的整數編碼。

### Regression
- 若訓練時對 target 做了轉換(log1p、Box-Cox、standardise),**預測後一定要逆轉換**。
  這是 range 檢查最常抓到的東西。
- RMSLE 競賽:預測不得為負(`np.clip(pred, 0, None)`),否則 Kaggle 直接報錯。
- 目標若為整數計數,依 metric 決定要不要 round;RMSE 通常**不要** round。

### Ordinal / QWK
- 交整數等第,不是連續值。threshold 邊界要在 OOF 上最佳化後固定下來,不能用測試集調。

## 提交前最後三問 / Three questions before the upload

1. 這個檔案是不是由**非 diagnostic** 的最佳實驗產生的?
   (見 `references/06_submission.md` 第 1 步與 `utils/experiment_log.py` 的 `_entry_is_diagnostic`)
2. 今天的額度還夠嗎?(`daily_submission_limit`)
3. experiments.json 裡有沒有這一筆,LB 分數回來時對得回去嗎?
