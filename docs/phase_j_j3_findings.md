# Phase J J-3 歸因發現:階段5 注入 hook 已建置,但尚未接入搜尋消費端

> 這是一個**誠實的負面/架構級結果**,需要決策(見「下一步的分岔」)。日期:2026-07-08。

## 一句話

外部想法注入(階段5)的 hook 本身正確,但現行樹搜尋架構下 **`tree['priors']` 是 write-only**
(只寫來列印/記錄,沒有任何搜尋算子讀它),所以「純 [INT] vs [INT]+[EXT]」的單一變因對照
得到**構造性的 delta = 0**——不是噪音級的接近零,是**逐位元相同**。

## 怎麼測的(單一變因)

以 s3e3(AUC,資料小、全搜尋 ~17s,可乾淨確定性重跑)為對象:

- 建 `tree_search/run_s3e3_v4.py`,**唯一搜尋相關改動** = 把 driver 開頭那次
  `suggest_priors(COMP_META)` 換成 `harness_v4.suggest_priors_v4(COMP_META, mode=V4_MODE)`
  (mode 讀環境變數)。其餘種子、折、預算、mutation、評估器全部不動。
- 正確性閘門:(1) `mode='off'` 的先驗與 `hv2.suggest_priors` **逐字相同**(通過);
  (2) `mode='ext'` 對本場**淨新注入 2 條** [EXT]——EXT-12(rank averaging)、EXT-14
  (對抗驗證)(通過)。
- 同 seed=42、同折、各跑到底,比較全域最佳 OOF。

## 結果

| | 全域最佳節點 | AUC | tree['priors'] 條數 |
|---|---|---|---|
| off([INT] only) | node #11 | 0.841442 | 8 |
| ext([INT]+[EXT]) | node #11 | 0.841442 | 10(多 2 條 [EXT]) |

- **delta (ext − off) = +0.00000000**。
- 逐節點比對(22 節點):結構性差異 0 處、分數差異 0 處(max|Δscore|=0)、18 個快取 OOF
  向量 max|Δ|=0。**off 與 ext 逐位元相同。**

## 根因(grep + 實證雙重確認)

`suggest_priors` 的回傳只寫入 `tree['priors']` 供列印/記錄;種子(`SOLO_SEEDS`)、mutation
佇列、`propose_child`、`select_next_parent`、評估器**沒有一個讀 `tree['priors']`**。driver 裡
`[PRIOR Pk]` 標籤是作者在種子/佇列裡**手寫的字串常數**,`prior_usage_summary` 也只 regex 解析
這些手寫標籤、從不回溯 `tree['priors']`。harness_v2/v3 內查無任何 `tree['priors']` 讀取點。
**bit-identical 的結果本身就是證明**:唯一差異(2 條額外先驗)若被任何算子讀到必會改變某處;
它什麼都沒改 → `tree['priors']` 確為 write-only。此為**架構級**(四個候選 driver 同一模式),
非 s3e3 特性。

## 重要的區辨(不要誤讀這個 0)

1. **注入機制本身是對的**:harness_v4 正確併池 [INT]+[EXT]、去重、閘門通過。問題在**消費端**。
2. **階段 4 的價值仍然真實**:樹搜尋 stage4 > stage3 的增益來自**搜尋機制本身**(候選樹、
   回溯、OOF 權重搜尋、探索爆發),與 `suggest_priors` 無關——這些結果不受本發現影響。
3. **經驗庫知識確實影響了搜尋,但透過的是「作者手寫種子」而非自動注入**:driver 作者讀
   experience.md 後把先驗手寫進種子配置;`suggest_priors` 這個**函式的輸出**是裝飾性的。
   因此各場報告「階段 4.3 先驗注入+去重 — 使用」在「知識有引導」意義上成立,但「自動注入
   plumbing 被搜尋消費」的隱含意義**不成立**——這點值得在報告語言上誠實校準(非緊急)。

## 下一步的分岔(需決策)

要真正量測階段5 的邊際價值,必須先讓 priors **通電**——把 `tree['priors']` 接進搜尋決策,例如:

- 把 **EXT-12(rank averaging)** 轉成「加一個 rank-average blend 成員」的候選節點;
- 把 **EXT-14(對抗驗證)** 轉成「跑對抗驗證、據以重排/加權 CV」的候選動作;
- 更一般地:讓 `propose_child` / 種子邏輯真的讀 `tree['priors']`,把每條先驗映射成一個
  可執行的 mutation/候選,並在 mutation 記錄裡標 provenance([INT]/[EXT])。

**選項 A**:做這個 wiring(工程量中等,但會改變樹搜尋對**所有場**的行為 → 需重跑、可能重開
既有 stage4 結果)。**選項 B**:接受此誠實發現,把階段5 描述為「注入 hook 建置完成、待接入
消費端」,不宣稱其增益。

> 這兩個選項牽動研究方向與既有結果,留待督導/使用者決定;在此之前不自行大改。
> 證據檔:`tree_search/run_s3e3_v4.py`(+ 過程狀態檔,未入版控的實驗產物)。
