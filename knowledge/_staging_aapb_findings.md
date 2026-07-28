# 待併入 experience.md 的 AAPB 發現(暫存,RUN2 結束後再併)

**為何暫存而非直接寫入**:`knowledge/experience.md` 已於 2026-07-27 git-revert 成 benchmark 前狀態,
好讓 RUN2 的三個 my-agent 格(cell 3 patent、cell 6 ventilator、cell 9 siim)讀到**同一份**知識庫。
RUN2 進行中寫入會讓先跑的格與後跑的格帶著不同知識,那是 run 內部的不一致 —— 跟污染不同,但一樣
會讓格與格不可比。RUN2 九格跑完後再併入。

來源:AAPB(Autonomous Agent Prediction Beta)離線評測,`kaggle_aapb/analysis/baseline_scores_seed42.csv`
＋ `dataset_summary.csv`。**只有 6/16 個資料集有分數**,全部結論的樣本數都是 6,務必按此強度標註。

---

## 可寫進「依資料型態」節(證據紮實)

- **表格二分類的難度上限由訊號密度決定,不是樣本數 —— 「小資料比較難」是錯的先驗。**
  同一批合成資料中,500 列的集可以拿到 0.87,而 15000 列的集只有 0.71。據此,看到小資料時
  不該預期低分、也不該自動加重正則;該看的是特徵數與特徵型別組成。
  | 證據:AAPB train_15(n=500, feat=30, ordinal 17 個)private AUC 0.8716
  vs train_01(n=14957, feat=12)private AUC 0.7135。同一 pipeline、同一 seed。

## 只能寫進「待驗證想法」節(n=6,含單一反例,不可當規則用)

- **假說:混合(blend)在「同時最小又最稀疏」的資料上失效,其餘一律有效。** 6 場中 5 場
  blend 勝過單一 LGBM(+0.0011 ~ +0.0230),唯一反例是 n=500 且僅 9 特徵那場(−0.0004)。
  但同為 9 特徵、n=1060 的另一場 blend 卻 +0.0135,所以**不能用特徵數單一條件當閘門**——
  先前一度如此推論,已證偽。需要更多資料集才能定出閘門條件。
  | 證據:train_13(500×9)blend 0.6514 vs lgbm 0.6518;train_05(1060×9)blend 0.6860 vs lgbm 0.6725;
  train_15(500×30)blend 0.8716 vs lgbm 0.8486。

- **假說:OOF→private 落差最大的集,就是混合失效的那一集。** train_13 的 OOF 0.68229 →
  private 0.6514,落差 −0.031,是其餘五場(±0.010 內)的 3–15 倍。若此關聯在 16 場全評測後
  仍成立,則「OOF 與 private 落差」可當成執行期的自我診斷訊號 —— agent 不需要 private 分數
  也能從 fold 間變異偵測到自己在過擬,進而自動退回單模。這是可操作的,但目前 n=1。
  | 證據:同上表,train_13 OOF 0.68229 → private 0.6514。

## 待辦

1. 跑完整 16 資料集離線評測(需重算力 → 排 RUN2 格間空檔),把上述兩條假說的樣本數從 6 提到 16。
2. 若假說二成立,把「fold 間變異 → 自動退回單模」寫成 my-agent 的執行期規則,並評估包成
   AAPB skill(skills 上限 10 格,要挑)。
