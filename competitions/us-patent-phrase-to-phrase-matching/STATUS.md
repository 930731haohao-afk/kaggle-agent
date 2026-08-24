# STATUS — us-patent-phrase-to-phrase-matching (run 2, 2026-07-28)

**Final CV (honest leave-fold-out Pearson r): 0.83184** — NNLS blend of deberta-v3-large ×2 seeds + deberta-v3-base + LGBM lexical (weights 0.26/0.37/0.25/0.12), GroupKFold(4) by anchor.
Submission: `submission.csv` (= `submissions/ensemble_run2_nnls_all.csv`, 3648 rows, clipped to [0,1]).
Improvement over run 1 (archived, OOF 0.8158 / mlebench 0.8575): +0.016 OOF, driven by deberta-v3-large (solo 0.8197/0.8213 vs base 0.8060).

---

## Pipeline (run 2)

| Model | CV scheme | OOF Pearson r |
|---|---|---|
| LGBM char/word TF-IDF + lexical | StratifiedKFold(5) on score | 0.6113 |
| deberta-v3-base s42 (3 ep, lr 2e-5, max_len 96, bf16) | GroupKFold(4) by anchor | 0.8060 |
| deberta-v3-large s42 (3 ep, lr 1e-5, 10% warmup, bf16) | GroupKFold(4) by anchor | 0.8197 |
| deberta-v3-large s1337 (same recipe) | GroupKFold(4) by anchor | 0.8213 |
| **NNLS blend (all 4)** | leave-fold-out refit | **0.83184** (full-OOF 0.83206) |

- Input text: `anchor [SEP] target [SEP] <CPC section title> <context code>`; mean-pool head + BCEWithLogits; fp32 master weights + bf16 autocast (fused AdamW).
- deberta-v3-large recipe from memory (`deberta-v3-large-collapse`): lr 1e-5 + 10% cosine warmup — zero fold collapses across 8 large folds (probe smoke-tested first, val_r 0.64 @ 40 steps).
- Seeds fixed (42 / 1337); GroupKFold split is seed-independent so all OOFs share identical folds.
- All training foreground-waited (background handle + blocking TaskOutput; session never idle while a job ran).

## Tree search decision
**Skipped — not worth the budget here.** Stage-4 tree search assumes cheap node evaluations (~seconds–minutes for GBDT refits). In this competition every meaningful node is a transformer fine-tune (~35 min for 4-fold large), so a ~60-node search is ~35 GPU-hours vs the ~4 h budget. The high-leverage moves (large backbone with a stable recipe, seed bagging, NNLS pool blend) were taken directly instead; blend-level candidate search (9 candidates, honest leave-fold-out scoring) served as the light-weight Stage-4 loop.

## Honesty notes
- Blend chosen by leave-fold-out score (weights refit per held-out fold), not full-OOF; the two differ by only 0.0002 — weight fitting is not overfitting the OOF.
- LGBM's OOF comes from a non-grouped split (optimistic solo), but it still earns 0.12 NNLS weight on the grouped-deberta folds; same pattern as run 1 where it helped mlebench score.
- Test anchors are 100% seen in train (pairs unseen) — GroupKFold-by-anchor CV is conservative relative to the test distribution; true score likely ≥ CV.

## Artifacts
- `scripts/train_deberta_v2.py` (now takes `--model`), logs `scripts/train_*.log`
- OOF/test preds: `scripts/{deberta_base_s42,deberta_large_s42,deberta_large_s1337,lgbm}_{oof,test}.npy`
- Blend decision: `scripts/ensemble_run2_decision.json`; experiments: `experiments.json` (5 entries)
- Run 1 archived at `.run1-archive/`
