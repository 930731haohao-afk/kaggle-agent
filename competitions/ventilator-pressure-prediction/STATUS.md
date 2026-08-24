# ventilator-pressure-prediction — STATUS (run2, 2026-07-29)

**Final CV (honest 5-fold OOF, masked MAE on u_out==0): 0.19222 snapped (raw 0.19361).**
**Model: BiLSTM-4x256 seq2seq, 53 per-timestep features, bf16 AMP, 260-epoch OneCycle, 5-fold KFold on breaths; submission = median across 5 fold models + snap-to-pressure-grid.**
**vs run1 (0.24004 OOF single fold, mlebench grade 0.2315): −0.048 CV; medal threshold 0.1638 likely still out of reach.**

## Pipeline
- Features: run1's validated 41 + 12 new (R×C cross one-hot 9, u_in cummax, cummean, reverse-cumsum). All per-timestep, (n_breaths, 80, 53), z-normalized.
- Model: Linear→LayerNorm→SiLU input proj, 4-layer bidirectional LSTM h=256, 2-layer head. Masked L1 loss (u_out==0 only). AdamW lr 2e-3, OneCycle 260 epochs, batch 1024, grad clip 5.
- Speed: bf16 autocast + GPU-resident tensors + cudnn.benchmark on GB10 → ~7 s/epoch vs run1's 46.8 s (≈6×). Made full 5-fold × 260 ep feasible (~2.6 h) inside the 4 h budget where run1 managed 1 fold × 100 ep.
- Foreground constraint: training chunked via checkpoint/resume (`scripts/train.py --chunk-seconds 520`, exit 3 = resume) — 18 sequential ~9-min foreground runs.
- Fold val MAEs: 0.19432 / 0.19599 / 0.19258 / 0.18934 / 0.19583.
- Postprocess: median across 5 fold test preds (MAE-optimal aggregator; scored-region fold spread std 0.115, mean-vs-median absdiff 0.037), then snap to the 950-value training pressure grid (validated on OOF: 0.19361→0.19222).

## Tree search decision
**Not used — deliberately.** One evaluation node here is a full LSTM training run (~30 min/fold, ~2.6 h for an honest 5-fold CV). A ~60-node search is impossible in a 4 h budget; even a shallow search would trade the one affordable full-CV model for several unconverged ones. Skill fallback applied: single linear iteration pass (run1 baseline → run2 improved recipe). Tree search stays appropriate for cheap-eval tabular comps, not per-node-hours deep-learning comps.

## Artifacts
- `submission.csv` — 603,600 rows, id order matches sample_submission.csv, median+snap.
- `scripts/train.py` — full pipeline (features→train→ensemble→submission), checkpoint/resume.
- `scripts/work/` — fold test preds, feature cache, checkpoint.
- `oof_lstm_run2.npy`, `cv_result.json`, `experiments.json` (v2 schema).
- Run1 archived under `.run1-archive/`.

## If more budget
Second seed per fold + median over 10 members (~+2.6 h); LSTM+GRU hybrid or deeper (5×384) model — public solutions suggest ~0.15 territory needs those plus ~300 epochs.
