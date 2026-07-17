# Stage V3 — Fine-tune the survivors (bounded, expensive tier)

Full fine-tune ONLY the configs promoted from Stage V2 — typically 3-6. This is the budget you
protected by doing discovery cheaply; do not blow it re-exploring here.

## Per-config protocol

- **Full training data** (not the discovery subsample), the competition CV scheme from config.yaml
  (usually 5-fold), fixed seeds.
- Full unfreeze, the tier-2-chosen augmentation + LR, cosine or one-cycle schedule, early stopping
  on the fold's validation metric. AMP (`torch.autocast`) on the GB10 for speed.
- **Outputs per config** (the ensemble contract, mirroring the tabular OOF cache):
  - OOF prediction matrix (N_train x n_classes, probabilities)
  - test prediction matrix (mean over folds)
  - per-fold scores + wall time
  - saved atomically to `competitions_vision/<name>/data/oof_<config_id>.npz`
- Log each config with `log_experiment_v2` the moment its CV completes.

## Execution discipline

- Run in the **background** with a log file; monitor the log, never block the session on a training.
- Wrap each fold's training in a subprocess-level timeout (kill the process group on hang —
  `signal.alarm` cannot interrupt native code; evidence: the tabular s3e7 CatBoost 28-min hang).
- Checkpoint per fold; a crashed run resumes from the last completed fold, never from scratch.
- If one config dominates all folds early and another is clearly failing, it is fine to stop the
  failing one — record the decision and partial evidence in experiments.json.

## Optional cheap add-ons (evidence-backed from tabular)

- **Seed bagging** the strongest fine-tuned config (retrain with 1-2 different seeds) buys a real,
  cheap ensemble-diversity gain when member variance is high — same finding as tabular s3e14/s3e9.
- Add, don't replace: keep the original config's OOF in the pool alongside the seed-bag variants
  and let the Stage-V4 weight solve decide (weights of 0 cost nothing).
