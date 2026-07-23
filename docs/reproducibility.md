# Reproducibility (against plan §4 "Modularity and Reproducibility" + Goal 5)

> Against *Summer Internship Plan* §4 and Goal 5, this describes the project's reproducibility mechanisms, the mapping to the plan, and
> the honest layering of "the reproducibility standard was raised midway through the project." Date: 2026-07-08.

## Plan requirements ↔ current status

| Plan §4 "Modularity and Reproducibility" | Current status |
|---|---|
| Fix torch/numpy/scikit-learn random seeds | ✅ **seed=42** in every competition (model `random_state`, `KFold(shuffle, seed=42)`, numpy `RandomState(42)`). This project's tabular tasks are mainly GBDT (LGB/XGB/CAT), also fixed; torch is rarely used but follows the same principle |
| Manage packages with uv | ✅ `pyproject.toml` + `uv.lock`; `uv sync` rebuilds the environment |
| Log experiments with MLflow | ✅ `docs/scripts/export_to_mlflow.py` mirrors each competition's `experiments.json` into a local **sqlite MLflow store** (`mlflow.db`; 284 runs / 38 competitions). The authoritative record remains experiments.json / facts / the report pipeline; MLflow is the corresponding standardized query layer |
| Follow the data/models/trainers/utils structure | Equivalent, differently named (mapping below) |

## Structural mapping (data/models/trainers/utils ↔ actual layout)

| Plan module | Actual location in this project |
|---|---|
| **data** | `competitions/<comp>/data` (gitignored raw data) + `config.yaml` (spec) |
| **models / trainers** | `competitions/<comp>/scripts` (`features.py` + `train*.py` / `04_train_blend.py` / `05_iterate.py`); `tree_search/` (`harness_v2/v3` + `run_*_v3.py` tree-search driver) |
| **utils** | `utils/` (data_loader / evaluation / experiment_log), `templates/`, `.claude/skills/kaggle-mlspec-report/assets` (collect / verify / md2pdf report pipeline) |

## Reproducibility layering (honest record: the standard was raised midway)

There are two strictness levels, both real, both honestly recorded:

**Layer 1 (plan standard) — all 15 competitions**: fixed seeds + fixed folds + traceable numbers (verify gate) + OOF cache
+ uv-locked environment + MLflow logging. S3 (the early batch of 10 competitions) is done at this layer, all CV-only (competitions closed, not submitted).

**Layer 2 (extra bit-level determinism) — the 5 cross-season competitions**: Layer 1 + the gate "retraining emits the exact same OOF bit-for-bit"
(pinning LightGBM `deterministic`/`force_row_wise`/`num_threads`).
- Introduced for **certifying the submission file** (the retrained test predictions must match the validated OOF); along the way discovered and fixed the LGBM
  cross-process non-determinism pitfall (see memory lgbm-determinism-oof-gate).
- This layer is **beyond the plan's requirement** — the plan asks for "fixed seeds," and **fixed seeds ≠ bit-level determinism**: that LGBM pitfall
  comes from ad-hoc timing choosing row/col-wise + thread accumulation order, which fixed seeds cannot escape.

**Why S3 is not brought back up to Layer 2**: S3 is entirely CV-only and not submitted, so Layer 2's purpose (certifying submissions) does not apply to it; S3 already
meets the plan standard (Layer 1). It is retained as reasonable progress in the project's evolution, honestly recorded, without retraining. (Finalized 2026-07-08,
see `docs/plan_compliance_audit.md`.)

## How to reproduce

```bash
uv sync                        # rebuild the environment from uv.lock
# per-competition reproduction = its scripts/ (04_train_blend -> 05_iterate -> 06_rebuild_tree_best) + the tree_search/ driver; report is docs/ml_specs/md/playground-series-<comp>.md
# the one-command delivery-layer build + self-check is in setup.sh / REPRODUCE.md
uv run python3 docs/scripts/export_to_mlflow.py                       # regenerate the MLflow records
uv run --with mlflow mlflow ui --backend-store-uri sqlite:///mlflow.db  # open the MLflow UI to inspect
```
