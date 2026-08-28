# Kaggle Agent Skills — packaging and installation

This directory holds this project's five Claude Code skills. Four are live and can be copied
as a set into another Claude Code environment; the fifth is deprecated and kept only as a
historical path. They divide the work as follows.

## The five skills

| skill | what it does | when it triggers | status |
|-------|--------------|------------------|--------|
| **kaggle-agent** | Core tabular competition pipeline: experience-library retrieval gate → Stage 0.5 problem dossier → EDA → CV design → features → modeling (LGB/XGB/CAT + Optuna) → evaluation with tree search → submission | starting or taking over a Kaggle tabular competition | live |
| **kaggle-vision-agent** | Discovery-first pipeline for image competitions: frozen backbones as infrastructure → cheap embedding/probe discovery → fine-tune the survivors → convex ensemble over real OOFs → distil into the vision [INT] library | image competitions, backbone choice, fine-tuning, `competitions_vision/` | live |
| **kaggle-safe-submit** | Validate a submission CSV against `sample_submission` and check the remaining daily quota BEFORE uploading, log the run, then submit via the Kaggle CLI | any submit / upload-predictions / "is this file valid?" request | live |
| **kaggle-mlspec-report** | Turn a finished run's raw records into a structured report: `collect.py` extracts numbers → `verify_report.py` number-traceability check → `md2pdf.sh` emits a PDF with TOC and page numbers | a competition has finished and needs a reviewable, reproducible report | live |
| **kaggle-agent-self-improvement** | Superseded predecessor of `kaggle-agent` | — | **DEPRECATED — do not invoke** |

**Relationships**: `kaggle-agent` is the tabular entry point and `kaggle-vision-agent` is its
image counterpart; the two never share an experience library, because a tabular lesson is not
evidence for an image model. `kaggle-safe-submit` and `kaggle-mlspec-report` are independent and
pair with either.

`kaggle-agent-self-improvement` **must not be used.** Its own SKILL.md says so: it carries a
stale Stage 3 that reads MEMORY.md and scans competition directories, which is exactly the
cross-competition contamination path the benchmark's isolation protocol forbids, and it predates
Stage 0.5, the typed injection contract and the binding experience-library retrieval gate. It is
kept only because historical run logs cite its path. Use `kaggle-agent` instead.

## Installation

1. **Copy the skills**: copy each skill directory into the target environment's
   `.claude/skills/`, for example
   ```bash
   cp -r .claude/skills/kaggle-agent          <target-project>/.claude/skills/
   cp -r .claude/skills/kaggle-vision-agent   <target-project>/.claude/skills/
   cp -r .claude/skills/kaggle-safe-submit    <target-project>/.claude/skills/
   cp -r .claude/skills/kaggle-mlspec-report  <target-project>/.claude/skills/
   ```
   Claude Code detects `.claude/skills/<name>/SKILL.md` automatically and activates it from
   its `description` / trigger phrases.

2. **Python environment (managed with uv)**: the skills' scripts run via `uv run python3`. The
   target project needs a `pyproject.toml` with the following dependencies; `uv sync` then
   installs them:
   ```
   modeling:       lightgbm xgboost catboost scikit-learn optuna
   data handling:  pandas numpy pyyaml tqdm
   Kaggle:         kaggle (CLI; needed only to submit)
   ```
   The full list is this project's `pyproject.toml`. `kaggle-vision-agent` additionally needs
   torch, which is installed separately (`bash setup.sh --torch`) and is not in `uv.lock`.

3. **PDF dependencies (kaggle-mlspec-report only)**: `md2pdf.sh` prefers **weasyprint** (the only
   backend supporting TOC page numbers via target-counter plus a page footer) and falls back to
   **chromium** (no page numbers); it also needs Python's **markdown** package.

4. **Kaggle credentials (needed only to submit)**: injected as an environment variable, **never
   written to a file**:
   ```bash
   export KAGGLE_API_TOKEN=$(cat ~/.kaggle/kaggle_api_token.txt | tr -d '[:space:]')
   ```
   See the security rules in the project-root `CLAUDE.md`.

## Directory layout (it differs per skill)

```
kaggle-agent/                    kaggle-vision-agent/
├── SKILL.md                     ├── SKILL.md
├── references/                  ├── references/          # 01_setup … 06_reproducibility
│   └── 00_problem_dossier.md    ├── assets/
│       … 07_tree_search.md      │   ├── embed_cache.py
└── assets/                      │   └── torch_determinism.py
    ├── templates/               └── evals/evals.json
    │   # eda / feature / train / submit script skeletons
    └── utils/
        # data_loader / evaluation / experiment_log / kaggle_auth

kaggle-mlspec-report/            kaggle-safe-submit/
├── SKILL.md                     ├── SKILL.md
├── references/                  ├── references/submission_checklist.md
│   ├── mlspec_structure.md      └── scripts/validate_submission.py
│   └── rubric.md
└── assets/
    ├── collect.py               # deterministic number extraction → facts.json
    ├── verify_report.py         # number-traceability check
    ├── eda_summary.py
    ├── md2pdf.sh
    └── report_style.css
```

`kaggle-agent-self-improvement/` mirrors `kaggle-agent/`'s layout with one extra reference file,
`07_self_improvement.md`, and without `00_problem_dossier.md`.

## Versioning and environment

- The skill sources are version-controlled in this repo; `__pycache__/` and `*.pyc` are excluded
  by `.gitignore`.
- Randomness: modeling scripts fix seeds. Bit-for-bit reproduction across processes additionally
  requires LightGBM `deterministic=True, force_row_wise=True, num_threads=<fixed>` (see
  `knowledge/experience.md`).
- The development machine is arm64 Linux. Every package listed above is pure Python or has an
  arm64 wheel, so `uv sync` suffices and no Docker image is needed.
