---
name: kaggle-vision-agent
description: 'Discovery-first agent pipeline for IMAGE/VISION Kaggle competitions (classification-style: digit-recognizer, cifar-10, dogs-vs-cats, plant/leaf/medical image datasets). Runs the staged flow: frozen pretrained backbones as infrastructure -> cheap discovery experiments (cached embeddings + linear/LightGBM probes, then short fine-tunes) -> full fine-tune of the few survivors -> convex ensemble over real OOFs -> distil results into the vision [INT] experience library. Use whenever the user wants to work on an image competition, train/evaluate image models, build an image submission, extract embeddings, choose a backbone, or says things like "vision", "image comp", "CNN", "backbone", "fine-tune", "digit-recognizer", "cifar". Do NOT use for tabular or NLP competitions (use kaggle-agent) or for general PyTorch debugging not tied to a competition workspace. Core identity: the agent DISCOVERS what works from this data — community/competition recipes ([EXT]) are deliberately NOT injected first.'
---

# Kaggle Vision Agent — discovery-first image-competition pipeline

Design doc: `docs/vision/vision-pipeline-discovery-first.drawio.png` (+
`docs/vision/vision-adapter-design-v2.drawio.png` for the tree-search-as-ensembler variant).
This skill turns that design into an executable staged workflow.

## Identity: discover, don't copy

This project's structural differentiator (measured in the three-way benchmark — report
`docs/REPORT_v7.tex`, score table `benchmark_results/rerun_three_way.csv`; do not open either
during a run, both name competitions and their results) is the
**original-solution engine**: the agent discovers what works **from this competition's data**, instead of
reproducing the strongest public kernel. Therefore:

- **Pretrained backbones are infrastructure, not idea injection.** Everyone starts from pretrained
  weights; using them does not compromise originality.
- **[EXT] community recipes are deferred.** Do not seed the run with competition write-ups / public
  kernels. External-idea injection is a later, explicitly opt-in stage (same [INT]-first philosophy as
  the tabular [INT] / [EXT] split; the [EXT] prose bank was archived on 2026-08-07 to
  `knowledge/archive_pre_structured/idea_bank.md` and its live successor is the structured
  `knowledge/knowledge_base.json`, rendered per competition by `knowledge/task_priors_for.py`).
- **Every claim needs score evidence** — the house rule. Log every experiment.

## The cost ladder (why the pipeline is staged)

Vision evals are expensive; the pipeline is ordered so each tier's cost is justified by the previous
tier's evidence:

| Tier | Cost/config | What it can discover | What it CANNOT see |
|------|------------|----------------------|---------------------|
| 0. frozen embeddings (once) | ~minutes total | — (produces the substrate) | — |
| 1. frozen probe (linear/LightGBM head) | **seconds** | which backbone's features fit, head type, resolution effects | **augmentation, LR, fine-tune dynamics** |
| 2. cheap fine-tune (last block, 1-2 epochs, low res) | **minutes** | augmentation policy, LR regime, unfreeze depth | long-schedule effects |
| 3. full fine-tune | **hours** | final performance | — |

The frozen probe is **blind to augmentation** (embeddings come from un-augmented images) — never use
tier 1 to judge augmentation; that is tier 2's job. Promote only tier-N winners to tier N+1.

## Workflow stages

Work through stages in order; each has a reference file with the detailed procedure.

### Stage V1: Setup & data
See [references/01_setup.md](references/01_setup.md).
Environment check (torch+CUDA on the GB10, timm), competition data loading (CSV-pixel and
image-folder formats), config.yaml, and the CV design for images (stratified k-fold; GROUP by
patient/scene/source when leakage risk exists).

### Stage V2: Discovery (the heart)
See [references/02_discovery.md](references/02_discovery.md).
Tier 0-2: extract & cache frozen embeddings for a small backbone menu, probe-rank them, run the
agent's own cheap experiments (backbone, head, resolution at tier 1; augmentation & LR at tier 2).
Check `vision/derisk_results.json` first if present — it holds this machine's measured evidence on
how much to trust probe rankings.

### Stage V3: Fine-tune the survivors
See [references/03_finetune.md](references/03_finetune.md).
Full fine-tune ONLY the few configs tier 1-2 justified (typically 3-6, chosen for diversity, not
just top-k score). Produce real OOF + test predictions per config; cache as npz.

### Stage V4: Ensemble & submit
See [references/04_ensemble_submit.md](references/04_ensemble_submit.md).
Thin convex weight solve (scipy/NNLS — a search tree is not needed for a convex problem) over the
real OOFs + TTA + post-processing/calibration; format and validate the submission.

### Stage V5: Distil to the vision [INT] library
See [references/05_int_library.md](references/05_int_library.md).
After the run, write score-cited lessons to `knowledge/vision_experience.md` — the vision analogue
of `experience.md`. These become priors for the NEXT image competition.

## Data separation: vision NEVER mixes with tabular

Vision and tabular artifacts are fully separated at every layer — do not write vision results into
any tabular file, and vice versa:

| Layer | Tabular | Vision |
|---|---|---|
| competition workspaces | `competitions/<comp>/` | **`competitions_vision/<comp>/`** |
| experiments.json / reports (REPORT.md, PDFs) | inside `competitions/<comp>/` | inside `competitions_vision/<comp>/` |
| [INT] experience library | `knowledge/experience.md` | **`knowledge/vision_experience.md`** |
| [EXT] idea bank (deferred) | `knowledge/knowledge_base.json` (live; prose bank archived at `knowledge/archive_pre_structured/idea_bank.md`) | `knowledge/vision_idea_bank.md` (when the opt-in stage arrives) |
| MLflow mirror | `mlflow.db` | **`mlflow_vision.db`** |
| cross-comp aggregate docs | `docs/` | `docs/vision/` |
| infra experiments (derisk etc.) | `tree_search/` | `vision/` |

Prior matching (`suggest_priors`-style) for a vision run reads ONLY the vision library — a tabular
lesson citing s3eX evidence is not evidence for an image model, and cross-domain matches would
pollute attribution.

## Reproducibility (same two-layer standard as tabular)

See [references/06_reproducibility.md](references/06_reproducibility.md) — Layer 1 (fixed
seeds/folds, traceable numbers, OOF cache, uv-locked env, separate MLflow store) applies to every
run; Layer 2 (bit-level determinism gate: retrain → byte-identical OOF) certifies submissions.
The torch determinism preamble is bundled at `assets/torch_determinism.py`; its bit-exactness is
**measured** on this machine (GATE: PASS, `vision/derisk_determinism_gate.py`, 2026-07-17).

Pre-run lint gate: same rule as the tabular agent — before a NEW script's first logged run,
`uv run --with ruff==0.16.1 ruff check <script> --select F821,F841,B023,B006,E722,PLW1510,RUF059 --isolated`
and fix all findings; after the run is logged the script is frozen as-run, never re-linted.
Not `uvx`: it installs into uv's own tool directory, which the benchmark sandbox mounts
read-only, so the gate fails with `Read-only file system` on the first script of every
competition. `--with` resolves into the per-run ephemeral environment under `~/.cache`, which
is writable. See `.claude/skills/kaggle-agent/SKILL.md` — the two skills share this gate.

## Non-negotiables (inherited from the tabular agent)

- **Experiment logging is MANDATORY** via `experiment_log.log_experiment_v2()`
  (`.claude/skills/kaggle-agent/assets/utils/experiment_log.py`) — same logger, same schema,
  writing to the VISION workspace's experiments.json.
- **Never compare scores across CV schemes**; log the scheme with every experiment.
- **Fixed seeds + fixed folds** shared across every arm you intend to compare (byte-comparable);
  certification runs use the full determinism preamble (see reproducibility reference).
- **Respect compute**: GPU runs go through background execution with logs; long trainings need a
  subprocess-level timeout (`signal.alarm` cannot interrupt a native fit/forward).
- **uv for everything**: `uv run python`, `.venv/bin/python` for GPU scripts.
- **Never print or commit credentials** (`KAGGLE_API_TOKEN` etc.).

## Utilities bundled with this skill

- `assets/embed_cache.py` — extract & cache frozen-backbone embeddings (CSV-pixel or image-folder
  input; incremental, atomic writes). The tier-0 workhorse; import it rather than rewriting it.
- Existing evidence: `vision/derisk_probe_vs_finetune.py` (+ `vision/derisk_results.json` when the
  run has finished) — the probe-vs-finetune ranking-agreement experiment.

## When something is out of scope

Detection / segmentation / video need task-specific heads, losses, and metrics this skill does not
yet cover — say so and plan explicitly rather than forcing the classification flow onto them.
