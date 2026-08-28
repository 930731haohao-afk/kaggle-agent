# Reproduction Instructions (REPRODUCE)

This project's deliverable aims to be **lightweight and reproducible**, replacing Docker containerization (rationale at the end). The core approach:
a frozen `uv.lock` + a single `setup.sh` + these instructions. Anyone who follows this document can
rebuild an environment consistent with this project on their own machine and reproduce the results.

---

## 0. Prerequisites

| Requirement | Description |
|------|------|
| **uv** | The only hard requirement. `curl -LsSf https://astral.sh/uv/install.sh \| sh`, then after installing `export PATH="$HOME/.local/bin:$PATH"`. |
| **Python 3.13** | No need to install it yourself — `uv sync` will automatically download the corresponding version according to `.python-version` (3.13). |
| Platform | The development machine is **arm64 Linux (Ubuntu 24.04)**. x86_64 also works (see "Known Limitations"). |
| Kaggle credentials | **Only needed when downloading data / submitting**; not needed for purely reproducing local CV scores. |
| torch | **Only needed for image/NLP competitions**; not needed for the core tabular results. |

---

## 1. One-Command Build + Self-Check

```bash
bash setup.sh
```

`setup.sh` will, in order: find uv → print platform/arm64 hints → `uv sync` (build `.venv` with frozen versions
per `uv.lock`) → self-check that the core ML stack imports and print versions → report torch / PDF backend status.
The whole process is idempotent and can be run repeatedly.

To install torch (for image/NLP) at the same time:

```bash
bash setup.sh --torch
```

Expected self-check output (versions per uv.lock):

```
python 3.13.12
OK  numpy  2.4.2   OK  pandas 3.0.1   OK  sklearn 1.8.0
OK  lightgbm 4.6.0  OK  xgboost 3.2.0  OK  catboost 1.2.10
OK  optuna 4.9.0    OK  yaml 6.0.3     OK  tqdm 4.67.3
```

---

## 2. Reproduce Results (from shallow to deep)

```bash
# (a) full test suite -- the fastest overall health check
uv run pytest -q

# (b) rebuild the cross-competition benchmark facts table (extracted from each competition's experiments.json, with consistency asserts)
uv run python3 docs/scripts/build_benchmark_table.py

# (c) verify that all numbers in a report are traceable to facts.json (gate, exit 0 = pass)
uv run python3 .claude/skills/kaggle-mlspec-report/assets/verify_report.py \
    docs/ml_specs/md/ventilator-pressure-prediction.md competitions/ventilator-pressure-prediction/facts.json

# (d) reproduce a single competition's best solution (including the digit-for-digit OOF reproduction gate; produces a submission only on pass), using s5e10 as an example
uv run python3 competitions/playground-series-s5e10/scripts/06_rebuild_tree_best.py

# (e) generate a PDF with table of contents / page numbers from Markdown
bash .claude/skills/kaggle-mlspec-report/assets/md2pdf.sh \
    docs/ml_specs/md/playground-series-s6e1.md docs/ml_specs/pdf/playground-series-s6e1.pdf
```

> (b)–(e) start from `experiments.json` / cached OOF and **do not need the raw competition data**.
> For (d), to rerun from raw data you first need to download that competition's `data/` (see Section 3); the full "from raw data to
> best solution" path for each competition is its `scripts/` (`04_train_blend.py` → `05_iterate.py` → `06_rebuild_tree_best.py`) plus the `tree_search/` driver.

---

## 3. Data and Credentials (only needed when downloading/submitting)

```bash
# the Kaggle token is always injected via an environment variable, never written to a file (see the CLAUDE.md security rules)
export KAGGLE_API_TOKEN=$(cat ~/.kaggle/kaggle_api_token.txt | tr -d '[:space:]')

# download a competition's data into that competition's data/ (already gitignored, not version-controlled)
uv run kaggle competitions download -c playground-series-s5e10 \
    -p competitions/playground-series-s5e10/data
```

---

## 4. Known Limitations (trade-offs relative to Docker)

The lightweight alternative freezes the **Python version + all package versions** (`uv.lock`), but **does not freeze the operating system and
system libraries**. Practical impact:

- **arm64 (development machine)**: `uv.lock` works directly, zero risk.
- **x86_64**: The core ML packages also have prebuilt wheels for x86_64, usually without issue. If some package has no
  wheel for your platform and needs on-the-spot compilation that fails, it is most likely a missing compiler or system development library;
  install what the error indicates.
- **torch is not in `uv.lock`**: it is installed separately (`setup.sh --torch`) due to a triton dependency-resolution issue,
  and is only needed for image/NLP competitions. The core tabular benchmark and tree search do not depend on it.
- **PDF backend**: `weasyprint` (preferred, with table of contents and page numbers) or `chromium` (fallback, no page numbers);
  if neither is available, the report `.md` can still be produced, only without a PDF.

---

## Why Not Docker

The goal (a reproducible deliverable) can be achieved either way; the trade-offs are as follows:

| | Docker image | **This approach (uv.lock + setup.sh)** |
|---|---|---|
| Reproduction strength | Strongest (freezes even OS/system libs) | Strong (freezes Python and all package versions) |
| arm64 risk | High: installing ML packages from scratch in a clean container easily hits "no arm64 wheel" → on-the-spot compilation failure | Low: uses a "known to work" environment as the baseline |
| Build/delivery | Slow, image is GB-scale | Fast, deliverable is a few text files |
| What the other party needs | Install Docker | Install uv (a single executable) |

The development machine is arm64, and building ML packages inside a clean arm64 container carries a real risk of failure; this approach uses an
already-working environment as the baseline, which is sufficient for the actual scenario of "reproducing on a similar machine," hence its adoption.
