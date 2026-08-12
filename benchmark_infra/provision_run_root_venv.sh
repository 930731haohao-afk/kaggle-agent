#!/bin/bash
# Build the run root's virtualenv, completely, and prove it before a lane depends on it.
#
# WHY THIS IS A SCRIPT AND NOT A COMMAND SOMEONE REMEMBERS.
#
# The run root's .venv was created by hand with `uv sync` inside the sandbox. That resolves
# pyproject.toml and uv.lock, and torch is in NEITHER: pyproject.toml records that torch,
# torchvision and torchaudio are installed separately because of triton resolution issues,
# and setup.sh uses `uv sync --inexact` in the repo specifically so a hand-installed torch
# survives. The run root got a plain sync, so it had no torch at all.
#
# What that costs is a manufactured LOSS, which is the half of "a benchmark can lie" that
# eleven audit rounds never checked. conway-s-reverse-game-of-life's pinned evaluator calls
# stage2_inputs.require_module(..., "cnn_lib", ...) at MODULE IMPORT, and cnn_lib imports
# torch -- so with no torch the evaluator does not import, the lgb node kind is not a
# fallback, and the whole conway tree search is unusable. conway is a competition my-agent
# won on the record (private 0.10875 against 0.11065 and 0.11189); its own recorded champion
# is a CNN at OOF MAE 0.110352 versus 0.12245 for the best LightGBM member. The margin is
# 0.0019 and the gap is 0.012, so the cell flips from first to last.
#
# It also made two deliberate guarantees inert: the run root's SKILL.md promises "PyTorch
# 2.11.0 + CUDA 13.0", and lane_sandbox.sh uses --dev-bind rather than --dev precisely to
# keep /dev/nvidia* -- and no pinned evaluator uses a GPU except through torch.
#
# The root cause is in setup.sh's own comment: "the core benchmark is 15 tabular
# competitions using only lightgbm/xgboost/catboost, torch not needed". True for the old
# 15-competition set, stale from the moment the manifest became 20 and absorbed conway.
#
# Usage:  bash provision_run_root_venv.sh <run-root>
set -uo pipefail

ROOT=${1:?usage: provision_run_root_venv.sh <run-root>}
ROOT=$(cd "$ROOT" 2>/dev/null && pwd) || { echo "no such run root: $1" >&2; exit 4; }
REPO=$(dirname "$(dirname "$(readlink -f "$0")")")
UV=${UV:-/home/tjyen/.local/bin/uv}

[ -f "$ROOT/pyproject.toml" ] || { echo "$ROOT has no pyproject.toml — not a built run root" >&2; exit 4; }

echo "== syncing the locked dependencies =="
# --inexact for the same reason setup.sh uses it: a plain sync REMOVES anything not in the
# lock file, which is exactly how torch would disappear again on the next provisioning run.
( cd "$ROOT" && VIRTUAL_ENV= "$UV" sync --inexact ) || { echo "uv sync failed" >&2; exit 5; }

echo "== installing the out-of-lock deep-learning stack =="
if ( cd "$ROOT" && VIRTUAL_ENV= "$UV" run --no-sync python3 -c "import torch" ) >/dev/null 2>&1; then
  echo "torch already present"
else
  ( cd "$ROOT" && VIRTUAL_ENV= "$UV" pip install torch torchvision torchaudio \
      --index-url https://download.pytorch.org/whl/nightly/cu130 ) \
    || { echo "torch install FAILED — conway cannot run and its result would be a fabricated loss" >&2; exit 5; }
fi

echo "== preflight =="
exec python3 "$REPO/benchmark_infra/preflight_run_root_env.py" --root "$ROOT"
