#!/usr/bin/env python3
"""Prove the run root's environment can run every competition, before lane 1 starts.

WHY. The run root's .venv was built with a plain `uv sync`, and torch is not in uv.lock --
pyproject.toml says so explicitly. Nothing checked. conway-s-reverse-game-of-life's pinned
evaluator imports a Stage-2 module that imports torch, so that lane would have failed to
import its evaluator at all, and the recorded result says conway is a competition my-agent
WINS. A missing package produces a fabricated LOSS, which is the direction eleven audit
rounds never looked at: leakage inflates my-agent and makes a WIN false, a broken
environment deflates it and makes a LOSS false, and the two are equally fatal to the one
question this benchmark exists to answer.

WHAT IS CHECKED, and why each is here rather than in a comment somewhere:

  the GBDT stack        every one of the 20 pinned evaluators trains through it
  torch                 the only GPU consumer in the whole tree; conway's model class
  torch.cuda            the sandbox keeps /dev/nvidia* on purpose (--dev-bind, not --dev);
                        a CPU-only torch silently reintroduces the handicap that mistake
                        would have caused, and the frozen reference lanes had the GPU
  the skill's own claim  SKILL.md tells the lane "Linux with NVIDIA GPU ... PyTorch + CUDA".
                        An instruction the environment does not honour is a lie the lane
                        plans around.

Deliberately NOT a full import of each evaluator: several build features at module import,
which costs minutes each and would make this too expensive to run before every launch.

Usage:  preflight_run_root_env.py --root <run-root>
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

# (module, why it must be present, fatal)
REQUIRED = [
    ("lightgbm", "every pinned evaluator's default learner", True),
    ("xgboost", "a node kind in the search space of most competitions", True),
    ("catboost", "13 of the 20 evaluators pair signal.alarm with a CatBoost runner", True),
    ("sklearn", "CV splitters, metrics and the linear family", True),
    ("pandas", "every evaluator's data path", True),
    ("numpy", "every evaluator's data path", True),
    ("scipy", "blend weight search", True),
    ("torch", "conway-s-reverse-game-of-life's cnn_lib; the only GPU consumer", True),
]

PROBE = r"""
import json, importlib.util
out = {}
for m in %(mods)r:
    out[m] = importlib.util.find_spec(m) is not None
try:
    import torch
    out["_cuda"] = bool(torch.cuda.is_available())
    out["_torch_version"] = torch.__version__
except Exception as exc:
    out["_cuda"] = False
    out["_torch_version"] = f"unimportable: {type(exc).__name__}: {exc}"
print(json.dumps(out))
"""


def probe(root: str) -> dict:
    py = os.path.join(root, ".venv/bin/python")
    if not os.path.exists(py):
        return {"_error": f"no interpreter at {py} — the run root has no virtualenv"}
    code = PROBE % {"mods": [m for m, _w, _f in REQUIRED]}
    # Same PYTHONPATH the sandbox gives a lane: torch lives OUTSIDE .venv, because uv
    # rebuilds .venv from uv.lock and torch is not in it. Probing without this would report
    # a missing package that the lane can actually import, and probing the venv alone once
    # reported a torch that a single uv invocation then deleted.
    env = dict(os.environ, PYTHONPATH=os.path.join(root, ".extra-site"))
    r = subprocess.run([py, "-c", code], capture_output=True, text=True, check=False, env=env)
    if r.returncode != 0:
        return {"_error": f"the run root's interpreter failed: {r.stderr.strip()[-400:]}"}
    try:
        return json.loads(r.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {"_error": f"unreadable probe output: {r.stdout[-200:]}"}


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", required=True)
    ap.add_argument("--allow-cpu", action="store_true",
                    help="accept a torch without CUDA. Only for a machine that has no GPU: "
                         "on this one it means the lanes train on CPU while the frozen "
                         "reference lanes did not, which voids the comparison.")
    a = ap.parse_args(argv)
    root = os.path.abspath(a.root)

    got = probe(root)
    if "_error" in got:
        print(f"PREFLIGHT FAILED — {got['_error']}", file=sys.stderr)
        return 5

    missing = [(m, why) for m, why, fatal in REQUIRED if fatal and not got.get(m)]
    if missing:
        print(f"PREFLIGHT FAILED — {len(missing)} package(s) the run needs are not in "
              f"{root}/.venv:", file=sys.stderr)
        for m, why in missing:
            print(f"  {m}: {why}", file=sys.stderr)
        print("\nRun: bash benchmark_infra/provision_run_root_venv.sh " + root,
              file=sys.stderr)
        return 5

    if not got.get("_cuda"):
        msg = (f"torch reports no CUDA device (version {got.get('_torch_version')}). The "
               f"sandbox keeps /dev/nvidia* deliberately and the frozen reference lanes had "
               f"the GPU; running without it handicaps my-agent in a way that reads as a "
               f"fair loss.")
        if not a.allow_cpu:
            print("PREFLIGHT FAILED — " + msg, file=sys.stderr)
            print("Pass --allow-cpu only if this machine genuinely has no GPU.",
                  file=sys.stderr)
            return 5
        print("WARNING — " + msg)

    print(f"preflight OK — {len(REQUIRED)} required package(s) present in {root}/.venv, "
          f"torch {got.get('_torch_version')}, CUDA {got.get('_cuda')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
