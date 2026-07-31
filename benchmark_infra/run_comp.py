#!/usr/bin/env python3
"""Launch an AIDE run on a competition from ~/ai_agents/kaggle/competitions.

Usage:
    .venv/bin/python run_comp.py <competition-name> [--steps N] [extra aide args...]

Reads the competition's config.yaml (description, metric, target) to build
aide's goal/eval strings. Runs live under ~/ai_agents/aideml-runs/<comp>/
(aide creates logs/ and workspaces/ there). Requires an LLM API key in the
environment (OPENAI_API_KEY / ANTHROPIC_API_KEY / OPENROUTER_API_KEY / GEMINI_API_KEY).
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml


def load_comp_cfg(cfg_path: Path) -> dict:
    """Parse a competition config.yaml tolerantly.

    Handles two quirks found in the fleet: unquoted colons that break strict
    YAML (fall back to scalar extraction, first occurrence wins), and the
    nested `competition:` schema with alias field names.
    """
    text = cfg_path.read_text()
    try:
        cfg = yaml.safe_load(text)
    except yaml.YAMLError:
        cfg = {}
        for m in re.finditer(r"^\s*([A-Za-z_]+):[ \t]+([^#\n]+)", text, re.M):
            cfg.setdefault(m.group(1), m.group(2).strip())
    if isinstance(cfg.get("competition"), dict):
        cfg = {**cfg, **cfg["competition"]}
    for src, dst in [
        ("metric", "evaluation_metric"),
        ("metric_direction", "optimization_direction"),
        ("target", "target_column"),
    ]:
        if src in cfg and dst not in cfg:
            cfg[dst] = cfg[src]
    cfg.setdefault("description", cfg.get("name", cfg_path.parent.name))
    return cfg

# Competition source root. Defaults to the user's own agent project (playground
# comps live there), but benchmark runs that must stay isolated from that
# project override it with AIDE_COMP_ROOT -- e.g. the 3-agent MLE-bench
# comparison points at ~/ai_agents/bench-comps so AIDE never shares a directory
# with another agent's workspace or submissions.
COMP_ROOT = Path(
    os.environ.get("AIDE_COMP_ROOT", Path.home() / "ai_agents/kaggle/competitions")
)
RUNS_ROOT = Path.home() / "ai_agents/aideml-runs"
AIDE_BIN = Path(__file__).resolve().parent / ".venv/bin/aide"

KEY_VARS = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY", "GEMINI_API_KEY")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("comp", help="competition folder name under competitions/")
    parser.add_argument("--steps", type=int, default=10, help="aide agent.steps (default 10)")
    parser.add_argument(
        "--shim",
        action="store_true",
        help="route LLM calls through the local claude_shim (start claude_shim.py first)",
    )
    args, extra = parser.parse_known_args()

    comp_dir = COMP_ROOT / args.comp
    cfg_path = comp_dir / "config.yaml"
    if not cfg_path.exists():
        sys.exit(f"no config.yaml in {comp_dir}")
    if args.shim:
        os.environ.setdefault("OPENAI_BASE_URL", "http://127.0.0.1:8765/v1")
        os.environ.setdefault("OPENAI_API_KEY", "local-shim")
        extra = [
            "agent.code.model=cc-sonnet",
            "agent.feedback.model=cc-sonnet",
            "report.model=cc-sonnet",
            *extra,
        ]
    elif not any(os.environ.get(v) for v in KEY_VARS):
        sys.exit(f"set one of {KEY_VARS} first — aide needs an LLM API key (or use --shim)")

    cfg = load_comp_cfg(cfg_path)
    direction = cfg.get("optimization_direction", "")
    # aide's CLI parses arg values as YAML — a "word: word" inside the text
    # would silently become a dict and crash config validation
    plain = lambda s: str(s).replace(": ", " - ")
    goal = (
        f"{cfg.get('description', args.comp)}. "
        f"Predict `{cfg.get('target_column')}` for each row of test.csv. "
        "Save the test predictions as submission.csv in the working directory, "
        "with exactly the same columns and id order as sample_submission.csv."
    )
    eval_desc = f"{cfg.get('evaluation_metric', 'see description')}"
    if direction:
        eval_desc += f" ({direction})"

    run_dir = RUNS_ROOT / args.comp
    run_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(AIDE_BIN),
        # hard guarantee: aide works on a private copy, never symlinks back into
        # the kaggle project's data
        "copy_data=True",
        f"data_dir={comp_dir / 'data'}",
        f"goal={plain(goal)}",
        f"eval={plain(eval_desc)}",
        f"agent.steps={args.steps}",
        *extra,
    ]
    print("run dir:", run_dir)
    print("cmd:", " ".join(cmd[:1] + [f'"{a}"' for a in cmd[1:]]))
    sys.exit(subprocess.run(cmd, cwd=run_dir, check=False).returncode)


if __name__ == "__main__":
    main()
