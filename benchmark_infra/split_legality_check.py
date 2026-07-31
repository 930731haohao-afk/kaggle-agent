"""Split-legality auto-check: flag single-step metric jumps that coincide with a
validation-split change.

Motivating incident (main report, failure-mode analysis): on s3e19 AIDE rewrote its
training loop to fix a LightGBM API issue and incidentally replaced time-based validation
with shuffled KFold -- the metric "improved" 18.42 -> 4.41 in one step and nothing flagged
it; nine further steps tuned against the leaked number. A controlled split-only experiment
put the optimism at 4.5x. This tool is the mechanical alarm that incident argued for.

Scope, stated honestly: the jump trigger catches the *split-switch* failure class (a large
one-step improvement whose code diff touches validation splitting). AIDE's other failure
(s3e16: an optimistic in-sample node chosen over its own nested-CV estimate) produces no
jump -- its guard is the nested-CV gate in the evaluation protocol, not this tool.

Usage:
  VIRTUAL_ENV= uv run python3 benchmark_infra/split_legality_check.py --aide-journal <journal.json>
  VIRTUAL_ENV= uv run python3 benchmark_infra/split_legality_check.py --selftest
Exit status is non-zero when any step is flagged, so lane audits can gate on it.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# one-step relative improvement beyond this triggers inspection (s3e19's was 0.76)
DEFAULT_THRESHOLD = 0.40

# split-related code patterns; a *change* in their presence between parent and child is
# what constitutes evidence, not their mere existence
SPLIT_PATTERNS = {
    "shuffled_kfold": re.compile(r"KFold\([^)]*shuffle\s*=\s*True", re.S),
    "plain_kfold": re.compile(r"(?<!Stratified)(?<!Group)(?<!TimeSeries)KFold\("),
    "train_test_split": re.compile(r"train_test_split\("),
    "timeseries_split": re.compile(r"TimeSeriesSplit\("),
    "group_split": re.compile(r"Group(KFold|ShuffleSplit)\("),
    "date_mask_split": re.compile(r"(year|date|month)\s*[<>=]{1,2}", re.I),
}
# losing one of these while gaining one of the above is the classic leak signature
TIME_AWARE = {"timeseries_split", "group_split", "date_mask_split"}


def _signature(code: str) -> dict[str, bool]:
    return {name: bool(pat.search(code or "")) for name, pat in SPLIT_PATTERNS.items()}


def check_steps(steps: list[dict], threshold: float,
                lower_is_better: bool = True) -> list[dict]:
    """steps: [{step, metric, code}] in execution order. Returns flag records."""
    flags = []
    prev = None
    for s in steps:
        m = s.get("metric")
        if prev is not None and isinstance(m, (int, float)) \
                and isinstance(prev.get("metric"), (int, float)) and prev["metric"] != 0:
            p = prev["metric"]
            improvement = (p - m) / abs(p) if lower_is_better else (m - p) / abs(p)
            if improvement > threshold:
                sig_now, sig_prev = _signature(s.get("code", "")), _signature(prev.get("code", ""))
                gained = [k for k in sig_now if sig_now[k] and not sig_prev[k]]
                lost = [k for k in sig_prev if sig_prev[k] and not sig_now[k]]
                split_changed = bool(gained or lost)
                lost_time_awareness = any(k in TIME_AWARE for k in lost) or "shuffled_kfold" in gained
                flags.append({
                    "step": s.get("step"),
                    "metric_before": p, "metric_after": m,
                    "improvement": round(improvement, 4),
                    "split_signature_gained": gained, "split_signature_lost": lost,
                    "verdict": ("SPLIT_SWITCH_SUSPECTED" if split_changed and lost_time_awareness
                                else "SPLIT_CHANGED" if split_changed
                                else "JUMP_ONLY (inspect manually: no split-pattern change detected)"),
                })
        if isinstance(m, (int, float)):
            prev = s
    return flags


def load_aide_journal(path: Path) -> list[dict]:
    j = json.loads(path.read_text())
    nodes = j.get("nodes", j) if isinstance(j, dict) else j
    steps = []
    for i, n in enumerate(nodes):
        metric = n.get("metric")
        value = metric.get("value") if isinstance(metric, dict) else metric
        steps.append({"step": n.get("step", i), "metric": value, "code": n.get("code", "")})
    return steps


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aide-journal", type=Path)
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--higher-is-better", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)

    if args.selftest:
        return selftest()

    if not args.aide_journal:
        ap.error("--aide-journal required (or --selftest)")
    steps = load_aide_journal(args.aide_journal)
    flags = check_steps(steps, args.threshold, lower_is_better=not args.higher_is_better)
    for f in flags:
        print(json.dumps(f, ensure_ascii=False))
    print(f"{len(flags)} flag(s) over {len(steps)} steps "
          f"(threshold {args.threshold:.0%} one-step improvement)")
    return 1 if flags else 0


def selftest() -> int:
    # synthetic: time-aware -> shuffled KFold with a big jump must flag as SPLIT_SWITCH
    a = {"step": 0, "metric": 20.0, "code": "cv = TimeSeriesSplit(n_splits=5)"}
    b = {"step": 1, "metric": 4.5, "code": "cv = KFold(n_splits=5, shuffle=True, random_state=0)"}
    flags = check_steps([a, b], DEFAULT_THRESHOLD)
    assert len(flags) == 1 and flags[0]["verdict"] == "SPLIT_SWITCH_SUSPECTED", flags
    # same jump, same split -> JUMP_ONLY
    flags = check_steps([a, {**b, "code": a["code"]}], DEFAULT_THRESHOLD)
    assert flags and flags[0]["verdict"].startswith("JUMP_ONLY"), flags
    # small improvement -> no flag
    assert not check_steps([a, {"step": 1, "metric": 19.0, "code": b["code"]}], DEFAULT_THRESHOLD)

    # the real motivating journal, when present on this machine
    real = Path.home() / ("ai_agents/aideml-runs/playground-series-s3e19/logs/"
                          "0-meaty-zebra-of-improvement/journal.json")
    if real.exists():
        flags = check_steps(load_aide_journal(real), DEFAULT_THRESHOLD)
        assert any(f["step"] == 11 and f["verdict"] == "SPLIT_SWITCH_SUSPECTED" for f in flags), \
            f"failed to flag the s3e19 step-11 incident: {flags}"
        print("real-journal replay: s3e19 step 11 flagged as SPLIT_SWITCH_SUSPECTED")
    print("split_legality_check selftest: all sections passed")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
