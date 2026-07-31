"""Reproduce-then-verify: compare a reproduced kernel's submitted public score against the
score its source kernel claims.

Motivating incident (main report, "Votes Do Not Equal Quality"): the s3e19 kernel NVIDIA
reproduced claims "same idea as the 2nd-place solution", but the port missed the
year-scaling step that decides the outcome (48.27 submitted vs. a claimed neighbourhood of
4.67) -- and nothing in the frozen method ever compared the reproduced score against the
source's claim. s5e1 is the positive case: the reproduction matched the kernel's claimed
public LB digit-for-digit, verified manually. This tool mechanizes that comparison.

POST-FREEZE IMPROVEMENT. The benchmark froze the NVIDIA method with no improvements of any
kind; this check therefore gates nothing retroactively and is not part of any reported run.
It exists for post-study use, where a reproduction that cannot match its source's claimed
score should be flagged as an incomplete port before it burns a submission.

Contract: the reproduction lane records `claimed_lb` (the source kernel's self-reported
public score, read at selection time) in its result.json next to `kernel`/`votes`; after
submission, run this tool with the realized public score.

Usage:
  VIRTUAL_ENV= uv run python3 benchmark_infra/verify_reproduction.py <run-dir> --public <score>
  VIRTUAL_ENV= uv run python3 benchmark_infra/verify_reproduction.py --selftest
Exit non-zero on MISMATCH.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_REL_TOL = 0.02


def verify(result: dict, public: float, rel_tol: float = DEFAULT_REL_TOL) -> dict:
    claimed = result.get("claimed_lb")
    out = {"kernel": result.get("kernel"), "public": public, "claimed_lb": claimed}
    if claimed is None:
        out["status"] = "UNVERIFIABLE"
        out["detail"] = ("result.json records no claimed_lb; record the source kernel's "
                         "self-reported public score at selection time")
        return out
    claimed = float(claimed)
    rel = abs(public - claimed) / max(abs(claimed), 1e-12)
    out["rel_diff"] = round(rel, 4)
    if rel <= rel_tol:
        out["status"] = "VERIFIED"
        out["detail"] = f"within {rel_tol:.0%} of the source's claimed public score"
    else:
        out["status"] = "MISMATCH"
        out["detail"] = (f"reproduction is {rel:.0%} away from the claimed score -- "
                         f"the port is likely missing a decisive step (cf. s3e19's "
                         f"missing year-scaling, 48.27 vs ~4.67)")
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", nargs="?", type=Path)
    ap.add_argument("--public", type=float)
    ap.add_argument("--rel-tol", type=float, default=DEFAULT_REL_TOL)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)

    if args.selftest:
        r = verify({"kernel": "k", "claimed_lb": 0.05726}, 0.05726)
        assert r["status"] == "VERIFIED", r
        r = verify({"kernel": "k", "claimed_lb": 4.67}, 48.27)
        assert r["status"] == "MISMATCH", r
        r = verify({"kernel": "k"}, 1.0)
        assert r["status"] == "UNVERIFIABLE", r
        print("verify_reproduction selftest: all sections passed")
        return 0

    if not args.run_dir or args.public is None:
        ap.error("run_dir and --public required (or --selftest)")
    result = json.loads((args.run_dir / "result.json").read_text())
    out = verify(result, args.public, args.rel_tol)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 1 if out["status"] == "MISMATCH" else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
