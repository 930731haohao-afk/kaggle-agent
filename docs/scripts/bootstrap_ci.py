#!/usr/bin/env python3
"""Paired-bootstrap confidence intervals for the stage ladder (OOF-noise rigor check).

WHAT THIS ANSWERS
-----------------
Each competition's pipeline reports a "ladder" of cross-validated (OOF) scores:
    stage 2 (base 3-way blend) -> stage 3 (tuned/seed-bagged blend) -> stage 4
    (tree-search mega-blend winner).
The per-step gains are often tiny (0.00x%). This script quantifies, for every stage
transition, whether the improvement is larger than the *sampling noise of the OOF
estimate itself* -- i.e. whether a 95% paired-bootstrap CI on the metric difference
excludes 0 (significant) or contains 0 (inside the noise).

METHOD (paired bootstrap, B=5000)
---------------------------------
* All stages of one competition share the SAME cached OOF row order (train file order,
  identical KFold(5, shuffle, seed=42) across every stage), so predictions are paired
  row-for-row.
* Each bootstrap iteration draws ONE resample of row indices (with replacement, size n)
  and recomputes EVERY stage's metric on that SAME resample. Using one shared resample
  per iteration preserves the between-stage correlation, which is exactly what makes a
  paired test powerful for highly-correlated blends.
* For a transition before->after we collect the signed improvement
      delta = (metric_after - metric_before)      if the metric is maximized
      delta = (metric_before - metric_after)      if the metric is minimized
  so "improvement" is always positive. We report delta's point estimate (full data),
  its 95% percentile CI [2.5, 97.5], whether that CI contains 0, and P(delta>0) over
  the bootstrap draws.

Metrics are recomputed with the SAME post-processing each competition's pipeline used
(rmse clipped to [0,1]; R2 clipped to [0,100]; ROC-AUC; accuracy at the best swept
threshold 0.10..0.90 + 0.5). Fast count-weighted evaluators are used so B=5000 x 5
stages x 5 competitions runs in minutes; each is validated to reproduce the reference
(features.py / sklearn) implementation to machine precision, and every stage's
unresampled metric is asserted to equal the recorded ladder value before bootstrapping
(the gate -- a mismatch means an OOF file maps to the wrong stage and the run aborts).

SCOPE LIMIT (read before quoting a p-value)
-------------------------------------------
This bootstrap quantifies ONLY the sampling variability of the OOF *estimate* on the
fixed 517K-630K training rows -- "if we had drawn a slightly different validation set,
how much would this gap wobble?". It does NOT capture the variance of re-running the
whole search with a different seed / different fold split / different Optuna draws
(that needs retraining, which this analysis deliberately does not do). A gap can be
bootstrap-significant here and still be seed-fragile under a full re-run. Treat these
CIs as a *lower bound* on the uncertainty of each reported gain, not the whole story.

Stage-1 (the February baseline) OOF vectors are not cached, so the 1->2 transition is
omitted; the ladder analysed here starts at stage 2.

Reproduce:  uv run python3 docs/scripts/bootstrap_ci.py
Outputs a human table and, with --json, a machine-readable results blob on stdout.
Read-only w.r.t. all data/ and cache/ files.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, roc_auc_score

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
B_DEFAULT = 5000
SEED = 42
GATE_TOL = 1e-5  # recorded ladder values are rounded to 6 dp

# --------------------------------------------------------------------------- #
# competition registry: metric, direction, target loader, stage-4 winner node,
# and the recorded ladder values (from competitions/<comp>/experiments.json).
# --------------------------------------------------------------------------- #
def _y_s5e10(tr): return tr["accident_risk"].to_numpy(np.float64)
def _y_s6e1(tr):  return tr["exam_score"].to_numpy(np.float64)
def _y_s6e2(tr):  return (tr["Heart Disease"] == "Presence").astype(np.int64).to_numpy()
def _y_s4e1(tr):  return tr["Exited"].to_numpy(np.int64)
def _y_s4e11(tr): return tr["Depression"].to_numpy(np.int64)

COMPS = {
    "s5e10": dict(full="playground-series-s5e10", pre="stage", metric="rmse",
                  direction="minimize", yload=_y_s5e10, winner=29,
                  recorded=dict(stage2=0.056027, stage3_r1=0.056017, stage3_r2=0.055982,
                                stage3_r3=0.055976, stage4=0.055968)),
    "s6e1":  dict(full="playground-series-s6e1", pre="stage", metric="r2",
                  direction="maximize", yload=_y_s6e1, winner=27,
                  recorded=dict(stage2=0.786793, stage3_r1=0.786794, stage3_r2=0.786998,
                                stage3_r3=0.787060, stage4=0.787183)),
    "s6e2":  dict(full="playground-series-s6e2", pre="stage", metric="auc",
                  direction="maximize", yload=_y_s6e2, winner=43,
                  recorded=dict(stage2=0.955197, stage3_r1=0.955310, stage3_r2=0.955501,
                                stage3_r3=0.955510, stage4=0.955529)),
    "s4e1":  dict(full="playground-series-s4e1", pre="tier", metric="auc",
                  direction="maximize", yload=_y_s4e1, winner=44,
                  recorded=dict(stage2=0.894302, stage3_r1=0.894302, stage3_r2=0.894354,
                                stage3_r3=0.894354, stage4=0.894393)),
    "s4e11": dict(full="playground-series-s4e11", pre="tier", metric="accuracy",
                  direction="maximize", yload=_y_s4e11, winner=26,
                  recorded=dict(stage2=0.939488, stage3_r1=0.939545, stage3_r2=0.939758,
                                stage3_r3=0.939900, stage4=0.940235)),
}

# canonical headline transitions + fine-grained round-by-round breakdown
TRANSITIONS = [
    ("2->3", "stage2", "stage3_r3"),      # base blend  -> final stage-3 blend
    ("3->4", "stage3_r3", "stage4"),      # stage-3     -> tree-search winner
    ("2->4", "stage2", "stage4"),         # total ladder (base -> winner)
]
FINE = [
    ("2->3r1",   "stage2",    "stage3_r1"),
    ("3r1->3r2", "stage3_r1", "stage3_r2"),
    ("3r2->3r3", "stage3_r2", "stage3_r3"),
    ("3r3->4",   "stage3_r3", "stage4"),
]

# --------------------------------------------------------------------------- #
# reference metrics (bit-exact re-impl of each comp's features.py post-processing)
# --------------------------------------------------------------------------- #
def ref_metric(kind, y, p):
    if kind == "rmse":
        p = np.clip(p, 0.0, 1.0); return float(np.sqrt(np.mean((p - y) ** 2)))
    if kind == "r2":
        p = np.clip(p, 0.0, 100.0); return float(r2_score(y, p))
    if kind == "auc":
        return float(roc_auc_score(y, p))
    if kind == "accuracy":
        yb = np.asarray(y).astype(bool); p = np.asarray(p, np.float64)
        ths = np.union1d(np.arange(0.10, 0.90 + 1e-9, 0.01, dtype=np.float64), [0.5])
        return float(((p[:, None] >= ths[None, :]) == yb[:, None]).mean(axis=0).max())
    raise ValueError(kind)

# --------------------------------------------------------------------------- #
# fast count-weighted evaluators: metric(resample) as a function of per-row counts c.
# c = bincount of the drawn indices; sum(c) == n. Each validated == ref to ~1e-8.
# --------------------------------------------------------------------------- #
class _RMSE:
    def __init__(self, y, p): self.e2 = (np.clip(p, 0, 1) - y) ** 2; self.n = len(y)
    def __call__(self, c): return float(np.sqrt(c.dot(self.e2) / self.n))

class _R2:
    def __init__(self, y, p):
        self.e2 = (np.clip(p, 0, 100) - y) ** 2
        self.y = y.astype(np.float64); self.y2 = self.y ** 2; self.n = len(y)
    def __call__(self, c):
        sse = c.dot(self.e2); sy = c.dot(self.y); syy = c.dot(self.y2)
        return float(1.0 - sse / (syy - sy * sy / self.n))

class _AUC:
    """Zero-tie rank-based weighted ROC-AUC (all original OOF scores are distinct, so a
    positive draw and a negative draw can never share a score -> no 0.5 tie term)."""
    def __init__(self, y, p):
        self.order = np.argsort(p, kind="stable")
        sy = y[self.order].astype(np.float64)
        self.sy = sy; self.sneg = 1.0 - sy
    def __call__(self, c):
        cs = c[self.order]
        pos = cs * self.sy; neg = cs * self.sneg
        cum_neg_below = np.concatenate(([0.0], np.cumsum(neg)[:-1]))
        return float(pos.dot(cum_neg_below) / (pos.sum() * neg.sum()))

class _ACC:
    """Accuracy at the best swept threshold, count-weighted. C[k,t] = (p_k>=t)==y_k."""
    def __init__(self, y, p):
        yb = np.asarray(y).astype(bool); p = np.asarray(p, np.float64)
        ths = np.union1d(np.arange(0.10, 0.90 + 1e-9, 0.01, dtype=np.float64), [0.5])
        self.C = ((p[:, None] >= ths[None, :]) == yb[:, None]).astype(np.float64)
        self.n = len(y)
    def __call__(self, c): return float((self.C.T.dot(c) / self.n).max())

_FAST = {"rmse": _RMSE, "r2": _R2, "auc": _AUC, "accuracy": _ACC}

# --------------------------------------------------------------------------- #
def load_stage_oofs(comp, cfg):
    """Return {stage_name: oof_vector}. Stage 4 is reconstructed from the tree winner's
    members + dirichlet weights (node_results) via the cached member OOFs."""
    cache = os.path.join(ROOT, "competitions", cfg["full"], "scripts", "cache")
    pre = cfg["pre"]
    oofs = {
        "stage2":    np.load(os.path.join(cache, f"blend_{pre}2.npz"))["oof"],
        "stage3_r1": np.load(os.path.join(cache, f"blend_{pre}3_r1.npz"))["oof"],
        "stage3_r2": np.load(os.path.join(cache, f"blend_{pre}3_r2.npz"))["oof"],
        "stage3_r3": np.load(os.path.join(cache, f"blend_{pre}3_r3.npz"))["oof"],
    }
    tree = json.load(open(os.path.join(ROOT, "competitions", cfg["full"],
                                       "experiments_tree_v3.json")))
    res = tree["search_state"]["node_results"][str(cfg["winner"])]
    members, weights = res["members"], np.asarray(res["weights"], np.float64)
    tcache = os.path.join(ROOT, "tree_search", f"cache_{comp}")
    blend = None
    for m, w in zip(members, weights):
        v = np.load(os.path.join(tcache, f"solo_{m}.npz"))["oof"]
        blend = w * v if blend is None else blend + w * v
    oofs["stage4"] = blend
    return oofs


def run_comp(comp, cfg, B, rng):
    tr = pd.read_csv(os.path.join(ROOT, "competitions", cfg["full"], "data", "train.csv"))
    y = cfg["yload"](tr).astype(np.float64)
    n = len(y)
    kind = cfg["metric"]
    oofs = load_stage_oofs(comp, cfg)
    stages = ["stage2", "stage3_r1", "stage3_r2", "stage3_r3", "stage4"]

    # ---- gate: unresampled metric must equal recorded ladder value ----
    point = {}
    for st in stages:
        assert len(oofs[st]) == n, f"{comp}/{st}: length {len(oofs[st])} != n {n}"
        m = ref_metric(kind, y, oofs[st])
        point[st] = m
        rec = cfg["recorded"][st]
        if abs(m - rec) > GATE_TOL:
            raise SystemExit(f"GATE FAIL {comp}/{st}: recompute {m:.6f} != recorded "
                             f"{rec:.6f} (|diff|={abs(m-rec):.2e}). OOF maps to wrong "
                             f"stage -- aborting per protocol.")

    # ---- build fast evaluators ----
    fast = {st: _FAST[kind](y, oofs[st]) for st in stages}

    # ---- paired bootstrap: shared resample per iteration across all stages ----
    M = {st: np.empty(B) for st in stages}
    for b in range(B):
        c = np.bincount(rng.integers(0, n, n), minlength=n).astype(np.float64)
        for st in stages:
            M[st][b] = fast[st](c)

    dir_sign = 1.0 if cfg["direction"] == "maximize" else -1.0

    def transition(before, after):
        d_point = dir_sign * (point[after] - point[before])
        d_boot = dir_sign * (M[after] - M[before])
        lo, hi = np.percentile(d_boot, [2.5, 97.5])
        return dict(before=before, after=after,
                    delta=d_point, ci_lo=float(lo), ci_hi=float(hi),
                    contains_zero=bool(lo <= 0.0 <= hi),
                    p_improve=float(np.mean(d_boot > 0)),
                    boot_mean=float(d_boot.mean()), boot_se=float(d_boot.std(ddof=1)))

    return dict(comp=comp, metric=kind, direction=cfg["direction"], n=n,
                point={k: float(v) for k, v in point.items()},
                transitions={name: transition(a, b) for name, a, b in TRANSITIONS},
                fine={name: transition(a, b) for name, a, b in FINE})


def fmt(x, sig="%.7f"):
    return sig % x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=B_DEFAULT)
    ap.add_argument("--comps", default=",".join(COMPS))
    ap.add_argument("--json", action="store_true", help="also dump results JSON to stdout")
    args = ap.parse_args()

    results = {}
    for comp in args.comps.split(","):
        comp = comp.strip()
        rng = np.random.default_rng(SEED)  # same seed per comp -> reproducible
        r = run_comp(comp, COMPS[comp], args.B, rng)
        results[comp] = r
        d = COMPS[comp]["direction"]
        print("=" * 92)
        print(f"{comp}   metric={r['metric']}  ({d})   n={r['n']}   B={args.B}   seed={SEED}")
        print("  per-stage OOF metric:  " +
              "  ".join(f"{k}={v:.6f}" for k, v in r["point"].items()))
        print(f"  {'transition':10} {'delta':>12} {'95% CI low':>13} {'95% CI high':>13} "
              f"{'P(improve)':>11}  verdict")
        for group, label in ((r["transitions"], "HEADLINE"), (r["fine"], "fine-grained")):
            print(f"  -- {label} --")
            for name, t in group.items():
                verdict = "INSIDE NOISE (CI contains 0)" if t["contains_zero"] \
                    else "significant (CI excludes 0)"
                print(f"  {name:10} {t['delta']:12.6e} {t['ci_lo']:13.6e} "
                      f"{t['ci_hi']:13.6e} {t['p_improve']:11.3f}  {verdict}")
        print()

    if args.json:
        print("<<<JSON>>>")
        print(json.dumps(results))


if __name__ == "__main__":
    main()
