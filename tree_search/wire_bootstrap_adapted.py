"""wire_bootstrap_adapted.py — clean paired bootstrap for the 6 adapted GBDT comps.

The v1/v2 committed champion is NOT the v3 convex optimum, so we do NOT compare against it.
Instead: OFF = Arm-A pool-only v3 reblend ([PRIOR-EXT-12] pool blend), B = Arm-B pool+external
v3 reblend ([PRIOR-LLM-REBLEND]). Both under the same deterministic v3 weight search, so their
delta is the clean injection effect. B=10,000 paired bootstrap on the metric delta.
Per-comp metric applied to the blend OOF exactly as the evaluator scores it; a bit-check vs the
stored node score guards against OOF-space mismatch (comps that fail the bit-check are flagged).

Usage: uv run python3 tree_search/wire_bootstrap_adapted.py
"""
import importlib
import json
import os
import sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import harness_v2 as hv2  # noqa: E402
REPO = os.path.dirname(_HERE)
B = 10_000

SPEC = {
    "s3e1":  dict(eval_mod="eval_s3e1",  metric="rmse",  disp=lambda s: s,  higher_better=False),
    "s3e3":  dict(eval_mod="eval_s3e3",  metric="auc",   disp=lambda s: -s, higher_better=True),
    "s3e9":  dict(eval_mod="eval_s3e9_v2", metric="rmse", disp=lambda s: s, higher_better=False),
    "s3e11": dict(eval_mod="eval_s3e11", metric="rmsle", disp=lambda s: s,  higher_better=False),
    "s3e16": dict(eval_mod="eval_s3e16_v2", metric="mae_round", disp=lambda s: s, higher_better=False),
    "s3e19": dict(eval_mod="eval_s3e19", metric="smape", disp=lambda s: s,  higher_better=False),
}


def auc_fast(y, s):
    order = np.argsort(s, kind="mergesort"); ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    sv = s[order]; i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]: j += 1
        if j > i: ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    pos = y == 1; n1, n0 = pos.sum(), (~pos).sum()
    return (ranks[pos].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def blend_vec(res, cache_dir):
    members, W = res["members"], np.asarray(res["weights"], dtype=float)
    oofs = np.stack([hv2.load_oof(cache_dir, m) for m in members], axis=1)
    return oofs @ W


def pick(tree, marker):
    ns = [n for n in tree["nodes"] if n["status"] == "evaluated" and isinstance(n["score"], (int, float))
          and marker in n["mutation"]]
    return min(ns, key=lambda n: n["score"]) if ns else None


def metric_scalar(comp, y, v):
    m = SPEC[comp]["metric"]
    if m == "rmse":   return float(np.sqrt(((y - v) ** 2).mean()))
    if m == "auc":    return float(auc_fast(y, v))
    if m == "rmsle":  return float(np.sqrt(((np.log1p(y) - np.log1p(np.clip(v, 0, None))) ** 2).mean()))
    if m == "mae_round": return float(np.abs(y - np.rint(v)).mean())
    if m == "smape":
        d = np.abs(y) + np.abs(v)
        return float((np.where(d == 0, 0.0, 2 * np.abs(v - y) / d)).mean() * 100)


def delta_on(comp, y, v_off, v_b, idx):
    yy = y[idx]; a, b = v_off[idx], v_b[idx]
    # improvement = OFF metric - B metric for lower-better; B - OFF for AUC
    if SPEC[comp]["higher_better"]:
        return metric_scalar(comp, yy, b) - metric_scalar(comp, yy, a)
    return metric_scalar(comp, yy, a) - metric_scalar(comp, yy, b)


def main():
    rows = []
    for comp, sp in SPEC.items():
        try:
            ev = importlib.import_module(sp["eval_mod"]); y = np.asarray(ev._y, dtype=float)
            A = json.load(open(os.path.join(_HERE, f"wire_{comp}_A.json")))
            Bt = json.load(open(os.path.join(_HERE, f"wire_{comp}_B.json")))
            n_off = pick(A, "pool blend"); n_b = pick(Bt, "REBLEND")
            if n_off is None or n_b is None:
                print(f"{comp}: missing reblend node"); continue
            v_off = blend_vec(n_off["config"]["result"], ev.CACHE_DIR)
            v_b = blend_vec(n_b["config"]["result"], ev.CACHE_DIR)
            m_off, m_b = metric_scalar(comp, y, v_off), metric_scalar(comp, y, v_b)
            t_off, t_b = sp["disp"](n_off["score"]), sp["disp"](n_b["score"])
            ok = round(m_off, 4) == round(t_off, 4) and round(m_b, 4) == round(t_b, 4)
            rng = np.random.default_rng(42); n = len(y)
            ds = np.array([delta_on(comp, y, v_off, v_b, rng.integers(0, n, n)) for _ in range(B)])
            lo, hi = np.percentile(ds, [2.5, 97.5])
            verdict = "SIGNIFICANT ext-benefit" if lo > 0 else ("SIGNIFICANT ext-harm" if hi < 0 else "not significant")
            flag = "" if ok else "  [!! bit-check FAILED — metric/OOF-space mismatch, treat as unreliable]"
            print(f"{comp:6} {sp['metric']:9} pool-only={m_off:.5f} pool+ext={m_b:.5f} "
                  f"Δmean={ds.mean():+.6f} CI95=[{lo:+.6f},{hi:+.6f}] -> {verdict}{flag}")
            rows.append(dict(comp=comp, metric=sp["metric"], pool_only=round(m_off, 5), pool_ext=round(m_b, 5),
                             delta_mean=round(float(ds.mean()), 6), ci95=[round(float(lo), 6), round(float(hi), 6)],
                             significant=bool(lo > 0 or hi < 0), bit_check=bool(ok)))
        except Exception as e:  # noqa: BLE001
            print(f"{comp}: ERROR {type(e).__name__}: {e}")
    json.dump(rows, open(os.path.join(_HERE, "injection_bootstrap_adapted.json"), "w"), indent=2)
    print("\n-> tree_search/injection_bootstrap_adapted.json")


if __name__ == "__main__":
    main()
