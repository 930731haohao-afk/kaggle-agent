"""tree_search/wire_bootstrap_v3.py — paired bootstrap for the v3-comp wiring arms.

For a given comp, reconstructs the TRUE champion blend (weights from
search_state.node_results, metric recomputed and bit-checked against the tree score)
and the arm-B reblend, then runs a B=10,000 paired bootstrap on the metric delta.
Metric handling per comp:
  s4e1  AUC       — rank-based AUC per resample (monotone metric, no postprocess)
  s4e11 accuracy  — each blend gets its own full-OOF-refit threshold (same procedure
                    as the evaluator), thresholds then FIXED across resamples
  s5e10 RMSE      — clip to [0,1] then RMSE per resample
Writes the result into wire_<comp>_ledger_b.json under "paired_bootstrap".

Usage: uv run python3 tree_search/wire_bootstrap_v3.py --comp s4e1
"""

import argparse
import importlib
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import harness_v2 as hv2  # noqa: E402

REPO = os.path.dirname(_HERE)
SPEC = {
    "s4e1": dict(slug="playground-series-s4e1", eval_mod="eval_s4e1", metric="auc",
                 disp=lambda s: -s, higher_better=True),
    "s4e11": dict(slug="playground-series-s4e11", eval_mod="eval_s4e11", metric="accuracy",
                  disp=lambda s: -s, higher_better=True),
    "s5e10": dict(slug="playground-series-s5e10", eval_mod="eval_s5e10", metric="rmse",
                  disp=lambda s: s, higher_better=False),
}
B = 10_000


def blend_vec(res, cache_dir):
    members, W = res["members"], np.asarray(res["weights"], dtype=float)
    oofs = np.stack([hv2.load_oof(cache_dir, m) for m in members], axis=1)
    return oofs @ W, dict(zip(members, res["weights"]))


def auc_fast(y, s):
    """Rank-based AUC (ties get average rank)."""
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    sv = s[order]
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks over ties
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    pos = y == 1
    n1, n0 = pos.sum(), (~pos).sum()
    return (ranks[pos].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def best_threshold_accuracy(y, s):
    """Full-OOF-refit threshold, evaluator-style: scan midpoints of sorted unique scores."""
    order = np.argsort(s)
    ss, yy = s[order], y[order]
    # accuracy(t) = (#neg below t) + (#pos at/above t); scan all cut positions
    pos_total = yy.sum()
    neg_cum = np.cumsum(yy == 0)
    pos_above = pos_total - np.cumsum(yy)
    acc = (neg_cum + pos_above) / len(yy)          # cut AFTER index i
    k = int(np.argmax(acc))
    thr = ss[k] + 1e-12
    return thr, float(acc[k])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comp", choices=sorted(SPEC), required=True)
    args = ap.parse_args()
    sp = SPEC[args.comp]
    ev = importlib.import_module(sp["eval_mod"])
    y = ev._y.astype(float)

    off = json.load(open(os.path.join(REPO, "competitions", sp["slug"],
                                      "experiments_tree_v3.json")))
    off_champ = min([n for n in off["nodes"] if n["status"] == "evaluated"
                     and isinstance(n["score"], (int, float))], key=lambda n: n["score"])
    res_off = off["search_state"]["node_results"][str(off_champ["id"])]

    treeB = json.load(open(os.path.join(_HERE, f"wire_{args.comp}_B.json")))
    reblend = [n for n in treeB["nodes"]
               if n["mutation"].startswith("[PRIOR-LLM-REBLEND]")]
    assert reblend, "no arm-B reblend node found"
    nB = reblend[0]
    res_b = nB["config"]["result"]

    v_off, _w_off = blend_vec(res_off, ev.CACHE_DIR)
    v_b, w_b = blend_vec(res_b, ev.CACHE_DIR)

    # per-comp metric machinery + bit-check vs tree scores
    if args.comp == "s4e1":
        m_off, m_b = auc_fast(y, v_off), auc_fast(y, v_b)
        e_off, e_b = v_off, v_b       # bootstrap re-ranks scores directly

        def delta(idx):
            return auc_fast(y[idx], e_b[idx]) - auc_fast(y[idx], e_off[idx])
    elif args.comp == "s4e11":
        thr_off, m_off = best_threshold_accuracy(y, v_off)
        thr_b, m_b = best_threshold_accuracy(y, v_b)
        lab_off, lab_b = (v_off >= thr_off).astype(float), (v_b >= thr_b).astype(float)
        hit_off, hit_b = (lab_off == y).astype(float), (lab_b == y).astype(float)

        def delta(idx):
            return hit_b[idx].mean() - hit_off[idx].mean()
    else:  # s5e10 rmse (lower better) -> report delta as improvement = rmse_off - rmse_b
        c_off, c_b = np.clip(v_off, 0, 1), np.clip(v_b, 0, 1)
        m_off = float(np.sqrt(((y - c_off) ** 2).mean()))
        m_b = float(np.sqrt(((y - c_b) ** 2).mean()))
        se_off, se_b = (y - c_off) ** 2, (y - c_b) ** 2

        def delta(idx):
            return float(np.sqrt(se_off[idx].mean()) - np.sqrt(se_b[idx].mean()))

    tgt_off = sp["disp"](off_champ["score"])
    tgt_b = sp["disp"](nB["score"])
    ok_off = round(m_off, 6) == round(tgt_off, 6)
    ok_b = round(m_b, 6) == round(tgt_b, 6)
    print(f"bit-check: OFF {m_off:.6f} vs tree {tgt_off:.6f} -> {ok_off} | "
          f"B {m_b:.6f} vs tree {tgt_b:.6f} -> {ok_b}")

    rng = np.random.default_rng(42)
    n = len(y)
    deltas = np.empty(B)
    for b in range(B):
        deltas[b] = delta(rng.integers(0, n, n))
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    sig = lo > 0
    print(f"paired bootstrap Δ({sp['metric']} improvement, B-arm − champion): "
          f"mean={deltas.mean():+.6f}  95% CI=[{lo:+.6f}, {hi:+.6f}]  "
          f"-> {'SIGNIFICANT' if sig else 'not significant'}")

    new_ids = [r["node_id"] for r in
               json.load(open(os.path.join(_HERE, f"wire_{args.comp}_ledger_b.json")))["ledger"]
               if r.get("node_id") is not None and r["tag"].startswith("PRIOR-LLM-0")]
    llm_w = {str(m): w for m, w in w_b.items() if m in set(new_ids) and w > 0}

    led_path = os.path.join(_HERE, f"wire_{args.comp}_ledger_b.json")
    led = json.load(open(led_path))
    led["paired_bootstrap"] = dict(
        method=f"paired bootstrap B={B} seed=42; TRUE champion weights from "
               f"search_state.node_results; bit_check_off={bool(ok_off)}, bit_check_b={bool(ok_b)}",
        metric=sp["metric"], champion=m_off, reblend=m_b,
        delta_mean=float(deltas.mean()), ci95=[float(lo), float(hi)],
        significant=bool(sig))
    led["llm_member_weights_in_reblend"] = llm_w
    led["llm_member_weights_sum"] = float(sum(llm_w.values()))
    json.dump(led, open(led_path, "w"), ensure_ascii=False, indent=2)
    print(f"LLM member weights: {llm_w} (sum={sum(llm_w.values()):.4f})")
    print(f"ledger updated -> {led_path}")


if __name__ == "__main__":
    main()
