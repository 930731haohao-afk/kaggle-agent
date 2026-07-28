"""Honest gate + final submission for tabular-playground-series-jan-2022.

1. LOFO honest gate (experience.md aug-2022 lesson): for each candidate blend,
   refit weights on ONE val year and score on the other; mean = honest SMAPE.
   Compare against the root solo (no fitted weights -> its pooled score is
   already honest). Pick the winner.
2. Final test predictions: weighted sum of cached member test preds (weights
   from the full 2017+2018 OOF for the chosen candidate), rounded to int.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

_REPO = "/home/tjyen/ai_agents/kaggle"
sys.path.insert(0, os.path.join(_REPO, "tree_search"))
sys.path.insert(0, os.path.dirname(__file__))
import common  # noqa: E402
import harness_v2 as hv2  # noqa: E402
import harness_v3 as hv3  # noqa: E402
import eval_tpsjan22 as ev  # noqa: E402

COMP_DIR = os.path.join(_REPO, "competitions", "tabular-playground-series-jan-2022")
TREE_PATH = os.path.join(COMP_DIR, "experiments_tree_v3.json")

tree = hv3.load_search_state(TREE_PATH)
nodes = {n["id"]: n for n in tree["nodes"]}

train, test = common.load_data()
year = train["date"].dt.year.values
Y = train["num_sold"].values.astype(float)
M2017, M2018 = year == 2017, year == 2018
IDX = ev.IDX


def masked_metric(mask):
    def f(vec):
        return common.smape(Y[mask], np.round(vec[mask]))
    return f


def fit_weights(members, mask):
    w, s, oofs = hv3.eval_blend(ev.CACHE_DIR, members, masked_metric(mask))
    return w, s, oofs


def honest_blend(members):
    """Refit weights on one year, score on the other; also full-OOF (optimistic)."""
    scores = {}
    for fit_mask, eval_mask, tag in [(M2017, M2018, "fit17->18"), (M2018, M2017, "fit18->17")]:
        w, _, oofs = fit_weights(members, fit_mask)
        blended = oofs @ w
        scores[tag] = common.smape(Y[eval_mask], np.round(blended[eval_mask]))
    w_full, s_full, _ = fit_weights(members, IDX)
    return (scores["fit17->18"] + scores["fit18->17"]) / 2, scores, s_full, w_full


CANDIDATES = {
    "champion_#59": nodes[59]["config"]["members"],
    "megablend_#48": nodes[48]["config"]["members"],
    "blend_#17": nodes[17]["config"]["members"],
    "seed_blend_#9": nodes[9]["config"]["members"],
}

root_score = nodes[tree["root_id"]]["score"]
print(f"root solo (honest by construction): {root_score:.6f}")

results = {}
for name, members in CANDIDATES.items():
    hon, per, full, w_full = honest_blend(members)
    results[name] = dict(honest=hon, per_fold=per, full_oof=full,
                         members=members, weights=[float(x) for x in w_full])
    print(f"{name:16s} honest {hon:.6f} ({per}) | full-OOF {full:.6f} "
          f"| optimism {hon - full:+.6f} | {len(members)} members")

best_name = min(results, key=lambda k: results[k]["honest"])
best = results[best_name]
print(f"\nBest honest candidate: {best_name} @ {best['honest']:.6f} vs root {root_score:.6f}")

if best["honest"] < root_score:
    chosen, chosen_kind = best_name, "blend"
    members = best["members"]
    w = np.array(best["weights"])
    print(f"DECISION: blend {best_name} passes the honest gate "
          f"({best['honest']:.6f} < {root_score:.6f}) — final weights refit on full OOF.")
else:
    chosen, chosen_kind = "root_solo", "solo"
    print(f"DECISION: no blend beats the root honestly — submit root solo.")

# --- build test predictions ---
def load_pred(node_id):
    d = np.load(os.path.join(ev.CACHE_DIR, f"solo_{node_id}.npz"), allow_pickle=True)
    return np.asarray(d["pred"], dtype=float)


if chosen_kind == "blend":
    test_pred = np.round(np.stack([load_pred(m) for m in members], 1) @ w)
else:
    test_pred = np.round(load_pred(tree["root_id"]))

test_pred = np.clip(test_pred, 1, None)

sub = pd.DataFrame({"row_id": test["row_id"].values, "num_sold": test_pred.astype(int)})
sample = pd.read_csv(os.path.join(COMP_DIR, "data", "sample_submission.csv"))
assert list(sub.columns) == list(sample.columns)
assert (sub["row_id"].values == sample["row_id"].values).all()
assert len(sub) == len(sample)
out = os.path.join(COMP_DIR, "submission.csv")
sub.to_csv(out, index=False)
print(f"\nWrote {out}: {len(sub)} rows, pred range [{sub.num_sold.min()}, {sub.num_sold.max()}], "
      f"mean {sub.num_sold.mean():.1f} (train 2018 mean {Y[year == 2018].mean():.1f})")

with open(os.path.join(COMP_DIR, "final_decision.json"), "w") as f:
    json.dump(dict(chosen=chosen, kind=chosen_kind, root_score=root_score,
                   candidates={k: {kk: vv for kk, vv in v.items() if kk != "members"}
                               for k, v in results.items()}), f, indent=2, default=str)
