"""Final ensemble: DeBERTa seed 42 + DeBERTa seed 1337 + LGBM lexical baseline.

All three OOF vectors come from the SAME GroupKFold(4)-by-anchor split, so the
weight search is unbiased. Weights are found by non-negative least squares on
standardised OOF preds, then sanity-checked against the simple alternatives.
Only overwrites submission.csv if the ensemble beats the current best OOF.
"""
import json
import os

import numpy as np
import pandas as pd
from scipy.optimize import nnls

COMP = "/home/tjyen/ai_agents/kaggle/competitions/us-patent-phrase-to-phrase-matching"


def pearson(a, b):
    return float(np.corrcoef(a, b)[0, 1])


train = pd.read_csv(f"{COMP}/data/train.csv")
test = pd.read_csv(f"{COMP}/data/test.csv")
y = train.score.values

members = {}
for tag in ["deberta_v3_base", "deberta_s1337"]:
    o = f"{COMP}/scripts/{tag}_oof.npy"
    t = f"{COMP}/scripts/{tag}_test.npy"
    if os.path.exists(o) and os.path.exists(t):
        members[tag] = (np.load(o), np.load(t))
members["lgbm"] = (np.load(f"{COMP}/scripts/lgbm_oof_group.npy"),
                   np.load(f"{COMP}/scripts/lgbm_test_group.npy"))

names = list(members)
for n in names:
    print(f"{n:18s} OOF r = {pearson(y, members[n][0]):.5f}")

O = np.column_stack([members[n][0] for n in names])
T = np.column_stack([members[n][1] for n in names])

# --- candidate 1: NNLS weights (with intercept via centring) ---
w, _ = nnls(O, y)
if w.sum() > 0:
    w = w / w.sum()
r_nnls = pearson(y, O @ w)
print(f"NNLS weights {dict(zip(names, np.round(w, 4)))} -> OOF r = {r_nnls:.5f}")

# --- candidate 2: equal-weight DeBERTas only ---
deb = [n for n in names if n.startswith("deberta")]
r_deb_avg = pearson(y, np.mean([members[n][0] for n in deb], axis=0))
print(f"DeBERTa mean ({len(deb)} models)        -> OOF r = {r_deb_avg:.5f}")

# --- candidate 3: DeBERTa mean + small LGBM weight ---
best_c3, best_w3 = -1, 0.0
deb_o = np.mean([members[n][0] for n in deb], axis=0)
deb_t = np.mean([members[n][1] for n in deb], axis=0)
for wl in np.arange(0.0, 0.35, 0.025):
    r = pearson(y, (1 - wl) * deb_o + wl * members["lgbm"][0])
    if r > best_c3:
        best_c3, best_w3 = r, wl
print(f"DeBERTa mean + {best_w3:.3f}*LGBM      -> OOF r = {best_c3:.5f}")

cands = [("nnls", r_nnls, T @ w),
         ("deberta_mean", r_deb_avg, deb_t),
         (f"deberta_mean+lgbm{best_w3:.3f}", best_c3,
          (1 - best_w3) * deb_t + best_w3 * members["lgbm"][1])]
name, r_best, final_test = max(cands, key=lambda c: c[1])
print(f"\nCHOSEN: {name}  OOF r = {r_best:.5f}")

prev = json.load(open(f"{COMP}/scripts/blend_decision.json"))
print(f"previous submission OOF r = {prev['best_blend_r']:.5f} "
      f"(graded 0.85596)")

sub = pd.read_csv(f"{COMP}/data/sample_submission.csv")
assert list(sub.id) == list(test.id)
sub["score"] = np.clip(final_test, 0, 1)
sub.to_csv(f"{COMP}/submissions/ensemble_{name}.csv", index=False)

improved = r_best > prev["best_blend_r"]
if improved:
    sub.to_csv(f"{COMP}/submission.csv", index=False)
    print("submission.csv UPDATED to the ensemble")
else:
    print("ensemble did NOT beat the previous OOF; submission.csv left as is")

json.dump({"members": names,
           "member_oof": {n: pearson(y, members[n][0]) for n in names},
           "candidates": {c[0]: c[1] for c in cands},
           "chosen": name, "chosen_oof": r_best,
           "previous_oof": prev["best_blend_r"], "updated": bool(improved)},
          open(f"{COMP}/scripts/ensemble_decision.json", "w"), indent=2)
