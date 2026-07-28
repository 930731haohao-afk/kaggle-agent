"""Decide whether blending the LGBM lexical baseline into the DeBERTa preds helps.

Caveat printed explicitly: the LGBM OOF came from StratifiedKFold on score
(anchors leak across folds), while the DeBERTa OOF uses GroupKFold by anchor.
That biases the LGBM component OPTIMISTICALLY, so a blend must win by a clear
margin on the shared OOF rows before it is adopted.
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import rankdata

COMP = "/home/tjyen/ai_agents/kaggle/competitions/us-patent-phrase-to-phrase-matching"
TAG = "deberta_v3_base"


def pearson(a, b):
    return float(np.corrcoef(a, b)[0, 1])


train = pd.read_csv(f"{COMP}/data/train.csv")
y = train.score.values

deb_oof = np.load(f"{COMP}/scripts/{TAG}_oof.npy")
deb_test = np.load(f"{COMP}/scripts/{TAG}_test.npy")
lgb_oof = np.load(f"{COMP}/scripts/lgbm_oof_group.npy")
lgb_test = np.load(f"{COMP}/scripts/lgbm_test_group.npy")

mask = ~np.isnan(deb_oof)
print(f"rows used: {mask.sum()} / {len(y)}")
r_deb = pearson(y[mask], deb_oof[mask])
r_lgb = pearson(y[mask], lgb_oof[mask])
print(f"DeBERTa OOF r = {r_deb:.5f}")
print(f"LGBM    OOF r = {r_lgb:.5f}  (GroupKFold by anchor - same split as DeBERTa, unbiased)")

best = (r_deb, 0.0, "none")
for w in np.arange(0.0, 0.55, 0.05):
    b = (1 - w) * deb_oof[mask] + w * lgb_oof[mask]
    r = pearson(y[mask], b)
    rb = pearson(y[mask],
                 (1 - w) * rankdata(deb_oof[mask]) + w * rankdata(lgb_oof[mask]))
    print(f"  w_lgbm={w:.2f}  linear r={r:.5f}   rank r={rb:.5f}")
    if r > best[0]:
        best = (r, w, "linear")
    if rb > best[0]:
        best = (rb, w, "rank")

gain = best[0] - r_deb
print(f"\nbest: {best[2]} w_lgbm={best[1]:.2f} r={best[0]:.5f} "
      f"gain over solo DeBERTa = {gain:+.5f}")

MIN_GAIN = 0.002
adopt = best[2] != "none" and gain >= MIN_GAIN
print(f"adopt blend? {adopt} (threshold {MIN_GAIN})")

if adopt and best[2] == "linear":
    final_test = (1 - best[1]) * deb_test + best[1] * lgb_test
elif adopt:  # rank blend -> map back to the DeBERTa value distribution
    rk = ((1 - best[1]) * rankdata(deb_test) + best[1] * rankdata(lgb_test))
    final_test = np.interp(rankdata(rk), np.arange(1, len(rk) + 1),
                           np.sort(deb_test))
else:
    final_test = deb_test

sub = pd.read_csv(f"{COMP}/data/sample_submission.csv")
test = pd.read_csv(f"{COMP}/data/test.csv")
assert list(sub.id) == list(test.id)
sub["score"] = np.clip(final_test, 0, 1)
sub.to_csv(f"{COMP}/submission.csv", index=False)
name = "deberta_solo" if not adopt else f"blend_{best[2]}_w{best[1]:.2f}"
sub.to_csv(f"{COMP}/submissions/final_{name}.csv", index=False)
json.dump({"deberta_oof": r_deb, "lgbm_oof": r_lgb,
           "best_blend_r": best[0], "best_w_lgbm": best[1],
           "best_kind": best[2], "gain": gain, "adopted": bool(adopt),
           "final": name},
          open(f"{COMP}/scripts/blend_decision.json", "w"), indent=2)
print(f"wrote submission.csv ({name})")
