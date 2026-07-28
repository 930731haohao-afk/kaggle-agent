"""Baselines + synthetic-generator sanity check."""
import sys

sys.path.insert(0, "competitions/conway-s-reverse-game-of-life/scripts")
import numpy as np

import common

a = common.load_arrays()
S, P, delta = a["S"], a["P"], a["delta"]
folds = common.get_folds(delta)

# baseline 1: all-zero
zero_scores = [float(S[folds == f].mean()) for f in range(common.N_SPLITS)]
# baseline 2: per-delta best of {zero, stop-as-start}
pred = np.where((delta == 1)[:, None, None], P, 0)
pd_scores = [common.mae(pred[folds == f], S[folds == f]) for f in range(common.N_SPLITS)]

print("all-zero fold MAE:", np.round(zero_scores, 5), "mean", np.mean(zero_scores))
print("per-delta(stop@d1, zero@d2-5) fold MAE:", np.round(pd_scores, 5), "mean", np.mean(pd_scores))

common.experiment_log.log_experiment_v2(
    common.COMP_DIR, model="baseline-all-zero", metric="mae", direction="minimize",
    score=float(np.mean(zero_scores)),
    cv={"strategy": "5-fold stratified by delta, seed 42", "scores": [round(s, 6) for s in zero_scores]},
    notes="Predict all cells dead. Start density 0.145.")
common.experiment_log.log_experiment_v2(
    common.COMP_DIR, model="baseline-perdelta-stop-or-zero", metric="mae", direction="minimize",
    score=float(np.mean(pd_scores)),
    cv={"strategy": "5-fold stratified by delta, seed 42", "scores": [round(s, 6) for s in pd_scores]},
    notes="stop board as start for delta=1, zeros otherwise.")

# synthetic generator distribution check
gS, gP, gD = common.gen_synthetic(20000, seed=123)
print("\nsynthetic vs train:")
print("start density: synth %.4f train %.4f" % (gS.mean(), S.mean()))
print("stop density:  synth %.4f train %.4f" % (gP.mean(), P.mean()))
qs = [5, 25, 50, 75, 95]
print("start board-density percentiles synth:", np.round(np.percentile(gS.mean(axis=(1, 2)), qs), 4))
print("start board-density percentiles train:", np.round(np.percentile(S.mean(axis=(1, 2)), qs), 4))
for d in range(1, 6):
    print(f"delta={d}: synth stop {gP[gD == d].mean():.4f} train stop {P[delta == d].mean():.4f}")
