"""Linear-protocol blend: cnn_v1 + lgb_v1 full-OOF weight search (MAE@0.5), logs
cnn_v1 / lgb_v1 / blend experiments."""
import json
import sys

import numpy as np

sys.path.insert(0, "competitions/conway-s-reverse-game-of-life/scripts")
import common

a = common.load_arrays()
S, delta = a["S"], a["delta"]
folds = common.get_folds(delta)
Y = S.reshape(-1).astype(np.int8)

members = ["cnn_v1", "lgb_v1"]
oofs, metas = [], {}
for m in members:
    npz = np.load(f"{common.CACHE}/preds_{m}.npz")
    oofs.append(npz["oof"].reshape(-1).astype(np.float32))
    with open(f"{common.CACHE}/score_{m}.json") as fh:
        metas[m] = json.load(fh)
X = np.stack(oofs, axis=1)


def mae_w(w):
    return float((((X @ w) > 0.5).astype(np.int8) != Y).mean())


# weight grid search (2 members -> 1D grid) + fold scores
grid = np.linspace(0, 1, 101)
scores = [mae_w(np.array([g, 1 - g])) for g in grid]
gi = int(np.argmin(scores))
w = np.array([grid[gi], 1 - grid[gi]])
blend_score = scores[gi]

blended = X @ w
fold_scores = [float(((blended[np.repeat(folds == f, 400)] > 0.5).astype(np.int8)
                      != Y[np.repeat(folds == f, 400)]).mean())
               for f in range(common.N_SPLITS)]

for m in members:
    meta = metas[m]
    common.experiment_log.log_experiment_v2(
        common.COMP_DIR, model=meta["cfg"].get("model", "cnn") if m == "lgb_v1" else "cnn-delta-conditioned",
        metric="mae", direction="minimize", score=meta["oof_mae"],
        cv={"strategy": "5-fold stratified by delta, seed 42",
            "scores": [round(s, 6) for s in meta["fold_scores"]]},
        notes=f"{m}: {json.dumps(meta['cfg'])[:400]}")

common.experiment_log.log_experiment_v2(
    common.COMP_DIR, model="blend-cnn_v1+lgb_v1", metric="mae", direction="minimize",
    score=blend_score,
    cv={"strategy": "5-fold stratified by delta, seed 42",
        "scores": [round(s, 6) for s in fold_scores]},
    ensemble={"members": members, "weights": [round(float(x), 3) for x in w],
              "search": "101-point 1D grid on full OOF, MAE@0.5"},
    notes="Linear-protocol first blend (Stage 3->4 gate).")

print("member scores:", {m: metas[m]["oof_mae"] for m in members})
print(f"blend weights {w} OOF MAE {blend_score:.6f}")
print("fold scores:", np.round(fold_scores, 6))
np.savez_compressed(f"{common.CACHE}/preds_blend_v1.npz",
                    oof=blended.reshape(-1, 20, 20).astype(np.float32),
                    test=(np.load(f"{common.CACHE}/preds_cnn_v1.npz")["test"] * w[0] +
                          np.load(f"{common.CACHE}/preds_lgb_v1.npz")["test"] * w[1]))
