"""Final champion pipeline:
1. blend members' full OOF probs with per-delta weight search (+ per-delta threshold),
2. apply to test preds,
3. write submission.csv,
4. log experiment.

Usage: final_pipeline.py <member1,member2,...> [--no-thresh]
Members are preds_<name>.npz files in scripts/cache (must contain full 5-fold oof+test).
"""
import json
import sys

import numpy as np

sys.path.insert(0, "competitions/conway-s-reverse-game-of-life/scripts")
import common
from make_submission import main as write_submission

members = sys.argv[1].split(",") if len(sys.argv) > 1 else ["cnn_v1", "cnn_v2", "lgb_v1"]
use_thresh = "--no-thresh" not in sys.argv

a = common.load_arrays()
S, delta, tdelta = a["S"], a["delta"], a["tdelta"]
folds = common.get_folds(delta)
Y = S.reshape(-1).astype(np.int8)
cell_delta = np.repeat(delta, 400)

oofs, tests, metas = [], [], {}
for m in members:
    npz = np.load(f"{common.CACHE}/preds_{m}.npz")
    oofs.append(npz["oof"].reshape(-1).astype(np.float32))
    tests.append(npz["test"].astype(np.float32))
    with open(f"{common.CACHE}/score_{m}.json") as fh:
        metas[m] = json.load(fh)
X = np.stack(oofs, axis=1)


def wsearch(Xs, Ys, k=400, seed=42):
    rng = np.random.RandomState(seed)
    nm = Xs.shape[1]
    best_w = np.ones(nm) / nm
    best_s = float((((Xs @ best_w) > 0.5).astype(np.int8) != Ys).mean())
    for _ in range(k):
        w = rng.dirichlet(np.ones(nm))
        s = float((((Xs @ w) > 0.5).astype(np.int8) != Ys).mean())
        if s < best_s:
            best_w, best_s = w, s
    for _ in range(5):
        improved = False
        for j in range(nm):
            for step in (0.1, -0.1, 0.05, -0.05, 0.02, -0.02):
                w = best_w.copy()
                w[j] = max(0.0, w[j] + step)
                if w.sum() == 0:
                    continue
                w /= w.sum()
                s = float((((Xs @ w) > 0.5).astype(np.int8) != Ys).mean())
                if s < best_s - 1e-9:
                    best_w, best_s, improved = w, s, True
        if not improved:
            break
    return best_w, best_s


weights, thresholds = {}, {}
oof_blend = np.zeros(len(Y), np.float32)
test_blend = np.zeros((len(tdelta), 20, 20), np.float32)
for d in range(1, 6):
    mc = cell_delta == d
    w, s = wsearch(X[mc], Y[mc], seed=42 + d)
    oof_blend[mc] = X[mc] @ w
    mt = tdelta == d
    test_blend[mt] = sum(tests[i][mt] * w[i] for i in range(len(members)))
    weights[str(d)] = [round(float(x), 4) for x in w]
    print(f"delta={d}: weights {weights[str(d)]} MAE@0.5 {s:.6f}")

# per-delta threshold sweep on full OOF
pred_flat = np.zeros_like(Y)
for d in range(1, 6):
    mc = cell_delta == d
    if use_thresh:
        cand = np.arange(0.40, 0.61, 0.01)
        scs = [((oof_blend[mc] > t).astype(np.int8) != Y[mc]).mean() for t in cand]
        t = float(cand[int(np.argmin(scs))])
    else:
        t = 0.5
    thresholds[str(d)] = round(t, 2)
    pred_flat[mc] = (oof_blend[mc] > t).astype(np.int8)
final_oof = float((pred_flat != Y).mean())

fold_scores = []
for f in range(common.N_SPLITS):
    mf = np.repeat(folds == f, 400)
    fold_scores.append(float((pred_flat[mf] != Y[mf]).mean()))

print(f"\nFINAL blend {members}: OOF MAE {final_oof:.6f}")
print("thresholds:", thresholds)
print("fold scores:", np.round(fold_scores, 6))

np.savez_compressed(f"{common.CACHE}/preds_final_blend.npz",
                    oof=oof_blend.reshape(-1, 20, 20), test=test_blend)
sub_path = "competitions/conway-s-reverse-game-of-life/submission.csv"
write_submission(f"{common.CACHE}/preds_final_blend.npz", sub_path,
                 thresholds if use_thresh else None)

common.experiment_log.log_experiment_v2(
    common.COMP_DIR, model="final-blend-" + "+".join(members), metric="mae",
    direction="minimize", score=final_oof,
    cv={"strategy": "5-fold stratified by delta, seed 42",
        "scores": [round(s, 6) for s in fold_scores]},
    ensemble={"members": members, "per_delta_weights": weights,
              "search": "per-delta dirichlet(400)+coord-ascent on full OOF, MAE@thresh"},
    postprocess=[f"per-delta threshold {thresholds}"],
    submission=sub_path,
    notes="Final submission blend. Weights+thresholds fit on full OOF (in-sample for "
          "the weight fit; members' OOFs are honest).")
print("logged final experiment")
