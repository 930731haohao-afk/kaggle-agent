"""Per-cell LightGBM blend member: per-delta models on 7x7 stop-board windows.

Features per (board, cell): 49 binary window values (dead-padded) + row + col.
One model per (fold, delta). OOF probs + test preds saved like CNN members.
"""
import json
import sys
import time

import lightgbm as lgb
import numpy as np

sys.path.insert(0, "competitions/conway-s-reverse-game-of-life/scripts")
import common

W = 3  # half-window -> 7x7

LGB_PARAMS = dict(
    objective="binary", learning_rate=0.1, num_leaves=63, max_depth=-1,
    min_child_samples=100, subsample=0.9, subsample_freq=1, colsample_bytree=0.9,
    n_estimators=150, num_threads=10, deterministic=True, force_row_wise=True,
    random_state=42, verbosity=-1)


def board_features(P: np.ndarray) -> np.ndarray:
    """P (N,20,20) -> (N*400, 51) uint8 features: 7x7 window + row + col."""
    n = len(P)
    pad = np.pad(P, ((0, 0), (W, W), (W, W)))
    # windows for every cell via stride tricks
    from numpy.lib.stride_tricks import sliding_window_view
    win = sliding_window_view(pad, (2 * W + 1, 2 * W + 1), axis=(1, 2))  # (N,20,20,7,7)
    feats = win.reshape(n, 400, (2 * W + 1) ** 2)
    rows = np.tile(np.repeat(np.arange(20, dtype=np.uint8), 20), (n, 1))[..., None]
    cols = np.tile(np.tile(np.arange(20, dtype=np.uint8), 20), (n, 1))[..., None]
    out = np.concatenate([feats.astype(np.uint8), rows, cols], axis=2)
    return out.reshape(n * 400, (2 * W + 1) ** 2 + 2)


def main(name="lgb_v1", n_estimators=150):
    LGB_PARAMS["n_estimators"] = n_estimators
    a = common.load_arrays()
    S, P, delta, T, tdelta = a["S"], a["P"], a["delta"], a["T"], a["tdelta"]
    folds = common.get_folds(delta)
    oof = np.full((len(S), 20, 20), np.nan, np.float32)
    test_acc = np.zeros((len(T), 20, 20), np.float32)
    t0 = time.time()
    Xte_by_d = {}
    for d in range(1, 6):
        mte = tdelta == d
        Xte_by_d[d] = board_features(T[mte])
    for f in range(common.N_SPLITS):
        tr, va = np.where(folds != f)[0], np.where(folds == f)[0]
        for d in range(1, 6):
            trd = tr[delta[tr] == d]
            vad = va[delta[va] == d]
            Xtr = board_features(P[trd])
            ytr = S[trd].reshape(-1)
            clf = lgb.LGBMClassifier(**LGB_PARAMS)
            clf.fit(Xtr, ytr)
            pv = clf.predict_proba(board_features(P[vad]))[:, 1]
            oof[vad] = pv.reshape(-1, 20, 20).astype(np.float32)
            mte = tdelta == d
            test_acc[mte] += clf.predict_proba(Xte_by_d[d])[:, 1].reshape(-1, 20, 20) / common.N_SPLITS
            del Xtr
        sc = common.mae((oof[va] > 0.5).astype(np.int8), S[va])
        print(f"[{name}] fold {f}: MAE {sc:.5f} ({time.time() - t0:.0f}s)", flush=True)
    oof_mae = common.mae((oof > 0.5).astype(np.int8), S)
    print(f"[{name}] OOF MAE {oof_mae:.5f} total {time.time() - t0:.0f}s", flush=True)
    np.savez_compressed(f"{common.CACHE}/preds_{name}.npz",
                        oof=oof, test=test_acc, folds_run=np.arange(common.N_SPLITS))
    fold_scores = [common.mae((oof[folds == f] > 0.5).astype(np.int8), S[folds == f])
                   for f in range(common.N_SPLITS)]
    with open(f"{common.CACHE}/score_{name}.json", "w") as fh:
        json.dump({"name": name, "oof_mae": oof_mae, "fold_scores": fold_scores,
                   "cfg": {"model": "lgb-percell-perdelta", "window": 7,
                           **{k: str(v) for k, v in LGB_PARAMS.items()}}}, fh, indent=2)


if __name__ == "__main__":
    main()
