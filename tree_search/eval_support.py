"""tree_search/eval_support.py — shared evaluator capabilities for v5 operators.

Exists to close the two encoding gaps the coverage audit exposed (engineering log,
"what still cannot be executed"): `target` encoding needs per-fold computation inside the
evaluator — computing it anywhere else is the s4e1 leakage path — and `onehot_sparse`
needs a linear model family the GBDT evaluators did not have. Both are implemented here
once and imported by the per-competition evaluators, so an arm's codegen copy inherits
them without further patching.

Also home to the runtime attestation writer: an evaluator that actually honours a
dossier-requested `postprocess` or `target_encoding` records that fact next to the
competition's artifacts, so external_data/verify_advisory.py can report "verified
(runtime)" instead of trusting a static grep. The design rule is the report's own:
a produced artifact needs a consumer trace, and a claimed execution needs evidence.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

DEFAULT_SMOOTHING = 20.0


# ---------------------------------------------------------------------------
# per-fold target encoding
# ---------------------------------------------------------------------------
def per_fold_target_encode(X_tr: pd.DataFrame, X_va: pd.DataFrame, X_te: pd.DataFrame,
                           y_tr: np.ndarray, columns: list[str] | None,
                           smoothing: float = DEFAULT_SMOOTHING,
                           cat_fallback: list[str] | None = None):
    """Smoothed mean-target encoding computed from THIS fold's training rows only.

    Returns augmented copies (X_tr, X_va, X_te) with one added `te_<col>` column per
    encoded column; originals are never replaced (same invariant as the data-layer
    encoders in external_data/apply.py). The encoding map is fit on `X_tr`/`y_tr`
    exclusively: validation and test rows only ever look values up, and unseen
    categories fall back to the fold's global prior. This is the fold-aligned
    computation the operator contract requires (s4e1: an out-of-fold approximation
    inflated AUC to 0.89653 vs ~0.8937 fold-aligned).
    """
    cols = [c for c in (columns or cat_fallback or []) if c in X_tr.columns]
    if not cols:
        raise ValueError("target encoding requested but no encodable column exists "
                         f"(requested={columns!r}, available={list(X_tr.columns)[:8]}...)")
    y = np.asarray(y_tr, dtype=np.float64)
    if len(y) != len(X_tr):
        raise ValueError(f"y_tr length {len(y)} != X_tr rows {len(X_tr)}")
    prior = float(y.mean())
    out_tr, out_va, out_te = X_tr.copy(), X_va.copy(), X_te.copy()
    for c in cols:
        grp = pd.Series(y, index=X_tr.index).groupby(X_tr[c].astype(str))
        stats = grp.agg(["mean", "count"])
        enc = ((stats["count"] * stats["mean"] + smoothing * prior)
               / (stats["count"] + smoothing))
        name = f"te_{c}"
        out_tr[name] = X_tr[c].astype(str).map(enc).fillna(prior).astype(np.float64)
        out_va[name] = X_va[c].astype(str).map(enc).fillna(prior).astype(np.float64)
        out_te[name] = X_te[c].astype(str).map(enc).fillna(prior).astype(np.float64)
    return out_tr, out_va, out_te


# ---------------------------------------------------------------------------
# sparse linear family (unlocks encoding scheme "onehot_sparse")
# ---------------------------------------------------------------------------
def run_linear(params: dict, Xdf: pd.DataFrame, Xtestdf: pd.DataFrame,
               cat_feats: list[str], folds, y_target: np.ndarray, n_rows: int,
               invert=None):
    """Ridge on a sparse one-hot design, same OOF/pred contract as the GBDT runners.

    Per fold, the one-hot encoder and the numeric scaler are fit on the fold's training
    rows only (`handle_unknown="ignore"` absorbs categories the fold never saw). The
    target vector is whatever the calling evaluator trains its GBDTs on (log space for
    the firing-class evaluators) and `invert` is the same inversion those runners apply
    (e.g. np.expm1), so linear and tree members are pooled on identical semantics.
    """
    from scipy import sparse
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    alpha = float((params or {}).get("alpha", 1.0))
    cats = [c for c in cat_feats if c in Xdf.columns]
    nums = [c for c in Xdf.columns if c not in cats]
    invert = invert or (lambda v: v)

    oof = np.zeros(n_rows)
    pred = np.zeros(len(Xtestdf))
    n_folds = 0
    for tr_mask, va_mask in folds:
        X_tr, X_va = Xdf[tr_mask], Xdf[va_mask]
        parts_tr, parts_va, parts_te = [], [], []
        if cats:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
            ohe.fit(X_tr[cats].astype(str))
            parts_tr.append(ohe.transform(X_tr[cats].astype(str)))
            parts_va.append(ohe.transform(X_va[cats].astype(str)))
            parts_te.append(ohe.transform(Xtestdf[cats].astype(str)))
        if nums:
            sc = StandardScaler()
            tr_num = sc.fit_transform(X_tr[nums].fillna(0.0).to_numpy(np.float64))
            parts_tr.append(sparse.csr_matrix(tr_num))
            parts_va.append(sparse.csr_matrix(
                sc.transform(X_va[nums].fillna(0.0).to_numpy(np.float64))))
            parts_te.append(sparse.csr_matrix(
                sc.transform(Xtestdf[nums].fillna(0.0).to_numpy(np.float64))))
        A_tr = sparse.hstack(parts_tr).tocsr()
        A_va = sparse.hstack(parts_va).tocsr()
        A_te = sparse.hstack(parts_te).tocsr()
        m = Ridge(alpha=alpha, random_state=0)
        m.fit(A_tr, np.asarray(y_target)[tr_mask])
        oof[va_mask] = invert(m.predict(A_va))
        pred += invert(m.predict(A_te))
        n_folds += 1
    pred /= max(n_folds, 1)
    return oof, pred


# ---------------------------------------------------------------------------
# runtime attestation
# ---------------------------------------------------------------------------
def write_attestation(comp_dir: str, kind: str, payload: dict) -> None:
    """Record that this evaluator actually executed a dossier-requested mechanism.

    Merge-updates `<comp_dir>/evaluator_attestation.json` under `kind` (e.g.
    "postprocess", "target_encoding", "linear_family"). Never raises: attestation is
    evidence, not a dependency, and an eval must not fail because a disk write did.
    """
    path = os.path.join(comp_dir, "evaluator_attestation.json")
    try:
        doc = json.load(open(path)) if os.path.exists(path) else {}
    except Exception:  # noqa: BLE001
        doc = {}
    try:
        doc[kind] = payload
        json.dump(doc, open(path, "w"), indent=2)
    except Exception:  # noqa: BLE001
        pass


if __name__ == "__main__":  # self-test: leakage, smoothing math, linear smoke, attestation
    import tempfile

    rng = np.random.default_rng(0)

    # --- target encoding: fold isolation + smoothing formula + unseen fallback ---
    tr = pd.DataFrame({"cat": ["a"] * 6 + ["b"] * 4})
    va = pd.DataFrame({"cat": ["a", "b", "c"]})          # "c" unseen in training fold
    te = pd.DataFrame({"cat": ["b", "c"]})
    y = np.array([1.0] * 6 + [3.0] * 4)                  # a->1, b->3, prior=1.8
    s = 2.0
    otr, ova, ote = per_fold_target_encode(tr, va, te, y, ["cat"], smoothing=s)
    prior = 1.8
    exp_a = (6 * 1.0 + s * prior) / (6 + s)
    exp_b = (4 * 3.0 + s * prior) / (4 + s)
    assert np.isclose(ova["te_cat"].iloc[0], exp_a), (ova["te_cat"].iloc[0], exp_a)
    assert np.isclose(ova["te_cat"].iloc[1], exp_b)
    assert np.isclose(ova["te_cat"].iloc[2], prior), "unseen category must take the prior"
    assert "te_cat" not in tr.columns, "inputs must not be mutated"
    # leakage probe: encoding must be identical whatever the validation targets are —
    # recompute with a poisoned copy of va and check the map does not move
    ova2 = per_fold_target_encode(tr, va.assign(cat=["a", "a", "a"]), te, y, ["cat"],
                                  smoothing=s)[1]
    assert np.allclose(ova2["te_cat"], exp_a), "encoding may depend on training rows only"

    # --- linear runner: fits an OHE-recoverable signal ---
    n = 200
    df = pd.DataFrame({"g": rng.choice(list("xyz"), n), "num": rng.normal(size=n)})
    signal_map = {"x": 0.0, "y": 1.0, "z": 2.0}
    yv = df["g"].map(signal_map).to_numpy() + 0.01 * rng.normal(size=n)
    half = n // 2
    masks = [(np.arange(n) < half, np.arange(n) >= half)]
    oof, pred = run_linear({"alpha": 1e-3}, df, df.iloc[:5], ["g"], masks, yv, n)
    err = float(np.abs(oof[half:] - yv[half:]).mean())
    assert err < 0.1, f"linear family failed to recover a one-hot signal (mae={err})"

    # --- attestation roundtrip ---
    with tempfile.TemporaryDirectory() as d:
        write_attestation(d, "postprocess", {"honored_params": ["clip_min"]})
        write_attestation(d, "target_encoding", {"columns": ["g"]})
        doc = json.load(open(os.path.join(d, "evaluator_attestation.json")))
        assert doc["postprocess"]["honored_params"] == ["clip_min"]
        assert doc["target_encoding"]["columns"] == ["g"]

    print("eval_support self-test: all sections passed")
