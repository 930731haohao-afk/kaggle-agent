"""Coverage audit: how many of the dossiers' ideas can the execution layer actually run?

Before the operator contract this number could not be computed at all — ideas were prose and
whether anything executed them was invisible. That invisibility was the bug: on s3e19 a correct
"GDP as a scale covariate" judgment reached a layer that could only append a feature column.

The audit reports two regimes side by side:

  legacy prose  — ideas are free text; classified heuristically by keyword into operator
                  families, then split into families that have an operator and families that
                  do not. The MISSING count is the invisible gap.
  typed         — ideas are operators from the vocabulary, so executability is exact rather
                  than heuristic, and everything the vocabulary cannot express is counted in
                  `not_recorded` instead of hiding in prose.

Usage: VIRTUAL_ENV= uv run python3 external_data/coverage_audit.py
"""
from __future__ import annotations

import collections
import glob
import json
import os
import re

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

# families with a working operator today (keep in sync with apply.REALIZED/CONFIG_ONLY)
IMPLEMENTED = {
    "join_feature":  r"join .*(gdp|world bank|holiday|iso|covid|exchange|macro)|external (data|covariate)",
    "ratio_target":  r"ratio target|normali[sz]e.*(gdp|covariate)|divide.*by.*gdp|log\(.*/",
    "trend_term":    r"trend term|linear year|year_c|time index",
    "flag_feature":  r"holiday flag|is_holiday|holiday (indicator|calendar)",
    "sample_weight": r"sample weight|down-?weight|re-?weight|regime (flag|indicator)|covid",
    "objective":     r"l1 |mae objective|objective|loss function|tweedie|poisson|huber|logloss",
    "blend_member":  r"blend|ensemble|stack|shallow.*regulari|decorrelat|diverse",
    "split_policy":  r"kfold|groupkfold|stratified|timeseriessplit|split|fold",
    "postprocess":   r"round|clip|inside the metric|post-?process|threshold",
    "encoding":      r"target encoding|ordinal|one-?hot|categor|encod|cross(es)? |interaction",
}
MISSING = {
    "transform":  r"derivative|savitzky|smooth|pca|dimension|scal(e|ing)|log1p|spectral",
    "structure":  r"per-cell|neighborhood|decompos|multiplicative|seasonal-naive|per-country|per-product|hierarch",
    "feature_eng": r"fourier|lag |rolling|aggregat|group statistic|frequency|count feature",
    "tuning":     r"optuna|hyperparameter|bayesian|grid search|early stopping",
}


def classify(text: str) -> tuple[str, str]:
    t = str(text).lower()
    for k, pat in IMPLEMENTED.items():
        if re.search(pat, t):
            return "IMPLEMENTED", k
    for k, pat in MISSING.items():
        if re.search(pat, t):
            return "MISSING", k
    return "UNCLASSIFIED", "-"


def main() -> None:
    legacy_rows, typed_rows = [], []
    fam = collections.Counter()
    for p in sorted(glob.glob(os.path.join(_ROOT, "competitions", "*", "dossier*.json"))):
        comp = os.path.basename(os.path.dirname(p))
        d = json.load(open(p))
        ideas = d.get("injection_ideas")
        nrec = len(d.get("not_recorded") or [])
        if isinstance(ideas, dict):                                  # legacy prose
            texts = [t for v in ideas.values() if isinstance(v, list) for t in v]
            st = collections.Counter()
            for t in texts:
                s, f = classify(t)
                st[s] += 1
                fam[(s, f)] += 1
            legacy_rows.append((comp, len(texts), st["IMPLEMENTED"], st["MISSING"],
                                st["UNCLASSIFIED"], nrec))
        elif isinstance(ideas, list) and ideas and isinstance(ideas[0], dict):   # typed
            ops = [i.get("operator") for i in ideas]
            known = sum(1 for o in ops if o in IMPLEMENTED)
            typed_rows.append((comp, len(ops), known, nrec))

    print("=== LEGACY PROSE SCHEMA (the state the contract replaced)")
    print(f"{'comp':36s} {'ideas':>5s} {'exec':>5s} {'gap':>5s} {'?':>3s} {'not_rec':>7s}")
    ti = te = tg = tu = 0
    for c, n, i, m, u, nr in legacy_rows:
        print(f"{c[:36]:36s} {n:>5d} {i:>5d} {m:>5d} {u:>3d} {nr:>7d}")
        ti += n; te += i; tg += m; tu += u
    if ti:
        print(f"{'TOTAL':36s} {ti:>5d} {te:>5d} {tg:>5d} {tu:>3d}"
              f"        -> {te/ti*100:.0f}% executable, {tg/ti*100:.0f}% invisible gap")
    print("\n  gap by family (no operator exists):")
    for (s, f), n in sorted(fam.items(), key=lambda x: -x[1]):
        if s == "MISSING":
            print(f"    {f:12s} {n:4d}")

    print("\n=== TYPED SCHEMA (executability is exact, the gap is explicit)")
    print(f"{'comp':36s} {'ops':>4s} {'known':>6s} {'not_recorded':>13s}")
    to = tk = tn = 0
    for c, n, k, nr in typed_rows:
        print(f"{c[:36]:36s} {n:>4d} {k:>6d} {nr:>13d}")
        to += n; tk += k; tn += nr
    if to:
        print(f"{'TOTAL':36s} {to:>4d} {tk:>6d} {tn:>13d}"
              f"   -> {tk/to*100:.0f}% of emitted operators executable; "
              f"{tn} inexpressible ideas recorded rather than hidden")


if __name__ == "__main__":
    main()
