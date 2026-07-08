"""Phase J stage-5 接線 pilot:量「外部想法注入」對搜尋最佳解的邊際價值(可重現)。

做法(快、真、不重訓):對一場競賽,取其**階段4樹搜尋贏家**(=搜尋收斂/plateau 點)的
成員,呼叫 `idea_injection.plateau_inject` 產生「外部想法翻成的候選」,再用**已快取的成員
OOF** 直接評估該候選,與階段4的 prob 空間最佳做同成員、同權重搜尋的對照。

目前登錄的乾淨翻譯器是 EXT-12(rank averaging)→ rank 空間 blend。此腳本回答:
「在搜尋收斂點注入這條外部想法,能不能贏過階段4?」

用法:
    uv run python3 tree_search/stage5_inject_pilot.py s4e1
    uv run python3 tree_search/stage5_inject_pilot.py s4e1 --subsample 150000

結論(2026-07-08 實測 s3e3/s4e1/s6e2):rank averaging 在所有測試 AUC 場的 delta 皆 ≤1e-5
(噪音級),外部注入無可量測系統性增益——見 docs/phase_j_j3_findings.md。
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import yaml
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
import idea_injection as inj  # noqa: E402

# 各場的 comp_meta(交給 idea_injection 篩選外部想法)+ AUC 目標欄解析
COMP_META = {
    "s3e3": {"metric": "auc", "tags": ["10k", "small_sample"]},
    "s4e1": {"metric": "auc", "tags": []},
    "s6e2": {"metric": "auc", "tags": []},
    "s3e7": {"metric": "auc", "tags": []},
}


def _load_y(comp_dir):
    cfg = yaml.safe_load(open(os.path.join(comp_dir, "config.yaml")))
    tgt = cfg.get("target_column") or cfg.get("target")
    tr = pd.read_csv(os.path.join(comp_dir, "data", "train.csv"))
    if tgt not in tr.columns:
        cand = [c for c in tr.columns if "disease" in c.lower() or c.lower() in ("target", "y")]
        tgt = cand[0] if cand else tr.columns[-1]
    y = tr[tgt]
    if y.dtype == object or str(y.dtype).startswith("string"):
        y = y.astype(str).str.strip().str.lower().isin(
            ["1", "yes", "presence", "true", "pos"]).astype(int)
    return y.values.astype(float), tgt


def _weight_search_auc(mat, y, k=800, seed=42):
    rng = np.random.RandomState(seed)
    best = -1.0
    cands = [np.ones(mat.shape[1]) / mat.shape[1]]
    cands += [rng.dirichlet(np.ones(mat.shape[1])) for _ in range(k)]
    for w in cands:
        best = max(best, roc_auc_score(y, mat @ w))
    return best


def run(comp, subsample=None, k=800):
    comp_dir = os.path.join(_ROOT, "competitions", f"playground-series-{comp}")
    y, tgt = _load_y(comp_dir)
    tree = json.load(open(os.path.join(comp_dir, "experiments_tree_v3.json")))
    scored = [n for n in tree["nodes"] if isinstance(n.get("score"), (int, float))]
    best = min(scored, key=lambda n: n["score"])  # AUC 存為 -AUC(minimize)
    members = best["config"].get("members", [])
    cache = os.path.join(_HERE, f"cache_{comp}")
    oofs = {m: np.load(os.path.join(cache, f"solo_{m}.npz"), allow_pickle=True)["oof"]
            for m in members if os.path.exists(os.path.join(cache, f"solo_{m}.npz"))}
    ok = [m for m in members if m in oofs]
    assert len(ok) >= 2, f"{comp}: 成員 OOF 不足({len(ok)})"

    # idea_injection 產生的注入候選(EXT-12 -> rank 空間 blend,同這批成員)
    injected = inj.plateau_inject(COMP_META.get(comp, {"metric": "auc"}), ok, tried_sigs=set(), k=3)
    inj_ids = [e for _, _, e in injected]

    M = np.column_stack([oofs[m] for m in ok])
    y_use, M_use = y, M
    if subsample and subsample < len(y):
        idx = np.random.RandomState(0).choice(len(y), subsample, replace=False)
        y_use, M_use = y[idx], M[idx]

    prob = _weight_search_auc(M_use, y_use, k=k)
    Mr = np.column_stack([rankdata(M_use[:, j]) for j in range(M_use.shape[1])]) / len(y_use)
    rank = _weight_search_auc(Mr, y_use, k=k)

    print(f"=== {comp} stage-5 注入 pilot(AUC,{len(ok)} 成員,"
          f"{'子樣本 %d' % subsample if subsample else '全量'})===")
    print(f"  注入的外部想法候選: {inj_ids or '(無—已去重/無翻譯器)'}")
    print(f"  prob 空間(階段4做法)      AUC = {prob:.6f}")
    print(f"  rank 空間(EXT-12 注入候選) AUC = {rank:.6f}")
    print(f"  delta(rank - prob)          = {rank - prob:+.6f}")
    verdict = "有系統性增益" if rank - prob > 1e-4 else "無可量測增益(≤1e-4,噪音級)"
    print(f"  → 外部注入(rank averaging)在 {comp}:{verdict}")
    return dict(comp=comp, members=len(ok), injected=inj_ids, prob=prob, rank=rank, delta=rank - prob)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("comp", help="競賽代號,如 s4e1")
    ap.add_argument("--subsample", type=int, default=None, help="子抽樣列數(大場加速)")
    ap.add_argument("--k", type=int, default=800, help="dirichlet 權重搜尋樣本數")
    a = ap.parse_args()
    run(a.comp, subsample=a.subsample, k=a.k)
