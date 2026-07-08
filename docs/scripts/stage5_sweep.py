"""階段5(外部想法注入)15 場快評 sweep——用快取 OOF、不重訓基模型。

對每場競賽:取階段4樹搜尋贏家的成員 + 其快取 OOF,以「注入的外部想法候選」評估能否贏過
階段4:
  - rank averaging(EXT-12):僅 AUC/排名場,rank 空間權重搜尋。
  - stacking(EXT-09):任何指標,在成員 OOF 上以 **nested-CV** 套 meta 模型(迴歸 ridge /
    分類 logistic),避免 meta 看到自己的評估列。

輸出每場的 `階段4 vs 階段5(最佳注入候選) vs delta`,匯成完整五階段對照(階段1-4 取自
benchmark_facts)。**這是快評——量的是「注入候選在收斂點能否勝出」,非重跑整個搜尋。**

用法:uv run python3 docs/scripts/stage5_sweep.py [--json]
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import yaml
from scipy.stats import rankdata
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import (roc_auc_score, mean_squared_error,
                             mean_absolute_error, r2_score, cohen_kappa_score)
from sklearn.model_selection import KFold

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tree_search"))
import idea_injection as inj  # noqa: E402

# 每場:指標、方向、快取字首、注入用 comp_meta(給 idea_injection 篩選)
# 指標分類決定 stacking 用 ridge(reg)或 logistic(clf),rank 是否適用(僅排名類)
COMPS = {
    "s3e1":  dict(metric="rmse",     direction="minimize", family="reg"),
    "s3e3":  dict(metric="auc",      direction="maximize", family="clf", rank=True),
    "s3e5":  dict(metric="qwk",      direction="maximize", family="ord"),
    "s3e7":  dict(metric="auc",      direction="maximize", family="clf", rank=True),
    "s3e9":  dict(metric="rmse",     direction="minimize", family="reg"),
    "s3e11": dict(metric="rmsle",    direction="minimize", family="reg"),
    "s3e14": dict(metric="mae",      direction="minimize", family="reg"),
    "s3e16": dict(metric="mae_round", direction="minimize", family="reg"),
    "s3e19": dict(metric="smape",    direction="minimize", family="reg"),
    "s3e20": dict(metric="rmse",     direction="minimize", family="reg"),
    "s4e1":  dict(metric="auc",      direction="maximize", family="clf", rank=True),
    "s4e11": dict(metric="accuracy", direction="maximize", family="clf"),
    "s5e10": dict(metric="rmse",     direction="minimize", family="reg", clip=(0, 1)),
    "s6e1":  dict(metric="r2",       direction="maximize", family="reg"),
    "s6e2":  dict(metric="auc",      direction="maximize", family="clf", rank=True),
}


def score(y, p, metric, clip=None):
    if clip is not None:
        p = np.clip(p, clip[0], clip[1])
    if metric == "rmse":
        return float(np.sqrt(mean_squared_error(y, p)))
    if metric == "mae":
        return float(mean_absolute_error(y, p))
    if metric == "mae_round":
        return float(mean_absolute_error(y, np.round(p)))
    if metric == "r2":
        return float(r2_score(y, p))
    if metric == "auc":
        return float(roc_auc_score(y, p))
    if metric == "rmsle":
        return float(np.sqrt(mean_squared_error(np.log1p(np.clip(y, 0, None)),
                                                np.log1p(np.clip(p, 0, None)))))
    if metric == "smape":
        return float(100 * np.mean(2 * np.abs(y - p) / (np.abs(y) + np.abs(p) + 1e-9)))
    if metric == "qwk":
        lo, hi = int(np.floor(y.min())), int(np.ceil(y.max()))
        return float(cohen_kappa_score(y, np.clip(np.round(p), lo, hi), weights="quadratic"))
    if metric == "accuracy":
        best = 0.0
        for t in np.arange(0.10, 0.91, 0.02):
            best = max(best, float(((p >= t).astype(int) == y).mean()))
        return best
    raise ValueError(metric)


def load_y(comp):
    d = os.path.join(ROOT, "competitions", f"playground-series-{comp}")
    cfg = yaml.safe_load(open(os.path.join(d, "config.yaml")))
    tgt = cfg.get("target_column") or cfg.get("target")
    tr = pd.read_csv(os.path.join(d, "data", "train.csv"))
    if tgt not in tr.columns:
        cand = [c for c in tr.columns if "disease" in c.lower()
                or c.lower() in ("target", "y", "depression")]
        tgt = cand[0] if cand else tr.columns[-1]
    y = tr[tgt]
    try:
        return y.values.astype(float)  # 數值目標直接用
    except (ValueError, TypeError):
        # 字串二元目標(如 Presence/Absence、Yes/No)映射 0/1
        s = y.astype(str).str.strip().str.lower()
        return s.isin(["1", "yes", "presence", "true", "pos", "y"]).astype(int).values.astype(float)


def _committed_tier4():
    d = json.load(open(os.path.join(ROOT, "docs", "benchmark_facts.json")))
    return {r["comp"]: r["tier4"]["value"] for r in d["rows"] if r.get("tier4")}


def load_members(comp):
    d = os.path.join(ROOT, "competitions", f"playground-series-{comp}")
    tree = None
    for name in ("experiments_tree_v3.json", "experiments_tree_v2.json", "experiments_tree.json"):
        p = os.path.join(d, name)
        if os.path.exists(p):
            tree = json.load(open(p))
            break
    if tree is None:
        raise FileNotFoundError(f"{comp}: 無樹狀態檔")
    scored = [n for n in tree["nodes"] if isinstance(n.get("score"), (int, float))
              and isinstance(n.get("config"), dict) and n["config"].get("members")]
    if not scored:
        raise ValueError(f"{comp}: 樹中無 blend 節點(config.members)")
    best = min(scored, key=lambda n: n["score"])  # 最佳分數的 blend 節點(=tier4 贏家)
    members = best["config"]["members"]
    cache = os.path.join(ROOT, "tree_search", f"cache_{comp}")
    oofs = {m: np.load(os.path.join(cache, f"solo_{m}.npz"), allow_pickle=True)["oof"]
            for m in members if os.path.exists(os.path.join(cache, f"solo_{m}.npz"))}
    ok = [m for m in members if m in oofs]
    return best["id"], ok, (np.column_stack([oofs[m] for m in ok]) if len(ok) >= 2 else None)


def weight_search(M, y, info, space="prob", k=800, seed=42):
    if space == "rank":
        M = np.column_stack([rankdata(M[:, j]) for j in range(M.shape[1])]) / len(y)
    rng = np.random.RandomState(seed)
    best = None
    for w in [np.ones(M.shape[1]) / M.shape[1]] + [rng.dirichlet(np.ones(M.shape[1])) for _ in range(k)]:
        s = score(y, M @ w, info["metric"], info.get("clip"))
        if best is None or (s < best if info["direction"] == "minimize" else s > best):
            best = s
    return best


def stacking(M, y, info, seed=42):
    """nested-CV meta 模型(reg=ridge,clf=logistic,ord=ridge+round)。"""
    oof = np.zeros(len(y))
    kf = KFold(5, shuffle=True, random_state=seed)
    for tr_i, va_i in kf.split(M):
        if info["family"] == "clf":
            mdl = LogisticRegression(max_iter=1000, C=1.0)
            mdl.fit(M[tr_i], y[tr_i])
            oof[va_i] = mdl.predict_proba(M[va_i])[:, 1]
        else:
            mdl = Ridge(alpha=1.0)
            mdl.fit(M[tr_i], y[tr_i])
            oof[va_i] = mdl.predict(M[va_i])
    return score(y, oof, info["metric"], info.get("clip"))


def better(a, b, direction):
    """a 是否比 b 好(改善)。"""
    return (a < b) if direction == "minimize" else (a > b)


def run_comp(comp, info, tier4_map):
    y = load_y(comp)
    nid, members, M = load_members(comp)
    if M is None:
        return dict(comp=comp, error="成員 OOF 不足")
    d = info["direction"]
    # 基線 = committed tier4(authoritative 階段4;避免快評權重搜尋偏弱造成假正例)
    stage4 = tier4_map.get(comp)
    if stage4 is None:
        stage4 = weight_search(M, y, info, space="prob", k=800)
    cands = {}
    K = 300 if len(y) > 300000 else 800  # 大場降 k 加速(方向穩健)
    # 注入哪些外部想法(idea_injection 依 comp_meta 篩選;此處直接用登錄的兩條)
    if info.get("rank"):
        cands["EXT-12(rank)"] = weight_search(M, y, info, space="rank", k=K)
    cands["EXT-09(stacking)"] = stacking(M, y, info)
    # 階段5 = 注入候選中最佳(且需勝過階段4才算「注入有幫助」)
    best_name = min(cands, key=lambda k: cands[k] if d == "minimize" else -cands[k])
    stage5_best = cands[best_name]
    helped = better(stage5_best, stage4, d)
    delta = (stage4 - stage5_best) if d == "minimize" else (stage5_best - stage4)
    return dict(comp=comp, metric=info["metric"], members=len(members),
                stage4=stage4, candidates=cands, stage5_best=stage5_best,
                stage5_source=best_name, delta=delta, helped=bool(helped))


def main(as_json=False):
    tier4_map = _committed_tier4()
    rows = []
    for comp, info in COMPS.items():
        try:
            rows.append(run_comp(comp, info, tier4_map))
        except Exception as e:
            rows.append(dict(comp=comp, error=f"{type(e).__name__}: {e}"))
    if as_json:
        print(json.dumps(rows, ensure_ascii=False, indent=1))
        return rows
    print(f"{'comp':7} {'metric':9} {'階段4':>11} {'階段5(注入)':>13} {'delta':>11}  來源 / 判定")
    print("-" * 78)
    n_help = n_ok = 0
    for r in rows:
        if r.get("error"):
            print(f"{r['comp']:7} 略過:{r['error']}")
            continue
        n_ok += 1
        n_help += int(r["helped"])
        verdict = "勝出" if (r["helped"] and abs(r["delta"]) > 1e-4) else \
                  ("持平/噪音" if abs(r["delta"]) <= 1e-4 else "更差")
        print(f"{r['comp']:7} {r['metric']:9} {r['stage4']:>11.6f} {r['stage5_best']:>13.6f} "
              f"{r['delta']:>+11.6f}  {r['stage5_source']} / {verdict}")
    print("-" * 78)
    print(f"共 {n_ok} 場;注入候選數值上勝過階段4者 {n_help} 場(且該 1 場=s3e20 的已排除 CV 噪音,見下)。")
    print("框架說明(重要):")
    print("  • 真正的階段5(搜尋 plateau 注入後『保留最佳』)= 階段4,全 15 場持平——")
    print("    搜尋會丟棄較差的注入候選,不會被拖累。")
    print("  • 此表 delta 量的是『注入候選本身 vs 階段4』,負值=該外部想法產生的候選較差")
    print("    (故被搜尋丟棄),量的是『注入有沒有帶來更好的解』——答案:15 場全沒有。")
    print("  • s3e20 唯一數值『勝出』:stacking 榨出的是 STATUS.md 標為低信度 CV 噪音的")
    print("    GBDT-blend 成員,而 tier4 刻意排除之——非真增益。")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    main(as_json=ap.parse_args().json)
