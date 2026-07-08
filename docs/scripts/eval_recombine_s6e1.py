"""評估 s6e1 任意 (members, weights) 的 clipped-OOF R2,用於 LLM 重組提案。"""
import json, os, sys
import numpy as np, pandas as pd
from sklearn.metrics import r2_score
ROOT="/home/tjyen/ai_agents/kaggle"
y=pd.read_csv(f"{ROOT}/competitions/playground-series-s6e1/data/train.csv")["exam_score"].to_numpy(float)
CACHE=f"{ROOT}/tree_search/cache_s6e1"
def oof(m): return np.load(f"{CACHE}/solo_{m}.npz")["oof"]
def score(members, weights=None, strategy="given"):
    M=np.column_stack([oof(m) for m in members])
    if strategy=="equal" or weights is None:
        w=np.ones(M.shape[1])/M.shape[1]
    elif strategy=="rank":
        from scipy.stats import rankdata
        M=np.column_stack([rankdata(M[:,j]) for j in range(M.shape[1])])/len(y)
        w=np.ones(M.shape[1])/M.shape[1]
    elif strategy=="dirichlet":
        rng=np.random.RandomState(42); best=None
        for _ in range(2000):
            ww=rng.dirichlet(np.ones(M.shape[1])); s=r2_score(y,np.clip(M@ww,0,100))
            if best is None or s>best[0]: best=(s,ww)
        return best[0]
    else:
        w=np.asarray(weights,float); w=w/w.sum()
    return float(r2_score(y, np.clip(M@w,0,100)))
if __name__=="__main__":
    champ=[0,1,2,3,4,5,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]
    tj=json.load(open(f"{ROOT}/competitions/playground-series-s6e1/experiments_tree_v3.json"))
    w=tj["search_state"]["node_results"]["27"]["weights"]
    print(f"冠軍(給定權重)R2 = {score(champ, w):.6f}  (應≈0.787183)")
    print(f"冠軍成員等權   R2 = {score(champ, strategy='equal'):.6f}")
    print(f"冠軍成員重搜dirichlet R2 = {score(champ, strategy='dirichlet'):.6f}")
