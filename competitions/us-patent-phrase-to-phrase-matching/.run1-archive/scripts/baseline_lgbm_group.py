"""Baseline: lexical/TF-IDF features + LightGBM. Secures a valid submission fast."""
import importlib.util
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupKFold
import lightgbm as lgb

SEED = 42
COMP = "/home/tjyen/ai_agents/kaggle/competitions/us-patent-phrase-to-phrase-matching"
DATA = f"{COMP}/data"

_spec = importlib.util.spec_from_file_location(
    "experiment_log",
    "/home/tjyen/ai_agents/kaggle/.claude/skills/kaggle-agent/assets/utils/experiment_log.py")
experiment_log = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(experiment_log)

train = pd.read_csv(f"{DATA}/train.csv")
test = pd.read_csv(f"{DATA}/test.csv")
sub = pd.read_csv(f"{DATA}/sample_submission.csv")

def build_features(df, tfidf_char, tfidf_word):
    a, t = df.anchor.str.lower(), df.target.str.lower()
    va, vt = tfidf_char.transform(a), tfidf_char.transform(t)
    wa, wt = tfidf_word.transform(a), tfidf_word.transform(t)
    cos_char = np.asarray(va.multiply(vt).sum(axis=1)).ravel()
    cos_word = np.asarray(wa.multiply(wt).sum(axis=1)).ravel()
    aw = a.str.split().apply(set)
    tw = t.str.split().apply(set)
    inter = [len(x & y) for x, y in zip(aw, tw)]
    union = [len(x | y) for x, y in zip(aw, tw)]
    jac = np.array(inter) / np.maximum(union, 1)
    feats = pd.DataFrame({
        "cos_char": cos_char,
        "cos_word": cos_word,
        "jaccard": jac,
        "n_common": inter,
        "len_a": a.str.len(), "len_t": t.str.len(),
        "nw_a": a.str.split().str.len(), "nw_t": t.str.split().str.len(),
        "len_ratio": a.str.len() / np.maximum(t.str.len(), 1),
        "a_in_t": [int(x in y) for x, y in zip(a, t)],
        "t_in_a": [int(y in x) for x, y in zip(a, t)],
        "same_first_word": (a.str.split().str[0] == t.str.split().str[0]).astype(int),
    })
    feats["context_code"] = df.context.astype("category").cat.codes
    feats["section"] = df.context.str[0].astype("category").cat.codes
    return feats

all_text = pd.concat([train.anchor, train.target, test.anchor, test.target]).str.lower()
tfidf_char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2).fit(all_text)
tfidf_word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1).fit(all_text)

X = build_features(train, tfidf_char, tfidf_word)
X_test = build_features(test, tfidf_char, tfidf_word)
y = train.score.values

params = dict(objective="regression", metric="rmse", learning_rate=0.05,
              num_leaves=63, feature_fraction=0.9, bagging_fraction=0.9,
              bagging_freq=1, seed=SEED, deterministic=True, force_row_wise=True,
              num_threads=8, verbosity=-1)

skf = GroupKFold(n_splits=4)
oof = np.zeros(len(train))
pred = np.zeros(len(test))
strat = (y * 4).astype(int)
for fold, (tr, va) in enumerate(skf.split(X, groups=train.anchor)):
    dtr = lgb.Dataset(X.iloc[tr], y[tr])
    dva = lgb.Dataset(X.iloc[va], y[va])
    m = lgb.train(params, dtr, num_boost_round=3000, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    oof[va] = m.predict(X.iloc[va], num_iteration=m.best_iteration)
    pred += m.predict(X_test, num_iteration=m.best_iteration) / 4
    print(f"fold {fold}: r={pearsonr(y[va], oof[va])[0]:.4f} iters={m.best_iteration}")

r = pearsonr(y, oof)[0]
print(f"OOF Pearson r = {r:.4f}")
np.save(f"{COMP}/scripts/lgbm_oof_group.npy", oof)
np.save(f"{COMP}/scripts/lgbm_test_group.npy", pred)

print("group-CV lgbm done")
