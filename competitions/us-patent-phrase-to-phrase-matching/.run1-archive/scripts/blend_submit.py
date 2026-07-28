"""Blend DeBERTa + LGBM on OOF Pearson, write final submission."""
import importlib.util
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

COMP = "/home/tjyen/ai_agents/kaggle/competitions/us-patent-phrase-to-phrase-matching"
DATA = f"{COMP}/data"

train = pd.read_csv(f"{DATA}/train.csv")
test = pd.read_csv(f"{DATA}/test.csv")
sub = pd.read_csv(f"{DATA}/sample_submission.csv")
y = train.score.values

oof_d = np.load(f"{COMP}/scripts/deberta_oof.npy")
oof_l = np.load(f"{COMP}/scripts/lgbm_oof.npy")
t_d = np.load(f"{COMP}/scripts/deberta_test.npy")
t_l = np.load(f"{COMP}/scripts/lgbm_test.npy")

r_d = pearsonr(y, oof_d)[0]
r_l = pearsonr(y, oof_l)[0]
print(f"deberta OOF r={r_d:.4f}  lgbm OOF r={r_l:.4f}")

best_w, best_r = 1.0, r_d
for w in np.arange(0.5, 1.001, 0.01):
    r = pearsonr(y, w * oof_d + (1 - w) * oof_l)[0]
    if r > best_r:
        best_w, best_r = w, r
print(f"best blend: w_deberta={best_w:.2f}  OOF r={best_r:.4f} "
      f"(+{best_r - r_d:.4f} vs deberta solo)")

# Only blend if it actually improves OOF; otherwise deberta solo.
if best_r > r_d + 1e-5:
    final_test = best_w * t_d + (1 - best_w) * t_l
    model_name = f"blend_deberta{best_w:.2f}_lgbm{1-best_w:.2f}"
    final_r = best_r
else:
    final_test = t_d
    model_name = "deberta-v3-base-5fold-solo"
    final_r = r_d

final_test = np.clip(final_test, 0, 1)
assert list(sub.id) == list(test.id)
sub["score"] = final_test
sub.to_csv(f"{COMP}/submissions/final_blend.csv", index=False)
sub.to_csv(f"{COMP}/submission.csv", index=False)
print(f"final submission written ({model_name}), rows={len(sub)}")

_spec = importlib.util.spec_from_file_location(
    "experiment_log",
    "/home/tjyen/ai_agents/kaggle/.claude/skills/kaggle-agent/assets/utils/"
    "experiment_log.py")
experiment_log = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(experiment_log)
experiment_log.log_experiment_v2(
    COMP, model=model_name, metric="pearson_r", direction="maximize",
    score=final_r,
    ensemble={"type": "weighted_average",
              "weights": {"deberta": round(best_w, 2),
                          "lgbm": round(1 - best_w, 2)},
              "selected": model_name},
    submission="submissions/final_blend.csv",
    notes="Grid-searched blend weight on OOF; falls back to deberta solo if "
          "blend does not improve OOF Pearson.")
