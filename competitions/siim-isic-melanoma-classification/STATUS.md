# siim-isic-melanoma-classification — run2 (kaggle-agent)

**Final CV (ROC-AUC): 0.93214** — honest leave-fold-out score of a 6-member convex rank blend
(vit_small @384/256/320 px + efficientnet_b0 + CatBoost/LightGBM on metadata & image statistics).
Optimistic full-OOF counterpart 0.93291; patient-grouped cross-check 0.93278; best solo 0.92596
(vit_small @384 px). Deliverable: `submission.csv`, 4,142 rows, id order verified against
`sample_submission.csv`. Run wall time ~6.5 h, all training in the foreground.

---

## 1. What was run

Fresh run2. Everything from the earlier run was moved to `run1_archive/` before starting and was
not consulted for any modelling decision; the image caches were rebuilt from `data/jpeg` by this
run's own `scripts/prep_images.py`.

| Stage | Output |
|---|---|
| 0.5 Problem dossier | `dossier.json` |
| 1 EDA | `scripts/eda.py`, `eda_summary.json`, `cache/{train,test}_sizes.csv` |
| 2 Features | `scripts/prep_images.py` (pixels + image statistics), `scripts/common.py` (`build_features`) |
| 3 Modelling | `scripts/train_meta.py`, `scripts/discover.py`, `scripts/vision.py`, `scripts/finetune.py` |
| 4 Evaluation / blend | `scripts/blend.py`, `scripts/stack.py`, `blend_results.json` |
| 5 Submission | `scripts/submit.py`, `submission.csv` |

All 20 experiments are in `experiments.json` via `experiment_log.log_experiment_v2()`.

## 2. The decision that mattered: the split is NOT patient-disjoint

The real SIIM-ISIC competition has patient-disjoint train/test, which makes
`StratifiedGroupKFold(patient_id)` the reflexive choice. **That reflex is wrong here.** EDA found
that mle-bench re-split the original 33,126-image train set **randomly by image** into
28,984 / 4,142:

- all **1,457** test patients also appear in train (overlap 1457/1457, 599 patients train-only);
- per-patient test share mean 0.124, median 0.111, vs an overall test share of 0.125.

So the graded test set is iid-by-image. A patient-grouped CV would have been measuring a strictly
harder problem than the one being scored. Primary CV is therefore `StratifiedKFold(5, seed=42)`
on images; every model additionally reports a patient-grouped `StratifiedGroupKFold` score as an
honesty check and as a leakage detector.

**The two agreed everywhere** — the largest gap across all 10 members was 0.0004 AUC
(blend 0.93214 image-level vs 0.93278 patient-grouped). That is a useful negative result: no
member is exploiting patient identity, so the choice cost nothing here. The one-line overlap
check remains worth doing every time.

## 3. Results

### Metadata + image-statistic models (no CNN)
Colour/texture statistics are computed for free during JPEG decoding (central-disc "lesion" vs
outer-ring "skin" colour, contrast, gradient energy, vignetting), plus resolution and
patient-relative "ugly duckling" z-scores.

| model | CV AUC | patient-grouped |
|---|---|---|
| LogisticRegression, sex+age+site only | 0.67148 | 0.67121 |
| LogisticRegression, all 54 features | 0.85467 | 0.85418 |
| LightGBM | 0.88574 | 0.88556 |
| XGBoost | 0.88668 | 0.88633 |
| **CatBoost** | **0.89171** | 0.89157 |

Feature-block ablation (LightGBM, identical folds, vs 0.88574 full):

| removed block | CV AUC | Δ |
|---|---|---|
| image colour/texture statistics | 0.82114 | **−0.06460** |
| image resolution | 0.87158 | −0.01417 |
| patient-relative (ugly duckling) | 0.87244 | −0.01330 |

The cheap image statistics are by far the largest tabular lever, and `ud_*` patient-relative
z-scores occupy 8 of the top-20 LightGBM importances — the clinical "ugly duckling" sign
(a lesion is suspicious *relative to that patient's other moles*) shows up directly in the model.

### Vision — discovery sweep (fold 0, 2 epochs, 256 px, 18 configs, 52 min)
Per-backbone LR mini-sweep, as `knowledge/vision_experience.md` requires:

| backbone | 1e-4 | 3e-4 | 1e-3 | best |
|---|---|---|---|---|
| vit_small_patch16_224 | **0.8853** | 0.8686 | 0.8173 | 1e-4 |
| efficientnet_b0 | 0.7686 | 0.8231 | **0.8782** | 1e-3 |
| efficientnet_b3 | 0.7868 | 0.8489 | **0.8716** | 1e-3 |
| resnet50 | 0.7608 | 0.8393 | **0.8694** | 1e-3 |
| resnet18 | 0.7936 | **0.8672** | 0.8551 | 3e-4 |
| convnext_tiny | **0.8643** | 0.8468 | 0.7775 | 1e-4 |

CNNs peak at 1e-3, ViT/ConvNeXt at 1e-4 — the pattern now holds on a 5th domain.

### Vision — full 5-fold fine-tunes (bf16, dihedral aug, 4× flip TTA, no class weighting)

| tag | backbone | px | ep | CV AUC | patient-grouped | wall |
|---|---|---|---|---|---|---|
| vit_small_384 | vit_small_patch16_224 | 384 | 8 | **0.92596** | 0.92589 | 89 min |
| vit_small_256 | vit_small_patch16_224 | 256 | 10 | 0.92309 | 0.92295 | 49 min |
| vit_small_320 | vit_small_patch16_224 (seed 2024) | 320 | 9 | 0.92024 | 0.92007 | 69 min |
| resnet50_256 | resnet50 | 256 | 10 | 0.89166 | 0.89179 | 70 min |
| effb0_256 | efficientnet_b0 | 256 | 10 | 0.88447 | 0.88440 | 48 min |

### Blend (Stage 4)
Rank-transform each member, then Caruana greedy selection with replacement. Reported score is the
**leave-fold-out** estimate: weights are refitted on 4 folds and scored on the held-out fold.

| | AUC |
|---|---|
| best solo (vit_small_384) | 0.92596 |
| equal weight, 10 members | 0.92517 |
| **optimised, full OOF (optimistic)** | 0.93291 |
| **leave-fold-out (honest — reported CV)** | **0.93214** |
| patient-grouped cross-check | 0.93278 |

Weights: vit_small_384 0.388, vit_small_256 0.286, vit_small_320 0.102, lgb_meta 0.102,
cat_meta 0.082, effb0_256 0.041. `resnet50_256`, `xgb_meta`, `lr_allfeat` and the level-2 stack
were given zero weight.

## 4. What did not work

- **A level-2 LightGBM stack lost to the convex blend.** Stacking member ranks together with
  context features (age, site, resolution, patient size) scored 0.92566 solo and earned ~0.02
  weight before being dropped entirely once vit_small_320 joined. Same direction as the s3e14
  Ridge-stacking counterexample in `knowledge/experience.md`.
- **Extra CNN families bought nothing.** Member-set comparison on the honest score:
  {vit384, vit256, 3× metadata} = 0.93204; + resnet50 = 0.93204; all 10 members = 0.93203. Both
  CNNs were zero-weighted or near-zero despite resnet50's respectable 0.89166 solo. Architecture
  diversity is not automatically useful; **input-resolution** diversity within the ViT family was,
  taking 0.78 of the total weight across three scales.
- **The discovery-scale gap widened rather than shrank.** vit_small led efficientnet_b0 by 0.007
  at 2 epochs and by 0.039 at 10 epochs — a direct counterexample to the digit-recognizer prior
  "don't over-prune on discovery scores". Both have been written back to the vision library.

## 5. Tree search: deliberately not used — reason

The skill designates tree search as the preferred Stage 4 loop *"where the budget justifies it"*,
and `references/07_tree_search.md` names the exact fallback that applies here: *"tiny data /
extremely expensive evaluation, where a ~60-node budget is unaffordable."*

A solo node in this competition is a 5-fold CNN fine-tune costing **48–89 min**. The prescribed
60-node budget is therefore **50–90 GPU-hours** against a ~8 h budget — two orders of magnitude
over. Spending the budget instead on one LR sweep plus five full fine-tunes was the right trade:
the single largest gain in the run (+0.037 AUC, metadata 0.892 → vit_small 0.926) came from
training a strong backbone at all, which no amount of node-space search over cheap models reaches.

Tree search restricted to the *blend layer* (nodes = member subsets and weights, seconds each) was
considered and rejected as redundant: with 10 members the greedy-with-replacement search already
covers that space, and the member-set comparison in §4 shows the blend is saturated — three
structurally different member sets land within 0.00001 of each other.

## 6. Reproducibility

Seeds fixed (`SEED=42`; vit_small_320 deliberately uses 2024 for decorrelation), folds persisted
to `cache/r2_folds.csv` and shared by every model so all scores are digit-for-digit comparable.
`torch.manual_seed` / `cuda.manual_seed_all` / `np.random.seed` are set per fold; cuDNN benchmark
is on (autotuning is not bit-deterministic across runs — a deliberate speed trade, and the
consequence is that CNN OOFs reproduce to ~1e-3 AUC, not bitwise).

Order: `prep_images.py 256` → `prep_images.py 384` → `prep_images.py 320` → `eda.py` →
`train_meta.py` → `discover.py` → `finetune.py <backbone> <lr> <px> <ep> <bs> <tag>` ×5 →
`stack.py` → `blend.py` → `submit.py blend`.

## 7. Not submitted to Kaggle

This is an offline mle-bench lane graded against the held-out private split; no Kaggle upload was
made and none was authorised. `submission.csv` is the deliverable.
