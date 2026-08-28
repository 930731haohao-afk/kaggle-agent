# ML Specification Report — siim-isic-melanoma-classification

### Melanoma Classification from Dermoscopy Images + Patient Metadata · our from-scratch agent (ViT/CNN fine-tunes + metadata GBDT, convex rank blend — deep-learning fallback lane)

> *Figures grounded in the competition's `facts.json` (built by `benchmark_infra/collect_dl_facts.py` from `config.yaml`, `experiments.json`, `eda_summary.json`, `discovery.json`, `res_*.json`, `meta_results.json`, `blend_results.json`, `dossier.json`, `STATUS.md`, `headless_run2.log`, `scripts/vision.py`), with grading figures from the offline MLE-bench records in `benchmark_results/run2/`. This is a **special (deep-learning fallback) lane**: grading is **offline MLE-bench** against a held-out private split — there is no live Kaggle leaderboard, and the framework's DL fields (backbone, learning-rate schedule, batch size, epochs, augmentation, transfer learning) are native here rather than translated from GBDT.*

## Overview

The task is to predict the probability that a skin-lesion image is malignant melanoma, from the dermoscopy image plus patient-level metadata (age, sex, anatomical site); the metric is **ROC-AUC (maximize)** on a heavily imbalanced label (~1.8% positive). Image classification is outside the GBDT tree-search pipeline, so the agent ran the vision fallback: cheap metadata/image-statistic models, a discovery LR sweep, five full backbone fine-tunes, and a convex rank blend. The champion is a **6-member convex rank blend — vit_small at 384/256/320 px + efficientnet_b0 + CatBoost/LightGBM on metadata & image statistics** — scoring **honest leave-fold-out CV AUC 0.93214** (optimistic full-OOF 0.93291; patient-grouped cross-check 0.93278; best solo vit_small @384 px 0.92596) and grading **0.93529** on the offline MLE-bench private split — **above median (0.9128) but short of bronze (0.937): no medal**. The NVIDIA reproduce-agent grades 0.89822 and AIDE 0.88579 — **our agent wins the three-way comparison**.

**Why it matters.** Melanoma causes the large majority of skin-cancer deaths, yet is highly curable when caught early; dermatologist-grade triage from a dermoscopy photo plus context (the clinical "ugly duckling" sign — a lesion suspicious relative to the patient's *other* moles) is exactly what this model family automates.

---

## Data

**Purpose of Data.** Each row is one lesion image (`image_name`) with patient metadata; predict `target`, the malignancy probability — binary classification scored as ranking. **Data Format** is **JPEG/DICOM images + a tabular CSV** of metadata. **Data Volume**: **28,984 training images / 4,142 test images** (8 vs 5 metadata columns; `benign_malignant` and `diagnosis` are train-only), re-split by mle-bench from the original 33,126-image train set. **513 positives** in train — a positive rate of 0.0177.

**Data Quality.** Metadata missingness is small: fractions 0.0019 (`sex`), 0.0020 (`age_approx`), 0.0161 (`anatom_site_general_challenge`). Images span 81 distinct resolutions in train / 28 in test, with a 0.0017 fraction of test rows at an unseen resolution. Patients: 2,056 in train / 1,457 in test, with images-per-patient mean 14.1 (max 104). **The defining wrinkle is the split's structure**: the run's EDA *refuted* the dossier's patient-disjoint hypothesis — **all 1,457 test patients also appear in train** (overlap 1457/1457; 599 patients are train-only), with per-patient test share mean 0.124 / median 0.111 vs an overall test share of 0.125. mle-bench re-split the images **randomly by image**, so the graded test set is iid-by-image, not patient-disjoint like the real SIIM competition.

Strongest recorded EDA/modelling signals:

- **Cheap image colour/texture statistics are the largest tabular lever**: removing them drops the LightGBM metadata model 0.88574 → 0.82114 (−0.06460 AUC); removing resolution features −0.01417; removing patient-relative features −0.01330.
- **The ugly-duckling sign appears directly in the model**: patient-relative `ud_*` z-scores occupy 8 of the top-20 LightGBM importances.
- **Class imbalance needs no reweighting** — AUC is a pure ranking metric, so all members train without class weighting.

**Annotation Guidelines.** The label is `target` ∈ {0, 1} (malignant or not); a submission is a real-valued malignancy score per image, scored by ROC-AUC — raw scores, no threshold or calibration.

**Feature Set.** The vision members consume raw pixels; the metadata members consume **54 features**: sex/age/site, image resolution, prep-time colour/texture statistics (central-disc "lesion" vs outer-ring "skin" colour, contrast, gradient energy, vignetting), and patient-relative ugly-duckling z-scores.

**Splitting strategy.** Primary CV is **image-level StratifiedKFold(5, seed=42)** — deliberately, because the graded split is iid-by-image; a patient-grouped `StratifiedGroupKFold` would measure a strictly harder problem than the one being scored. Every member additionally reports a **patient-grouped score as a leakage detector**: the two agreed everywhere, largest gap 0.0004 AUC across all 10 members — no member exploits patient identity. Folds are persisted to `cache/r2_folds.csv` and shared by every model, so all scores are digit-for-digit comparable. **LB status**: offline MLE-bench grade only — no live public/private leaderboard.

## Models & Architecture

**Purpose of Architecture.** Maximize ROC-AUC for malignancy ranking. **Architecture Type** is a **heterogeneous ensemble of fine-tuned ImageNet-pretrained backbones plus metadata GBDTs, combined by a convex rank blend** — the DL fields are native. The champion carries 6 non-zero members:

| Member | Backbone / model | px | Seed | Solo CV AUC | Blend weight |
|--------|------------------|---:|-----:|------------:|-------------:|
| vit_small_384 | `vit_small_patch16_224` | 384 | 42 | 0.92596 | 0.388 |
| vit_small_256 | `vit_small_patch16_224` | 256 | 42 | 0.92309 | 0.286 |
| vit_small_320 | `vit_small_patch16_224` | 320 | 2024 | 0.92024 | 0.102 |
| lgb_meta | LightGBM, 54 metadata features | — | 42 | 0.88574 | 0.102 |
| cat_meta | CatBoost, 54 metadata features | — | 42 | 0.89171 | 0.082 |
| effb0_256 | `efficientnet_b0` | 256 | 42 | 0.88447 | 0.041 |

Zero-weighted candidates: `resnet50_256` (solo 0.89166), `xgb_meta` (0.88668), `lr_allfeat` (0.85467), and a level-2 LightGBM stack (0.92566).

**Input Format / Dimension.** Vision members take square RGB crops at 384 / 256 / 320 px (uint8 arrays preloaded to GPU; resize/augment/normalize on-GPU in batch); metadata members take a 54-dimensional feature vector.

**Architecture Description.** Each vision member is a `timm` backbone with `pretrained=True`, a 1-logit head, and drop-path 0.1 (`scripts/vision.py`). The blend layer **rank-transforms each member's OOF predictions, then runs Caruana greedy selection with replacement**; the reported champion score is the **honest leave-fold-out** estimate (weights refit on 4 folds, scored on the held-out fold). **Model Complexity**: parameter counts were **not recorded**; the members are fully identified by their published backbone names. A recorded structural finding: **input-resolution diversity within the ViT family beat architecture diversity** — the three vit_small scales take 0.78 of the total weight while both extra CNN families end at or near zero.

## Training procedures

Training was a **staged discovery-first ladder** (tree search deliberately skipped — see below):

| Stage | Configuration | AUC |
|-------|---------------|----:|
| 1 | metadata + image statistics, best tabular (CatBoost, 54 features) | 0.89171 |
| 2 | discovery sweep: 18 backbone×LR configs, fold 0, 2 epochs, 256 px, 52 min | best probe 0.8853 |
| 3 | five full 5-fold fine-tunes, best solo (vit_small @384) | 0.92596 |
| 4 | convex rank blend, honest leave-fold-out | **0.93214** |

The discovery sweep is the native learning-rate selection step (per-backbone LR mini-sweep, mandated by the vision experience library):

| Backbone | 1e-4 | 3e-4 | 1e-3 | best LR |
|----------|-----:|-----:|-----:|---------|
| vit_small_patch16_224 | **0.8853** | 0.8686 | 0.8173 | 1e-4 |
| efficientnet_b0 | 0.7686 | 0.8231 | **0.8782** | 1e-3 |
| efficientnet_b3 | 0.7868 | 0.8489 | **0.8716** | 1e-3 |
| resnet50 | 0.7608 | 0.8393 | **0.8694** | 1e-3 |
| resnet18 | 0.7936 | **0.8672** | 0.8551 | 3e-4 |
| convnext_tiny | **0.8643** | 0.8468 | 0.7775 | 1e-4 |

CNNs peak at 1e-3, ViT/ConvNeXt at 1e-4 — the fifth domain confirming this pattern. The full fine-tunes:

| Tag | px | Epochs | Batch | LR | CV AUC | patient-grouped | Wall |
|-----|---:|-------:|------:|----|-------:|----------------:|-----:|
| vit_small_384 | 384 | 8 | 64 | 1e-4 | **0.92596** | 0.92589 | 5345.5 s |
| vit_small_256 | 256 | 10 | 96 | 1e-4 | 0.92309 | 0.92295 | 2942.0 s |
| vit_small_320 | 320 | 9 | 80 | 1e-4 | 0.92024 | 0.92007 | 4133.1 s |
| resnet50_256 | 256 | 10 | 96 | 1e-3 | 0.89166 | 0.89179 | 4218.9 s |
| effb0_256 | 256 | 10 | 96 | 1e-3 | 0.88447 | 0.88440 | 2883.2 s |

The **Loss Function** is **binary cross-entropy with logits** (`scripts/vision.py`), with model selection on ROC-AUC and **no class weighting** (a ranking metric gains nothing from resampling a 0.0177 positive rate). The **Optimization Algorithm** is **AdamW** (weight decay 1e-4). The **Learning Rate** (native) is backbone-specific per the discovery sweep — 1e-4 for ViT members, 1e-3 for the CNNs — under a native **Learning Rate Scheduler: linear warmup over the first 10% of steps, then cosine decay** (per-step, `scripts/vision.py`). The **Batch Size** (native) is 64–96 depending on member (table above), for **8–10 epochs**, under **bf16 autocast**.

**Data Augmentation** (native DL field): **dihedral-group augmentation** ("medium" setting — random h-flip, v-flip, and transpose, all label-preserving since dermoscopy is orientation-free), plus **4× flip TTA** at prediction time. One honestly recorded gap: dihedral augmentation was adopted but never ablated against no-aug, so its contribution is unmeasured. **Transfer Learning** (native DL field): **ImageNet-pretrained `timm` backbones** for every vision member — allowed by the rules (`pretrained_models_allowed: true`) and named by the dossier as the run's main source of injected prior knowledge.

**Training Duration**: **~6.5 h** for the whole run, all in the foreground — 52 min discovery plus fine-tune walls of 48–89 min each (per-member seconds in the table). **Training Memory**: **not recorded**.

**Reproducibility Standards.** `SEED=42` throughout (vit_small_320 deliberately uses 2024 for decorrelation); folds persisted to `cache/r2_folds.csv` and shared by every member; `torch.manual_seed` / `cuda.manual_seed_all` / `np.random.seed` set per fold. `cudnn.benchmark` is on — a deliberate speed trade whose recorded consequence is that CNN OOFs reproduce to ~1e-3 AUC, not bitwise. **Tree search was deliberately skipped**: a solo node is a 5-fold fine-tune costing 48–89 min, so the prescribed 60-node budget is 50–90 GPU-hours against ~8 h. Blend-layer-only tree search was considered and rejected as redundant — the greedy-with-replacement search already covers that space, and three structurally different member sets land within 0.00001 of each other (0.93204 / 0.93204 / 0.93203), i.e. the blend is saturated.

Recorded negative results: the **level-2 LightGBM stack lost to the convex blend** (0.92566 vs 0.93214, same direction as the s3e14 Ridge-stacking counterexample); **extra CNN families bought nothing** (resnet50 zero-weighted despite a respectable 0.89166 solo); and the **discovery-scale gap widened rather than shrank** (vit_small led efficientnet_b0 by 0.007 at 2 epochs and 0.039 at 10 epochs), a direct counterexample to the digit-recognizer prior "don't over-prune on discovery scores".

## Inference procedures

**Decision Threshold** is *N/A* — ROC-AUC is a pure ranking metric; raw malignancy scores are submitted with no threshold or calibration. Inference-time post-processing: **4× flip TTA** per vision member, then **rank-transform and convex-weighted averaging** of the 6 members' test predictions. The submission has 4,142 rows with id order verified against `sample_submission.csv`. **Inference Duration / Inference Memory**: **not recorded**.

## Evaluation & Benchmarking

**Task for Performance Evaluation.** Rank test lesions by malignancy probability. **Performance Metrics.** ROC-AUC (maximize): metadata 0.89171 → best solo fine-tune 0.92596 → blend **0.93214** honest leave-fold-out (the single largest gain in the run, +0.037 AUC, came from training a strong backbone at all); offline MLE-bench grade **0.93529** against thresholds gold 0.9455 / silver 0.9401 / bronze 0.937 / median 0.9128 — **above median, no medal** (0.93529 falls short of bronze 0.937). The grade sits slightly above the honest CV (0.93214) and close to the optimistic full-OOF (0.93291), consistent with a well-calibrated validation.

**Performance Benchmarking.** Grading is **offline MLE-bench** (all three agents scored on the same held-out private split — not a live leaderboard):

| Agent | Approach | ROC-AUC (offline grade) | Medal |
|-------|----------|------------------------:|-------|
| Our agent | from-scratch ViT/CNN fine-tunes + metadata GBDT, convex rank blend | **0.93529** | none (above median) |
| NVIDIA | (approach not recorded in the grade record) | 0.89822 | none (below median) |
| AIDE | (approach not recorded in the grade record) | 0.88579 | none (below median) |

Our agent is the only one above the median threshold, with a comfortable margin over both comparators. The honest caveats: nobody medalled, the comparison is grade-only (the comparators' pipelines are not described in the run2 grade records), and our margin traces to the discovery-first backbone selection plus resolution-diverse ViT blending rather than any exotic component.

---

*Fields marked "not recorded": model parameter counts; training peak memory; inference duration and memory; the contribution of dihedral augmentation (adopted but never ablated); NVIDIA/AIDE approach details. Grading is offline MLE-bench — no public/private live leaderboard exists for this lane.*
