# LLM Classification Finetuning Competition

**Competition**: [LLM Classification Finetuning](https://www.kaggle.com/competitions/llm-classification-finetuning)
**Type**: Code Competition (Notebook Submission)
**Task**: Multi-class Classification (3 classes)
**Metric**: Log Loss (lower is better)
**Status**: ⚠️ Active - Submitted v1, gap between val/LB needs investigation

---

## Competition Overview

### Problem Statement
Predict which of two LLM responses is better for a given prompt, or if they are equally good (tie).

**Classes**:
- `winner_model_a` - Response A is better
- `winner_model_b` - Response B is better
- `winner_tie` - Both responses are equally good

**Input Features**:
- `prompt` - Multi-turn conversation (JSON array)
- `response_a` - First model's response (JSON array)
- `response_b` - Second model's response (JSON array)
- `model_a` - Name of first model (e.g., "gpt-4-1106-preview")
- `model_b` - Name of second model (e.g., "claude-3-opus")

**Output**: Probabilities for each class (must sum to 1.0)

### Dataset
- **Train**: 57,477 samples
- **Test**: 3 samples (extremely small!)
- **Label Distribution** (approx):
  - Model A wins: ~33%
  - Model B wins: ~33%
  - Tie: ~33%
- **Models**: Multiple LLM variants (GPT-4, Claude, Gemini, etc.)
- **Prompt Structure**: JSON arrays representing multi-turn conversations
- **Text Length**: Varies widely (avg ~1000-2000 chars per response)

---

## Approaches Tried

### Experiment 1: Initial Transformer (DeBERTa)
**Date**: 2026-02-13
**Script**: `scripts/train_transformer.py`

**Approach**:
- Model: DeBERTa-v3-base
- Max length: 512 tokens
- Batch size: 8
- Epochs: 3
- Learning rate: 2e-5

**Issues**:
- Training instability with NaN losses
- Gradient explosion issues
- Did not complete successfully

**Result**: ❌ Failed

---

### Experiment 2: Simple Baseline
**Date**: 2026-02-13
**Script**: `scripts/train_simple.py`

**Approach**:
- Simpler transformer approach
- Reduced complexity for debugging

**Result**: ❌ Issues with stability, moved to DistilBERT

---

### Experiment 3: DistilBERT (Current Best)
**Date**: 2026-02-13
**Script**: `scripts/train_distilbert.py`
**Notebook**: `kaggle_notebook_distilbert.py` / `distilbert_submission.ipynb`

**Configuration**:
```python
{
    'model_name': 'distilbert-base-uncased',
    'max_length': 384,
    'batch_size': 16,
    'epochs': 3,
    'learning_rate': 3e-5,
    'seed': 42,
    'num_classes': 3,
}
```

**Architecture**:
- Base: DistilBERT-base-uncased (66M parameters)
- Input: `prompt [SEP] Response A: response_a [SEP] Response B: response_b`
- Pooling: CLS token ([CLS] position)
- Dropout: 0.1
- Classifier: Linear layer (768 → 3)
- Loss: CrossEntropyLoss

**Training Details**:
- Train/Val split: 80/20 stratified
- Train samples: 45,982
- Val samples: 11,495
- Optimizer: AdamW
- Scheduler: Linear warmup
- Gradient clipping: max_norm=1.0
- Device: CUDA GPU

**Results**:
- **Validation Log Loss**: 1.0548 (best epoch)
- **Public Leaderboard**: 1.11084
- **Gap**: 0.056 (5.3% higher on LB)

**Training Progress**:
- Epoch 1: Train Loss 1.0827, Val Loss 1.0548 ✓
- Epoch 2: Train Loss 1.0369, Val Loss 1.0589
- Epoch 3: Train Loss 0.9396, Val Loss 1.1165 (overfitting)

**Submission**:
- Submitted to Kaggle: 2026-02-13 15:04 UTC
- Status: Complete
- Public Score: **1.11084**

**Test Predictions**:
```
        id  winner_model_a  winner_model_b  winner_tie
0   136060        0.118      0.229          0.653
1   211333        0.571      0.298          0.131
2  1233961        0.379      0.383          0.237
```

**Result**: ✅ Working, but val/LB gap needs investigation

---

## Key Findings

### What Worked
1. **DistilBERT stability**: Much more stable than DeBERTa for this task
2. **Text preprocessing**: Parsing JSON arrays and joining with `[SEP]` tokens
3. **Gradient clipping**: Essential to prevent NaN losses
4. **Reduced max_length (384)**: Faster training without hurting performance
5. **Stratified split**: Maintains class balance in train/val

### What Didn't Work
1. **DeBERTa-v3**: Training instability, NaN losses
2. **Longer sequences (512)**: Slower with minimal benefit
3. **Epoch 3**: Clear overfitting (train 0.94 vs val 1.12)

### Challenges
1. **Tiny test set (3 samples)**: Public LB is very unstable
2. **Val/LB gap (1.055 → 1.111)**: Suggests:
   - Possible distribution shift in test set
   - Overfitting to validation set
   - Random variance due to tiny test set size
3. **Long text inputs**: Truncation at 384 tokens may lose information
4. **Model name information**: Currently not used as features

### Observations
- **Best epoch**: Epoch 1 (val loss 1.0548), not final epoch
- **Overfitting signal**: Train loss drops to 0.94 but val increases to 1.12
- **Probability calibration**: Test predictions look reasonable (sum to ~1.0)
- **Class imbalance**: Relatively balanced, no special handling needed

---

## Next Steps / Ideas for Improvement

### High Priority
1. **Investigate val/LB gap**:
   - Analyze test predictions in detail
   - Check if test distribution differs from train
   - With only 3 test samples, gap might be noise

2. **Try different models**:
   - RoBERTa-base (more robust than BERT variants)
   - DeBERTa-v3-small (if stability can be fixed)
   - Larger variants (bert-large, roberta-large) if compute allows

3. **Add model name features**:
   - Embed model_a and model_b names
   - Some models may consistently win/lose
   - Could be valuable signal

### Medium Priority
4. **Longer context**:
   - Try max_length=512 or 768
   - Some conversations may be getting truncated

5. **Better early stopping**:
   - Stop at epoch 1 or 2 (before overfitting)
   - Use patience-based early stopping

6. **Ensemble**:
   - Train multiple seeds
   - Average predictions from different checkpoints
   - Combine DistilBERT + RoBERTa

### Low Priority
7. **Different input formats**:
   - Try different separator tokens
   - Experiment with token type IDs
   - Add position markers for response A/B

8. **Regularization**:
   - Increase dropout (0.1 → 0.2)
   - Add weight decay
   - Use label smoothing

---

## Files

### Scripts
- `scripts/eda.py` - Exploratory data analysis
- `scripts/train_transformer.py` - Initial DeBERTa attempt (failed)
- `scripts/train_simple.py` - Simple baseline attempt
- `scripts/train_distilbert.py` - ✅ Working DistilBERT trainer
- `scripts/test_inference.py` - Test inference logic
- `scripts/test_notebook_locally.py` - Local notebook validation
- `scripts/monitor_training.sh` - Training monitor script

### Notebooks
- `kaggle_notebook_distilbert.py` - Kaggle Code Competition submission (Python)
- `distilbert_submission.ipynb` - Jupyter notebook version
- `distilbert_submission_test.ipynb` - Test notebook

### Submissions
- `submissions/submission_distilbert.csv` - ✅ Main submission (LB: 1.11084)
- `submissions/submission_lightgbm.csv` - Earlier baseline attempt
- `submissions/test_notebook_submission.csv` - Local test run

### Logs
- `training_log.txt` - Initial transformer training log
- `training_log_simple.txt` - Simple baseline log
- `training_log_distilbert.txt` - ✅ DistilBERT training log
- `test_notebook_log.txt` - Local notebook test log

### Config
- `config.yaml` - Competition configuration
- `experiments.json` - Experiment tracking (empty - need to populate)

---

## Technical Notes

### Code Competition Requirements
- Must be a **Kaggle Notebook** (not a script submission)
- Notebook must output `submission.csv` (exact name required)
- Limited compute: GPU T4 x 2 (9-hour limit)
- Internet access: Enabled (can download models from HuggingFace)

### Model Loading
- HuggingFace transformers automatically download models
- DistilBERT (~250MB) downloads quickly in Kaggle
- Larger models may hit internet quota limits

### GPU Training
- Local GPU: NVIDIA GB10 (confirmed working)
- Kaggle GPU: T4 x 2 (available)
- Training time: ~30 min per epoch on local GPU

### Key Dependencies
- PyTorch 2.11.0 (CUDA 13.0)
- Transformers 5.1.0
- scikit-learn
- pandas, numpy

---

## Conclusion

**Current Best**: DistilBERT with validation log loss **1.0548**, public LB **1.11084**

**Status**: Working solution submitted, but val/LB gap suggests room for improvement.

**Recommendation**:
1. Try RoBERTa-base (more robust architecture)
2. Add model name embeddings as features
3. Use early stopping at epoch 1-2
4. Consider ensemble of multiple seeds

**Note**: With only 3 test samples, public LB score is highly unstable. Focus on improving validation score and ensuring robust generalization rather than chasing LB improvements.
