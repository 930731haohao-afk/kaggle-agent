"""
LLM Classification Fine-tuning with DeBERTa-v3
Uses GPU for training
"""
import pandas as pd
import numpy as np
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import log_loss
import warnings
warnings.filterwarnings('ignore')

# Check GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Configuration
CONFIG = {
    'model_name': 'microsoft/deberta-v3-base',  # Can also try 'roberta-base' or 'bert-base-uncased'
    'max_length': 512,  # Max tokens (responses can be long, but we'll truncate)
    'batch_size': 32,  # Balanced for GPU memory and speed
    'epochs': 2,
    'learning_rate': 1e-5,  # Reduced for stability
    'n_folds': 1,  # Single fold - no CV for speed
    'seed': 42,
    'num_classes': 3,
}

np.random.seed(CONFIG['seed'])
torch.manual_seed(CONFIG['seed'])

# Load data
print("Loading data...")
train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')

# Parse JSON prompts
def parse_prompt(prompt_str):
    try:
        prompt_list = json.loads(prompt_str)
        return " [SEP] ".join(prompt_list)
    except:
        return str(prompt_str)

def parse_response(response_str):
    try:
        response_list = json.loads(response_str)
        return " [SEP] ".join(response_list)
    except:
        return str(response_str)

print("Parsing JSON fields...")
train['prompt_text'] = train['prompt'].apply(parse_prompt)
train['response_a_text'] = train['response_a'].apply(parse_response)
train['response_b_text'] = train['response_b'].apply(parse_response)

test['prompt_text'] = test['prompt'].apply(parse_prompt)
test['response_a_text'] = test['response_a'].apply(parse_response)
test['response_b_text'] = test['response_b'].apply(parse_response)

# Create labels (convert to single column)
train['label'] = train.apply(lambda x: 0 if x['winner_model_a'] == 1 else (1 if x['winner_model_b'] == 1 else 2), axis=1)

print(f"Label distribution: {train['label'].value_counts().sort_index().to_dict()}")

# Dataset class
class LLMDataset(Dataset):
    def __init__(self, df, tokenizer, max_length, is_test=False):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Create input text: [CLS] prompt [SEP] response_a [SEP] response_b [SEP]
        text = f"{row['prompt_text']} [SEP] Response A: {row['response_a_text']} [SEP] Response B: {row['response_b_text']}"

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        item = {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
        }

        if not self.is_test:
            item['label'] = torch.tensor(row['label'], dtype=torch.long)

        return item

# Model class
class LLMClassifier(nn.Module):
    def __init__(self, model_name, num_classes):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.backbone.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]  # CLS token
        pooled = pooled.float()  # Convert to float32 to match classifier
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits

# Training function
def train_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0

    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = nn.CrossEntropyLoss()(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

# Validation function
def validate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            logits = model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1)

            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_labels = np.concatenate(all_labels)

    # Calculate log loss
    loss = log_loss(all_labels, all_preds)
    return loss, all_preds

# Initialize tokenizer
print(f"Loading tokenizer: {CONFIG['model_name']}")
tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])

# Cross-validation or single split
if CONFIG['n_folds'] == 1:
    print(f"\nUsing single train/validation split (80/20)...")
    train_idx, val_idx = train_test_split(
        range(len(train)),
        test_size=0.2,
        random_state=CONFIG['seed'],
        stratify=train['label']
    )
    splits = [(train_idx, val_idx)]
else:
    print(f"\nStarting {CONFIG['n_folds']}-fold cross-validation...")
    kfold = StratifiedKFold(n_splits=CONFIG['n_folds'], shuffle=True, random_state=CONFIG['seed'])
    splits = list(kfold.split(train, train['label']))

oof_preds = np.zeros((len(train), CONFIG['num_classes']))
test_preds = np.zeros((len(test), CONFIG['num_classes']))

for fold, (train_idx, val_idx) in enumerate(splits):
    print(f"\n{'='*80}")
    if CONFIG['n_folds'] == 1:
        print(f"Training (80% train, 20% validation)")
    else:
        print(f"Fold {fold + 1}/{CONFIG['n_folds']}")
    print(f"{'='*80}")

    train_fold = train.iloc[train_idx]
    val_fold = train.iloc[val_idx]

    # Create datasets
    train_dataset = LLMDataset(train_fold, tokenizer, CONFIG['max_length'], is_test=False)
    val_dataset = LLMDataset(val_fold, tokenizer, CONFIG['max_length'], is_test=False)

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=2)

    # Initialize model
    model = LLMClassifier(CONFIG['model_name'], CONFIG['num_classes']).to(device)

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'])
    total_steps = len(train_loader) * CONFIG['epochs']
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # Training loop
    best_val_loss = float('inf')

    for epoch in range(CONFIG['epochs']):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        val_loss, val_preds = validate(model, val_loader, device)

        print(f"Epoch {epoch+1}/{CONFIG['epochs']} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save OOF predictions
            oof_preds[val_idx] = val_preds

            # Generate test predictions
            test_dataset = LLMDataset(test, tokenizer, CONFIG['max_length'], is_test=True)
            test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=2)

            model.eval()
            fold_test_preds = []
            with torch.no_grad():
                for batch in test_loader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    logits = model(input_ids, attention_mask)
                    probs = torch.softmax(logits, dim=1)
                    fold_test_preds.append(probs.cpu().numpy())

            fold_test_preds = np.vstack(fold_test_preds)
            if CONFIG['n_folds'] == 1:
                test_preds = fold_test_preds
            else:
                test_preds += fold_test_preds / CONFIG['n_folds']

    print(f"Best Val Loss: {best_val_loss:.4f}")

# Calculate overall OOF score
oof_score = log_loss(train['label'], oof_preds)
print(f"\n{'='*80}")
print(f"Overall OOF Log Loss: {oof_score:.5f}")
print(f"{'='*80}")

# Save predictions
print("\nSaving OOF predictions...")
oof_df = train[['id']].copy()
oof_df['winner_model_a'] = oof_preds[:, 0]
oof_df['winner_model_b'] = oof_preds[:, 1]
oof_df['winner_tie'] = oof_preds[:, 2]
oof_df.to_csv('../submissions/oof_transformer.csv', index=False)

print("Creating submission...")
submission = pd.DataFrame({
    'id': test['id'],
    'winner_model_a': test_preds[:, 0],
    'winner_model_b': test_preds[:, 1],
    'winner_tie': test_preds[:, 2]
})
submission.to_csv('../submissions/submission_transformer.csv', index=False)

print("\nSubmission preview:")
print(submission)
print("\nSubmission probabilities sum:")
print(submission[['winner_model_a', 'winner_model_b', 'winner_tie']].sum(axis=1))

# Log experiment
experiment = {
    'model': 'DeBERTa-v3-base Transformer',
    'config': CONFIG,
    'oof_logloss': float(oof_score),
    'n_folds': CONFIG['n_folds'],
    'features': 'prompt + response_a + response_b (concatenated)',
    'notes': 'Fine-tuned transformer with GPU, 3-class classification'
}

print(f"\nExperiment: {experiment}")
print("\nTraining completed successfully!")
