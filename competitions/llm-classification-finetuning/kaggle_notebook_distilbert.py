"""
LLM Classification Fine-tuning with DistilBERT - Kaggle Notebook Version
Code Competition Submission
"""
import pandas as pd
import numpy as np
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertModel, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
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
    'model_name': 'distilbert-base-uncased',
    'max_length': 384,
    'batch_size': 16,
    'epochs': 3,
    'learning_rate': 3e-5,
    'seed': 42,
    'num_classes': 3,
}

np.random.seed(CONFIG['seed'])
torch.manual_seed(CONFIG['seed'])

# Load data - Kaggle paths
print("Loading data...")
train = pd.read_csv('/kaggle/input/llm-classification-finetuning/train.csv')
test = pd.read_csv('/kaggle/input/llm-classification-finetuning/test.csv')

print(f"Train: {len(train)} samples")
print(f"Test: {len(test)} samples")

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

# Create labels
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

        # Create input text
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
        self.backbone = DistilBertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.backbone.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits

# Training function
def train_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0

    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)

        # Check for NaN in logits
        if torch.isnan(logits).any():
            print(f"WARNING: NaN in logits at batch {batch_idx}")
            continue

        loss = nn.CrossEntropyLoss()(logits, labels)

        if torch.isnan(loss):
            print(f"WARNING: NaN in loss at batch {batch_idx}")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        # Progress indicator
        if (batch_idx + 1) % 100 == 0:
            print(f"  Batch {batch_idx + 1}/{len(dataloader)}, Loss: {loss.item():.4f}")

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

    # Check for NaN in predictions
    if np.isnan(all_preds).any():
        print("ERROR: NaN detected in validation predictions!")
        all_preds = np.nan_to_num(all_preds, nan=0.33)

    loss = log_loss(all_labels, all_preds)
    return loss, all_preds

# Initialize tokenizer
print(f"Loading tokenizer: {CONFIG['model_name']}")
tokenizer = DistilBertTokenizer.from_pretrained(CONFIG['model_name'])

# Split data
print(f"\nUsing single train/validation split (80/20)...")
train_idx, val_idx = train_test_split(
    range(len(train)),
    test_size=0.2,
    random_state=CONFIG['seed'],
    stratify=train['label']
)

train_fold = train.iloc[train_idx]
val_fold = train.iloc[val_idx]

print(f"Train: {len(train_fold)}, Val: {len(val_fold)}")

# Create datasets
train_dataset = LLMDataset(train_fold, tokenizer, CONFIG['max_length'], is_test=False)
val_dataset = LLMDataset(val_fold, tokenizer, CONFIG['max_length'], is_test=False)

train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=2)

# Initialize model
print("\n" + "="*80)
print("Initializing DistilBERT Model")
print("="*80)
model = LLMClassifier(CONFIG['model_name'], CONFIG['num_classes']).to(device)

# Optimizer and scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'])
total_steps = len(train_loader) * CONFIG['epochs']
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Training loop
best_val_loss = float('inf')
best_model_state = None

print("\n" + "="*80)
print(f"Training for {CONFIG['epochs']} epochs")
print("="*80)

for epoch in range(CONFIG['epochs']):
    print(f"\nEpoch {epoch+1}/{CONFIG['epochs']}")
    print("-" * 80)

    train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
    val_loss, val_preds = validate(model, val_loader, device)

    print(f"Epoch {epoch+1}/{CONFIG['epochs']} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = model.state_dict().copy()
        print(f"  ✓ New best model! Val Loss: {val_loss:.4f}")

# Load best model for test predictions
model.load_state_dict(best_model_state)

# Generate test predictions
print("\n" + "="*80)
print("Generating Test Predictions")
print("="*80)

test_dataset = LLMDataset(test, tokenizer, CONFIG['max_length'], is_test=True)
test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=2)

model.eval()
test_preds = []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        logits = model(input_ids, attention_mask)
        probs = torch.softmax(logits, dim=1)
        test_preds.append(probs.cpu().numpy())

test_preds = np.vstack(test_preds)

# Create submission - MUST be named submission.csv for Code Competition
submission = pd.DataFrame({
    'id': test['id'],
    'winner_model_a': test_preds[:, 0],
    'winner_model_b': test_preds[:, 1],
    'winner_tie': test_preds[:, 2]
})

# Save as submission.csv (required name for Code Competition)
submission.to_csv('submission.csv', index=False)

print("\n" + "="*80)
print("Training Completed!")
print("="*80)
print(f"Best Validation Log Loss: {best_val_loss:.5f}")
print(f"Submission saved to: submission.csv")
print("\nSubmission preview:")
print(submission)
print(f"\nProbability sums: {submission[['winner_model_a', 'winner_model_b', 'winner_tie']].sum(axis=1).values}")
