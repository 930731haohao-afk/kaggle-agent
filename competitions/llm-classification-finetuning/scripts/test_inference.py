"""
Test inference to check for numerical issues
"""
import pandas as pd
import numpy as np
import json
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

print("Testing inference for numerical stability...")

# Check GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Load a small sample
train = pd.read_csv('../data/train.csv').head(10)
print(f"Loaded {len(train)} samples for testing")

# Parse JSON
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

train['prompt_text'] = train['prompt'].apply(parse_prompt)
train['response_a_text'] = train['response_a'].apply(parse_response)
train['response_b_text'] = train['response_b'].apply(parse_response)

# Model
class LLMClassifier(nn.Module):
    def __init__(self, model_name, num_classes):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.backbone.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        pooled = pooled.float()
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits

# Initialize
print("\nLoading model...")
model_name = 'microsoft/deberta-v3-base'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = LLMClassifier(model_name, num_classes=3).to(device)
model.eval()

print("\nRunning inference on test samples...")
for i in range(len(train)):
    row = train.iloc[i]
    text = f"{row['prompt_text']} [SEP] Response A: {row['response_a_text']} [SEP] Response B: {row['response_b_text']}"

    encoding = tokenizer(
        text,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probs = torch.softmax(logits, dim=1)

    logits_cpu = logits.cpu().numpy()[0]
    probs_cpu = probs.cpu().numpy()[0]

    print(f"Sample {i+1}:")
    print(f"  Logits: {logits_cpu}")
    print(f"  Probs: {probs_cpu}")
    print(f"  Sum: {probs_cpu.sum():.6f}")
    print(f"  Has NaN: {np.isnan(logits_cpu).any() or np.isnan(probs_cpu).any()}")
    print()

print("✓ Inference test completed!")
print("\nIf you see NaN values above, there's a numerical issue.")
print("If all values are normal, the model architecture is fine.")
