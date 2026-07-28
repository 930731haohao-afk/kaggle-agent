"""
EDA for LLM Classification Finetuning Competition
"""
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Load data
print("Loading data...")
train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')
sample_sub = pd.read_csv('../data/sample_submission.csv')

print("\n" + "="*80)
print("DATASET OVERVIEW")
print("="*80)
print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"Sample submission shape: {sample_sub.shape}")

print("\n" + "="*80)
print("TRAIN COLUMNS")
print("="*80)
print(train.columns.tolist())
print("\nData types:")
print(train.dtypes)

print("\n" + "="*80)
print("MISSING VALUES")
print("="*80)
print(train.isnull().sum())

print("\n" + "="*80)
print("TARGET DISTRIBUTION")
print("="*80)
print(f"winner_model_a: {train['winner_model_a'].sum()}")
print(f"winner_model_b: {train['winner_model_b'].sum()}")
print(f"winner_tie: {train['winner_tie'].sum()}")
print(f"\nPercentages:")
print(f"Model A wins: {train['winner_model_a'].sum() / len(train) * 100:.2f}%")
print(f"Model B wins: {train['winner_model_b'].sum() / len(train) * 100:.2f}%")
print(f"Tie: {train['winner_tie'].sum() / len(train) * 100:.2f}%")

print("\n" + "="*80)
print("MODEL DISTRIBUTION")
print("="*80)
print("\nTop 10 Model A:")
print(train['model_a'].value_counts().head(10))
print("\nTop 10 Model B:")
print(train['model_b'].value_counts().head(10))
print(f"\nUnique models in model_a: {train['model_a'].nunique()}")
print(f"Unique models in model_b: {train['model_b'].nunique()}")

# Get all unique models
all_models = set(train['model_a'].unique()) | set(train['model_b'].unique())
print(f"\nTotal unique models: {len(all_models)}")

print("\n" + "="*80)
print("TEXT LENGTH ANALYSIS")
print("="*80)

# Parse JSON prompts and calculate lengths
def get_prompt_length(prompt_str):
    try:
        prompt_list = json.loads(prompt_str)
        return sum(len(p) for p in prompt_list)
    except:
        return len(str(prompt_str))

train['prompt_length'] = train['prompt'].apply(get_prompt_length)
train['response_a_length'] = train['response_a'].apply(lambda x: len(str(x)))
train['response_b_length'] = train['response_b'].apply(lambda x: len(str(x)))

print("\nPrompt length statistics:")
print(train['prompt_length'].describe())
print("\nResponse A length statistics:")
print(train['response_a_length'].describe())
print("\nResponse B length statistics:")
print(train['response_b_length'].describe())

print("\n" + "="*80)
print("PROMPT TURNS ANALYSIS")
print("="*80)

def get_prompt_turns(prompt_str):
    try:
        prompt_list = json.loads(prompt_str)
        return len(prompt_list)
    except:
        return 1

train['prompt_turns'] = train['prompt'].apply(get_prompt_turns)
print("\nPrompt turns distribution:")
print(train['prompt_turns'].value_counts().sort_index())

print("\n" + "="*80)
print("SAMPLE EXAMPLES")
print("="*80)
for i in range(min(3, len(train))):
    print(f"\n--- Example {i+1} ---")
    print(f"ID: {train.iloc[i]['id']}")
    print(f"Model A: {train.iloc[i]['model_a']}")
    print(f"Model B: {train.iloc[i]['model_b']}")
    print(f"Prompt: {train.iloc[i]['prompt'][:200]}...")
    print(f"Response A: {train.iloc[i]['response_a'][:200]}...")
    print(f"Response B: {train.iloc[i]['response_b'][:200]}...")
    print(f"Winner: A={train.iloc[i]['winner_model_a']}, B={train.iloc[i]['winner_model_b']}, Tie={train.iloc[i]['winner_tie']}")

print("\n" + "="*80)
print("TEST SET ANALYSIS")
print("="*80)
print("\nTest columns:")
print(test.columns.tolist())
print("\nTest sample:")
print(test.head())

print("\n" + "="*80)
print("KEY INSIGHTS")
print("="*80)
print("""
1. This is a 3-class classification problem (model_a wins, model_b wins, or tie)
2. Output should be probabilities that sum to 1.0
3. Input features: prompt (conversation), response_a, response_b, model_a, model_b
4. Prompts are JSON arrays representing multi-turn conversations
5. Very small test set (only 3 examples)
6. Need to use transformers (BERT/RoBERTa/DeBERTa) for this NLP task
7. GPU training is essential due to large text inputs
""")

print("\nEDA completed!")
