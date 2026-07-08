# Stage 0: Competition Setup

## Objective
Establish a workspace for the competition and build an understanding of the problem before any analysis or modeling begins.

## Steps

### 1. Gather Competition Information
Ask the user:
- **Competition name** — Used as the workspace folder name (use lowercase, hyphens for spaces)
- **Data location** — Where the user has downloaded/placed the competition data files
- **Competition URL** (optional) — For reference in config

If the user provides a Kaggle URL, use `WebFetch` to read the competition overview page and extract:
- Problem description
- Evaluation metric
- Submission format requirements
- Timeline / deadlines
- Any special rules (external data, pretrained models, etc.)

### 2. Create Workspace
Create the following directory structure:
```
competitions/<competition-name>/
├── config.yaml
├── data/          (symlink or copy data here)
├── scripts/
├── submissions/
└── experiments.json  (initialize as empty array: [])
```

### 3. Create config.yaml
Populate with competition metadata:
```yaml
name: <competition-name>
url: <competition-url>
description: <one-line summary>
problem_type: <regression | binary_classification | multiclass_classification | multilabel | ranking>
evaluation_metric: <e.g., rmse, auc, f1, log_loss, map@k>
optimization_direction: <minimize | maximize>
target_column: <name of target column>
id_column: <name of ID column for submission>
train_file: <filename>
test_file: <filename>
sample_submission_file: <filename, if available>
special_rules:
  external_data_allowed: <true/false>
  pretrained_models_allowed: <true/false>
  internet_access_allowed: <true/false>
  daily_submission_limit: <number>
```

If any fields are unknown, ask the user or mark as `TBD`.

### 4. Initial Data Inspection
Run a quick Python script to report:
- **File inventory**: List all files in the data directory with sizes
- **Train set**: Shape, column names, dtypes, first 5 rows
- **Test set**: Shape, column names, dtypes, first 5 rows
- **Target variable**: dtype, unique values count, distribution summary
- **Missing values**: Count and percentage per column
- **Sample submission**: Expected format (columns, dtypes, shape)

Use this template structure:
```python
import pandas as pd
import os

data_dir = "competitions/<name>/data"

# List all files
for f in os.listdir(data_dir):
    size = os.path.getsize(os.path.join(data_dir, f))
    print(f"{f}: {size / 1024:.1f} KB")

# Load and inspect
train = pd.read_csv(os.path.join(data_dir, "<train_file>"))
test = pd.read_csv(os.path.join(data_dir, "<test_file>"))

print(f"\nTrain shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"\nTrain dtypes:\n{train.dtypes}")
print(f"\nTrain head:\n{train.head()}")
print(f"\nMissing values:\n{train.isnull().sum()}")
print(f"\nTarget distribution:\n{train['<target>'].describe()}")
```

### 5. Summarize Findings
Present a brief summary to the user:
- What the competition is about
- Key characteristics of the data (size, feature types, target distribution)
- Any immediate concerns (heavy class imbalance, lots of missing data, high cardinality categoricals)
- Recommended next step (proceed to EDA)

## Completion Criteria
- Workspace directory exists with config.yaml populated
- Data files are accessible in the workspace
- Initial data inspection has been run and summarized
- experiments.json is initialized as `[]`
- User has confirmed the setup looks correct
