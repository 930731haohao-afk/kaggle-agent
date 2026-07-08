"""
Data Loader Utility
====================
Common functions for loading and validating competition data.
"""

import pandas as pd
import yaml
import os
from typing import Tuple, Optional


def load_config(competition_dir: str) -> dict:
    """Load competition config.yaml."""
    config_path = os.path.join(competition_dir, "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_data(
    competition_dir: str,
    config: Optional[dict] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Load train, test, and sample submission data for a competition.

    Returns:
        Tuple of (train_df, test_df, sample_submission_df)
    """
    if config is None:
        config = load_config(competition_dir)

    data_dir = os.path.join(competition_dir, "data")

    train = pd.read_csv(os.path.join(data_dir, config["train_file"]))
    test = pd.read_csv(os.path.join(data_dir, config["test_file"]))

    sample_sub = None
    if config.get("sample_submission_file"):
        sub_path = os.path.join(data_dir, config["sample_submission_file"])
        if os.path.exists(sub_path):
            sample_sub = pd.read_csv(sub_path)

    print(f"Loaded: train={train.shape}, test={test.shape}", end="")
    if sample_sub is not None:
        print(f", sample_sub={sample_sub.shape}")
    else:
        print()

    return train, test, sample_sub


def validate_data(
    train: pd.DataFrame,
    test: pd.DataFrame,
    config: dict
) -> bool:
    """Run basic validation checks on loaded data."""
    target = config["target_column"]
    id_col = config["id_column"]
    all_ok = True

    # Target exists in train
    if target not in train.columns:
        print(f"FAIL: Target '{target}' not in train columns")
        all_ok = False

    # Target not in test
    if target in test.columns:
        print(f"WARNING: Target '{target}' found in test (possible leakage)")

    # ID column exists
    if id_col not in train.columns:
        print(f"FAIL: ID column '{id_col}' not in train")
        all_ok = False
    if id_col not in test.columns:
        print(f"FAIL: ID column '{id_col}' not in test")
        all_ok = False

    # No duplicate IDs
    if id_col in train.columns:
        dupes = train[id_col].duplicated().sum()
        if dupes > 0:
            print(f"WARNING: {dupes} duplicate IDs in train")

    # Feature columns match (excluding target)
    train_features = set(train.columns) - {target}
    test_features = set(test.columns)
    if train_features != test_features:
        extra_in_train = train_features - test_features
        extra_in_test = test_features - train_features
        if extra_in_train:
            print(f"WARNING: Columns in train but not test: {extra_in_train}")
        if extra_in_test:
            print(f"WARNING: Columns in test but not train: {extra_in_test}")

    if all_ok:
        print("Data validation: PASS")

    return all_ok
