#!/bin/bash
# Source this script to set KAGGLE_API_TOKEN from kaggle.json
# Usage: source utils/kaggle_auth.sh && kaggle competitions list
export KAGGLE_API_TOKEN=$(python -c "import json; print(json.load(open('C:/Users/user/.kaggle/kaggle.json'))['key'])")
