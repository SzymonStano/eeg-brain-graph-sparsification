#!/bin/bash
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH="${PYTHONPATH}:${PROJECT_ROOT}/src"

# Activate venv if available
if [ -d "$PROJECT_ROOT/venv" ]; then
    source "$PROJECT_ROOT/venv/bin/activate"
fi

python scripts/preprocess_data.py
python scripts/train_baselines.py --kfold_cv 5
python scripts/train_automatic_methods.py --kfold_cv 5