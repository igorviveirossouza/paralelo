#!/bin/bash
set -euo pipefail
PARALELO_ROOT="/sonic_home/igor.viveiros/paralelo"
TFB_ROOT="/sonic_home/igor.viveiros/src/TFB"
PYTHON_BIN="/sonic_home/igor.viveiros/py310/bin/python"
cd "$PARALELO_ROOT"
MODELOS=("DUET" "TimesNet" "FEDformer" "Nonstationary_Transformer")
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
MODEL_KEY="${MODELOS[$TASK_ID]}"
echo "$MODEL_KEY"
