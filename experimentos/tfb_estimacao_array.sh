#!/bin/bash
#SBATCH -p medusas_shr
#SBATCH --gres=gpu:1
#SBATCH --array=0-179%6
#SBATCH --time=48:00:00
#SBATCH --job-name=tfb_estimacao
#SBATCH --output=/sonic_home/igor.viveiros/paralelo/logs/tfb-estimacao-%A_%a.out
#SBATCH --error=/sonic_home/igor.viveiros/paralelo/logs/tfb-estimacao-%A_%a.err

set -euo pipefail

PARALELO_ROOT="${PARALELO_ROOT:-/sonic_home/igor.viveiros/paralelo}"
source "$PARALELO_ROOT/experimentos/tfb_pipeline_config.sh"

cd "$PARALELO_ROOT"
mkdir -p logs "$TFB_DECODED_ROOT"
export MPLCONFIGDIR="/tmp/${USER}-mpl"

tfb_resolve_task "${SLURM_ARRAY_TASK_ID:-0}"
tfb_print_task

[[ -d "$TFB_ROOT" ]] || { echo "ERRO: repositório TFB ausente: $TFB_ROOT" >&2; exit 1; }
[[ -f "$TFB_RUN_SCRIPT" ]] || { echo "ERRO: script do TFB ausente: $TFB_RUN_SCRIPT" >&2; exit 1; }
[[ -f "$ORIGINAL_DATASET" ]] || { echo "ERRO: dataset TFB ausente: $ORIGINAL_DATASET" >&2; exit 1; }
[[ -x "$PYTHON_BIN" ]] || { echo "ERRO: Python não executável: $PYTHON_BIN" >&2; exit 1; }

if [[ "${TFB_CLEAN_RAW:-false}" == "true" ]]; then
  echo "Limpando saída TFB anterior: $RAW_DECODED_DIR"
  rm -rf "$RAW_DECODED_DIR"
fi

mkdir -p "$RAW_DECODED_DIR"
START_TS=$(date +%s)

TFB_ROOT="$TFB_ROOT" \
PYTHON_BIN="$PYTHON_BIN" \
DATA_NAME="$DATA_NAME" \
SEQ_LEN="$LOOKBACK" \
PRED_LEN="$PRED_LEN" \
DECODED_ROOT="$TFB_DECODED_ROOT/$DATASET_STEM" \
SAVE_SUBDIR="$TFB_SAVE_SUBDIR" \
bash "$TFB_RUN_SCRIPT" "$MODEL_KEY"

if ! find "$RAW_DECODED_DIR" -maxdepth 1 -type f \
  -name '*csv_sample_*_inference_data.csv' -print -quit | grep -q .; then
  echo "ERRO: o TFB terminou sem gerar previsões em $RAW_DECODED_DIR" >&2
  exit 1
fi

END_TS=$(date +%s)
echo "✅ Estimação TFB concluída em $((END_TS - START_TS))s"
