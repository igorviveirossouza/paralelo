#!/bin/bash
#SBATCH -p gorgonas
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --array=0-179%16
#SBATCH --mem=6G
#SBATCH --time=12:00:00
#SBATCH --job-name=tfb_carteiras
#SBATCH --output=/sonic_home/igor.viveiros/paralelo/logs/tfb-carteiras-%A_%a.out
#SBATCH --error=/sonic_home/igor.viveiros/paralelo/logs/tfb-carteiras-%A_%a.err

set -euo pipefail

PARALELO_ROOT="${PARALELO_ROOT:-/sonic_home/igor.viveiros/paralelo}"
source "$PARALELO_ROOT/experimentos/tfb_pipeline_config.sh"

cd "$PARALELO_ROOT"
mkdir -p logs

tfb_resolve_task "${SLURM_ARRAY_TASK_ID:-0}"
tfb_print_task

MAX_ASSETS="${TFB_MAX_ASSETS:-9}"
ONLY_POSITIVE="${TFB_ONLY_POSITIVE:-true}"
ANNUAL_RF="${ANNUAL_RF:-0.043}"
BT_PRED_DIR="$PRED_DIR/_backtest_janelas_only"

mapfile -t PRED_FILES < <(
  find "$PRED_DIR" -maxdepth 1 -type f -name 'janela_*.csv' -print | sort
)

if ((${#PRED_FILES[@]} == 0)); then
  echo "ERRO: previsões convertidas não encontradas em $PRED_DIR" >&2
  exit 1
fi

rm -rf "$BT_PRED_DIR"
mkdir -p "$BT_PRED_DIR" "$SIM_DIR"
cp -- "${PRED_FILES[@]}" "$BT_PRED_DIR/"

for K in "${TFB_REBALANCE_WINDOWS_ARR[@]}"; do
  if (( K > PRED_LEN )); then
    continue
  fi

  echo "[BACKTEST] modelo=$MODEL_KEY | lookback=$LOOKBACK | pred_len=$PRED_LEN | k=$K"

  "$PYTHON_BIN" "$PARALELO_ROOT/estrategias/ranking_backtest.py" \
    --pred_dir "$BT_PRED_DIR" \
    --price_path "$PRICE_PATH" \
    --output_dir "$SIM_DIR" \
    --model_output "$MODEL_OUTPUT" \
    --rebalance_k "$K" \
    --max_assets "$MAX_ASSETS" \
    --horizon "$K" \
    --only_positive_pred "$ONLY_POSITIVE" \
    --returns_mode step \
    --annual_rf "$ANNUAL_RF" \
    --run_name "k_${K}"
done

echo "✅ Carteiras TFB concluídas: $SIM_DIR"
