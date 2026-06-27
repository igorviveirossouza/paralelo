#!/bin/bash
set -euo pipefail

PARALELO_ROOT="/sonic_home/igor.viveiros/paralelo"
TFB_ROOT="/sonic_home/igor.viveiros/src/TFB"
PYTHON_BIN="/sonic_home/igor.viveiros/py310/bin/python"
export MPLCONFIGDIR="/tmp/${USER}-mpl"

cd "$PARALELO_ROOT"
mkdir -p logs

DATA_NAME="b3_daily_return_financeiro.csv"
ORIGINAL_DATASET="${TFB_ROOT}/dataset/forecasting/${DATA_NAME}"
PRICE_DATASET="b3_daily_tfb.csv"
LOOKBACK=32
PRED_LEN=24
MODEL_OUTPUT="returns"
TIPO_SERIE="retornos_simples"
MAX_ASSETS="${MAX_ASSETS:-9}"
ONLY_POSITIVE="${ONLY_POSITIVE:-true}"
ANNUAL_RF="${ANNUAL_RF:-0.043}"

MODELOS=("DUET" "TimesNet" "FEDformer" "Nonstationary_Transformer")
JANELAS_REBALANCEAMENTO=(10 20 24)
EXPERIMENTO="tfb_predlen24_lb32_carteiras"
PRED_ROOT="previsoes/${EXPERIMENTO}"
TFB_DECODED_ROOT="${PARALELO_ROOT}/${PRED_ROOT}/_tfb_decoded"
SIM_ROOT="simulacoes/${EXPERIMENTO}"
COMPARATIVO="${SIM_ROOT}/comparativo_metricas.csv"
COMPARATIVO_LONG="${SIM_ROOT}/comparativo_metricas_long.csv"

resolve_data_path() {
  local fname="$1"
  for p in "$fname" "data/$fname" "attachments/$fname"; do
    if [[ -f "$p" ]]; then
      echo "$p"
      return 0
    fi
  done
  echo "ERRO: arquivo não encontrado: $fname" >&2
  exit 1
}

prepare_backtest_pred_dir() {
  local pred_dir="$1"
  local bt_dir="$pred_dir/_backtest_janelas_only"
  rm -rf "$bt_dir"
  mkdir -p "$bt_dir"
  cp -f "$pred_dir"/janela_*.csv "$bt_dir"/
  echo "$bt_dir"
}

N_MODELOS=${#MODELOS[@]}
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if (( TASK_ID >= N_MODELOS )); then
  echo "TASK_ID=$TASK_ID fora do intervalo N_MODELOS=$N_MODELOS"
  exit 0
fi

MODEL_KEY="${MODELOS[$TASK_ID]}"
PRICE_PATH="$(resolve_data_path "$PRICE_DATASET")"
PRED_DIR="${PRED_ROOT}/${TIPO_SERIE}/${MODEL_KEY}/lookback_${LOOKBACK}/pred_len_${PRED_LEN}"
RAW_DECODED_DIR="${TFB_DECODED_ROOT}/seq_len_${LOOKBACK}/pred_len_${PRED_LEN}/${MODEL_KEY}"
SIM_DIR="${SIM_ROOT}/${TIPO_SERIE}/${MODEL_KEY}/lookback_${LOOKBACK}/pred_len_${PRED_LEN}"
TFB_SAVE_SUBDIR="paralelo_${EXPERIMENTO}/seq_len_${LOOKBACK}/pred_len_${PRED_LEN}/${MODEL_KEY}"
mkdir -p "$PRED_DIR" "$SIM_DIR" "$TFB_DECODED_ROOT"

START_TIME=$(date +%s)
echo "=== Experimento TFB dentro do paralelo iniciado em $(date) ==="
echo "TASK_ID=$TASK_ID / modelo=$MODEL_KEY"
echo "Previsões finais: $PRED_DIR"
echo "Simulações: $SIM_DIR"

TFB_ROOT="$TFB_ROOT" \
PYTHON_BIN="$PYTHON_BIN" \
DATA_NAME="$DATA_NAME" \
SEQ_LEN="$LOOKBACK" \
PRED_LEN="$PRED_LEN" \
DECODED_ROOT="$TFB_DECODED_ROOT" \
SAVE_SUBDIR="$TFB_SAVE_SUBDIR" \
bash "$TFB_ROOT/scripts/run_b3_financeiro_predlen24_lb32.sh" "$MODEL_KEY"

rm -rf "$PRED_DIR"
mkdir -p "$PRED_DIR"

"$PYTHON_BIN" "$PARALELO_ROOT/utils/timeStamp_to_tfbPred.py" \
  --pred-dir "$RAW_DECODED_DIR" \
  --original-dataset "$ORIGINAL_DATASET" \
  --pred-len "$PRED_LEN" \
  --lookback "$LOOKBACK" \
  --output-dir "$PRED_DIR" \
  --output-name-template 'janela_{sample_idx:06d}.csv'

BT_PRED_DIR="$(prepare_backtest_pred_dir "$PRED_DIR")"

for K in "${JANELAS_REBALANCEAMENTO[@]}"; do
  echo "Backtest | modelo=$MODEL_KEY | lookback=$LOOKBACK | pred_len=$PRED_LEN | k=$K"
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

"$PYTHON_BIN" "$PARALELO_ROOT/utils/comparar_simulacoes_financeiro.py" \
  --root "$SIM_ROOT" \
  --output "$COMPARATIVO" \
  --long_output "$COMPARATIVO_LONG" \
  --pred_len

END_TIME=$(date +%s)
echo "=== Experimento finalizado em $(date) ==="
echo "Tempo total: $((END_TIME - START_TIME)) segundos"
echo "Comparativo: $COMPARATIVO"
