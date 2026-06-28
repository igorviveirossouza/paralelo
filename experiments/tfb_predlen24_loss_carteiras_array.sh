#!/bin/bash
#SBATCH -p medusas_shr
#SBATCH --gres=gpu:1
#SBATCH --array=0-179%6
#SBATCH --time=48:00:00
#SBATCH --job-name=all_tfb_carteira
#SBATCH --output=/sonic_home/igor.viveiros/paralelo/logs/all_tfb_carteira-%A_%a.out
#SBATCH --error=/sonic_home/igor.viveiros/paralelo/logs/all_tfb_carteira-%A_%a.err

set -euo pipefail

PARALELO_ROOT="/sonic_home/igor.viveiros/paralelo"
TFB_ROOT="/sonic_home/igor.viveiros/src/TFB"
PYTHON_BIN="/sonic_home/igor.viveiros/py310/bin/python"
export MPLCONFIGDIR="/tmp/${USER}-mpl"
cd "$PARALELO_ROOT"
mkdir -p logs

# tipo_saida:data_name_tfb:price_dataset_paralelo:model_output
DATASETS=(
  "retornos_simples:b3_daily_return.csv:b3_daily_tfb.csv:returns"
  "log_retornos:b3_log_returns.csv:b3_daily_tfb.csv:log_returns"
  "prices:b3_daily_tfb.csv:b3_daily_tfb.csv:prices"
)
LOOKBACKS=(32 104 246)
PRED_LENS=(1 5 10 15 24)
MODELOS=("DUET" "TimesNet" "FEDformer" "Nonstationary_Transformer")
JANELAS_REBALANCEAMENTO=(1 5 10 15 20 24)

MAX_ASSETS="${MAX_ASSETS:-9}"
ONLY_POSITIVE="${ONLY_POSITIVE:-true}"
ANNUAL_RF="${ANNUAL_RF:-0.043}"
EXPERIMENTO="tfb_multi_lb_predlen_carteiras"
PRED_ROOT="previsoes/${EXPERIMENTO}"
TFB_DECODED_ROOT="${PARALELO_ROOT}/${PRED_ROOT}/_tfb_decoded"
SIM_ROOT="simulacoes/${EXPERIMENTO}"
COMPARATIVO="${SIM_ROOT}/comparativo_metricas.csv"
COMPARATIVO_LONG="${SIM_ROOT}/comparativo_metricas_long.csv"

resolve_data_path() {
  local fname="$1"
  for p in "$fname" "data/$fname" "attachments/$fname"; do
    [[ -f "$p" ]] && echo "$p" && return 0
  done
  echo "ERRO: arquivo não encontrado: $fname" >&2
  exit 1
}

stem_csv() {
  local fname="$1"
  fname="${fname##*/}"
  echo "${fname%.csv}"
}

prepare_backtest_pred_dir() {
  local pred_dir="$1"
  local bt_dir="$pred_dir/_backtest_janelas_only"
  rm -rf "$bt_dir"
  mkdir -p "$bt_dir"
  cp -f "$pred_dir"/janela_*.csv "$bt_dir"/
  echo "$bt_dir"
}

N_DATASETS=${#DATASETS[@]}
N_MODELOS=${#MODELOS[@]}
N_LOOKBACKS=${#LOOKBACKS[@]}
N_PRED_LENS=${#PRED_LENS[@]}
N_TASKS=$((N_DATASETS * N_MODELOS * N_LOOKBACKS * N_PRED_LENS))
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

if (( TASK_ID >= N_TASKS )); then
  echo "TASK_ID=$TASK_ID fora do intervalo N_TASKS=$N_TASKS"
  exit 0
fi

PRED_LEN_IDX=$(( TASK_ID % N_PRED_LENS ))
TMP=$(( TASK_ID / N_PRED_LENS ))
LOOKBACK_IDX=$(( TMP % N_LOOKBACKS ))
TMP=$(( TMP / N_LOOKBACKS ))
MODEL_IDX=$(( TMP % N_MODELOS ))
DATASET_IDX=$(( TMP / N_MODELOS ))

IFS=":" read -r TIPO_SERIE DATA_NAME PRICE_DATASET MODEL_OUTPUT <<< "${DATASETS[$DATASET_IDX]}"
MODEL_KEY="${MODELOS[$MODEL_IDX]}"
LOOKBACK="${LOOKBACKS[$LOOKBACK_IDX]}"
PRED_LEN="${PRED_LENS[$PRED_LEN_IDX]}"
DATASET_STEM="$(stem_csv "$DATA_NAME")"
ORIGINAL_DATASET="${TFB_ROOT}/dataset/forecasting/${DATA_NAME}"
PRICE_PATH="$(resolve_data_path "$PRICE_DATASET")"

[[ -f "$ORIGINAL_DATASET" ]] || { echo "ERRO: dataset TFB não encontrado: $ORIGINAL_DATASET" >&2; exit 1; }

PRED_DIR="${PRED_ROOT}/${TIPO_SERIE}/${MODEL_KEY}/lookback_${LOOKBACK}/pred_len_${PRED_LEN}"
RAW_DECODED_DIR="${TFB_DECODED_ROOT}/${DATASET_STEM}/seq_len_${LOOKBACK}/pred_len_${PRED_LEN}/${MODEL_KEY}"
SIM_DIR="${SIM_ROOT}/${TIPO_SERIE}/${MODEL_KEY}/lookback_${LOOKBACK}/pred_len_${PRED_LEN}"
TFB_SAVE_SUBDIR="paralelo_${EXPERIMENTO}/${DATASET_STEM}/seq_len_${LOOKBACK}/pred_len_${PRED_LEN}/${MODEL_KEY}"
mkdir -p "$PRED_DIR" "$SIM_DIR" "$TFB_DECODED_ROOT"

START_TIME=$(date +%s)
echo "TASK_ID=$TASK_ID / N_TASKS=$N_TASKS"
echo "dataset=$DATA_NAME | tipo=$TIPO_SERIE | modelo=$MODEL_KEY | lookback=$LOOKBACK | pred_len=$PRED_LEN"

TFB_ROOT="$TFB_ROOT" PYTHON_BIN="$PYTHON_BIN" DATA_NAME="$DATA_NAME" SEQ_LEN="$LOOKBACK" PRED_LEN="$PRED_LEN" DECODED_ROOT="${TFB_DECODED_ROOT}/${DATASET_STEM}" SAVE_SUBDIR="$TFB_SAVE_SUBDIR" bash "$TFB_ROOT/scripts/run_b3_financeiro_predlen24_lb32.sh" "$MODEL_KEY"

rm -rf "$PRED_DIR"
mkdir -p "$PRED_DIR"
"$PYTHON_BIN" "$PARALELO_ROOT/utils/timeStamp_to_tfbPred.py" --pred-dir "$RAW_DECODED_DIR" --original-dataset "$ORIGINAL_DATASET" --pred-len "$PRED_LEN" --lookback "$LOOKBACK" --output-dir "$PRED_DIR" --output-name-template 'janela_{sample_idx:06d}.csv'
BT_PRED_DIR="$(prepare_backtest_pred_dir "$PRED_DIR")"

for K in "${JANELAS_REBALANCEAMENTO[@]}"; do
  (( K > PRED_LEN )) && continue
  echo "Backtest | tipo=$TIPO_SERIE | modelo=$MODEL_KEY | lookback=$LOOKBACK | pred_len=$PRED_LEN | k=$K"
  "$PYTHON_BIN" "$PARALELO_ROOT/estrategias/ranking_backtest.py" --pred_dir "$BT_PRED_DIR" --price_path "$PRICE_PATH" --output_dir "$SIM_DIR" --model_output "$MODEL_OUTPUT" --rebalance_k "$K" --max_assets "$MAX_ASSETS" --horizon "$K" --only_positive_pred "$ONLY_POSITIVE" --returns_mode step --annual_rf "$ANNUAL_RF" --run_name "k_${K}"
done

"$PYTHON_BIN" "$PARALELO_ROOT/utils/comparar_simulacoes_financeiro.py" --root "$SIM_ROOT" --output "$COMPARATIVO" --long_output "$COMPARATIVO_LONG" --pred_len
END_TIME=$(date +%s)
echo "Tempo total: $((END_TIME - START_TIME)) segundos"
echo "Comparativo: $COMPARATIVO"
