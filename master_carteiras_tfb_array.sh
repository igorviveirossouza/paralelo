#!/bin/bash
#SBATCH -p medusas_shr
#SBATCH --gres=gpu:1
#SBATCH --array=0-89%2
#SBATCH --time=48:00:00
#SBATCH --job-name=master_tfb_cart
#SBATCH --output=/sonic_home/igor.viveiros/paralelo/logs/master-tfb-cart-%A_%a.out
#SBATCH --error=/sonic_home/igor.viveiros/paralelo/logs/master-tfb-cart-%A_%a.err

set -euo pipefail

PARALELO_ROOT="${PARALELO_ROOT:-/sonic_home/igor.viveiros/paralelo}"
PYTHON_BIN="${PYTHON_BIN:-/sonic_home/igor.viveiros/py310/bin/python}"
export MPLCONFIGDIR="/tmp/${USER}-mpl"

cd "$PARALELO_ROOT"
mkdir -p logs

# Configuração geral
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-16}"
TEST_RATIO="${TEST_RATIO:-0.2}"
LOSS_NAME="${LOSS_NAME:-mse}"
MAX_ASSETS="${MAX_ASSETS:-5}"
ONLY_POSITIVE_PRED="${ONLY_POSITIVE_PRED:-true}"
ANNUAL_RF="${ANNUAL_RF:-0.043}"

FORECAST_ROOT="${FORECAST_ROOT:-previsoes/master_tfb_experimento}"
SIM_ROOT="${SIM_ROOT:-simulacoes/master_tfb_experimento}"
OHLCV_FEATURE_FILE="${OHLCV_FEATURE_FILE:-b3_daily_tfb_ohlcv.csv}"
MARKET_FEATURE_FILES_STR="${MARKET_FEATURE_FILES:-indices.csv}"
read -r -a MARKET_FEATURE_FILES_ARR <<< "$MARKET_FEATURE_FILES_STR"

LOOKBACKS=(32 104 246)
PRED_LENS=(1 5 10 15 24)
JANELAS_REBALANCEAMENTO=(1 5 10 15 20 24)

# tipo_saida:data_name_tfb:price_dataset_paralelo:model_output
DATASETS=(
  "retornos_simples:b3_daily_return.csv:b3_daily_tfb.csv:returns"
  "log_retornos:b3_log_returns.csv:b3_daily_tfb.csv:log_returns"
  "prices:b3_daily_tfb.csv:b3_daily_tfb.csv:prices"
)

# Comparação solicitada:
# full       = Alpha158-like + OHLCV relativo + market_x
# ohlcv_only = OHLCV relativo + market_x, sem Alpha158-like
VARIANTS=("full" "ohlcv_only")

N_DATASETS=${#DATASETS[@]}
N_VARIANTS=${#VARIANTS[@]}
N_LOOKBACKS=${#LOOKBACKS[@]}
N_PRED_LENS=${#PRED_LENS[@]}
TOTAL=$((N_DATASETS * N_VARIANTS * N_LOOKBACKS * N_PRED_LENS))

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if (( TASK_ID < 0 || TASK_ID >= TOTAL )); then
  echo "TASK_ID=$TASK_ID fora do intervalo [0, $((TOTAL - 1))]."
  exit 1
fi

idx=$TASK_ID
pred_idx=$((idx % N_PRED_LENS)); idx=$((idx / N_PRED_LENS))
lookback_idx=$((idx % N_LOOKBACKS)); idx=$((idx / N_LOOKBACKS))
variant_idx=$((idx % N_VARIANTS)); idx=$((idx / N_VARIANTS))
dataset_idx=$((idx % N_DATASETS))

IFS=':' read -r TIPO_SAIDA DATA_NAME_TFB PRICE_DATASET MODEL_OUTPUT <<< "${DATASETS[$dataset_idx]}"
VARIANT="${VARIANTS[$variant_idx]}"
LOOKBACK="${LOOKBACKS[$lookback_idx]}"
PRED_LEN="${PRED_LENS[$pred_idx]}"

case "$VARIANT" in
  full)
    VARIANT_DIR="master_full_alpha158_ohlcv_market"
    USE_STOCK_FACTORS="true"
    USE_CANDLE_ENCODER="true"
    ;;
  ohlcv_only)
    VARIANT_DIR="master_ohlcv_market"
    USE_STOCK_FACTORS="false"
    USE_CANDLE_ENCODER="true"
    ;;
  *)
    echo "VARIANT inválida: $VARIANT"
    exit 1
    ;;
esac

DATA_STEM="${DATA_NAME_TFB%.csv}"
PRED_DIR="${FORECAST_ROOT}/${DATA_STEM}/MASTER/${VARIANT_DIR}/lookback_${LOOKBACK}/pred_len_${PRED_LEN}/loss_${LOSS_NAME}"

RUN_PREFIX="${DATA_STEM}__${VARIANT_DIR}__lb${LOOKBACK}__pl${PRED_LEN}__loss${LOSS_NAME}"

cat <<EOF
============================================================
MASTER carteira TFB
TASK_ID:                 $TASK_ID / $((TOTAL - 1))
Dataset alvo:            $DATA_NAME_TFB
Tipo saída:              $TIPO_SAIDA
Dataset preços carteira: $PRICE_DATASET
Model output:            $MODEL_OUTPUT
Variant:                 $VARIANT_DIR
Lookback:                $LOOKBACK
Pred len:                $PRED_LEN
Loss:                    $LOSS_NAME
Forecast dir:            $PRED_DIR
Sim root:                $SIM_ROOT
OHLCV feature file:      $OHLCV_FEATURE_FILE
Market feature files:    ${MARKET_FEATURE_FILES_ARR[*]}
============================================================
EOF

"$PYTHON_BIN" main_test.py \
  --model_name MASTER \
  --base_de_dados "$DATA_NAME_TFB" \
  --lookback "$LOOKBACK" \
  --pred_len "$PRED_LEN" \
  --test_ratio "$TEST_RATIO" \
  --batch_size "$BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --loss_name "$LOSS_NAME" \
  --output_dir "$FORECAST_ROOT" \
  --extra_dirs "$VARIANT_DIR" "lookback_${LOOKBACK}" "pred_len_${PRED_LEN}" "loss_${LOSS_NAME}" \
  --use_stock_factors "$USE_STOCK_FACTORS" \
  --stock_factor_mode alpha158 \
  --stock_factor_normalize true \
  --use_candle_encoder "$USE_CANDLE_ENCODER" \
  --candle_feature_mode ohlcv_relative \
  --ohlcv_feature_file "$OHLCV_FEATURE_FILE" \
  --use_market_features true \
  --market_feature_files "${MARKET_FEATURE_FILES_ARR[@]}" \
  --market_feature_mode master

for K in "${JANELAS_REBALANCEAMENTO[@]}"; do
  if (( K > PRED_LEN )); then
    echo "Pulando rebalance_k=$K porque é maior que pred_len=$PRED_LEN."
    continue
  fi

  RUN_NAME="${RUN_PREFIX}__k${K}__${MODEL_OUTPUT}"
  echo "Gerando carteira: $RUN_NAME"

  "$PYTHON_BIN" estrategias/ranking_backtest.py \
    --pred_dir "$PRED_DIR" \
    --price_path "data/${PRICE_DATASET}" \
    --output_dir "$SIM_ROOT" \
    --model_output "$MODEL_OUTPUT" \
    --rebalance_k "$K" \
    --max_assets "$MAX_ASSETS" \
    --horizon "$PRED_LEN" \
    --only_positive_pred "$ONLY_POSITIVE_PRED" \
    --returns_mode step \
    --annual_rf "$ANNUAL_RF" \
    --run_name "$RUN_NAME"
done

echo "✅ Experimento MASTER concluído: $RUN_PREFIX"
