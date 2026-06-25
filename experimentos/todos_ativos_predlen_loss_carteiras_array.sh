#!/bin/bash
#SBATCH -p medusas_shr
#SBATCH --gres=gpu:1
#SBATCH --array=0-119%2
#SBATCH --time=48:00:00
#SBATCH --job-name=all_predlen
#SBATCH --output=/sonic_home/igor.viveiros/paralelo/logs/all-predlen-%A_%a.out
#SBATCH --error=/sonic_home/igor.viveiros/paralelo/logs/all-predlen-%A_%a.err

set -euo pipefail

cd /sonic_home/igor.viveiros/paralelo

source /sonic_home/igor.viveiros/py310/bin/activate
PYTHON_BIN=/sonic_home/igor.viveiros/py310/bin/python

mkdir -p logs

EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-16}"
TEST_RATIO="${TEST_RATIO:-0.2}"
LOSS_NAME="${LOSS_NAME:-mse}"
MAX_ASSETS="${MAX_ASSETS:-9}"
ONLY_POSITIVE="${ONLY_POSITIVE:-true}"
ANNUAL_RF="${ANNUAL_RF:-0.043}"

MODELOS=(
  "AttentionSoloChannelIndependent"
  "AttentionSoloChannelIndependentSharedSpecific"
  "AttentionSoloChannelIndependentShrINSpec"
  "TransformerSpecific"
  "TransformerShrINSpec"
)

LOOKBACKS=(32 104 246)

# Não inclui 20 porque o exercício equivalente já foi feito.
PRED_LENS=(1 5 10 15)

JANELAS_REBALANCEAMENTO=(1 5 10 15 20)

# tipo_saida:arquivo_dataset:model_output
DATASETS=(
  "prices:b3_daily_tfb.csv:prices"
  "retornos:b3_log_returns.csv:returns"
)

EXPERIMENTO="predlen_loss_todos_ativos"
PRED_ROOT="previsoes/${EXPERIMENTO}"
RAW_PRED_ROOT="${PRED_ROOT}/_raw"
SIM_ROOT="simulacoes/${EXPERIMENTO}"
PRICE_DATASET="b3_daily_tfb.csv"

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

stem_csv() {
  basename "$1" .csv
}

copy_predictions_to_target() {
  local raw_dir="$1"
  local target_dir="$2"

  if [[ ! -d "$raw_dir" ]]; then
    echo "ERRO: diretório bruto de previsões não encontrado: $raw_dir" >&2
    exit 1
  fi

  rm -rf "$target_dir"
  mkdir -p "$target_dir"

  cp -f "$raw_dir"/janela_*.csv "$target_dir"/
  cp -f "$raw_dir"/train_loss.csv "$target_dir"/ 2>/dev/null || true
  cp -f "$raw_dir"/train_loss.png "$target_dir"/ 2>/dev/null || true
}

prepare_backtest_pred_dir() {
  local pred_dir="$1"
  local bt_dir="$pred_dir/_backtest_janelas_only"

  rm -rf "$bt_dir"
  mkdir -p "$bt_dir"

  cp -f "$pred_dir"/janela_*.csv "$bt_dir"/

  echo "$bt_dir"
}

# ============================================================
# Mapeamento SLURM_ARRAY_TASK_ID -> combinação única
# ============================================================

N_DATASETS=${#DATASETS[@]}
N_MODELOS=${#MODELOS[@]}
N_LOOKBACKS=${#LOOKBACKS[@]}
N_PRED_LENS=${#PRED_LENS[@]}

N_TASKS=$((N_DATASETS * N_MODELOS * N_LOOKBACKS * N_PRED_LENS))
TASK_ID=${SLURM_ARRAY_TASK_ID}

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

DATASET_SPEC="${DATASETS[$DATASET_IDX]}"
IFS=":" read -r TIPO_SERIE DATASET_FILE MODEL_OUTPUT <<< "$DATASET_SPEC"

MODEL_NAME="${MODELOS[$MODEL_IDX]}"
LOOKBACK="${LOOKBACKS[$LOOKBACK_IDX]}"
PRED_LEN="${PRED_LENS[$PRED_LEN_IDX]}"

DATASET_STEM="$(stem_csv "$DATASET_FILE")"
PRICE_PATH="$(resolve_data_path "$PRICE_DATASET")"
resolve_data_path "$DATASET_FILE" >/dev/null

mkdir -p "$PRED_ROOT" "$RAW_PRED_ROOT" "$SIM_ROOT"

echo "=== Job array iniciado em $(date) ==="
echo "TASK_ID=$TASK_ID / N_TASKS=$N_TASKS"
echo "experimento=$EXPERIMENTO"
echo "tipo=$TIPO_SERIE"
echo "dataset=$DATASET_FILE"
echo "model_output=$MODEL_OUTPUT"
echo "modelo=$MODEL_NAME"
echo "lookback=$LOOKBACK"
echo "pred_len=$PRED_LEN"
echo "max_assets=$MAX_ASSETS"
echo "python=$PYTHON_BIN"
echo "price_path=$PRICE_PATH"

EXTRA_DIRS=("lookback_${LOOKBACK}" "pred_len_${PRED_LEN}")

"$PYTHON_BIN" ./main_test.py \
  --base_de_dados "$DATASET_FILE" \
  --model_name "$MODEL_NAME" \
  --lookback "$LOOKBACK" \
  --pred_len "$PRED_LEN" \
  --test_ratio "$TEST_RATIO" \
  --batch_size "$BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --loss_name "$LOSS_NAME" \
  --output_dir "$RAW_PRED_ROOT" \
  --extra_dirs "${EXTRA_DIRS[@]}"

RAW_PRED_DIR="$RAW_PRED_ROOT/$DATASET_STEM/$MODEL_NAME/${EXTRA_DIRS[0]}/${EXTRA_DIRS[1]}"
PRED_DIR="$PRED_ROOT/$TIPO_SERIE/$MODEL_NAME/${EXTRA_DIRS[0]}/${EXTRA_DIRS[1]}"

copy_predictions_to_target "$RAW_PRED_DIR" "$PRED_DIR"
BT_PRED_DIR="$(prepare_backtest_pred_dir "$PRED_DIR")"

for K in "${JANELAS_REBALANCEAMENTO[@]}"; do
  if (( K > PRED_LEN )); then
    continue
  fi

  echo ""
  echo "Backtest | tipo=$TIPO_SERIE | modelo=$MODEL_NAME | lookback=$LOOKBACK | pred_len=$PRED_LEN | k=$K"

  "$PYTHON_BIN" -m estrategias.ranking_backtest \
    --pred_dir "$BT_PRED_DIR" \
    --price_path "$PRICE_PATH" \
    --output_dir "$SIM_ROOT/$TIPO_SERIE/$MODEL_NAME/lookback_${LOOKBACK}/pred_len_${PRED_LEN}" \
    --model_output "$MODEL_OUTPUT" \
    --rebalance_k "$K" \
    --max_assets "$MAX_ASSETS" \
    --horizon "$K" \
    --only_positive_pred "$ONLY_POSITIVE" \
    --returns_mode step \
    --annual_rf "$ANNUAL_RF" \
    --run_name "k_${K}"
done

echo ""
echo "=== Job array finalizado em $(date) ==="
echo "Previsões: $PRED_DIR"
echo "Simulações: $SIM_ROOT/$TIPO_SERIE/$MODEL_NAME/lookback_${LOOKBACK}/pred_len_${PRED_LEN}"
