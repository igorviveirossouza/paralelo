#!/bin/bash
#SBATCH -p medusas_shr
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --job-name=fin_cart
#SBATCH --output=/sonic_home/igor.viveiros/paralelo/logs/fin-cart-%j.out
#SBATCH --error=/sonic_home/igor.viveiros/paralelo/logs/fin-cart-%j.err

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

PRED_LEN=20

MODELOS=(
  "AttentionSoloChannelIndependent"
  "AttentionSoloChannelIndependentSharedSpecific"
  "AttentionSoloChannelIndependentShrINSpec"
  "TransformerSpecific"
  "TransformerShrINSpec"
)

LOOKBACKS=(32 104 246)
JANELAS_PREVISAO=(24 36 48)

# tipo_saida:arquivo_dataset:model_output
DATASETS=(
  "prices:b3_daily_financeiro.csv:prices"
  "retornos:b3_daily_return_financeiro.csv:returns"
)

PRED_ROOT="previsoes/financeiro"
RAW_PRED_ROOT="${PRED_ROOT}/_raw"
SIM_ROOT="simulacoes/financeiro"
PRICE_DATASET="b3_daily_financeiro.csv"

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

PRICE_PATH="$(resolve_data_path "$PRICE_DATASET")"

mkdir -p "$PRED_ROOT" "$RAW_PRED_ROOT" "$SIM_ROOT"

echo "=== Experimento iniciado em $(date) ==="
echo "Python: $PYTHON_BIN"
echo "Preço realizado para backtest: $PRICE_PATH"
echo "Previsões em: $PRED_ROOT/{prices,retornos}"
echo "Simulações em: $SIM_ROOT/{prices,retornos}"

for dataset_spec in "${DATASETS[@]}"; do
  IFS=":" read -r TIPO_SERIE DATASET_FILE MODEL_OUTPUT <<< "$dataset_spec"

  DATASET_STEM="$(stem_csv "$DATASET_FILE")"
  resolve_data_path "$DATASET_FILE" >/dev/null

  for MODEL_NAME in "${MODELOS[@]}"; do
    for LOOKBACK in "${LOOKBACKS[@]}"; do
      EXTRA_DIRS=("lookback_${LOOKBACK}" "pred_len_${PRED_LEN}")

      echo ""
      echo "------------------------------------------------------------"
      echo "Treinando/previsões | tipo=$TIPO_SERIE | modelo=$MODEL_NAME | lookback=$LOOKBACK"
      echo "------------------------------------------------------------"

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

      for K in "${JANELAS_PREVISAO[@]}"; do
        echo ""
        echo "Backtest | tipo=$TIPO_SERIE | modelo=$MODEL_NAME | lookback=$LOOKBACK | janela=$K"

        "$PYTHON_BIN" -m estrategias.ranking_backtest \
          --pred_dir "$BT_PRED_DIR" \
          --price_path "$PRICE_PATH" \
          --output_dir "$SIM_ROOT/$TIPO_SERIE/$MODEL_NAME/lookback_${LOOKBACK}" \
          --model_output "$MODEL_OUTPUT" \
          --rebalance_k "$K" \
          --max_assets "$MAX_ASSETS" \
          --horizon "$K" \
          --only_positive_pred "$ONLY_POSITIVE" \
          --returns_mode step \
          --annual_rf "$ANNUAL_RF" \
          --run_name "k_${K}"
      done
    done
  done
done

"$PYTHON_BIN" utils/comparar_simulacoes_financeiro.py \
  --root "$SIM_ROOT" \
  --output "$SIM_ROOT/comparativo_metricas.csv"

echo ""
echo "=== Experimento finalizado em $(date) ==="
echo "Comparativo: $SIM_ROOT/comparativo_metricas.csv"