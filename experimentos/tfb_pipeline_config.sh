#!/bin/bash

# Configuração compartilhada pelas etapas TFB.
# Este arquivo deve ser carregado com source.

PARALELO_ROOT="${PARALELO_ROOT:-/sonic_home/igor.viveiros/paralelo}"
TFB_ROOT="${TFB_ROOT:-/sonic_home/igor.viveiros/src/TFB}"
PYTHON_BIN="${PYTHON_BIN:-/sonic_home/igor.viveiros/py310/bin/python}"
TFB_RUN_SCRIPT="${TFB_RUN_SCRIPT:-$TFB_ROOT/scripts/run_b3_financeiro_predlen24_lb32.sh}"

TFB_EXPERIMENT="${TFB_EXPERIMENT:-tfb_multi_lb_predlen_carteiras}"
TFB_PRED_ROOT="${TFB_PRED_ROOT:-$PARALELO_ROOT/previsoes/$TFB_EXPERIMENT}"
TFB_DECODED_ROOT="${TFB_DECODED_ROOT:-$TFB_PRED_ROOT/_tfb_decoded}"
TFB_SIM_ROOT="${TFB_SIM_ROOT:-$PARALELO_ROOT/simulacoes/$TFB_EXPERIMENT}"

TFB_MODELS_STR="${TFB_MODELS:-DUET TimesNet FEDformer Nonstationary_Transformer}"
TFB_LOOKBACKS_STR="${TFB_LOOKBACKS:-32 104 246}"
TFB_PRED_LENS_STR="${TFB_PRED_LENS:-1 5 10 15 24}"
TFB_REBALANCE_WINDOWS_STR="${TFB_REBALANCE_WINDOWS:-1 5 10 15 20 24}"

read -r -a TFB_MODELS_ARR <<< "$TFB_MODELS_STR"
read -r -a TFB_LOOKBACKS_ARR <<< "$TFB_LOOKBACKS_STR"
read -r -a TFB_PRED_LENS_ARR <<< "$TFB_PRED_LENS_STR"
read -r -a TFB_REBALANCE_WINDOWS_ARR <<< "$TFB_REBALANCE_WINDOWS_STR"

# tipo_serie:dataset_no_TFB:dataset_de_precos_no_paralelo:model_output
TFB_DATASETS_ARR=(
  "retornos_simples:b3_daily_return.csv:b3_daily_tfb.csv:returns"
  "log_retornos:b3_log_returns.csv:b3_daily_tfb.csv:log_returns"
  "prices:b3_daily_tfb.csv:b3_daily_tfb.csv:prices"
)

TFB_N_DATASETS=${#TFB_DATASETS_ARR[@]}
TFB_N_MODELS=${#TFB_MODELS_ARR[@]}
TFB_N_LOOKBACKS=${#TFB_LOOKBACKS_ARR[@]}
TFB_N_PRED_LENS=${#TFB_PRED_LENS_ARR[@]}
TFB_TOTAL_TASKS=$((TFB_N_DATASETS * TFB_N_MODELS * TFB_N_LOOKBACKS * TFB_N_PRED_LENS))

stem_csv() {
  local name="${1##*/}"
  printf '%s\n' "${name%.csv}"
}

resolve_parallel_data_path() {
  local fname="$1"
  local candidate
  for candidate in \
    "$fname" \
    "$PARALELO_ROOT/$fname" \
    "$PARALELO_ROOT/data/$fname" \
    "$PARALELO_ROOT/attachments/$fname"; do
    if [[ -f "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  echo "ERRO: arquivo do paralelo não encontrado: $fname" >&2
  return 1
}

tfb_resolve_task() {
  local task_id="${1:-${SLURM_ARRAY_TASK_ID:-0}}"
  local idx pred_idx lookback_idx model_idx dataset_idx

  if (( task_id < 0 || task_id >= TFB_TOTAL_TASKS )); then
    echo "ERRO: TASK_ID=$task_id fora do intervalo [0, $((TFB_TOTAL_TASKS - 1))]." >&2
    return 1
  fi

  idx=$task_id
  pred_idx=$((idx % TFB_N_PRED_LENS)); idx=$((idx / TFB_N_PRED_LENS))
  lookback_idx=$((idx % TFB_N_LOOKBACKS)); idx=$((idx / TFB_N_LOOKBACKS))
  model_idx=$((idx % TFB_N_MODELS)); idx=$((idx / TFB_N_MODELS))
  dataset_idx=$((idx % TFB_N_DATASETS))

  IFS=':' read -r TIPO_SERIE DATA_NAME PRICE_DATASET MODEL_OUTPUT \
    <<< "${TFB_DATASETS_ARR[$dataset_idx]}"

  MODEL_KEY="${TFB_MODELS_ARR[$model_idx]}"
  LOOKBACK="${TFB_LOOKBACKS_ARR[$lookback_idx]}"
  PRED_LEN="${TFB_PRED_LENS_ARR[$pred_idx]}"
  DATASET_STEM="$(stem_csv "$DATA_NAME")"

  ORIGINAL_DATASET="$TFB_ROOT/dataset/forecasting/$DATA_NAME"
  PRICE_PATH="$(resolve_parallel_data_path "$PRICE_DATASET")"

  PRED_DIR="$TFB_PRED_ROOT/$TIPO_SERIE/$MODEL_KEY/lookback_${LOOKBACK}/pred_len_${PRED_LEN}"
  RAW_DECODED_DIR="$TFB_DECODED_ROOT/$DATASET_STEM/seq_len_${LOOKBACK}/pred_len_${PRED_LEN}/$MODEL_KEY"
  SIM_DIR="$TFB_SIM_ROOT/$TIPO_SERIE/$MODEL_KEY/lookback_${LOOKBACK}/pred_len_${PRED_LEN}"
  TFB_SAVE_SUBDIR="paralelo_${TFB_EXPERIMENT}/$DATASET_STEM/seq_len_${LOOKBACK}/pred_len_${PRED_LEN}/$MODEL_KEY"

  export TIPO_SERIE DATA_NAME PRICE_DATASET MODEL_OUTPUT MODEL_KEY
  export LOOKBACK PRED_LEN DATASET_STEM ORIGINAL_DATASET PRICE_PATH
  export PRED_DIR RAW_DECODED_DIR SIM_DIR TFB_SAVE_SUBDIR
}

tfb_print_task() {
  cat <<EOF
============================================================
Pipeline TFB
TASK_ID:          ${SLURM_ARRAY_TASK_ID:-0} / $((TFB_TOTAL_TASKS - 1))
Dataset:          $DATA_NAME
Tipo de série:    $TIPO_SERIE
Modelo:           $MODEL_KEY
Lookback:         $LOOKBACK
Pred len:         $PRED_LEN
Model output:     $MODEL_OUTPUT
TFB bruto:        $RAW_DECODED_DIR
Previsões finais: $PRED_DIR
Simulações:       $SIM_DIR
============================================================
EOF
}
