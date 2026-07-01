#!/bin/bash
#SBATCH -p medusas_shr
#SBATCH --array=0-17%6
#SBATCH --time=12:00:00
#SBATCH --job-name=bench_carteira
#SBATCH --output=/sonic_home/igor.viveiros/paralelo/logs/bench-carteira-%A_%a.out
#SBATCH --error=/sonic_home/igor.viveiros/paralelo/logs/bench-carteira-%A_%a.err

set -euo pipefail

cd /sonic_home/igor.viveiros/paralelo

source /sonic_home/igor.viveiros/py310/bin/activate
PYTHON_BIN=/sonic_home/igor.viveiros/py310/bin/python

mkdir -p logs

MAX_ASSETS="${MAX_ASSETS:-9}"
ONLY_POSITIVE="${ONLY_POSITIVE:-true}"
ANNUAL_RF="${ANNUAL_RF:-0.043}"
N_RANDOM_SEEDS="${N_RANDOM_SEEDS:-100}"

LOOKBACKS=(32 104 246)
PRED_LENS=(1 5 10 15 20 24)
JANELAS_REBALANCEAMENTO=(1 5 10 15 20 24)

# Experimento neural usado apenas para ancorar as mesmas janelas de previsão/teste.
EXPERIMENTO="${EXPERIMENTO:-meu_multi_lb_predlen_todos_ativos}"
REFERENCE_TIPO="${REFERENCE_TIPO:-retornos_simples}"
REFERENCE_MODEL="${REFERENCE_MODEL:-AttentionSoloChannelIndependent}"

PRED_ROOT="previsoes/${EXPERIMENTO}"
SIM_ROOT="simulacoes/${EXPERIMENTO}"
BENCH_ROOT="${SIM_ROOT}/benchmarks"
PRICE_DATASET="${PRICE_DATASET:-b3_daily_tfb.csv}"

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

N_LOOKBACKS=${#LOOKBACKS[@]}
N_PRED_LENS=${#PRED_LENS[@]}
N_TASKS=$((N_LOOKBACKS * N_PRED_LENS))
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

if (( TASK_ID >= N_TASKS )); then
  echo "TASK_ID=$TASK_ID fora do intervalo N_TASKS=$N_TASKS"
  exit 0
fi

PRED_LEN_IDX=$(( TASK_ID % N_PRED_LENS ))
LOOKBACK_IDX=$(( TASK_ID / N_PRED_LENS ))

LOOKBACK="${LOOKBACKS[$LOOKBACK_IDX]}"
PRED_LEN="${PRED_LENS[$PRED_LEN_IDX]}"
PRICE_PATH="$(resolve_data_path "$PRICE_DATASET")"

REF_PRED_DIR="$PRED_ROOT/$REFERENCE_TIPO/$REFERENCE_MODEL/lookback_${LOOKBACK}/pred_len_${PRED_LEN}"

if [[ ! -d "$REF_PRED_DIR" ]]; then
  echo "ERRO: referência de previsões não encontrada: $REF_PRED_DIR" >&2
  echo "Ajuste REFERENCE_TIPO/REFERENCE_MODEL ou rode as previsões neurais antes." >&2
  exit 1
fi

mkdir -p "$BENCH_ROOT"

echo "=== Benchmarks carteira iniciados em $(date) ==="
echo "TASK_ID=$TASK_ID / N_TASKS=$N_TASKS"
echo "experimento=$EXPERIMENTO"
echo "reference_tipo=$REFERENCE_TIPO"
echo "reference_model=$REFERENCE_MODEL"
echo "lookback=$LOOKBACK"
echo "pred_len=$PRED_LEN"
echo "price_path=$PRICE_PATH"
echo "max_assets=$MAX_ASSETS"
echo "n_random_seeds=$N_RANDOM_SEEDS"
echo "bench_root=$BENCH_ROOT"

for K in "${JANELAS_REBALANCEAMENTO[@]}"; do
  if (( K > PRED_LEN )); then
    continue
  fi

  echo ""
  echo "Benchmark Momentum | lookback=$LOOKBACK | pred_len=$PRED_LEN | k=$K"
  "$PYTHON_BIN" -m estrategias.benchmark_backtest \
    --strategy momentum \
    --reference_pred_dir "$REF_PRED_DIR" \
    --price_path "$PRICE_PATH" \
    --output_dir "$BENCH_ROOT/Momentum/lookback_${LOOKBACK}/pred_len_${PRED_LEN}" \
    --rebalance_k "$K" \
    --max_assets "$MAX_ASSETS" \
    --horizon "$K" \
    --only_positive_pred "$ONLY_POSITIVE" \
    --annual_rf "$ANNUAL_RF" \
    --run_name "k_${K}"

  echo ""
  echo "Benchmark EqualWeight | lookback=$LOOKBACK | pred_len=$PRED_LEN | k=$K"
  "$PYTHON_BIN" -m estrategias.benchmark_backtest \
    --strategy equal_weight \
    --reference_pred_dir "$REF_PRED_DIR" \
    --price_path "$PRICE_PATH" \
    --output_dir "$BENCH_ROOT/EqualWeight/lookback_${LOOKBACK}/pred_len_${PRED_LEN}" \
    --rebalance_k "$K" \
    --max_assets "$MAX_ASSETS" \
    --horizon "$K" \
    --only_positive_pred false \
    --annual_rf "$ANNUAL_RF" \
    --run_name "k_${K}"

  echo ""
  echo "Benchmark RandomTopJ | lookback=$LOOKBACK | pred_len=$PRED_LEN | k=$K | seeds=0..$((N_RANDOM_SEEDS - 1))"
  for SEED in $(seq 0 $((N_RANDOM_SEEDS - 1))); do
    "$PYTHON_BIN" -m estrategias.benchmark_backtest \
      --strategy random_topj \
      --reference_pred_dir "$REF_PRED_DIR" \
      --price_path "$PRICE_PATH" \
      --output_dir "$BENCH_ROOT/RandomTopJ/lookback_${LOOKBACK}/pred_len_${PRED_LEN}" \
      --rebalance_k "$K" \
      --max_assets "$MAX_ASSETS" \
      --horizon "$K" \
      --only_positive_pred false \
      --annual_rf "$ANNUAL_RF" \
      --random_seed "$SEED" \
      --run_name "k_${K}_seed_${SEED}"
  done
done

echo ""
echo "=== Benchmarks carteira finalizados em $(date) ==="
echo "Resultados: $BENCH_ROOT"
echo ""
echo "Após o array terminar, gere o resumo com:"
echo "$PYTHON_BIN -m estrategias.resumir_metricas --root $BENCH_ROOT --output_csv $SIM_ROOT/resumo_benchmarks.csv"
