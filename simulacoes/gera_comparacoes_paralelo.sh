#!/bin/bash
#SBATCH -p gorgonas
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --job-name=gera_comp_master
#SBATCH --output=/sonic_home/igor.viveiros/paralelo/logs/gera-comp-master-%j.out
#SBATCH --error=/sonic_home/igor.viveiros/paralelo/logs/gera-comp-master-%j.err

set -euo pipefail

PARALELO_ROOT="/sonic_home/igor.viveiros/paralelo"
PYTHON_BIN="/sonic_home/igor.viveiros/py310/bin/python"
WORKERS="${WORKERS:-16}"

ROOT="${ROOT:-$PARALELO_ROOT/simulacoes/master_tfb_experimento}"
OUTPUT="${OUTPUT:-$ROOT/comparativo_metricas_com_bench_neg.csv}"
LONG_OUTPUT="${LONG_OUTPUT:-$ROOT/comparativo_metricas_long_com_bench_neg.csv}"

cd "$PARALELO_ROOT"
mkdir -p logs

if (( WORKERS > ${SLURM_CPUS_PER_TASK:-16} )); then
    echo "ERRO: WORKERS=$WORKERS excede SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-16}." >&2
    exit 1
fi

# Evita que cada processo abra múltiplas threads internas.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "============================================================"
echo "Nó:       ${SLURMD_NODENAME:-desconhecido}"
echo "CPUs:     ${SLURM_CPUS_PER_TASK:-16}"
echo "Workers:  $WORKERS"
echo "Raiz:     $ROOT"
echo "============================================================"

"$PYTHON_BIN" utils/comparar_simulacoes_financeiro.py \
    --root "$ROOT" \
    --output "$OUTPUT" \
    --long_output "$LONG_OUTPUT" \
    --pred_len \
    --workers "$WORKERS"

echo "Comparativo final: $OUTPUT"
echo "Comparativo longo: $LONG_OUTPUT"
