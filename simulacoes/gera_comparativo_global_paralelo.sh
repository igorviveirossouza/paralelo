#!/bin/bash
#SBATCH -p gorgonas
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --job-name=comp_global_auc
#SBATCH --output=/sonic_home/igor.viveiros/paralelo/logs/comp-global-auc-%j.out
#SBATCH --error=/sonic_home/igor.viveiros/paralelo/logs/comp-global-auc-%j.err

set -euo pipefail

PARALELO_ROOT="/sonic_home/igor.viveiros/paralelo"
PYTHON_BIN="/sonic_home/igor.viveiros/py310/bin/python"
WORKERS="${WORKERS:-16}"
OUTPUT_DIR="${OUTPUT_DIR:-simulacoes/comparativo_global_master_tfb}"
TOP_N="${TOP_N:-20}"

cd "$PARALELO_ROOT"
mkdir -p logs

ALLOCATED_CPUS="${SLURM_CPUS_PER_TASK:-16}"
if (( WORKERS < 1 )); then
    echo "ERRO: WORKERS deve ser >= 1." >&2
    exit 1
fi
if (( WORKERS > ALLOCATED_CPUS )); then
    echo "ERRO: WORKERS=$WORKERS excede as CPUs alocadas: $ALLOCATED_CPUS." >&2
    exit 1
fi

# Cada worker deve permanecer com uma única thread interna.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

START_TS=$(date +%s)

echo "============================================================"
echo "Nó:          ${SLURMD_NODENAME:-desconhecido}"
echo "CPUs:        $ALLOCATED_CPUS"
echo "Workers:     $WORKERS"
echo "Saída:       $OUTPUT_DIR"
echo "Início:      $(date --iso-8601=seconds)"
echo "============================================================"

"$PYTHON_BIN" utils/comparativo_metricas_global.py \
    --base_dir "$PARALELO_ROOT" \
    --output_dir "$OUTPUT_DIR" \
    --top_n "$TOP_N" \
    --workers "$WORKERS"

END_TS=$(date +%s)
ELAPSED=$((END_TS - START_TS))

echo "============================================================"
echo "Concluído:   $(date --iso-8601=seconds)"
echo "Duração:     ${ELAPSED}s"
echo "Resultados:  $PARALELO_ROOT/$OUTPUT_DIR"
echo "============================================================"
