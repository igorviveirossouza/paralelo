#!/bin/bash
#SBATCH -p gorgonas
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --job-name=gera_comp_tfb
#SBATCH --output=/sonic_home/igor.viveiros/paralelo/logs/gera-comp-tfb-%j.out
#SBATCH --error=/sonic_home/igor.viveiros/paralelo/logs/gera-comp-tfb-%j.err

# Paralelismo: um processo principal e até 16 workers Python.
set -euo pipefail

PARALELO_ROOT="${PARALELO_ROOT:-/sonic_home/igor.viveiros/paralelo}"
PYTHON_BIN="${PYTHON_BIN:-/sonic_home/igor.viveiros/py310/bin/python}"
WORKERS="${WORKERS:-16}"
ROOT="${ROOT:-$PARALELO_ROOT/simulacoes/tfb_multi_lb_predlen_carteiras}"
OUTPUT="${OUTPUT:-$ROOT/comparativo_metricas_com_bench_neg.csv}"
LONG_OUTPUT="${LONG_OUTPUT:-$ROOT/comparativo_metricas_long_com_bench_neg.csv}"

cd "$PARALELO_ROOT"
mkdir -p logs

ALLOCATED_CPUS="${SLURM_CPUS_PER_TASK:-16}"
if (( WORKERS < 1 || WORKERS > ALLOCATED_CPUS )); then
  echo "ERRO: WORKERS=$WORKERS; CPUs alocadas=$ALLOCATED_CPUS." >&2
  exit 1
fi

# Evita multiplicação de threads dentro de cada processo worker.
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

START_TS=$(date +%s)
echo "============================================================"
echo "Comparação TFB em CPU"
echo "Nó:       ${SLURMD_NODENAME:-desconhecido}"
echo "CPUs:     $ALLOCATED_CPUS"
echo "Workers:  $WORKERS"
echo "Raiz:     $ROOT"
echo "Início:   $(date --iso-8601=seconds)"
echo "============================================================"

"$PYTHON_BIN" utils/comparar_simulacoes_financeiro.py \
  --root "$ROOT" \
  --output "$OUTPUT" \
  --long_output "$LONG_OUTPUT" \
  --pred_len \
  --workers "$WORKERS"

END_TS=$(date +%s)
echo "============================================================"
echo "✅ Comparativo TFB concluído em $((END_TS - START_TS))s"
echo "Comparativo:       $OUTPUT"
echo "Comparativo longo: $LONG_OUTPUT"
echo "============================================================"
