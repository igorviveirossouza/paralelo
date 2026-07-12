#!/bin/bash
#SBATCH -p gorgonas
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --array=0-179%16
#SBATCH --mem=4G
#SBATCH --time=08:00:00
#SBATCH --job-name=tfb_conversao
#SBATCH --output=/sonic_home/igor.viveiros/paralelo/logs/tfb-conversao-%A_%a.out
#SBATCH --error=/sonic_home/igor.viveiros/paralelo/logs/tfb-conversao-%A_%a.err

# Paralelismo: uma configuração TFB por tarefa do array.
# O limite %16 permite até 16 conversões simultâneas no cluster CPU.
set -euo pipefail

PARALELO_ROOT="${PARALELO_ROOT:-/sonic_home/igor.viveiros/paralelo}"
source "$PARALELO_ROOT/experimentos/tfb_pipeline_config.sh"

cd "$PARALELO_ROOT"
mkdir -p logs

# Cada tarefa usa apenas uma CPU e não deve abrir threads internas adicionais.
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

tfb_resolve_task "${SLURM_ARRAY_TASK_ID:-0}"
tfb_print_task

[[ -f "$ORIGINAL_DATASET" ]] || { echo "ERRO: dataset original ausente: $ORIGINAL_DATASET" >&2; exit 1; }
[[ -f "$PARALELO_ROOT/utils/timeStamp_to_tfbPred.py" ]] || {
  echo "ERRO: conversor ausente: utils/timeStamp_to_tfbPred.py" >&2
  exit 1
}

if ! find "$RAW_DECODED_DIR" -maxdepth 1 -type f \
  -name '*csv_sample_*_inference_data.csv' -print -quit | grep -q .; then
  echo "ERRO: previsões brutas do TFB não encontradas em $RAW_DECODED_DIR" >&2
  exit 1
fi

START_TS=$(date +%s)
echo "[CPU] nó=${SLURMD_NODENAME:-desconhecido} | job=${SLURM_ARRAY_JOB_ID:-local} | task=${SLURM_ARRAY_TASK_ID:-0}"

rm -rf "$PRED_DIR"
mkdir -p "$PRED_DIR"

"$PYTHON_BIN" "$PARALELO_ROOT/utils/timeStamp_to_tfbPred.py" \
  --pred-dir "$RAW_DECODED_DIR" \
  --original-dataset "$ORIGINAL_DATASET" \
  --pred-len "$PRED_LEN" \
  --lookback "$LOOKBACK" \
  --output-dir "$PRED_DIR" \
  --output-name-template 'janela_{sample_idx:06d}.csv'

N_FILES=$(find "$PRED_DIR" -maxdepth 1 -type f -name 'janela_*.csv' | wc -l)
if (( N_FILES == 0 )); then
  echo "ERRO: a conversão não gerou arquivos janela_*.csv em $PRED_DIR" >&2
  exit 1
fi

END_TS=$(date +%s)
echo "✅ Conversão TFB concluída: $N_FILES janelas em $((END_TS - START_TS))s"
