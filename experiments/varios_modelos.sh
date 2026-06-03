#!/bin/bash
#SBATCH -p medusas_shr
# ~  SBATCH -p gorgonas_dev
#SBATCH --gres=gpu:1
#SBATCH --time=06:30:00
#SBATCH --output=/sonic_home/igor.viveiros/paralelo/logs/tioms-%j.out
#SBATCH --error=/sonic_home/igor.viveiros/paralelo/logs/tioms-%j.err

set -euo pipefail

EXPERIMENTO=(
    "mse"
    "lookback_32"
    "epochs_100"
)

MODELOS=(
    #"AttentionSolo"
    #"AttentionSoloChannelIndependent"
    AttentionSoloChannelIndependentSharedSpecific
    #"Transformer"
    #"TransformerSpecific"

)

echo "=== Job iniciado em $(date) ==="
START_TIME=$(date +%s)

echo "Experimento: $EXPERIMENTO"
echo "Modelos: ${MODELOS[*]}"
echo "Hostname: $(hostname)"
echo "GPU disponível:"
nvidia-smi -L

source /sonic_home/igor.viveiros/py310/bin/activate
PYTHON_BIN=/sonic_home/igor.viveiros/py310/bin/python

cd /sonic_home/igor.viveiros/paralelo || exit 1

for MODEL_NAME in "${MODELOS[@]}"; do
    echo ""
    echo "======================================"
    echo "Iniciando modelo: $MODEL_NAME"
    echo "Horário: $(date)"
    echo "======================================"

    MODEL_START=$(date +%s)

    "$PYTHON_BIN" ./main_test.py \
        --base_de_dados b3_daily_financeiro.csv \
        --lookback 32 \
        --pred_len 24 \
        --batch_size 16 \
        --epochs 100 \
        --extra_dirs "${EXPERIMENTO[@]}" \
        --model_name "$MODEL_NAME" \
        --embedding_type "linear" \
        --embedding_lag_size 10 \
        --loss mse \
        --dilate_alpha 0.3 \
        --dilate_gamma 0.001

    MODEL_END=$(date +%s)
    MODEL_ELAPSED=$((MODEL_END - MODEL_START))

    echo "Modelo finalizado: $MODEL_NAME"
    echo "Tempo do modelo: $((MODEL_ELAPSED / 60)) min $((MODEL_ELAPSED % 60)) s"
done

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "=== Job finalizado em $(date) ==="
echo "Tempo total: $((ELAPSED / 60)) min $((ELAPSED % 60)) s"