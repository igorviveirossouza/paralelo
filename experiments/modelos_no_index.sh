#!/bin/bash
#SBATCH -p medusas_shr
#SBATCH --gres=gpu:1
#SBATCH --time=08:30:00
#SBATCH --output=/sonic_home/igor.viveiros/paralelo/logs/tioms-%j.out
#SBATCH --error=/sonic_home/igor.viveiros/paralelo/logs/tioms-%j.err

set -euo pipefail

MODELOS=(
    "AttentionSoloChannelIndependent"
    "AttentionSoloChannelIndependentSharedSpecific"
    "AttentionSoloChannelIndependentShrINSpec"
    "TransformerShrINSpec"
)

LOOKBACKS=(
    #32
    #104
    #369
    #246
    492
)

EPOCHS=(
    50
    100
    250
    500
)

loss='mse'

echo "=== Job iniciado em $(date) ==="
START_TIME=$(date +%s)

echo "Modelos: ${MODELOS[*]}"
echo "Epochs: ${EPOCHS[*]}"
echo "Loss fixa: mse"
echo "Lookback: ${LOOKBACKS[*]}"
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
    for L_LOOKBACK in "${LOOKBACKS[@]}"; do 
        for N_EPOCHS in "${EPOCHS[@]}"; do
            EXPERIMENTO=(
                "$loss"
                "lookback_${L_LOOKBACK}"
                "epochs_${N_EPOCHS}"
            )

            echo ""
            echo "--------------------------------------"
            echo "Modelo: $MODEL_NAME"
            echo "Lookback: $L_LOOKBACK"
            echo "Épocas: $N_EPOCHS"
            echo "Experimento: ${EXPERIMENTO[*]}"
            echo "Horário: $(date)"
            echo "--------------------------------------"

            RUN_START=$(date +%s)

            "$PYTHON_BIN" ./main_test.py \
                --base_de_dados b3_daily_financeiro_indice.csv \
                --lookback "$L_LOOKBACK" \
                --pred_len 24 \
                --batch_size 16 \
                --epochs "$N_EPOCHS" \
                --extra_dirs "${EXPERIMENTO[@]}" \
                --model_name "$MODEL_NAME" \
                --embedding_type "linear" \
                --embedding_lag_size 10 \
                --loss "$loss" \
                --dilate_alpha 0.3 \
                --dilate_gamma 0.001

            RUN_END=$(date +%s)
            RUN_ELAPSED=$((RUN_END - RUN_START))

            echo "Execução finalizada:"
            echo "Modelo: $MODEL_NAME"
            echo "Épocas: $N_EPOCHS"
            echo "Tempo da execução: $((RUN_ELAPSED / 60)) min $((RUN_ELAPSED % 60)) s"
        done
    done
    MODEL_END=$(date +%s)
    MODEL_ELAPSED=$((MODEL_END - MODEL_START))

    echo ""
    echo "Modelo finalizado: $MODEL_NAME"
    echo "Tempo total do modelo: $((MODEL_ELAPSED / 60)) min $((MODEL_ELAPSED % 60)) s"
done

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "=== Job finalizado em $(date) ==="
echo "Tempo total: $((ELAPSED / 60)) min $((ELAPSED % 60)) s"

