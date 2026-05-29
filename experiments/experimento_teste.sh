#!/bin/bash
#SBATCH -p medusas_shr
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/sonic_home/igor.viveiros/paralelo/logs/tioms-%j.out
#SBATCH --error=/sonic_home/igor.viveiros/paralelo/logs/tioms-%j.err

set -euo pipefail

echo "=== Job iniciado em $(date) ==="
echo "Hostname: $(hostname)"
echo "GPU disponível: $(nvidia-smi -L)"

source /sonic_home/igor.viveiros/py310/bin/activate
PYTHON_BIN=/sonic_home/igor.viveiros/py310/bin/python

cd /sonic_home/igor.viveiros/paralelo || exit 1

"$PYTHON_BIN" ./main_test.py \
    --base_de_dados b3_daily_financeiro.csv \
    --lookback 32 \
    --pred_len 24 \
    --batch_size 16\
    --epochs 250\
    --extra_dirs epochs_250 lookback_32 dilate\
    --model_name AttentionSoloChannelIndependent\
    --loss dilate \
    --dilate_alpha 0.7 \
    --dilate_gamma 0.05 \
    --dilate_normalize true
