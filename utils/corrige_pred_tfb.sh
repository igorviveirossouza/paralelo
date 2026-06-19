#!/usr/bin/env bash
set -euo pipefail

source /sonic_home/igor.viveiros/py310/bin/activate
cd /sonic_home/igor.viveiros/paralelo/utils || exit 1

lookbacks=(32 104)
modelos=(DUET)

for lookback in "${lookbacks[@]}"; do
  for modelo in "${modelos[@]}"; do
    echo "Processando lookback=${lookback}, modelo=${modelo}"

    python timeStamp_to_tfbPred.py \
      --pred-dir "/sonic_home/igor.viveiros/src/TFB/Previsoes/return_tfb/seq_len_${lookback}/DUET" \
      --original-dataset "/sonic_home/igor.viveiros/src/TFB/dataset/forecasting/b3_return_tfb.csv" \
      --pred-len 24 \
      --lookback "${lookback}" \
      --output-dir "/sonic_home/igor.viveiros/paralelo/previsoes/tfb/bolsa/returns/DUET/lookback_${lookback}" \
      --output-name-template 'janela_{sample_idx:06d}.csv'
  done
done

