#!/usr/bin/env bash
set -euo pipefail

source /sonic_home/igor.viveiros/py310/bin/activate
cd /sonic_home/igor.viveiros/paralelo/utils || exit 1

lookbacks=(32 104 246)
modelos=(DUET TimesNet Nonstationary_Transformer FEDformer)

for lookback in "${lookbacks[@]}"; do
  for modelo in "${modelos[@]}"; do
    echo "Processando lookback=${lookback}, modelo=${modelo}"

    python timeStamp_to_tfbPred.py \
      --pred-dir "/sonic_home/igor.viveiros/src/TFB/Previsoes/financeiro_indice/seq_len_${lookback}/${modelo}" \
      --original-dataset "/sonic_home/igor.viveiros/paralelo/data/b3_daily_financeiro.csv" \
      --pred-len 24 \
      --lookback "${lookback}" \
      --output-dir "/sonic_home/igor.viveiros/paralelo/previsoes/tfb/financeiro_indice/lookback_${lookback}/${modelo}" \
      --output-name-template 'janela_{sample_idx:06d}.csv'
  done
done