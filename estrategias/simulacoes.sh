source /sonic_home/igor.viveiros/py310/bin/activate
PYTHON_BIN=/sonic_home/igor.viveiros/py310/bin/python

cd /sonic_home/igor.viveiros/paralelo || exit 1

#previsoes="previsoes/tfb/financeiro/lookback_32/DUET"
previsoes="previsoes/b3_daily_financeiro/TransformerShrINSpec/PRICES/lookback_104/pred_len_01/epochs_100"

"$PYTHON_BIN" -m estrategias.ranking_backtest \
  --pred_dir "$previsoes" \
  --price_path data/b3_daily_financeiro.csv \
  --model_output prices \
  --rebalance_k 1 \
  --max_assets 9 \
  --horizon 5 \
  --only_positive_pred true \
  --output_dir simulacoes/prices/financeiro/Transformer/lookback_104/pred_len_01 \


