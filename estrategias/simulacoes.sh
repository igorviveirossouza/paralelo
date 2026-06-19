source /sonic_home/igor.viveiros/py310/bin/activate
PYTHON_BIN=/sonic_home/igor.viveiros/py310/bin/python

cd /sonic_home/igor.viveiros/paralelo || exit 1

#previsoes="previsoes/tfb/financeiro/lookback_32/DUET"
previsoes="previsoes/tfb/bolsa/returns/DUET/lookback_104"

"$PYTHON_BIN" -m estrategias.ranking_backtest \
  --pred_dir "$previsoes" \
  --price_path ../src/TFB/dataset/forecasting/b3_daily_tfb.csv \
  --model_output returns \
  --rebalance_k 1 \
  --max_assets 9 \
  --horizon 5 \
  --only_positive_pred true \
  --output_dir simulacoes/retornos/bolsa/DUET/lookback_104 \


