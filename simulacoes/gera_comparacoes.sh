cd /sonic_home/igor.viveiros/paralelo

source /sonic_home/igor.viveiros/py310/bin/activate

python utils/comparar_simulacoes_financeiro.py \
  --root simulacoes/tfb_multi_lb_predlen_carteiras \
  --output simulacoes/tfb_multi_lb_predlen_carteiras/comparativo_metricas_com_bench.csv \
  --long_output simulacoes/tfb_multi_lb_predlen_carteiras/comparativo_metricas_long_com_bench.csv \
  --pred_len \