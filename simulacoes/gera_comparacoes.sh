cd /sonic_home/igor.viveiros/paralelo

source /sonic_home/igor.viveiros/py310/bin/activate

python utils/comparar_simulacoes_financeiro.py \
  --root simulacoes/predlen_loss_todos_ativos \
  --output simulacoes/predlen_loss_todos_ativos/comparativo_metricas.csv \
  --long_output simulacoes/predlen_loss_todos_ativos/comparativo_metricas_long.csv \
  --pred_len \