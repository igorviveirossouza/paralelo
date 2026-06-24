cd /sonic_home/igor.viveiros/paralelo

source /sonic_home/igor.viveiros/py310/bin/activate

python utils/comparar_simulacoes_financeiro.py \
  --root simulacoes/financeiro_predlen_loss \
  --output simulacoes/financeiro_predlen_loss/comparativo_metricas.csv \
  --long_output simulacoes/financeiro_predlen_loss/comparativo_metricas_long.csv \
  --pred_len \