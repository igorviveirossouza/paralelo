#!/bin/bash
set -euo pipefail

cd /sonic_home/igor.viveiros/paralelo

source /sonic_home/igor.viveiros/py310/bin/activate
PYTHON_BIN=/sonic_home/igor.viveiros/py310/bin/python

"$PYTHON_BIN" -m estrategias.resumir_metricas \
  --root simulacoes/meu_multi_lb_predlen_todos_ativos/benchmarks \
  --output_csv simulacoes/meu_multi_lb_predlen_todos_ativos/resumo_benchmarks.csv