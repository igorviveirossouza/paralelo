#!/bin/bash
set -euo pipefail

cd /sonic_home/igor.viveiros/paralelo

source /sonic_home/igor.viveiros/py310/bin/activate
PYTHON_BIN=/sonic_home/igor.viveiros/py310/bin/python

EXPERIMENTO="${EXPERIMENTO:-meu_multi_lb_predlen_todos_ativos}"
ROOT="simulacoes/${EXPERIMENTO}"
BENCH_ROOT="${BENCH_ROOT:-$ROOT/benchmarks}"
RESUMO_CSV="${RESUMO_CSV:-$ROOT/resumo_benchmarks.csv}"
ACERTO_NEG_CSV="${ACERTO_NEG_CSV:-$ROOT/acerto_negativos.csv}"

"$PYTHON_BIN" -m estrategias.resumir_metricas \
  --root "$BENCH_ROOT" \
  --output_csv "$RESUMO_CSV"

"$PYTHON_BIN" -m estrategias.acerto_negativos \
  --root "$BENCH_ROOT" \
  --output_csv "$ACERTO_NEG_CSV" \
  --merge_csv "$RESUMO_CSV"

echo "Resumo final: $RESUMO_CSV"
echo "Acerto negativos: $ACERTO_NEG_CSV"
