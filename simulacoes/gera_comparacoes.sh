#!/bin/bash
set -euo pipefail

cd /sonic_home/igor.viveiros/paralelo

source /sonic_home/igor.viveiros/py310/bin/activate
PYTHON_BIN=/sonic_home/igor.viveiros/py310/bin/python

ROOT="${ROOT:-simulacoes/tfb_multi_lb_predlen_carteiras}"
OUTPUT="${OUTPUT:-$ROOT/comparativo_metricas_com_bench_neg.csv}"
LONG_OUTPUT="${LONG_OUTPUT:-$ROOT/comparativo_metricas_long_com_bench_neg.csv}"

"$PYTHON_BIN" utils/comparar_simulacoes_financeiro.py \
  --root "$ROOT" \
  --output "$OUTPUT" \
  --long_output "$LONG_OUTPUT" \
  --pred_len

echo "Comparativo final: $OUTPUT"
echo "Comparativo longo: $LONG_OUTPUT"
