#!/bin/bash
set -euo pipefail

cd /sonic_home/igor.viveiros/paralelo

source /sonic_home/igor.viveiros/py310/bin/activate
PYTHON_BIN=/sonic_home/igor.viveiros/py310/bin/python

ROOT="${ROOT:-simulacoes/tfb_multi_lb_predlen_carteiras}"
OUTPUT="${OUTPUT:-$ROOT/comparativo_metricas_com_bench.csv}"
LONG_OUTPUT="${LONG_OUTPUT:-$ROOT/comparativo_metricas_long_com_bench.csv}"
ACERTO_NEG_OUTPUT="${ACERTO_NEG_OUTPUT:-$ROOT/acerto_negativos.csv}"

"$PYTHON_BIN" -m estrategias.acerto_negativos \
  --root "$ROOT" \
  --output_csv "$ACERTO_NEG_OUTPUT"

"$PYTHON_BIN" utils/comparar_simulacoes_financeiro.py \
  --root "$ROOT" \
  --output "$OUTPUT" \
  --long_output "$LONG_OUTPUT" \
  --pred_len

echo "Comparativo final: $OUTPUT"
echo "Comparativo longo: $LONG_OUTPUT"
echo "Acerto negativos: $ACERTO_NEG_OUTPUT"
