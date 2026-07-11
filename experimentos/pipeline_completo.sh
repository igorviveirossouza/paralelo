#!/bin/bash

# Orquestrador completo: MASTER + TFB + comparações.
# Execute no nó de login: bash experimentos/pipeline_completo.sh
# Para pular uma etapa, comente a linha correspondente.
ENABLED_STAGES=(
  master_estimacao
  master_carteiras
  tfb_estimacao
  tfb_conversao
  tfb_carteiras
  comparacao_master
  comparacao_tfb
  comparacao_global
)

set -euo pipefail

PARALELO_ROOT="${PARALELO_ROOT:-/sonic_home/igor.viveiros/paralelo}"
MASTER_ESTIMATION_ARRAY="${MASTER_ESTIMATION_ARRAY:-0-107%8}"
MASTER_BACKTEST_ARRAY="${MASTER_BACKTEST_ARRAY:-0-107%16}"
TFB_ESTIMATION_ARRAY="${TFB_ESTIMATION_ARRAY:-0-179%6}"
TFB_CONVERSION_ARRAY="${TFB_CONVERSION_ARRAY:-0-179%16}"
TFB_BACKTEST_ARRAY="${TFB_BACKTEST_ARRAY:-0-179%16}"
COMPARISON_WORKERS="${COMPARISON_WORKERS:-16}"
DRY_RUN="${DRY_RUN:-false}"

cd "$PARALELO_ROOT"
mkdir -p logs

stage_enabled() {
  local wanted="$1"
  local stage
  for stage in "${ENABLED_STAGES[@]}"; do
    [[ "$stage" == "$wanted" ]] && return 0
  done
  return 1
}

validate_stages() {
  local stage
  for stage in "${ENABLED_STAGES[@]}"; do
    case "$stage" in
      master_estimacao|master_carteiras|tfb_estimacao|tfb_conversao|tfb_carteiras|comparacao_master|comparacao_tfb|comparacao_global) ;;
      *)
        echo "Etapa desconhecida em ENABLED_STAGES: $stage" >&2
        exit 1
        ;;
    esac
  done
}

submit_job() {
  local dependency="$1"
  shift

  local command=(sbatch --parsable)
  if [[ -n "$dependency" ]]; then
    command+=(--dependency="afterok:${dependency}")
  fi
  command+=("$@")

  echo "+ ${command[*]}" >&2

  if [[ "${DRY_RUN,,}" == "true" ]]; then
    echo "DRYRUN"
    return 0
  fi

  local output
  output=$("${command[@]}")
  echo "${output%%;*}"
}

join_dependencies() {
  local -a deps=()
  local dep existing found

  for dep in "$@"; do
    [[ -z "$dep" || "$dep" == "DRYRUN" ]] && continue
    found=false
    for existing in "${deps[@]:-}"; do
      [[ "$existing" == "$dep" ]] && found=true
    done
    [[ "$found" == "false" ]] && deps+=("$dep")
  done

  if ((${#deps[@]})); then
    local joined
    joined=$(IFS=:; echo "${deps[*]}")
    echo "$joined"
  fi
}

record_job() {
  local stage="$1"
  local job="$2"
  echo "$stage=$job" | tee -a "$SUBMISSION_LOG"
}

validate_stages

if ((${#ENABLED_STAGES[@]} == 0)); then
  echo "Nenhuma etapa habilitada."
  exit 0
fi

STAMP=$(date +%Y%m%d_%H%M%S)
SUBMISSION_LOG="logs/pipeline-completo-${STAMP}.log"
MASTER_DEP=""
TFB_DEP=""

{
  echo "============================================================"
  echo "Pipeline completo MASTER + TFB"
  echo "Data:                    $(date --iso-8601=seconds)"
  echo "Etapas:                  ${ENABLED_STAGES[*]}"
  echo "MASTER estimação:        $MASTER_ESTIMATION_ARRAY"
  echo "MASTER carteiras:        $MASTER_BACKTEST_ARRAY"
  echo "TFB estimação:           $TFB_ESTIMATION_ARRAY"
  echo "TFB conversão:           $TFB_CONVERSION_ARRAY"
  echo "TFB carteiras:           $TFB_BACKTEST_ARRAY"
  echo "Workers comparações:     $COMPARISON_WORKERS"
  echo "Dry run:                 $DRY_RUN"
  echo "============================================================"
} | tee "$SUBMISSION_LOG"

# ---------------------------- MASTER ----------------------------
if stage_enabled master_estimacao; then
  JOB=$(submit_job "$MASTER_DEP" \
    --array="$MASTER_ESTIMATION_ARRAY" \
    --export=ALL,RUN_ESTIMATION=true,RUN_BACKTEST=false \
    experimentos/carteiras_master_array.sh)
  [[ "$JOB" != "DRYRUN" ]] && MASTER_DEP="$JOB"
  record_job master_estimacao "$JOB"
else
  record_job master_estimacao PULADA
fi

if stage_enabled master_carteiras; then
  JOB=$(submit_job "$MASTER_DEP" \
    --array="$MASTER_BACKTEST_ARRAY" \
    experimentos/carteiras_master_backtest_array.sh)
  [[ "$JOB" != "DRYRUN" ]] && MASTER_DEP="$JOB"
  record_job master_carteiras "$JOB"
else
  record_job master_carteiras PULADA
fi

# ------------------------------ TFB ------------------------------
if stage_enabled tfb_estimacao; then
  JOB=$(submit_job "$TFB_DEP" \
    --array="$TFB_ESTIMATION_ARRAY" \
    experimentos/tfb_estimacao_array.sh)
  [[ "$JOB" != "DRYRUN" ]] && TFB_DEP="$JOB"
  record_job tfb_estimacao "$JOB"
else
  record_job tfb_estimacao PULADA
fi

if stage_enabled tfb_conversao; then
  JOB=$(submit_job "$TFB_DEP" \
    --array="$TFB_CONVERSION_ARRAY" \
    experimentos/tfb_conversao_array.sh)
  [[ "$JOB" != "DRYRUN" ]] && TFB_DEP="$JOB"
  record_job tfb_conversao "$JOB"
else
  record_job tfb_conversao PULADA
fi

if stage_enabled tfb_carteiras; then
  JOB=$(submit_job "$TFB_DEP" \
    --array="$TFB_BACKTEST_ARRAY" \
    experimentos/tfb_carteiras_array.sh)
  [[ "$JOB" != "DRYRUN" ]] && TFB_DEP="$JOB"
  record_job tfb_carteiras "$JOB"
else
  record_job tfb_carteiras PULADA
fi

# -------------------------- Comparações --------------------------
if stage_enabled comparacao_master; then
  JOB=$(submit_job "$MASTER_DEP" \
    --cpus-per-task="$COMPARISON_WORKERS" \
    --export=ALL,WORKERS="$COMPARISON_WORKERS" \
    simulacoes/gera_comparacoes_paralelo.sh)
  [[ "$JOB" != "DRYRUN" ]] && MASTER_DEP="$JOB"
  record_job comparacao_master "$JOB"
else
  record_job comparacao_master PULADA
fi

if stage_enabled comparacao_tfb; then
  JOB=$(submit_job "$TFB_DEP" \
    --cpus-per-task="$COMPARISON_WORKERS" \
    --export=ALL,WORKERS="$COMPARISON_WORKERS" \
    simulacoes/gera_comparacoes_tfb_paralelo.sh)
  [[ "$JOB" != "DRYRUN" ]] && TFB_DEP="$JOB"
  record_job comparacao_tfb "$JOB"
else
  record_job comparacao_tfb PULADA
fi

if stage_enabled comparacao_global; then
  GLOBAL_DEP=$(join_dependencies "$MASTER_DEP" "$TFB_DEP")
  JOB=$(submit_job "$GLOBAL_DEP" \
    --cpus-per-task="$COMPARISON_WORKERS" \
    --export=ALL,WORKERS="$COMPARISON_WORKERS" \
    simulacoes/gera_comparativo_global_paralelo.sh)
  record_job comparacao_global "$JOB"
else
  record_job comparacao_global PULADA
fi

{
  echo "============================================================"
  echo "Submissão concluída."
  echo "Último job MASTER: ${MASTER_DEP:-nenhum}"
  echo "Último job TFB:    ${TFB_DEP:-nenhum}"
  echo "Registro:          $PARALELO_ROOT/$SUBMISSION_LOG"
  echo "Acompanhe com:     squeue -u $USER"
  echo "============================================================"
} | tee -a "$SUBMISSION_LOG"
