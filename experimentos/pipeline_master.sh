#!/bin/bash

# Orquestrador do pipeline MASTER.
# Execute no nó de login: bash experimentos/pipeline_master.sh
#
# Para pular uma etapa, comente a linha correspondente abaixo.
ENABLED_STAGES=(
  estimacao
  carteiras
  comparacao_master
  comparacao_global
)

set -euo pipefail

PARALELO_ROOT="${PARALELO_ROOT:-/sonic_home/igor.viveiros/paralelo}"
ESTIMATION_ARRAY="${ESTIMATION_ARRAY:-0-107%8}"
BACKTEST_ARRAY="${BACKTEST_ARRAY:-0-107%16}"
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
      estimacao|carteiras|comparacao_master|comparacao_global) ;;
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

validate_stages

if ((${#ENABLED_STAGES[@]} == 0)); then
  echo "Nenhuma etapa habilitada."
  exit 0
fi

STAMP=$(date +%Y%m%d_%H%M%S)
SUBMISSION_LOG="logs/pipeline-master-${STAMP}.log"
LAST_JOB=""

{
  echo "============================================================"
  echo "Pipeline MASTER"
  echo "Data:                 $(date --iso-8601=seconds)"
  echo "Etapas:               ${ENABLED_STAGES[*]}"
  echo "Array estimação:      $ESTIMATION_ARRAY"
  echo "Array carteiras:      $BACKTEST_ARRAY"
  echo "Workers comparações:  $COMPARISON_WORKERS"
  echo "Dry run:              $DRY_RUN"
  echo "============================================================"
} | tee "$SUBMISSION_LOG"

if stage_enabled estimacao; then
  JOB_ESTIMATION=$(submit_job "$LAST_JOB" \
    --array="$ESTIMATION_ARRAY" \
    --export=ALL,RUN_ESTIMATION=true,RUN_BACKTEST=false \
    experimentos/carteiras_master_array.sh)
  [[ "$JOB_ESTIMATION" != "DRYRUN" ]] && LAST_JOB="$JOB_ESTIMATION"
  echo "estimacao=$JOB_ESTIMATION" | tee -a "$SUBMISSION_LOG"
else
  echo "estimacao=PULADA" | tee -a "$SUBMISSION_LOG"
fi

if stage_enabled carteiras; then
  JOB_BACKTEST=$(submit_job "$LAST_JOB" \
    --array="$BACKTEST_ARRAY" \
    experimentos/carteiras_master_backtest_array.sh)
  [[ "$JOB_BACKTEST" != "DRYRUN" ]] && LAST_JOB="$JOB_BACKTEST"
  echo "carteiras=$JOB_BACKTEST" | tee -a "$SUBMISSION_LOG"
else
  echo "carteiras=PULADA" | tee -a "$SUBMISSION_LOG"
fi

if stage_enabled comparacao_master; then
  JOB_MASTER_COMPARISON=$(submit_job "$LAST_JOB" \
    --cpus-per-task="$COMPARISON_WORKERS" \
    --export=ALL,WORKERS="$COMPARISON_WORKERS" \
    simulacoes/gera_comparacoes_paralelo.sh)
  [[ "$JOB_MASTER_COMPARISON" != "DRYRUN" ]] && LAST_JOB="$JOB_MASTER_COMPARISON"
  echo "comparacao_master=$JOB_MASTER_COMPARISON" | tee -a "$SUBMISSION_LOG"
else
  echo "comparacao_master=PULADA" | tee -a "$SUBMISSION_LOG"
fi

if stage_enabled comparacao_global; then
  JOB_GLOBAL_COMPARISON=$(submit_job "$LAST_JOB" \
    --cpus-per-task="$COMPARISON_WORKERS" \
    --export=ALL,WORKERS="$COMPARISON_WORKERS" \
    simulacoes/gera_comparativo_global_paralelo.sh)
  [[ "$JOB_GLOBAL_COMPARISON" != "DRYRUN" ]] && LAST_JOB="$JOB_GLOBAL_COMPARISON"
  echo "comparacao_global=$JOB_GLOBAL_COMPARISON" | tee -a "$SUBMISSION_LOG"
else
  echo "comparacao_global=PULADA" | tee -a "$SUBMISSION_LOG"
fi

{
  echo "============================================================"
  echo "Submissão concluída."
  echo "Último job da cadeia: ${LAST_JOB:-nenhum}"
  echo "Registro: $PARALELO_ROOT/$SUBMISSION_LOG"
  echo "Acompanhe com: squeue -u $USER"
  echo "============================================================"
} | tee -a "$SUBMISSION_LOG"
