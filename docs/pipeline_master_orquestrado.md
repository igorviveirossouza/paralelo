# Pipeline MASTER orquestrado

O controlador `experimentos/pipeline_master.sh` submete as etapas ao Slurm e liga os jobs por dependências `afterok`.

## Etapas

1. `estimacao`: treina o MASTER e gera as previsões em GPU.
2. `carteiras`: lê as previsões e recalcula as carteiras em CPU, usando `model_output=score` e sem filtro de sinal.
3. `comparacao_master`: consolida as métricas das simulações do MASTER.
4. `comparacao_global`: gera os arquivos finais de comparação entre MASTER, TFB, RandomTopJ e benchmarks, incluindo AUCs.

## Seleção das etapas

No início de `experimentos/pipeline_master.sh`:

```bash
ENABLED_STAGES=(
  estimacao
  carteiras
  comparacao_master
  comparacao_global
)
```

Para começar pelas carteiras, comente `estimacao`:

```bash
ENABLED_STAGES=(
  # estimacao
  carteiras
  comparacao_master
  comparacao_global
)
```

Para gerar apenas as comparações finais:

```bash
ENABLED_STAGES=(
  # estimacao
  # carteiras
  comparacao_master
  comparacao_global
)
```

## Execução

```bash
cd /sonic_home/igor.viveiros/paralelo
git checkout pipeline-master-orquestrado
git pull --ff-only origin pipeline-master-orquestrado
bash experimentos/pipeline_master.sh
```

## Teste sem submeter jobs

```bash
DRY_RUN=true bash experimentos/pipeline_master.sh
```

## Ajustes opcionais

```bash
ESTIMATION_ARRAY='0-107%4' \
BACKTEST_ARRAY='0-107%16' \
COMPARISON_WORKERS=16 \
bash experimentos/pipeline_master.sh
```

O script grava os IDs dos jobs em `logs/pipeline-master-YYYYMMDD_HHMMSS.log`.
