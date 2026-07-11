# Pipeline completo MASTER + TFB

O controlador `experimentos/pipeline_completo.sh` submete dois fluxos independentes ao Slurm:

```text
MASTER: estimação -> carteiras -> comparação MASTER -----\
                                                        -> comparação global
TFB:    estimação -> conversão -> carteiras -> comparação TFB --/
```

Os jobs são ligados por dependências `afterok`. MASTER e TFB podem executar em paralelo; a comparação global aguarda os últimos jobs habilitados dos dois fluxos.

## Etapas disponíveis

- `master_estimacao`: treina o MASTER em GPU e gera `janela_*.csv`.
- `master_carteiras`: forma carteiras em CPU com `model_output=score`.
- `tfb_estimacao`: chama o repositório externo em `/sonic_home/igor.viveiros/src/TFB` e gera previsões brutas.
- `tfb_conversao`: converte a saída do TFB para `janela_*.csv` no formato do repositório paralelo.
- `tfb_carteiras`: forma as carteiras para cada janela de rebalanceamento compatível com `pred_len`.
- `comparacao_master`: consolida as métricas do MASTER.
- `comparacao_tfb`: consolida as métricas do TFB.
- `comparacao_global`: gera rankings, resumos, AUCs e arquivos finais de MASTER, TFB, RandomTopJ e benchmarks.

## Grade padrão do TFB

```text
Datasets:  retornos_simples, log_retornos, prices
Modelos:   DUET, TimesNet, FEDformer, Nonstationary_Transformer
Lookbacks: 32, 104, 246
Pred lens: 1, 5, 10, 15, 24
```

Total:

```text
3 x 4 x 3 x 5 = 180 tarefas
```

As janelas de rebalanceamento são `1, 5, 10, 15, 20, 24`, usando apenas `k <= pred_len`.

## Selecionar etapas

Edite o início de `experimentos/pipeline_completo.sh`:

```bash
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
```

### Rodar somente o TFB desde a estimação

```bash
ENABLED_STAGES=(
  # master_estimacao
  # master_carteiras
  tfb_estimacao
  tfb_conversao
  tfb_carteiras
  # comparacao_master
  comparacao_tfb
  comparacao_global
)
```

### Usar previsões brutas do TFB já existentes

```bash
ENABLED_STAGES=(
  # master_estimacao
  # master_carteiras
  # tfb_estimacao
  tfb_conversao
  tfb_carteiras
  # comparacao_master
  comparacao_tfb
  comparacao_global
)
```

### Usar previsões convertidas já existentes

```bash
ENABLED_STAGES=(
  # master_estimacao
  # master_carteiras
  # tfb_estimacao
  # tfb_conversao
  tfb_carteiras
  # comparacao_master
  comparacao_tfb
  comparacao_global
)
```

### Gerar apenas comparações finais

```bash
ENABLED_STAGES=(
  # master_estimacao
  # master_carteiras
  # tfb_estimacao
  # tfb_conversao
  # tfb_carteiras
  comparacao_master
  comparacao_tfb
  comparacao_global
)
```

## Execução

```bash
cd /sonic_home/igor.viveiros/paralelo
git fetch origin
git checkout pipeline-master-orquestrado
git pull --ff-only origin pipeline-master-orquestrado

DRY_RUN=true bash experimentos/pipeline_completo.sh
bash experimentos/pipeline_completo.sh
```

## Arquivos do fluxo TFB

- `experimentos/tfb_pipeline_config.sh`: grade e resolução de cada `TASK_ID`.
- `experimentos/tfb_estimacao_array.sh`: execução dos modelos no repositório TFB.
- `experimentos/tfb_conversao_array.sh`: conversão das previsões.
- `experimentos/tfb_carteiras_array.sh`: backtests e métricas financeiras.
- `simulacoes/gera_comparacoes_tfb_paralelo.sh`: compilação paralela das métricas.

## Diretórios padrão

```text
Previsões brutas:     previsoes/tfb_multi_lb_predlen_carteiras/_tfb_decoded/
Previsões convertidas: previsoes/tfb_multi_lb_predlen_carteiras/
Carteiras e métricas: simulacoes/tfb_multi_lb_predlen_carteiras/
Comparativo global:  simulacoes/comparativo_global_master_tfb/
```

## Ajustes por variáveis de ambiente

```bash
TFB_ROOT=/sonic_home/igor.viveiros/src/TFB \
TFB_ESTIMATION_ARRAY='0-179%4' \
TFB_CONVERSION_ARRAY='0-179%16' \
TFB_BACKTEST_ARRAY='0-179%16' \
COMPARISON_WORKERS=16 \
bash experimentos/pipeline_completo.sh
```

A configuração da grade também pode ser alterada com `TFB_MODELS`, `TFB_LOOKBACKS`, `TFB_PRED_LENS` e `TFB_REBALANCE_WINDOWS`. Nesse caso, ajuste os intervalos dos arrays para o novo número de combinações.
