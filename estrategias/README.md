# Estratégias

Contém backtests de carteiras baseados em rankings de previsões.

## Ranking top-j

Entrada esperada: uma pasta com arquivos `janela_*.csv` no formato `h x papel`, como:

- linhas: horizontes futuros;
- colunas: papéis;
- coluna `step`: índice/data do alvo previsto.

Exemplo para modelo treinado em retornos:

```bash
python -m estrategias.ranking_backtest \
  --pred_dir previsoes/meu_modelo \
  --price_path data/b3_daily_financeiro.csv \
  --model_output returns \
  --rebalance_k 5 \
  --max_assets 10 \
  --horizon 24
```

Exemplo para modelo treinado em preços:

```bash
python -m estrategias.ranking_backtest \
  --pred_dir previsoes/meu_modelo \
  --price_path data/b3_daily_financeiro.csv \
  --model_output prices \
  --rebalance_k 5 \
  --max_assets 10 \
  --horizon 24
```

Os resultados são salvos em `simulacoes/`.
