
### Estrutura do projeto
```bash
my_tsf_project/
├── data/                     # Datasets (ETTh1, ETTm1, Weather, Electricity, etc.)
├── datasets/                 # Data loaders PyTorch
│   ├── __init__.py
│   ├── data_loader.py        # Principal (herdado/adaptado do TFB)
│   └── custom_datasets.py
├── models/                   # Modelos TIOMS adaptados
│   ├── __init__.py
│   ├── tioms/                # Copie/adapte os modelos daqui
│   └── base_model.py
├── trainer/                  # Loop de treino
│   ├── __init__.py
│   ├── trainer.py
│   └── early_stopping.py
├── forecaster/               # Inferência e avaliação
│   ├── __init__.py
│   └── forecaster.py
├── utils/
│   ├── metrics.py            # MAE, MSE, SMAPE, CRPS etc.
│   ├── visualization.py
│   └── normalization.py      # RevIN, InstanceNorm etc. (crucial para non-stationarity)
├── configs/                  # JSON/YAML por dataset + horizon
├── experiments/              # Scripts de execução
├── logs/
└── main.py
```
