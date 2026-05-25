import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os

import os
import pandas as pd
import numpy as np

class TimeSeriesDataset:
    def __init__(self, root_path, data_path, features=['data'], target='data', 
                 seq_len=96, label_len=48, pred_len=24, scale=True, timeenc=0, freq='d'):
        self.root_path = root_path
        self.data_path = data_path
        self.features = features
        self.target = target
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        
        self.__read_data__(root_path, data_path)

    def __read_data__(self, root_path, data_path):
        df_raw = pd.read_csv(os.path.join(root_path, data_path))
        
        print("Colunas originais:", df_raw.columns.tolist())
        print("Tipos de dados:\n", df_raw.dtypes)
        print("Primeiras 5 linhas:\n", df_raw.head())

        # === AJUSTE ESPECÍFICO PARA b3_daily_financeiro.csv ===
        if 'data' in df_raw.columns:
            df_raw = df_raw.rename(columns={'data': 'target_value'})
            numeric_cols = ['target_value']
        else:
            # fallback genérico
            numeric_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) == 0:
            raise ValueError(
                f"Nenhuma coluna numérica encontrada. Colunas disponíveis: {df_raw.columns.tolist()}\n"
                f"Verifique se o CSV tem colunas como 'data', 'price', etc."
            )

        # Selecionar apenas as colunas numéricas desejadas
        self.data = df_raw[numeric_cols].values.astype(np.float32)
        
        # Manter timestamp se existir
        if 'date' in df_raw.columns:
            self.date = pd.to_datetime(df_raw['date'], errors='coerce')
        else:
            self.date = None
            
        self.ticker = df_raw['cols'].values if 'cols' in df_raw.columns else None

        print(f"✅ Dataset carregado com sucesso!")
        print(f"   Shape: {self.data.shape}")
        print(f"   Colunas numéricas usadas: {numeric_cols}")