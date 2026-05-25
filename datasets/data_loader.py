import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
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

        # === Tratamento específico para b3_daily_financeiro.csv ===
        df_raw = df_raw.rename(columns={'data': 'target_value'})
        numeric_cols = ['target_value']

        if len(numeric_cols) == 0:
            raise ValueError(f"Nenhuma coluna numérica encontrada. Colunas: {df_raw.columns.tolist()}")

        self.data = df_raw[numeric_cols].values.astype(np.float32)   # shape: (N, 1)
        self.timestamps = df_raw['date'].values if 'date' in df_raw.columns else None
        self.tickers = df_raw['cols'].values if 'cols' in df_raw.columns else None

        print(f"✅ Dataset carregado com sucesso!")
        print(f"   Shape: {self.data.shape}")
        print(f"   Colunas numéricas usadas: {numeric_cols}\n")

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data[s_begin:s_end]           # input
        seq_y = self.data[r_begin:r_end]           # target (label + pred)

        # Converter para tensor
        seq_x = np.array(seq_x, dtype=np.float32)
        seq_y = np.array(seq_y, dtype=np.float32)

        return seq_x, seq_y   # (seq_len, 1), (label_len + pred_len, 1)