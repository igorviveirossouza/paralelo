import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    """
    DataLoader Multivariate compatível com modelos que esperam (B, T, N)
    """
    def __init__(self, root_path, data_path, seq_len=96, pred_len=24,
                 scale=True, cols=None):
        self.root_path = root_path
        self.data_path = data_path
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.scale = scale
        self.cols = cols  # None = multivariate

        self.__read_data__()

    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        
        print("Colunas originais:", df_raw.columns.tolist())
        print("Tickers únicos:", df_raw['cols'].nunique())

        if self.cols is not None:
            # Univariate
            df = df_raw[df_raw['cols'] == self.cols].copy()
            print(f"Modo Univariate - Ticker: {self.cols}")
        else:
            # Multivariate (principal modo)
            df = df_raw.pivot(index='date', columns='cols', values='data').reset_index()
            df = df.sort_values('date').fillna(method='ffill')
            print(f"Modo Multivariate - {df.shape[1]-1} séries")

        numeric_cols = [col for col in df.columns if col != 'date']
        self.data = df[numeric_cols].values.astype(np.float32)  # (T, N)
        self.num_series = self.data.shape[1]

        print(f"✅ Dataset carregado | Shape: {self.data.shape} | Séries (N): {self.num_series}\n")

    def __len__(self):
        return max(0, len(self.data) - self.seq_len - self.pred_len + 1)

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end + 1 
        r_end = r_begin + self.pred_len

        seq_x = self.data[s_begin:s_end]      # (T, N)
        seq_y = self.data[r_begin:r_end]      # (pred_len, N)

        # Retorna no formato esperado por modelos Transformer/MLP
        return seq_x, seq_y