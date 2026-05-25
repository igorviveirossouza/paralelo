import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    def __init__(self, root_path, data_path, features=['data'], target='data',
                 seq_len=96, label_len=48, pred_len=24, scale=True, 
                 timeenc=0, freq='d', ticker=None):
        self.root_path = root_path
        self.data_path = data_path
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.scale = scale
        self.ticker = ticker  # None = usa todas as séries (multivariate)
        
        self.__read_data__(root_path, data_path)

    def __read_data__(self, root_path, data_path):
        df_raw = pd.read_csv(os.path.join(root_path, data_path))
        
        # Renomeia para facilitar
        df_raw = df_raw.rename(columns={'data': 'value', 'cols': 'ticker', 'date': 'date'})
        
        print("Colunas:", df_raw.columns.tolist())
        print("Tickers únicos:", df_raw['ticker'].nunique())
        print("Exemplo de tickers:", df_raw['ticker'].unique()[:5].tolist())

        # === Lógica Multi-Série ===
        if self.ticker is not None:
            # Usa apenas um ticker (univariate)
            df = df_raw[df_raw['ticker'] == self.ticker].copy()
        else:
            # Usa todas as séries → Multivariate (uma coluna por ticker)
            df = df_raw.pivot(index='date', columns='ticker', values='value').reset_index()
            df = df.sort_values('date').fillna(method='ffill')
        
        numeric_cols = [col for col in df.columns if col != 'date']
        self.data = df[numeric_cols].values.astype(np.float32)  # (T, num_series)
        
        print(f"✅ Dataset carregado | Shape final: {self.data.shape} | Séries: {len(numeric_cols)}")
        self.num_series = self.data.shape[1]

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len 
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data[s_begin:s_end]      # (seq_len, num_series)
        seq_y = self.data[r_begin:r_end]      # (label+pred_len, num_series)

        return seq_x, seq_y