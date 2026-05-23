import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os

class TimeSeriesDataset(Dataset):
    """Dataset compatível com TFB e modelos TIOMS."""
    def __init__(self, root_path, data_path, flag='train', size=None,
                 features='M', target='OT', scale=True, timeenc=0,
                 freq='h', seasonal_patterns=None):
        # size = [seq_len, label_len, pred_len]
        self.seq_len, self.label_len, self.pred_len = size
        self.features = features
        self.target = target
        self.scale = scale
        self.flag = flag
        
        self.__read_data__(root_path, data_path)
        
    def __read_data__(self, root_path, data_path):
        df_raw = pd.read_csv(os.path.join(root_path, data_path))
        
        # Seleção de colunas (M = multivariate, S = univariate)
        if self.features == 'M' or self.features == 'MS':
            # Skip first column if it's 'date' or index
            cols_data = [col for col in df_raw.columns if col.lower() not in ['date', 'data']]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
        else:
            df_data = df_raw.iloc[:, 1:]
        
        data = df_data.values.astype(np.float32)
        
        # Normalização manual (sem scikit-learn)
        if self.scale:
            self.mean = data.mean(axis=0)
            self.std = data.std(axis=0) + 1e-5
            data = (data - self.mean) / self.std
        
        self.data_x = data
        self.data_y = data  # para forecasting supervisionado
    
    def __getitem__(self, index):
        # Proteção contra índices inválidos
        max_index = len(self.data_x) - self.seq_len - self.pred_len
        index = min(max(0, index), max_index)
        
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        
        # Garantia de shape correto
        seq_x = np.pad(seq_x, ((0, self.seq_len - len(seq_x)), (0, 0)), mode='edge') if len(seq_x) < self.seq_len else seq_x
        seq_y = np.pad(seq_y, ((0, self.label_len + self.pred_len - len(seq_y)), (0, 0)), mode='edge') if len(seq_y) < self.label_len + self.pred_len else seq_y
        
        return torch.tensor(seq_x, dtype=torch.float32), torch.tensor(seq_y, dtype=torch.float32)
    
    def __len__(self):
        return max(1, len(self.data_x) - self.seq_len - self.pred_len + 1)