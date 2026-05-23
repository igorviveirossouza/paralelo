# datasets/data_loader.py
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
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
        
        self.scaler = StandardScaler()
        self.__read_data__(root_path, data_path)
        
    def __read_data__(self, root_path, data_path):
        df_raw = pd.read_csv(os.path.join(root_path, data_path))
        
        # Seleção de colunas (M = multivariate, S = univariate)
        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:] if 'date' in df_raw.columns else df_raw.columns
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
        
        if self.scale:
            train_data = df_data  # ajuste para split real
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        
        self.data_x = data
        self.data_y = data  # para forecasting supervisionado
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        
        return torch.tensor(seq_x, dtype=torch.float32), torch.tensor(seq_y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1
