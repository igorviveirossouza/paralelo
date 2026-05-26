import torch
from torch.utils.data import Dataset
import pandas as pd

class TimeSeriesDataset(Dataset):
    """
    Dataset multivariado com múltiplos tickers, sem padding no target.
    Prioriza informações recentes (sem padding à direita).
    """
    def __init__(self, data_path, lookback=96, pred_len=24, stride=1, cols=None):
        super().__init__()
        self.lookback = lookback
        self.horizon = pred_len
        self.stride = stride
        self.cols = cols  # None = multivariate, str = univariate

        df = pd.read_csv(data_path)
        print("Colunas originais:", df.columns.tolist())

        # Pivot para formato [time, channels]
        df_pivot = df.pivot(index='date', columns='cols', values='data')
        df_pivot = df_pivot.fillna(method='ffill').fillna(method='bfill')
        
        if cols is None:
            self.data = torch.tensor(df_pivot.values, dtype=torch.float32)  # [T, N=9]
            self.feature_columns = df_pivot.columns.tolist()
            self.mode = "multivariate"
            print(f"✅ Modo Multivariate - {self.data.shape[1]} séries | Shape: {self.data.shape}")
        
        else:
            # Modo Univariate
            if cols not in df_pivot.columns:
                raise ValueError(f"Coluna '{cols}' não encontrada. Disponíveis: {list(df_pivot.columns)}")
            self.data = torch.tensor(df_pivot[cols].values, dtype=torch.float32).unsqueeze(1)  # [T, 1]
            self.feature_columns = [cols]
            self.mode = "univariate"
            print(f"✅ Modo Univariate - Ticker: {cols} | Shape: {self.data.shape}")    
            
        # Gerar apenas índices válidos (sem janelas truncadas)
        self.indices = []
        T = len(self.data)
        max_start = T - lookback - pred_len   # garante y completo
        
        for start in range(0, max_start + 1, stride):
            self.indices.append(start)
            
        print(f"Total de amostras válidas (sem truncamento): {len(self.indices)}")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        start = self.indices[idx]
        x = self.data[start : start + self.lookback]      # [96, 9]
        y = self.data[start + self.lookback : start + self.lookback + self.horizon]  # [24, 9]
        
        # Assert para debug (pode remover depois)
        assert len(y) == self.horizon, f"Janela truncada! idx={idx}, y_len={len(y)}"
        
        return x, y
    
    @property                        # Atributos a serem guardados
    def feature_columns(self):
        return self.feature_columns  # já definido no __init__
    
    def get_metadata_window(self, idx):
        """Retorna None por enquanto (pode expandir depois se precisar de metadata)"""
        return None