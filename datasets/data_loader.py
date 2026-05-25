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
        self.meta_columns = []
        self.meta_frame = None
        self.feature_columns = []
        
        self.__read_data__(root_path, data_path)
        
    def __read_data__(self, root_path, data_path):
        df_raw = pd.read_csv(os.path.join(root_path, data_path))
        df_raw.columns = [str(c).strip() for c in df_raw.columns]

        # Suporta formato longo: date (tempo), col (id da série), data (valor).
        required_long_cols = {"date", "col", "data"}
        lowered_cols = {c.lower(): c for c in df_raw.columns}
        if required_long_cols.issubset(set(lowered_cols.keys())):
            date_col = lowered_cols["date"]
            id_col = lowered_cols["col"]
            value_col = lowered_cols["data"]
            df_long = df_raw[[date_col, id_col, value_col]].copy()
            df_long[date_col] = pd.to_datetime(df_long[date_col], errors="coerce")
            df_long[value_col] = pd.to_numeric(df_long[value_col], errors="coerce")
            df_long = df_long.dropna(subset=[date_col, id_col, value_col])

            # Pivot para formato multivariado (linhas = datas, colunas = séries).
            df_pivot = (
                df_long.pivot_table(
                    index=date_col,
                    columns=id_col,
                    values=value_col,
                    aggfunc="last",
                )
                .sort_index()
                .reset_index()
            )
            df_raw = df_pivot
        
        lowered_cols = {c.lower(): c for c in df_raw.columns}

        # Seleção de colunas (M = multivariate, S = univariate)
        if self.features == 'M' or self.features == 'MS':
            # Skip first column if it's 'date' or index
            cols_data = [col for col in df_raw.columns if col.lower() not in ['date', 'data', 'col']]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
        else:
            df_data = df_raw.iloc[:, 1:]

        # Separa metadados não numéricos (ex.: ticker/canal) das features.
        # As features continuam numéricas para o modelo, mas os metadados
        # ficam disponíveis para rastreamento/identificação posterior.
        df_numeric = df_data.apply(pd.to_numeric, errors='coerce')
        valid_numeric_cols = df_numeric.columns[~df_numeric.isna().all(axis=0)]
        self.feature_columns = valid_numeric_cols.tolist()

        self.meta_columns = [col for col in df_data.columns if col not in self.feature_columns]
        if "date" in lowered_cols:
            self.meta_columns = list(dict.fromkeys([lowered_cols["date"]] + self.meta_columns))
        if self.meta_columns:
            # Busca metadados do df original para manter colunas de identificação.
            cols_available = [c for c in self.meta_columns if c in df_raw.columns]
            self.meta_frame = df_raw[cols_available].copy()
        else:
            self.meta_frame = None

        df_numeric = df_numeric[self.feature_columns]

        if df_numeric.shape[1] == 0:
            raise ValueError(
                "Nenhuma coluna numérica encontrada no dataset após o pré-processamento. "
                "Verifique `features`, `target` e o conteúdo do CSV."
            )

        # Preenche lacunas de conversão sem quebrar a sequência temporal.
        df_numeric = df_numeric.ffill().bfill().fillna(0.0)
        data = df_numeric.values.astype(np.float32)
        
        # Normalização manual (sem scikit-learn)
        if self.scale:
            self.mean = data.mean(axis=0)
            self.std = data.std(axis=0) + 1e-5
            data = (data - self.mean) / self.std
        
        self.data_x = data
        self.data_y = data  # para forecasting supervisionado

    def get_metadata_window(self, index):
        """Retorna metadados (não numéricos) para a mesma janela de seq_x."""
        if self.meta_frame is None:
            return None

        max_index = len(self.meta_frame) - self.seq_len - self.pred_len
        index = min(max(0, index), max_index)
        s_begin = index
        s_end = s_begin + self.seq_len
        return self.meta_frame.iloc[s_begin:s_end].reset_index(drop=True)
    
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
