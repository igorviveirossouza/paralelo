from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd


DEFAULT_CANDLE_COLS = ["abertura", "maxima", "minima", "data", "volume"]
DEFAULT_MARKET_WINDOWS = [5, 10, 20, 30, 60]


class TimeSeriesDataset(Dataset):
    """
    Dataset flexível para o projeto paralelo.
    Suporta modo Multivariate e Univariate + split Train/Test.

    Quando use_candle_encoder=True, retorna também candle_x:
        x:        [lookback, N]
        y:        [pred_len, N]
        candle_x: [lookback, N, F]

    Quando use_market_features=True, retorna também market_x:
        market_x: [lookback, K]
    """
    def __init__(self, data_path, lookback=96, pred_len=24, stride=1,
                 cols=None, train=True, test_ratio=0.2,
                 use_candle_encoder=False, candle_cols=None,
                 candle_feature_mode="ohlcv_relative",
                 use_market_features=False, market_feature_files=None,
                 market_feature_mode="master", market_windows=None,
                 market_date_col=None):
        super().__init__()
        self.lookback = lookback
        self.horizon = pred_len
        self.stride = stride
        self.cols = cols          # None = multivariate, str = ticker específico
        self.train = train
        self.use_candle_encoder = use_candle_encoder
        self.candle_cols = candle_cols or DEFAULT_CANDLE_COLS
        self.candle_feature_mode = candle_feature_mode
        self.candle_feature_names = []
        self.candle_data = None
        self.use_market_features = bool(use_market_features)
        self.market_feature_files = market_feature_files or []
        self.market_feature_mode = market_feature_mode
        self.market_windows = market_windows or DEFAULT_MARKET_WINDOWS
        self.market_date_col = market_date_col
        self.market_feature_names = []
        self.market_data = None
        self.date_index = []

        df = pd.read_csv(data_path)
        print("Colunas originais:", df.columns.tolist())

        required = {"date", "cols", "data"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"Colunas obrigatórias ausentes no CSV: {sorted(missing)}")

        df = df.copy()
        df["date"] = self._normalize_dates(df["date"])
        df["data"] = pd.to_numeric(df["data"], errors="coerce")
        df = df.sort_values(["cols", "date"])

        df_pivot = df.pivot(index="date", columns="cols", values="data")
        df_pivot = df_pivot.ffill().bfill().fillna(0.0)
        self.date_index = df_pivot.index.tolist()

        candle_np = None
        if use_candle_encoder:
            candle_frame, self.candle_feature_names = self._build_candle_features(df)
            candle_pivots = []
            for feature_name in self.candle_feature_names:
                feature_pivot = candle_frame.pivot(index="date", columns="cols", values=feature_name)
                feature_pivot = feature_pivot.reindex(index=df_pivot.index, columns=df_pivot.columns)
                feature_pivot = feature_pivot.ffill().bfill().fillna(0.0)
                candle_pivots.append(feature_pivot.values)

            candle_np = np.stack(candle_pivots, axis=-1).astype("float32")  # [T, N, F]
            print(
                f"✅ Candle Encoder ativo | features: {self.candle_feature_names} | "
                f"Shape OHLCV: {candle_np.shape}"
            )

        market_np = None
        if self.use_market_features:
            market_frame, self.market_feature_names = self._build_market_features(df_pivot.index)
            market_frame = market_frame.reindex(df_pivot.index)
            market_frame = market_frame.ffill().bfill().fillna(0.0)
            market_np = market_frame.values.astype("float32")  # [T, K]
            print(
                f"✅ Features de mercado ativas | K={len(self.market_feature_names)} | "
                f"Shape market: {market_np.shape}"
            )

        if cols is None:
            # === MULTIVARIATE ===
            self.data = torch.tensor(df_pivot.values, dtype=torch.float32)  # [T, N]
            self.feature_columns = df_pivot.columns.tolist()
            if candle_np is not None:
                self.candle_data = torch.tensor(candle_np, dtype=torch.float32)
            self.mode = "multivariate"
            print(f"✅ Modo Multivariate - {len(self.feature_columns)} séries | Shape: {self.data.shape}")
        else:
            # === UNIVARIATE ===
            if cols not in df_pivot.columns:
                raise ValueError(f"Coluna '{cols}' não encontrada. Disponíveis: {list(df_pivot.columns)}")
            col_idx = df_pivot.columns.get_loc(cols)
            self.data = torch.tensor(df_pivot[cols].values, dtype=torch.float32).unsqueeze(1)  # [T, 1]
            self.feature_columns = [cols]
            if candle_np is not None:
                self.candle_data = torch.tensor(candle_np[:, col_idx:col_idx + 1, :], dtype=torch.float32)
            self.mode = "univariate"
            print(f"✅ Modo Univariate - Ticker: {cols} | Shape: {self.data.shape}")

        if market_np is not None:
            self.market_data = torch.tensor(market_np, dtype=torch.float32)

        # === SPLIT TRAIN / TEST ===
        T = len(self.data)
        test_size = int(T * test_ratio)
        split_idx = T - test_size

        self.indices = []

        if train:
            # Janelas completamente dentro do treino
            max_start = split_idx - lookback - pred_len
            for start in range(0, max(0, max_start) + 1, stride):
                self.indices.append(start)
            print(f"✅ Train split | amostras: {len(self.indices)} | split_idx={split_idx}")
        else:
            # Janelas que começam a partir do split (usam dados de teste)
            start_min = max(0, split_idx - lookback)
            for start in range(start_min, T - lookback - pred_len + 1, stride):
                self.indices.append(start)
            print(f"✅ Test split | amostras: {len(self.indices)} | início global ≈ {split_idx}")

        print(f"Total de amostras válidas ({'train' if train else 'test'}): {len(self.indices)}")

    @staticmethod
    def _normalize_dates(values):
        series = pd.Series(values)
        parsed = pd.to_datetime(series, errors="coerce")
        out = series.astype(str)
        ok = parsed.notna()
        if ok.any():
            out.loc[ok] = parsed.loc[ok].dt.strftime("%Y-%m-%d")
        return out.tolist()

    @staticmethod
    def _safe_log_ratio(numerator, denominator, eps=1e-8):
        numerator = pd.to_numeric(numerator, errors="coerce").clip(lower=eps)
        denominator = pd.to_numeric(denominator, errors="coerce").clip(lower=eps)
        return np.log(numerator / denominator)

    def _build_candle_features(self, df):
        mode = self.candle_feature_mode.lower()
        work = df.copy()

        if mode == "raw":
            missing = [col for col in self.candle_cols if col not in work.columns]
            if missing:
                raise ValueError(f"Colunas de candle ausentes no CSV: {missing}")

            feature_names = []
            for col in self.candle_cols:
                feature_name = f"candle_{col}"
                work[feature_name] = pd.to_numeric(work[col], errors="coerce")
                feature_names.append(feature_name)

            work[feature_names] = (
                work.groupby("cols")[feature_names]
                .transform(lambda s: s.ffill().bfill())
                .fillna(0.0)
            )
            return work[["date", "cols", *feature_names]], feature_names

        if mode == "ohlcv_relative":
            missing = [col for col in DEFAULT_CANDLE_COLS if col not in work.columns]
            if missing:
                raise ValueError(
                    "Modo ohlcv_relative requer as colunas "
                    f"{DEFAULT_CANDLE_COLS}. Ausentes: {missing}"
                )

            for col in DEFAULT_CANDLE_COLS:
                work[col] = pd.to_numeric(work[col], errors="coerce")

            work[DEFAULT_CANDLE_COLS] = (
                work.groupby("cols")[DEFAULT_CANDLE_COLS]
                .transform(lambda s: s.ffill().bfill())
                .fillna(0.0)
            )

            prev_close = work.groupby("cols")["data"].shift(1)
            prev_close = prev_close.fillna(work["abertura"]).fillna(work["data"])

            work["candle_open_rel"] = self._safe_log_ratio(work["abertura"], prev_close)
            work["candle_high_rel"] = self._safe_log_ratio(work["maxima"], work["abertura"])
            work["candle_low_rel"] = self._safe_log_ratio(work["minima"], work["abertura"])
            work["candle_close_rel"] = self._safe_log_ratio(work["data"], work["abertura"])
            work["candle_log_volume"] = np.log1p(pd.to_numeric(work["volume"], errors="coerce").clip(lower=0.0))

            feature_names = [
                "candle_open_rel",
                "candle_high_rel",
                "candle_low_rel",
                "candle_close_rel",
                "candle_log_volume",
            ]
            work[feature_names] = work[feature_names].replace([np.inf, -np.inf], np.nan).fillna(0.0)
            return work[["date", "cols", *feature_names]], feature_names

        raise ValueError(f"candle_feature_mode inválido: {self.candle_feature_mode}")

    def _infer_market_date_col(self, df):
        if self.market_date_col is not None:
            if self.market_date_col not in df.columns:
                raise ValueError(f"Coluna de data '{self.market_date_col}' ausente no arquivo de mercado.")
            return self.market_date_col
        for col in ["date_pregao", "date", "datetime"]:
            if col in df.columns:
                return col
        raise ValueError("Arquivo de mercado precisa conter date_pregao, date ou datetime.")

    def _build_market_features(self, target_index):
        if not self.market_feature_files:
            raise ValueError("use_market_features=True requer pelo menos um arquivo em market_feature_files.")

        frames = []
        all_names = []
        for file_path in self.market_feature_files:
            frame, names = self._build_market_features_one(file_path)
            frames.append(frame)
            all_names.extend(names)

        market = pd.concat(frames, axis=1)
        market = market.replace([np.inf, -np.inf], np.nan)
        market = market.sort_index()
        target_index = pd.Index(target_index, name="date")
        market = market.reindex(target_index).ffill().bfill().fillna(0.0)
        return market, all_names

    def _build_market_features_one(self, file_path):
        mode = self.market_feature_mode.lower()
        path = Path(file_path)
        stem = path.stem
        df = pd.read_csv(path)
        date_col = self._infer_market_date_col(df)
        df = df.copy()
        df["date"] = self._normalize_dates(df[date_col])
        df = df.sort_values("date").drop_duplicates("date", keep="last")

        if mode == "raw":
            return self._build_raw_market_features(df, stem)

        if mode == "master":
            features = self._build_master_market_features(df, stem)
            if features is not None:
                return features
            return self._build_raw_market_features(df, stem)

        raise ValueError(f"market_feature_mode inválido: {self.market_feature_mode}")

    def _build_raw_market_features(self, df, stem):
        numeric_cols = []
        for col in df.columns:
            if col in {"date", "date_pregao", "datetime"}:
                continue
            values = pd.to_numeric(df[col], errors="coerce")
            if values.notna().any():
                numeric_cols.append(col)

        if not numeric_cols:
            raise ValueError("Arquivo de features de mercado não possui colunas numéricas utilizáveis.")

        out = pd.DataFrame(index=pd.Index(df["date"], name="date"))
        names = []
        for col in numeric_cols:
            name = f"{stem}__{col}"
            out[name] = pd.to_numeric(df[col], errors="coerce")
            names.append(name)

        out = out.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
        return out, names

    def _build_master_market_features(self, df, stem):
        close_cols = [col for col in df.columns if col.endswith("_Close")]
        if not close_cols:
            close_cols = [col for col in df.columns if col.endswith("_Adj Close")]
        if not close_cols:
            return None

        out = pd.DataFrame(index=pd.Index(df["date"], name="date"))
        names = []
        for close_col in close_cols:
            if close_col.endswith("_Adj Close"):
                prefix = close_col[: -len("_Adj Close")]
            else:
                prefix = close_col[: -len("_Close")]

            close = pd.to_numeric(df[close_col], errors="coerce").ffill().bfill()
            ret = close / close.shift(1) - 1.0

            name = f"{stem}__{prefix}_ret_1"
            out[name] = ret
            names.append(name)

            volume_col = f"{prefix}_Volume"
            volume = None
            if volume_col in df.columns:
                volume = pd.to_numeric(df[volume_col], errors="coerce").ffill().bfill()
                volume_denom = volume.replace(0.0, np.nan)

            for window in self.market_windows:
                ret_mean_name = f"{stem}__{prefix}_ret_mean_{window}"
                ret_std_name = f"{stem}__{prefix}_ret_std_{window}"
                out[ret_mean_name] = ret.rolling(window, min_periods=1).mean()
                out[ret_std_name] = ret.rolling(window, min_periods=1).std(ddof=0)
                names.extend([ret_mean_name, ret_std_name])

                if volume is not None:
                    vol_mean_name = f"{stem}__{prefix}_volume_mean_ratio_{window}"
                    vol_std_name = f"{stem}__{prefix}_volume_std_ratio_{window}"
                    out[vol_mean_name] = volume.rolling(window, min_periods=1).mean() / volume_denom
                    out[vol_std_name] = volume.rolling(window, min_periods=1).std(ddof=0) / volume_denom
                    names.extend([vol_mean_name, vol_std_name])

        out = out.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
        return out, names

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start = self.indices[idx]
        x = self.data[start: start + self.lookback]
        y = self.data[start + self.lookback: start + self.lookback + self.horizon]

        outputs = [x, y]
        if self.use_candle_encoder:
            outputs.append(self.candle_data[start: start + self.lookback])
        if self.use_market_features:
            outputs.append(self.market_data[start: start + self.lookback])

        return tuple(outputs)

    # Compatibilidade com rolling_forecast.py
    def get_metadata_window(self, idx):
        return None
