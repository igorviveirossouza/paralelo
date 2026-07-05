from pathlib import Path
import re
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd


DEFAULT_CANDLE_COLS = ["abertura", "maxima", "minima", "data", "volume"]
DEFAULT_MARKET_WINDOWS = [5, 10, 20, 30, 60]
DEFAULT_FACTOR_WINDOWS = [5, 10, 20, 30, 60]


class TimeSeriesDataset(Dataset):
    """
    Dataset flexível para o projeto paralelo.
    Suporta modo Multivariate e Univariate + split Train/Test.

    x/y vêm de data_path. As features por papel podem vir do próprio data_path
    ou de feature_source_path, útil quando o alvo é retorno/log-retorno, mas os
    fatores devem ser calculados sobre OHLCV/preço.

    Retorno:
        x:        [lookback, N]
        y:        [pred_len, N]
        candle_x: [lookback, N, F] opcional, com fatores e/ou OHLCV relativo
        market_x: [lookback, K] opcional
    """
    def __init__(self, data_path, lookback=96, pred_len=24, stride=1,
                 cols=None, train=True, test_ratio=0.2,
                 use_candle_encoder=False, candle_cols=None,
                 candle_feature_mode="ohlcv_relative",
                 feature_source_path=None, duplicate_policy="error",
                 use_market_features=False, market_feature_files=None,
                 market_feature_mode="master", market_windows=None,
                 market_date_col=None,
                 use_stock_factors=False, stock_factor_mode="alpha158",
                 stock_factor_windows=None, stock_factor_normalize=True):
        super().__init__()
        self.lookback = lookback
        self.horizon = pred_len
        self.stride = stride
        self.cols = cols
        self.train = train
        self.duplicate_policy = duplicate_policy
        self.raw_candle_enabled = bool(use_candle_encoder)
        self.use_stock_factors = bool(use_stock_factors)
        self.use_candle_encoder = self.raw_candle_enabled or self.use_stock_factors
        self.candle_cols = candle_cols or DEFAULT_CANDLE_COLS
        self.candle_feature_mode = candle_feature_mode
        self.feature_source_path = feature_source_path
        self.candle_feature_names = []
        self.stock_factor_names = []
        self.candle_data = None
        self.use_market_features = bool(use_market_features)
        self.market_feature_files = market_feature_files or []
        self.market_feature_mode = market_feature_mode
        self.market_windows = market_windows or DEFAULT_MARKET_WINDOWS
        self.market_date_col = market_date_col
        self.market_feature_names = []
        self.market_data = None
        self.stock_factor_mode = stock_factor_mode
        self.stock_factor_windows = stock_factor_windows or DEFAULT_FACTOR_WINDOWS
        self.stock_factor_normalize = bool(stock_factor_normalize)
        self.date_index = []

        df = self._load_main_data(data_path)
        print("Colunas originais:", df.columns.tolist())

        df_pivot = df.pivot(index="date", columns="cols", values="data")
        df_pivot = df_pivot.ffill().bfill().fillna(0.0)
        self.date_index = df_pivot.index.tolist()

        T_all = len(df_pivot)
        test_size = int(T_all * test_ratio)
        split_idx = T_all - test_size

        feature_df = df
        if self.use_candle_encoder and self.feature_source_path is not None:
            feature_df = self._load_main_data(self.feature_source_path)
            print(f"✅ Fonte OHLCV/fatores separada: {self.feature_source_path}")
            print("Colunas da fonte OHLCV/fatores:", feature_df.columns.tolist())

        stock_factor_np = None
        if self.use_stock_factors:
            stock_factor_frame, self.stock_factor_names = self._build_stock_factor_features(feature_df)
            factor_pivots = []
            for feature_name in self.stock_factor_names:
                feature_pivot = stock_factor_frame.pivot(index="date", columns="cols", values=feature_name)
                feature_pivot = feature_pivot.reindex(index=df_pivot.index, columns=df_pivot.columns)
                feature_pivot = feature_pivot.ffill().bfill().fillna(0.0)
                factor_pivots.append(feature_pivot.values)

            stock_factor_np = np.stack(factor_pivots, axis=-1).astype("float32")
            if self.stock_factor_normalize:
                stock_factor_np = self._robust_normalize_cube(stock_factor_np, split_idx)
            print(
                f"✅ Stock factors ativos | modo={self.stock_factor_mode} | "
                f"F={len(self.stock_factor_names)} | Shape factors: {stock_factor_np.shape}"
            )

        candle_np = None
        raw_candle_feature_names = []
        if self.raw_candle_enabled:
            candle_frame, raw_candle_feature_names = self._build_candle_features(feature_df)
            candle_pivots = []
            for feature_name in raw_candle_feature_names:
                feature_pivot = candle_frame.pivot(index="date", columns="cols", values=feature_name)
                feature_pivot = feature_pivot.reindex(index=df_pivot.index, columns=df_pivot.columns)
                feature_pivot = feature_pivot.ffill().bfill().fillna(0.0)
                candle_pivots.append(feature_pivot.values)

            candle_np = np.stack(candle_pivots, axis=-1).astype("float32")
            print(
                f"✅ Candle Encoder ativo | features: {raw_candle_feature_names} | "
                f"Shape OHLCV: {candle_np.shape}"
            )

        per_stock_parts = []
        per_stock_names = []
        if stock_factor_np is not None:
            per_stock_parts.append(stock_factor_np)
            per_stock_names.extend(self.stock_factor_names)
        if candle_np is not None:
            per_stock_parts.append(candle_np)
            per_stock_names.extend(raw_candle_feature_names)

        per_stock_np = None
        if per_stock_parts:
            per_stock_np = np.concatenate(per_stock_parts, axis=-1).astype("float32")
            self.candle_feature_names = per_stock_names
            print(f"✅ Features por papel no src | F_total={len(per_stock_names)} | Shape: {per_stock_np.shape}")

        market_np = None
        if self.use_market_features:
            market_frame, self.market_feature_names = self._build_market_features(df_pivot.index)
            market_frame = market_frame.reindex(df_pivot.index)
            market_frame = market_frame.ffill().bfill().fillna(0.0)
            market_np = market_frame.values.astype("float32")
            print(
                f"✅ Features de mercado ativas | K={len(self.market_feature_names)} | "
                f"Shape market: {market_np.shape}"
            )

        if cols is None:
            self.data = torch.tensor(df_pivot.values, dtype=torch.float32)
            self.feature_columns = df_pivot.columns.tolist()
            if per_stock_np is not None:
                self.candle_data = torch.tensor(per_stock_np, dtype=torch.float32)
            self.mode = "multivariate"
            print(f"✅ Modo Multivariate - {len(self.feature_columns)} séries | Shape: {self.data.shape}")
        else:
            if cols not in df_pivot.columns:
                raise ValueError(f"Coluna '{cols}' não encontrada. Disponíveis: {list(df_pivot.columns)}")
            col_idx = df_pivot.columns.get_loc(cols)
            self.data = torch.tensor(df_pivot[cols].values, dtype=torch.float32).unsqueeze(1)
            self.feature_columns = [cols]
            if per_stock_np is not None:
                self.candle_data = torch.tensor(per_stock_np[:, col_idx:col_idx + 1, :], dtype=torch.float32)
            self.mode = "univariate"
            print(f"✅ Modo Univariate - Ticker: {cols} | Shape: {self.data.shape}")

        if market_np is not None:
            self.market_data = torch.tensor(market_np, dtype=torch.float32)

        T = len(self.data)
        self.indices = []

        if train:
            max_start = split_idx - lookback - pred_len
            for start in range(0, max(0, max_start) + 1, stride):
                self.indices.append(start)
            print(f"✅ Train split | amostras: {len(self.indices)} | split_idx={split_idx}")
        else:
            start_min = max(0, split_idx - lookback)
            for start in range(start_min, T - lookback - pred_len + 1, stride):
                self.indices.append(start)
            print(f"✅ Test split | amostras: {len(self.indices)} | início global ≈ {split_idx}")

        print(f"Total de amostras válidas ({'train' if train else 'test'}): {len(self.indices)}")

    def _load_main_data(self, data_path):
        df = pd.read_csv(data_path)
        required = {"date", "cols", "data"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"Colunas obrigatórias ausentes no CSV {data_path}: {sorted(missing)}")

        df = df.copy()
        df = df.dropna(subset=["date", "cols"])
        df["_row_order"] = np.arange(len(df))
        df["date"] = self._normalize_dates(df["date"])
        df["cols"] = df["cols"].astype(str)
        df["data"] = pd.to_numeric(df["data"], errors="coerce")

        duplicated_rows = int(df.duplicated(["date", "cols"], keep=False).sum())
        if duplicated_rows > 0:
            dup_examples = (
                df.loc[df.duplicated(["date", "cols"], keep=False), ["date", "cols"]]
                .value_counts()
                .head(20)
                .reset_index(name="n")
            )
            if self.duplicate_policy == "error":
                raise ValueError(
                    f"{data_path}: {duplicated_rows} linhas duplicadas em (date, cols).\n"
                    "Isso indica problema na preparação/interpolação da base e não será corrigido silenciosamente.\n"
                    f"Exemplos:\n{dup_examples.to_string(index=False)}\n"
                    "Para depuração temporária, rode com --duplicate_policy last."
                )
            if self.duplicate_policy == "last":
                print(
                    f"⚠️ {data_path}: {duplicated_rows} linhas duplicadas em (date, cols). "
                    "Mantendo a última ocorrência por --duplicate_policy last."
                )
                df = df.drop_duplicates(["date", "cols"], keep="last")
            else:
                raise ValueError("duplicate_policy deve ser 'error' ou 'last'.")

        df = (
            df.sort_values(["cols", "date", "_row_order"])
            .drop(columns=["_row_order"])
            .sort_values(["cols", "date"])
        )
        return df

    @staticmethod
    def _normalize_dates(values):
        series = pd.Series(values)

        def normalize_one(value):
            if pd.isna(value):
                return value

            text = str(value).strip()
            if text == "":
                return text

            # Preserva índices numéricos de pregão: 1, 2, 3, ...
            if re.fullmatch(r"\d+(\.0)?", text):
                return str(int(float(text)))

            # Só converte datas textuais claras: YYYY-MM-DD, DD/MM/YYYY, etc.
            if any(sep in text for sep in ["-", "/", ":"]):
                parsed = pd.to_datetime(text, errors="coerce")
                if pd.notna(parsed):
                    return parsed.strftime("%Y-%m-%d")

            return text

        return [normalize_one(value) for value in series]

    @staticmethod
    def _safe_log_ratio(numerator, denominator, eps=1e-8):
        numerator = pd.to_numeric(numerator, errors="coerce").clip(lower=eps)
        denominator = pd.to_numeric(denominator, errors="coerce").clip(lower=eps)
        return np.log(numerator / denominator)

    @staticmethod
    def _robust_normalize_cube(cube, split_idx, eps=1e-12):
        train_values = cube[:split_idx].reshape(-1, cube.shape[-1])
        median = np.nanmedian(train_values, axis=0)
        mad = np.nanmedian(np.abs(train_values - median), axis=0)
        mad = np.where(np.isfinite(mad) & (mad > eps), mad, 1.0)
        normalized = (cube - median.reshape(1, 1, -1)) / mad.reshape(1, 1, -1)
        normalized = np.clip(normalized, -3.0, 3.0)
        return np.nan_to_num(normalized, nan=0.0, posinf=3.0, neginf=-3.0).astype("float32")

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

    @staticmethod
    def _rolling_rank_last(series, window):
        return series.rolling(window, min_periods=1).apply(
            lambda values: float(np.sum(values <= values[-1])) / max(len(values), 1), raw=True
        )

    @staticmethod
    def _rolling_idxmax(series, window):
        return series.rolling(window, min_periods=1).apply(
            lambda values: float(np.argmax(values)) / max(len(values) - 1, 1), raw=True
        )

    @staticmethod
    def _rolling_idxmin(series, window):
        return series.rolling(window, min_periods=1).apply(
            lambda values: float(np.argmin(values)) / max(len(values) - 1, 1), raw=True
        )

    @staticmethod
    def _rolling_regression_features(series, window, eps=1e-12):
        def beta_fn(values):
            y = np.asarray(values, dtype=float)
            x = np.arange(len(y), dtype=float)
            x_centered = x - x.mean()
            y_centered = y - np.nanmean(y)
            denom = np.sum(x_centered ** 2)
            if denom <= eps:
                return 0.0
            return float(np.nansum(x_centered * y_centered) / denom)

        def rsqr_fn(values):
            y = np.asarray(values, dtype=float)
            if len(y) < 2 or np.nanstd(y) <= eps:
                return 0.0
            x = np.arange(len(y), dtype=float)
            corr = np.corrcoef(x, y)[0, 1]
            if not np.isfinite(corr):
                return 0.0
            return float(corr ** 2)

        def resi_fn(values):
            y = np.asarray(values, dtype=float)
            x = np.arange(len(y), dtype=float)
            x_centered = x - x.mean()
            y_centered = y - np.nanmean(y)
            denom = np.sum(x_centered ** 2)
            if denom <= eps:
                return 0.0
            beta = np.nansum(x_centered * y_centered) / denom
            alpha = np.nanmean(y) - beta * x.mean()
            fitted_last = alpha + beta * x[-1]
            return float(y[-1] - fitted_last)

        beta = series.rolling(window, min_periods=2).apply(beta_fn, raw=True)
        rsqr = series.rolling(window, min_periods=2).apply(rsqr_fn, raw=True)
        resi = series.rolling(window, min_periods=2).apply(resi_fn, raw=True)
        return beta, rsqr, resi

    def _build_stock_factor_features(self, df):
        mode = self.stock_factor_mode.lower()
        if mode != "alpha158":
            raise ValueError(f"stock_factor_mode inválido: {self.stock_factor_mode}")

        required = ["abertura", "maxima", "minima", "data", "volume"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"stock_factor_mode=alpha158 requer colunas OHLCV. Ausentes: {missing}")

        frames = []
        factor_names = None
        for _, group in df.groupby("cols", sort=False):
            factor_frame, names = self._build_alpha158_one_asset(group)
            frames.append(factor_frame)
            if factor_names is None:
                factor_names = names

        out = pd.concat(frames, ignore_index=True)
        out[factor_names] = out[factor_names].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return out, factor_names

    def _build_alpha158_one_asset(self, group, eps=1e-12):
        group = group.sort_values("date").copy()
        open_ = pd.to_numeric(group["abertura"], errors="coerce").ffill().bfill()
        high = pd.to_numeric(group["maxima"], errors="coerce").ffill().bfill()
        low = pd.to_numeric(group["minima"], errors="coerce").ffill().bfill()
        close = pd.to_numeric(group["data"], errors="coerce").ffill().bfill()
        volume = pd.to_numeric(group["volume"], errors="coerce").ffill().bfill().clip(lower=0.0)

        if "volume_financeiro" in group.columns:
            amount = pd.to_numeric(group["volume_financeiro"], errors="coerce").ffill().bfill()
            vwap = amount / volume.replace(0.0, np.nan)
            vwap = vwap.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(close)
        else:
            vwap = close.copy()

        denom_open = open_.replace(0.0, np.nan)
        denom_close = close.replace(0.0, np.nan)
        amplitude = (high - low).replace(0.0, np.nan)
        max_oc = pd.concat([open_, close], axis=1).max(axis=1)
        min_oc = pd.concat([open_, close], axis=1).min(axis=1)

        out = pd.DataFrame({"date": group["date"].values, "cols": group["cols"].values})
        names = []

        base_features = {
            "KMID": (close - open_) / (denom_open + eps),
            "KLEN": (high - low) / (denom_open + eps),
            "KMID2": (close - open_) / (amplitude + eps),
            "KUP": (high - max_oc) / (denom_open + eps),
            "KUP2": (high - max_oc) / (amplitude + eps),
            "KLOW": (min_oc - low) / (denom_open + eps),
            "KLOW2": (min_oc - low) / (amplitude + eps),
            "KSFT": (2 * close - high - low) / (denom_open + eps),
            "KSFT2": (2 * close - high - low) / (amplitude + eps),
            "OPEN0": open_ / (denom_close + eps) - 1.0,
            "HIGH0": high / (denom_close + eps) - 1.0,
            "LOW0": low / (denom_close + eps) - 1.0,
            "VWAP0": vwap / (denom_close + eps) - 1.0,
        }
        for name, values in base_features.items():
            out[name] = values.values
            names.append(name)

        close_diff = close.diff()
        close_abs_diff = close_diff.abs()
        close_pos = close_diff.clip(lower=0.0)
        close_neg = (-close_diff.clip(upper=0.0))
        vol_diff = volume.diff()
        vol_abs_diff = vol_diff.abs()
        vol_pos = vol_diff.clip(lower=0.0)
        vol_neg = (-vol_diff.clip(upper=0.0))
        ret_ratio = close / close.shift(1)
        vol_ratio_log = np.log(volume / volume.shift(1).replace(0.0, np.nan) + 1.0)
        close_ret_abs_vol = close.pct_change().abs() * volume
        log_volume = np.log1p(volume)

        for window in self.stock_factor_windows:
            beta, rsqr, resi = self._rolling_regression_features(close, window)
            roll_high_max = high.rolling(window, min_periods=1).max()
            roll_low_min = low.rolling(window, min_periods=1).min()
            sump_denom = close_abs_diff.rolling(window, min_periods=1).sum() + eps
            vsump_denom = vol_abs_diff.rolling(window, min_periods=1).sum() + eps

            window_features = {
                f"ROC{window}": close.shift(window) / (close + eps),
                f"MA{window}": close.rolling(window, min_periods=1).mean() / (close + eps),
                f"STD{window}": close.rolling(window, min_periods=1).std(ddof=0) / (close + eps),
                f"BETA{window}": beta / (close + eps),
                f"RSQR{window}": rsqr,
                f"RESI{window}": resi / (close + eps),
                f"MAX{window}": roll_high_max / (close + eps) - 1.0,
                f"MIN{window}": roll_low_min / (close + eps) - 1.0,
                f"QTLU{window}": close.rolling(window, min_periods=1).quantile(0.8) / (close + eps) - 1.0,
                f"QTLD{window}": close.rolling(window, min_periods=1).quantile(0.2) / (close + eps) - 1.0,
                f"RANK{window}": self._rolling_rank_last(close, window),
                f"RSV{window}": (close - roll_low_min) / (roll_high_max - roll_low_min + eps),
                f"IMAX{window}": self._rolling_idxmax(high, window),
                f"IMIN{window}": self._rolling_idxmin(low, window),
                f"IMXD{window}": self._rolling_idxmax(high, window) - self._rolling_idxmin(low, window),
                f"CORR{window}": close.rolling(window, min_periods=2).corr(log_volume),
                f"CORD{window}": ret_ratio.rolling(window, min_periods=2).corr(vol_ratio_log),
                f"CNTP{window}": (close_diff > 0).astype(float).rolling(window, min_periods=1).mean(),
                f"CNTN{window}": (close_diff < 0).astype(float).rolling(window, min_periods=1).mean(),
                f"CNTD{window}": (close_diff > 0).astype(float).rolling(window, min_periods=1).mean()
                              - (close_diff < 0).astype(float).rolling(window, min_periods=1).mean(),
                f"SUMP{window}": close_pos.rolling(window, min_periods=1).sum() / sump_denom,
                f"SUMN{window}": close_neg.rolling(window, min_periods=1).sum() / sump_denom,
                f"SUMD{window}": (close_pos.rolling(window, min_periods=1).sum()
                               - close_neg.rolling(window, min_periods=1).sum()) / sump_denom,
                f"VMA{window}": volume.rolling(window, min_periods=1).mean() / (volume + eps),
                f"VSTD{window}": volume.rolling(window, min_periods=1).std(ddof=0) / (volume + eps),
                f"WVMA{window}": close_ret_abs_vol.rolling(window, min_periods=1).std(ddof=0)
                               / (close_ret_abs_vol.rolling(window, min_periods=1).mean() + eps),
                f"VSUMP{window}": vol_pos.rolling(window, min_periods=1).sum() / vsump_denom,
                f"VSUMN{window}": vol_neg.rolling(window, min_periods=1).sum() / vsump_denom,
                f"VSUMD{window}": (vol_pos.rolling(window, min_periods=1).sum()
                                - vol_neg.rolling(window, min_periods=1).sum()) / vsump_denom,
            }
            for name, values in window_features.items():
                out[name] = values.values
                names.append(name)

        if len(names) != 158:
            raise RuntimeError(f"Alpha158 deveria gerar 158 fatores, mas gerou {len(names)}.")

        return out, names

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
        if self.candle_data is not None:
            outputs.append(self.candle_data[start: start + self.lookback])
        if self.use_market_features:
            outputs.append(self.market_data[start: start + self.lookback])

        return tuple(outputs)

    def get_metadata_window(self, idx):
        return None
