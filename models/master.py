import torch
import torch.nn as nn

from utils.custom_losses import get_loss


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model > 1:
            pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.view(1, 1, max_len, d_model))

    def forward(self, x):
        # x: (B, N, T, D)
        return x + self.pe[:, :, : x.size(2), :]


class Gate(nn.Module):
    """Market-guided feature gate from MASTER.

    A saída é d_output * softmax(Wm / beta), preservando escala média ≈ 1.
    """

    def __init__(self, d_input, d_output, beta=1.0):
        super().__init__()
        if beta <= 0:
            raise ValueError("beta deve ser positivo.")
        self.trans = nn.Linear(d_input, d_output)
        self.d_output = d_output
        self.beta = beta

    def forward(self, market_state):
        gate = self.trans(market_state)
        gate = torch.softmax(gate / self.beta, dim=-1)
        return self.d_output * gate


class TemporalSelfAttention(nn.Module):
    """Intra-stock aggregation: atenção temporal dentro de cada papel."""

    def __init__(self, d_model, nhead, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout_attn = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-5)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: (B, N, T, D)
        batch, n_assets, seq_len, d_model = x.shape
        z = x.reshape(batch * n_assets, seq_len, d_model)
        z = self.norm1(z)
        attn_out, _ = self.attn(z, z, z, need_weights=False)
        z = z + self.dropout_attn(attn_out)
        z = self.norm2(z)
        z = z + self.ffn(z)
        return z.reshape(batch, n_assets, seq_len, d_model)


class SpatialSelfAttention(nn.Module):
    """Inter-stock aggregation: atenção entre papéis em cada instante."""

    def __init__(self, d_model, nhead, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout_attn = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-5)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: (B, N, T, D) -> atenção em N para cada t
        batch, n_assets, seq_len, d_model = x.shape
        z = x.permute(0, 2, 1, 3).reshape(batch * seq_len, n_assets, d_model)
        z = self.norm1(z)
        attn_out, _ = self.attn(z, z, z, need_weights=False)
        z = z + self.dropout_attn(attn_out)
        z = self.norm2(z)
        z = z + self.ffn(z)
        z = z.reshape(batch, seq_len, n_assets, d_model).permute(0, 2, 1, 3)
        return z


class TemporalAggregation(nn.Module):
    """Agregação temporal final do MASTER: último estado como query."""

    def __init__(self, d_model):
        super().__init__()
        self.trans = nn.Linear(d_model, d_model, bias=False)

    def forward(self, z):
        # z: (B, N, T, D)
        h = self.trans(z)
        query = h[:, :, -1, :].unsqueeze(-1)        # (B, N, D, 1)
        weights = torch.matmul(h, query).squeeze(-1) # (B, N, T)
        weights = torch.softmax(weights, dim=-1).unsqueeze(-2)
        return torch.matmul(weights, z).squeeze(-2)  # (B, N, D)


class MASTER(nn.Module):
    """MASTER adaptado ao formato do repositório paralelo.

    Entrada padrão:
        x:        (B, L, N)
        candle_x: (B, L, N, F), opcional

    O modelo preserva os blocos centrais do código oficial:
    feature gate guiado por mercado, agregação intra-stock,
    agregação inter-stock e agregação temporal final.
    """

    def __init__(
        self,
        lookback,
        pred_len,
        enc_in=1,
        d_model=64,
        t_nhead=4,
        s_nhead=2,
        dropout=0.3,
        beta=5.0,
        loss_name="mse",
        loss_kwargs=None,
        embedding_kwargs=None,
        candle_input_dim=None,
        use_candle_features=False,
        revin=False,
        revin_affine=False,
    ):
        super().__init__()
        del embedding_kwargs, revin, revin_affine  # mantidos por compatibilidade com main_test.py

        if d_model % t_nhead != 0:
            raise ValueError("d_model deve ser divisível por t_nhead.")
        if d_model % s_nhead != 0:
            raise ValueError("d_model deve ser divisível por s_nhead.")

        self.lookback = lookback
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.d_model = d_model
        self.forecast_model_name = "MASTER"
        self.use_candle_features = bool(use_candle_features)
        self.candle_input_dim = int(candle_input_dim or 0)

        self.stock_feature_dim = 1 + (self.candle_input_dim if self.use_candle_features else 0)
        self.market_state_dim = 4 * self.stock_feature_dim

        self.feature_gate = Gate(self.market_state_dim, self.stock_feature_dim, beta=beta)
        self.feature_projection = nn.Linear(self.stock_feature_dim, d_model)
        self.position = PositionalEncoding(d_model=d_model, max_len=max(lookback, 1))
        self.intra_stock = TemporalSelfAttention(d_model=d_model, nhead=t_nhead, dropout=dropout)
        self.inter_stock = SpatialSelfAttention(d_model=d_model, nhead=s_nhead, dropout=dropout)
        self.temporal_aggregation = TemporalAggregation(d_model=d_model)
        self.decoder = nn.Linear(d_model, pred_len)
        self.loss_fn = get_loss(loss_name, **(loss_kwargs or {}))

    def _build_stock_features(self, x, candle_x=None):
        # Retorna (B, L, N, F_stock)
        features = [x.unsqueeze(-1)]
        if self.use_candle_features:
            if candle_x is None:
                raise ValueError("MASTER foi configurado com use_candle_features=True, mas candle_x não foi informado.")
            if candle_x.dim() != 4:
                raise ValueError(f"candle_x deve ter shape (B, L, N, F). Recebido: {tuple(candle_x.shape)}")
            if candle_x.shape[:3] != x.shape:
                raise ValueError(f"candle_x desalinhado: x={tuple(x.shape)}, candle_x={tuple(candle_x.shape)}")
            if candle_x.size(-1) != self.candle_input_dim:
                raise ValueError(
                    f"Esperado candle_input_dim={self.candle_input_dim}, recebido {candle_x.size(-1)}"
                )
            features.append(candle_x)
        return torch.cat(features, dim=-1)

    def _market_state(self, stock_features):
        # stock_features: (B, L, N, F)
        last = stock_features[:, -1, :, :]
        last_mean = last.mean(dim=1)
        last_std = last.std(dim=1, unbiased=False)

        window = stock_features.reshape(stock_features.size(0), -1, stock_features.size(-1))
        window_mean = window.mean(dim=1)
        window_std = window.std(dim=1, unbiased=False)
        return torch.cat([last_mean, last_std, window_mean, window_std], dim=-1)

    def forward(self, x, y=None, return_loss=False, candle_x=None):
        if x.dim() != 3:
            raise ValueError(f"Esperado x com shape (batch, seq_len, channels), recebido {tuple(x.shape)}")

        batch, seq_len, channels = x.shape
        if seq_len != self.lookback:
            raise ValueError(f"Esperado seq_len={self.lookback}, recebido {seq_len}")
        if channels != self.enc_in:
            raise ValueError(f"Esperado {self.enc_in} canais, recebido {channels}")

        stock_features = self._build_stock_features(x, candle_x=candle_x)
        market_state = self._market_state(stock_features)
        gate = self.feature_gate(market_state).view(batch, 1, 1, self.stock_feature_dim)

        z = stock_features * gate
        z = z.permute(0, 2, 1, 3)  # (B, N, L, F)
        z = self.feature_projection(z)
        z = self.position(z)
        z = self.intra_stock(z)
        z = self.inter_stock(z)
        z = self.temporal_aggregation(z)  # (B, N, D)

        output = self.decoder(z).transpose(1, 2)  # (B, pred_len, N)

        if return_loss and y is not None:
            loss = self.loss_fn(output, y[:, -self.pred_len:, :])
            return output, loss
        return output

    def get_loss(self, pred, target):
        return self.loss_fn(pred, target)
