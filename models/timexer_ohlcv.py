import torch
import torch.nn as nn

from utils.custom_losses import get_loss
from utils.revin import RevIN


class TimeXerOHLCV(nn.Module):
    """Arquitetura TimeXer-like para OHLCV como variáveis exógenas.

    Fluxo:
        x        : (B, L, N)        série endógena/target
        candle_x : (B, L, N, F)     OHLCV/derivadas exógenas

    Ideia:
        1) x vira tokens temporais por patch, por canal;
        2) cria-se um global token endógeno por canal;
        3) candle_x vira tokens variate-level, um token por feature exógena;
        4) o global token faz cross-attention nas exógenas;
        5) projeção gera o horizonte por canal.
    """

    def __init__(self, lookback, pred_len, enc_in=1, d_model=32, n_heads=8,
                 dropout=0.1, loss_name="mse", loss_kwargs=None,
                 embedding_kwargs=None, candle_input_dim=5,
                 patch_len=16, patch_stride=None, num_layers=1,
                 dim_feedforward=None, channel_specific_projection=True,
                 revin=False, revin_affine=False):
        super().__init__()
        if patch_len < 1:
            raise ValueError("patch_len deve ser >= 1")
        if patch_len > lookback:
            raise ValueError("patch_len não pode ser maior que lookback")

        self.lookback = lookback
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.d_model = d_model
        self.candle_input_dim = candle_input_dim
        self.patch_len = patch_len
        self.patch_stride = patch_stride or patch_len
        self.num_patches = 1 + (lookback - patch_len) // self.patch_stride
        self.channel_specific_projection = channel_specific_projection
        self.revin_enabled = revin
        self.revin_affine = revin_affine
        self.revin = RevIN(enc_in, affine=revin_affine) if revin else None

        if dim_feedforward is None:
            dim_feedforward = 4 * d_model

        self.patch_projection = nn.Linear(patch_len, d_model)
        self.global_token = nn.Parameter(torch.zeros(1, 1, 1, d_model))
        self.endogenous_position = nn.Parameter(torch.zeros(1, 1, self.num_patches + 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.endogenous_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Variate-level embedding: cada feature exógena vira um token por canal.
        self.exogenous_projection = nn.Linear(lookback, d_model)
        self.exogenous_feature_embedding = nn.Parameter(torch.zeros(1, 1, candle_input_dim, d_model))

        self.cross_attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.cross_norm = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.ffn_norm = nn.LayerNorm(d_model)

        projection_in = (self.num_patches + 1) * d_model
        if channel_specific_projection:
            self.projections = nn.ModuleList([
                nn.Linear(projection_in, pred_len) for _ in range(enc_in)
            ])
        else:
            self.projection = nn.Linear(projection_in, pred_len)

        self.loss_fn = get_loss(loss_name, **(loss_kwargs or {}))

        nn.init.normal_(self.global_token, std=0.02)
        nn.init.normal_(self.endogenous_position, std=0.02)
        nn.init.normal_(self.exogenous_feature_embedding, std=0.02)

    def _build_endogenous_tokens(self, x):
        # x: (B, L, N) -> patches: (B, N, P, patch_len)
        x_by_channel = x.transpose(1, 2)
        patches = x_by_channel.unfold(dimension=-1, size=self.patch_len, step=self.patch_stride)
        tokens = self.patch_projection(patches)  # (B, N, P, D)

        batch, channels, _, _ = tokens.shape
        global_token = self.global_token.expand(batch, channels, 1, self.d_model)
        tokens = torch.cat([global_token, tokens], dim=2)
        tokens = tokens + self.endogenous_position[:, :, :tokens.size(2), :]
        return tokens

    def _build_exogenous_tokens(self, candle_x):
        # candle_x: (B, L, N, F) -> (B*N, F, D)
        batch, seq_len, channels, features = candle_x.shape
        if features != self.candle_input_dim:
            raise ValueError(
                f"Esperado candle_input_dim={self.candle_input_dim}, recebido {features}"
            )
        if seq_len != self.lookback:
            raise ValueError(f"Esperado candle seq_len={self.lookback}, recebido {seq_len}")
        if channels != self.enc_in:
            raise ValueError(f"Esperado candle channels={self.enc_in}, recebido {channels}")

        exog = candle_x.permute(0, 2, 3, 1).reshape(batch * channels, features, seq_len)
        exog_tokens = self.exogenous_projection(exog)
        feature_emb = self.exogenous_feature_embedding.expand(batch, channels, features, self.d_model)
        feature_emb = feature_emb.reshape(batch * channels, features, self.d_model)
        return exog_tokens + feature_emb

    def forward(self, x, y=None, return_loss=False, candle_x=None):
        if x.dim() != 3:
            raise ValueError(f"Esperado x com shape (batch, seq_len, channels), recebido {tuple(x.shape)}")
        if candle_x is None:
            raise ValueError("TimeXerOHLCV requer candle_x com shape (B, L, N, F).")

        batch, seq_len, channels = x.shape
        if seq_len != self.lookback:
            raise ValueError(f"Esperado seq_len={self.lookback}, recebido {seq_len}")
        if channels != self.enc_in:
            raise ValueError(f"Esperado {self.enc_in} canais, recebido {channels}")

        if self.revin_enabled:
            x = self.revin(x, mode="norm")

        endog_tokens = self._build_endogenous_tokens(x)
        endog_flat = endog_tokens.reshape(batch * channels, self.num_patches + 1, self.d_model)
        endog_encoded = self.endogenous_encoder(endog_flat)

        exog_tokens = self._build_exogenous_tokens(candle_x)

        # Cross-attention: somente o global token consulta as exógenas.
        global_query = endog_encoded[:, :1, :]
        cross_out, _ = self.cross_attention(global_query, exog_tokens, exog_tokens)
        global_out = self.cross_norm(global_query + self.dropout(cross_out))

        endog_encoded = torch.cat([global_out, endog_encoded[:, 1:, :]], dim=1)
        ffn_out = self.ffn(endog_encoded)
        endog_encoded = self.ffn_norm(endog_encoded + self.dropout(ffn_out))

        h = endog_encoded.reshape(batch, channels, -1)

        if self.channel_specific_projection:
            output = torch.cat([
                proj(h[:, i, :]).unsqueeze(-1)
                for i, proj in enumerate(self.projections)
            ], dim=-1)
        else:
            output = self.projection(h.reshape(batch * channels, -1))
            output = output.reshape(batch, channels, self.pred_len).transpose(1, 2)

        if self.revin_enabled:
            output = self.revin(output, mode="denorm")

        if return_loss and y is not None:
            loss = self.loss_fn(output, y[:, -self.pred_len:, :])
            return output, loss
        return output

    def get_loss(self, pred, target):
        return self.loss_fn(pred, target)
