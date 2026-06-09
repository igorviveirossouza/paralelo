import torch
import torch.nn as nn
from utils.embeddings import ChannelIndependentTemporalEmbedding
from utils.custom_losses import get_loss
from utils.revin import RevIN


class TransformerChannelIndependentSharedSpecific(nn.Module):
    def __init__(self, lookback, pred_len, enc_in=1, d_model=32, n_heads=8,
                 dropout=0.1, loss_name='mse', loss_kwargs=None, embedding_kwargs=None,
                 use_all_timesteps=True, channel_specific_embedding=True,
                 channel_specific_projection=True, num_layers=1,
                 dim_feedforward=None, revin=False, revin_affine=False):
        super().__init__()
        self.lookback = lookback
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.d_model = d_model
        self.use_all_timesteps = use_all_timesteps
        self.channel_specific_projection = channel_specific_projection
        self.revin_enabled = revin
        self.revin_affine = revin_affine
        self.revin = RevIN(enc_in, affine=revin_affine) if revin else None

        if dim_feedforward is None:
            dim_feedforward = 4 * d_model

        self.embedding = ChannelIndependentTemporalEmbedding(
            c_in=enc_in,
            d_model=d_model,
            dropout=dropout,
            channel_specific=channel_specific_embedding,
            **(embedding_kwargs or {})
        )

        shared_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False
        )
        self.shared_transformer = nn.TransformerEncoder(shared_layer, num_layers=num_layers)

        self.channel_transformers = nn.ModuleList([
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=n_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=False
                ),
                num_layers=num_layers
            )
            for _ in range(enc_in)
        ])

        projection_in = lookback * d_model if use_all_timesteps else d_model
        if channel_specific_projection:
            self.projections = nn.ModuleList([
                nn.Linear(projection_in, pred_len) for _ in range(enc_in)
            ])
        else:
            self.projection = nn.Linear(projection_in, pred_len)

        self.loss_fn = get_loss(loss_name, **(loss_kwargs or {}))

    def forward(self, x, y=None, return_loss=False):
        if x.dim() != 3:
            raise ValueError(f"Esperado x com shape (batch, seq_len, channels), recebido {tuple(x.shape)}")

        batch, seq_len, channels = x.shape
        if channels != self.enc_in:
            raise ValueError(f"Esperado {self.enc_in} canais, recebido {channels}")
        if self.use_all_timesteps and seq_len != self.lookback:
            raise ValueError(f"Esperado seq_len={self.lookback}, recebido {seq_len}")

        if self.revin_enabled:
            x = self.revin(x, mode='norm')

        x_emb = self.embedding(x)
        shared_input = x_emb.reshape(batch * channels, seq_len, self.d_model)
        shared_out = self.shared_transformer(shared_input)
        shared_out = shared_out.reshape(batch, channels, seq_len, self.d_model)

        specific_out = torch.cat([
            transformer(x_emb[:, i, :, :]).unsqueeze(1)
            for i, transformer in enumerate(self.channel_transformers)
        ], dim=1)

        z = shared_out + specific_out
        if self.use_all_timesteps:
            h = z.reshape(batch, channels, seq_len * self.d_model)
        else:
            h = z[:, :, -1, :].reshape(batch, channels, self.d_model)

        if self.channel_specific_projection:
            output = torch.cat([
                proj(h[:, i, :]).unsqueeze(-1)
                for i, proj in enumerate(self.projections)
            ], dim=-1)
        else:
            output = self.projection(h.reshape(batch * channels, -1))
            output = output.reshape(batch, channels, self.pred_len).transpose(1, 2)

        if self.revin_enabled:
            output = self.revin(output, mode='denorm')

        if return_loss and y is not None:
            loss = self.loss_fn(output, y[:, -self.pred_len:, :])
            return output, loss
        return output

    def get_loss(self, pred, target):
        return self.loss_fn(pred, target)
