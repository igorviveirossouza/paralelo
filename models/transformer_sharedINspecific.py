import torch
import torch.nn as nn
from utils.embeddings import ChannelIndependentTemporalEmbedding
from utils.custom_losses import get_loss


class TransformerChannelIndependentSharedINSpecific(nn.Module):
    """
    Transformer channel-independent com dois caminhos:
      1) Transformer temporal compartilhado entre canais;
      2) Transformer temporal específico por canal.

    Fluxo:
        x:            (B, L, N)
        emb:          (B, N, L, D)
        shared_out:   (B, N, L, D)
        specific_out: (B, N, L, D)
        soma:         (B, N, L, D)
        output:       (B, pred_len, N)

    Não há mistura temporal entre canais: cada canal olha apenas para a própria série.
    """
    def __init__(self, lookback, pred_len, enc_in=1, d_model=32, n_heads=8,
                 dropout=0.1, loss_name='mse', loss_kwargs=None, embedding_kwargs=None,
                 use_all_timesteps=True, channel_specific_embedding=True,
                 channel_specific_projection=True, num_layers=1,
                 dim_feedforward=None):
        super().__init__()
        self.lookback = lookback
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.d_model = d_model
        self.use_all_timesteps = use_all_timesteps
        self.channel_specific_projection = channel_specific_projection

        if dim_feedforward is None:
            dim_feedforward = 4 * d_model

        self.embedding = ChannelIndependentTemporalEmbedding(
            c_in=enc_in,
            d_model=d_model,
            dropout=dropout,
            channel_specific=channel_specific_embedding,
            **(embedding_kwargs or {})
        )

        # Caminho compartilhado: mesmos pesos para todos os canais.
        shared_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False
        )
        self.shared_transformer = nn.TransformerEncoder(
            shared_layer,
            num_layers=num_layers
        )

        # Caminho específico: um Transformer próprio para cada canal.
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
            raise ValueError(
                f"Esperado x com shape (batch, seq_len, channels), recebido {tuple(x.shape)}"
            )

        batch, seq_len, channels = x.shape
        if channels != self.enc_in:
            raise ValueError(f"Esperado {self.enc_in} canais, recebido {channels}")

        if self.use_all_timesteps and seq_len != self.lookback:
            raise ValueError(f"Esperado seq_len={self.lookback}, recebido {seq_len}")

        # (B, N, L, D)
        x_emb = self.embedding(x)

        # Caminho compartilhado: (B*N, L, D) -> (B, N, L, D)
        shared_input = x_emb.reshape(batch * channels, seq_len, self.d_model)
        shared_out = self.shared_transformer(shared_input)
        shared_out = shared_out.reshape(batch, channels, seq_len, self.d_model)

        z = x_emb + shared_out

        # Caminho específico por canal: cada papel usa seu próprio Transformer.
        specific_out = torch.cat([
            transformer(z[:, i, :, :]).unsqueeze(1)
            for i, transformer in enumerate(self.channel_transformers)
        ], dim=1)


        if self.use_all_timesteps:
            h = specific_out.reshape(batch, channels, seq_len * self.d_model)
        else:
            h = specific_out[:, :, -1, :].reshape(batch, channels, self.d_model)

        if self.channel_specific_projection:
            output = torch.cat([
                proj(h[:, i, :]).unsqueeze(-1)
                for i, proj in enumerate(self.projections)
            ], dim=-1)
        else:
            output = self.projection(h.reshape(batch * channels, -1))
            output = output.reshape(batch, channels, self.pred_len).transpose(1, 2)

        if return_loss and y is not None:
            loss = self.loss_fn(output, y[:, -self.pred_len:, :])
            return output, loss

        return output

    def get_loss(self, pred, target):
        return self.loss_fn(pred, target)
