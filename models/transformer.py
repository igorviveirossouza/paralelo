import torch
import torch.nn as nn
from utils.embeddings import ChannelIndependentTemporalEmbedding
from utils.custom_losses import get_loss


class TransformerChannelIndependent(nn.Module):
    """
    TRansformer com processamento independente por canal.

    Ideia teorica:
        Equivale a passar uma serie por vez pelo embedding, pela self-attention
        temporal e pela projecao.

    Implementacao pratica:
        O processamento e vetorizado em um unico forward. A dimensao de canal
        e incorporada temporariamente ao batch para que cada canal seja tratado
        como uma sequencia independente:

            x:       (B, L, N)
            emb:     (B, N, L, D)
            attn_in: (B*N, L, D)
            layer norm (B*N,L,D)
            feed_foward (B,N,L,D)
            output:  (B, pred_len, N)

    Nesta versao nao ha mistura entre canais mas atenção temporal compartilhada entre canais.
    Ou seja: atenção com parâmetros compartilhados no domínio dos canais porém, cada token de um canal só olha para a 
    própria série. 
    """
    def __init__(self, lookback, pred_len, enc_in=1, d_model=32, n_heads=8,
                 dropout=0.1, loss_name='mse', loss_kwargs=None, embedding_kwargs=None,
                 use_all_timesteps=True, channel_specific_embedding=True,
                 channel_specific_projection=True):
        super().__init__()
        self.lookback = lookback
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.d_model = d_model
        self.use_all_timesteps = use_all_timesteps
        self.channel_specific_projection = channel_specific_projection

        self.embedding = ChannelIndependentTemporalEmbedding(
            c_in=enc_in,
            d_model=d_model,
            dropout=dropout,
            channel_specific=channel_specific_embedding,
            **(embedding_kwargs or {})
        )

        self.attention = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=dropout,
            batch_first=True
        )

        self.dropout_attn = nn.Dropout(dropout)
        self.dropout_ffn = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)


        projection_in = lookback * d_model if use_all_timesteps else d_model

        if channel_specific_projection:
            self.projections = nn.ModuleList([
                nn.Linear(projection_in, pred_len) for _ in range(enc_in)
            ])
        else:
            self.projection = nn.Linear(projection_in, pred_len)

        self.loss_fn = get_loss(loss_name, **(loss_kwargs or {}))

    def forward(self, x, y=None, return_loss=False):
        # x: (batch, seq_len, channels)
        if x.dim() != 3:
            raise ValueError(
                f"Esperado x com shape (batch, seq_len, channels), recebido {tuple(x.shape)}"
            )

        batch, seq_len, channels = x.shape
        if channels != self.enc_in:
            raise ValueError(f"Esperado {self.enc_in} canais, recebido {channels}")

        if self.use_all_timesteps and seq_len != self.lookback:
            raise ValueError(f"Esperado seq_len={self.lookback}, recebido {seq_len}")

        # Embedding sem mistura de canais: (B, N, L, D)
        x_emb = self.embedding(x)

        # Processa cada canal como uma sequencia independente no batch: (B*N, L, D)
        attn_input = x_emb.reshape(batch * channels, seq_len, self.d_model)
        attn_output, _ = self.attention(attn_input, attn_input, attn_input)  # Q, K, V

        z = self.norm1(attn_input + self.dropout_attn(attn_output))

        ffn_output = self.ffn(z)

        z = self.norm2(z+self.dropout_ffn(ffn_output))


        if self.use_all_timesteps:
            h = z.reshape(batch, channels, seq_len * self.d_model)
        else:
            h = z[:, -1, :].reshape(batch, channels, self.d_model)

        # Projecao por canal para pred_len passos futuros.
        if self.channel_specific_projection:
            output = torch.cat([
                proj(h[:, i, :]).unsqueeze(-1)
                for i, proj in enumerate(self.projections)
            ], dim=-1)
        else:
            output = self.projection(h.reshape(batch * channels, -1))
            output = output.reshape(batch, channels, self.pred_len).transpose(1, 2)

        # output: (B, pred_len, N)
        if return_loss and y is not None:
            loss = self.loss_fn(output, y[:, -self.pred_len:, :])
            return output, loss

        return output

    def get_loss(self, pred, target):
        return self.loss_fn(pred, target)
