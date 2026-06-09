import torch
import torch.nn as nn
from utils.embeddings import TemporalEmbedding
from utils.custom_losses import get_loss
from utils.revin import RevIN

class AttentionSolo(nn.Module):
    """
    Modelo básico: Atenção temporal independente por canal (channel-independent).
    Compatível com interface do TFB / TIOMS.
    """
    def __init__(self, lookback, pred_len, enc_in=1, d_model=32, n_heads=8,
                 dropout=0.1, loss_name='mse', loss_kwargs=None, embedding_kwargs=None,
                 revin=False, revin_affine=False):
        super().__init__()
        self.lookback = lookback
        self.pred_len = pred_len
        self.enc_in = enc_in  # número de canais (variáveis)
        self.d_model = d_model
        self.revin_enabled = revin
        self.revin_affine = revin_affine
        self.revin = RevIN(enc_in, affine=revin_affine) if revin else None

        self.embedding = TemporalEmbedding(
            enc_in,
            d_model,
            dropout,
            **(embedding_kwargs or {})
        )

        # Camada de atenção multi-head simples
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        # Decoder linear (projection)
        # self.projection = nn.Linear(d_model, pred_len * enc_in)
        self.projection = nn.Linear(lookback * d_model, pred_len * enc_in)

        self.loss_fn = get_loss(loss_name, **(loss_kwargs or {}))

    def forward(self, x, y=None, return_loss=False):
        # x: (batch, seq_len, features)
        batch, seq, feat = x.shape

        if self.revin_enabled:
            x = self.revin(x, mode="norm")

        # Embedding
        x_emb = self.embedding(x)  # (B, L, D)

        # Self-attention temporal
        attn_output, _ = self.attention(x_emb, x_emb, x_emb)

        # Usa o último timestep como resumo da janela histórica
        # h = attn_output[:, -1, :]  # (B, D)
        h = attn_output.reshape(batch, self.lookback * self.d_model)

        # Projeção para horizonte futuro
        output = self.projection(h)

        # Reorganiza para matriz futura
        output = output.view(batch, self.pred_len, self.enc_in)  # (B, pred_len, enc_in)

        if self.revin_enabled:
            output = self.revin(output, mode="denorm")

        if return_loss and y is not None:
            loss = self.loss_fn(output, y[:, -self.pred_len:, :])
            return output, loss
        return output

    def get_loss(self, pred, target):
        return self.loss_fn(pred, target)
