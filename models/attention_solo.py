import torch
import torch.nn as nn
from utils.embeddings import TemporalEmbedding
from utils.custom_losses import get_loss

class AttentionSolo(nn.Module):
    """
    Modelo básico: Atenção temporal independente por canal (channel-independent).
    Compatível com interface do TFB / TIOMS.
    """
    def __init__(self, seq_len, pred_len, enc_in=1, d_model=32, n_heads=8,
                 dropout=0.1, loss_name='mse'):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in  # número de canais (variáveis)

        self.embedding = TemporalEmbedding(enc_in, d_model, dropout)

        # Camada de atenção multi-head simples
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        # Decoder linear (projection)
        self.projection = nn.Linear(d_model, enc_in)

        self.loss_fn = get_loss(loss_name)

    def forward(self, x, y=None, return_loss=False):
        # x: (batch, seq_len, features)
        batch, seq, feat = x.shape

        # Embedding
        x_emb = self.embedding(x)  # (B, L, D)

        # Self-attention temporal
        attn_output, _ = self.attention(x_emb, x_emb, x_emb)

        # Projeção para horizonte futuro
        output = self.projection(attn_output[:, -1:, :])  # usa último timestep
        output = output.repeat(1, self.pred_len, 1)  # naive repeat (melhorar depois)

        if return_loss and y is not None:
            loss = self.loss_fn(output, y[:, -self.pred_len:, :])
            return output, loss
        return output

    def get_loss(self, pred, target):
        return self.loss_fn(pred, target)
