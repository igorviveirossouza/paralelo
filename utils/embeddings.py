import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (Vaswani et al., 2017)."""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TemporalEmbedding(nn.Module):
    """Embedding para serie temporal multivariada.

    Observacao: esta versao mistura canais, pois aplica nn.Linear(c_in, d_model)
    em cada timestep.
    """
    def __init__(self, c_in, d_model=512, dropout=0.1):
        super().__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.position_embedding = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, seq_len, features)
        x = self.value_embedding(x)
        x = self.position_embedding(x)
        return self.dropout(x)

class ChannelIndependentTemporalEmbedding(nn.Module):
    """Embedding temporal sem mistura entre canais.

    Entrada:
        x: (batch, seq_len, channels)

    Saida:
        x_emb: (batch, channels, seq_len, d_model)

    A dimensao de canais e preservada. Cada canal recebe embedding escalar
    proprio e a codificacao posicional e aplicada ao lookback de cada serie.
    Isso permite processar os canais separadamente na atencao, mantendo tudo
    vetorizado em um unico forward.
    """
    def __init__(self, c_in, d_model=512, dropout=0.1, channel_specific=True):
        super().__init__()
        self.c_in = c_in
        self.d_model = d_model
        self.channel_specific = channel_specific

        if channel_specific:
            self.value_embeddings = nn.ModuleList([
                nn.Linear(1, d_model) for _ in range(c_in)
            ])
        else:
            self.value_embedding = nn.Linear(1, d_model)

        self.position_embedding = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, seq_len, channels)
        if x.dim() != 3:
            raise ValueError(
                f"Esperado x com shape (batch, seq_len, channels), recebido {tuple(x.shape)}"
            )

        batch, seq_len, channels = x.shape
        if channels != self.c_in:
            raise ValueError(f"Esperado {self.c_in} canais, recebido {channels}")

        if self.channel_specific:
            embedded_channels = [
                emb(x[:, :, i:i + 1]).unsqueeze(1)
                for i, emb in enumerate(self.value_embeddings)
            ]
            x_emb = torch.cat(embedded_channels, dim=1)  # (B, N, L, D)
        else:
            x_by_channel = x.transpose(1, 2).unsqueeze(-1)  # (B, N, L, 1)
            x_emb = self.value_embedding(x_by_channel)      # (B, N, L, D)

        # Aplica a mesma codificacao posicional temporal a cada canal sem misturar canais.
        x_flat = x_emb.reshape(batch * channels, seq_len, self.d_model)
        x_flat = self.position_embedding(x_flat)
        x_flat = self.dropout(x_flat)

        return x_flat.reshape(batch, channels, seq_len, self.d_model)
