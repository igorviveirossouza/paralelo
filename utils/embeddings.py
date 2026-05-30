import math

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class LinearScalarEmbedding(nn.Module):
    """Projeção linear escalar: (..., T, 1) -> (..., T, d_model)."""
    def __init__(self, d_model):
        super().__init__()
        self.proj = nn.Linear(1, d_model)

    def forward(self, x):
        return self.proj(x)


class NonlinearMultiFuncEmbedding(nn.Module):
    """Embedding escalar com base não linear simples antes da projeção."""
    def __init__(self, d_model, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(self, x):
        features = torch.cat([
            x,
            x.pow(2),
            torch.sin(x),
            torch.cos(x),
            torch.tanh(x),
        ], dim=-1)
        return self.net(features)


class LagLinearEmbedding(nn.Module):
    """Embedding linear usando janela local de lags do próprio canal."""
    def __init__(self, d_model, lag_size=4):
        super().__init__()
        if lag_size < 1:
            raise ValueError("lag_size deve ser >= 1")
        self.lag_size = lag_size
        self.proj = nn.Linear(lag_size, d_model)

    def forward(self, x):
        original_shape = x.shape[:-2]
        seq_len = x.size(-2)
        x_flat = x.reshape(-1, seq_len, 1).transpose(1, 2)  # (M, 1, T)
        x_pad = F.pad(x_flat, (self.lag_size - 1, 0), mode="replicate")
        windows = x_pad.unfold(dimension=-1, size=self.lag_size, step=1)  # (M, 1, T, lag)
        windows = windows.squeeze(1)
        out = self.proj(windows)
        return out.reshape(*original_shape, seq_len, -1)


class SpectralValueEmbedding(nn.Module):
    """Fourier features do valor escalar, seguidas de projeção."""
    def __init__(self, d_model, spectral_num_freqs=8):
        super().__init__()
        if spectral_num_freqs < 1:
            raise ValueError("spectral_num_freqs deve ser >= 1")
        self.spectral_num_freqs = spectral_num_freqs
        freqs = torch.arange(1, spectral_num_freqs + 1, dtype=torch.float32)
        self.register_buffer("freqs", freqs)
        self.proj = nn.Linear(2 * spectral_num_freqs + 1, d_model)

    def forward(self, x):
        angles = x * self.freqs.view(*([1] * (x.dim() - 1)), -1)
        features = torch.cat([x, torch.sin(angles), torch.cos(angles)], dim=-1)
        return self.proj(features)


class MixedEmbedding(nn.Module):
    """Combina embedding não linear e espectral."""
    def __init__(self, d_model, hidden_dim=64, spectral_num_freqs=8):
        super().__init__()
        self.nonlinear = NonlinearMultiFuncEmbedding(d_model, hidden_dim)
        self.spectral = SpectralValueEmbedding(d_model, spectral_num_freqs)
        self.proj = nn.Linear(2 * d_model, d_model)

    def forward(self, x):
        return self.proj(torch.cat([self.nonlinear(x), self.spectral(x)], dim=-1))


def build_embedding(
    embedding_type: str,
    d_model: int,
    hidden_dim: int = 64,
    lag_size: int = 4,
    spectral_num_freqs: int = 8,
) -> nn.Module:
    embedding_type = embedding_type.lower()
    if embedding_type == "linear":
        return LinearScalarEmbedding(d_model)
    if embedding_type == "nonlinear":
        return NonlinearMultiFuncEmbedding(d_model, hidden_dim)
    if embedding_type == "lag_linear":
        return LagLinearEmbedding(d_model, lag_size)
    if embedding_type == "mixed":
        return MixedEmbedding(d_model, hidden_dim, spectral_num_freqs)
    if embedding_type == "spectral":
        return SpectralValueEmbedding(d_model, spectral_num_freqs)
    raise ValueError(f"embedding_type inválido: {embedding_type}")


def add_embedding_arguments(parser):
    """Adiciona argumentos imputáveis via .sh para configurar o embedding."""
    embedding_group = parser.add_argument_group("embedding")
    embedding_group.add_argument(
        "--embedding_type",
        type=str,
        default="linear",
        choices=["linear", "nonlinear", "lag_linear", "mixed", "spectral"],
        help="Tipo de value embedding usado antes da codificação posicional.",
    )
    embedding_group.add_argument(
        "--embedding_hidden_dim",
        type=int,
        default=64,
        help="Dimensão oculta dos embeddings não lineares.",
    )
    embedding_group.add_argument(
        "--embedding_lag_size",
        type=int,
        default=4,
        help="Quantidade de lags usada pelo embedding lag_linear.",
    )
    embedding_group.add_argument(
        "--spectral_num_freqs",
        type=int,
        default=8,
        help="Número de frequências usado nos embeddings spectral/mixed.",
    )
    return parser


def get_embedding_kwargs_from_args(args):
    return {
        "embedding_type": getattr(args, "embedding_type", "linear"),
        "embedding_hidden_dim": getattr(args, "embedding_hidden_dim", 64),
        "embedding_lag_size": getattr(args, "embedding_lag_size", 4),
        "spectral_num_freqs": getattr(args, "spectral_num_freqs", 8),
    }


class TemporalEmbedding(nn.Module):
    """Embedding para serie temporal multivariada.

    Para c_in > 1, aplica o value embedding escalar em cada canal e agrega por média,
    evitando uma projeção linear direta do vetor multivariado para d_model.
    """
    def __init__(self, c_in, d_model=512, dropout=0.1, embedding_type="linear",
                 embedding_hidden_dim=64, embedding_lag_size=4, spectral_num_freqs=8):
        super().__init__()
        self.c_in = c_in
        self.d_model = d_model
        self.value_embedding = build_embedding(
            embedding_type=embedding_type,
            d_model=d_model,
            hidden_dim=embedding_hidden_dim,
            lag_size=embedding_lag_size,
            spectral_num_freqs=spectral_num_freqs,
        )
        self.position_embedding = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, seq_len, features)
        if x.dim() != 3:
            raise ValueError(
                f"Esperado x com shape (batch, seq_len, features), recebido {tuple(x.shape)}"
            )

        if x.size(-1) == 1:
            x = self.value_embedding(x)
        else:
            x_by_channel = x.unsqueeze(-1)             # (B, L, N, 1)
            x_emb = self.value_embedding(x_by_channel) # (B, L, N, D)
            x = x_emb.mean(dim=2)                      # (B, L, D)

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
    def __init__(self, c_in, d_model=512, dropout=0.1, channel_specific=True,
                 embedding_type="linear", embedding_hidden_dim=64,
                 embedding_lag_size=4, spectral_num_freqs=8):
        super().__init__()
        self.c_in = c_in
        self.d_model = d_model
        self.channel_specific = channel_specific

        embedding_kwargs = dict(
            embedding_type=embedding_type,
            d_model=d_model,
            hidden_dim=embedding_hidden_dim,
            lag_size=embedding_lag_size,
            spectral_num_freqs=spectral_num_freqs,
        )

        if channel_specific:
            self.value_embeddings = nn.ModuleList([
                build_embedding(**embedding_kwargs) for _ in range(c_in)
            ])
        else:
            self.value_embedding = build_embedding(**embedding_kwargs)

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
