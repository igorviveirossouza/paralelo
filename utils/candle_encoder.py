import torch
import torch.nn as nn


class CandleEncoder(nn.Module):
    """Codifica features OHLCV/candle para o mesmo tamanho do embedding temporal.

    Entrada esperada:
        candle_x: (..., candle_input_dim)

    Saida:
        candle_emb: (..., d_model)
    """

    def __init__(self, candle_input_dim=5, d_model=32, hidden_dim=64,
                 dropout=0.1, encoder_type="mlp"):
        super().__init__()
        if candle_input_dim < 1:
            raise ValueError("candle_input_dim deve ser >= 1")

        encoder_type = encoder_type.lower()
        self.candle_input_dim = candle_input_dim
        self.d_model = d_model
        self.encoder_type = encoder_type

        if encoder_type == "linear":
            self.encoder = nn.Linear(candle_input_dim, d_model)
        elif encoder_type == "mlp":
            self.encoder = nn.Sequential(
                nn.Linear(candle_input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, d_model),
            )
        else:
            raise ValueError(f"candle_encoder_type invalido: {encoder_type}")

    def forward(self, candle_x):
        if candle_x.size(-1) != self.candle_input_dim:
            raise ValueError(
                f"Esperado candle_input_dim={self.candle_input_dim}, "
                f"recebido {candle_x.size(-1)}"
            )
        return self.encoder(candle_x)
