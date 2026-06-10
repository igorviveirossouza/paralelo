import torch
import torch.nn as nn

from utils.candle_encoder import CandleEncoder
from utils.custom_losses import get_loss


class CandleEmbeddingWrapper(nn.Module):
    """Adiciona Candle Encoder Fusion ao embedding de qualquer modelo.

    O embedding original continua responsável por transformar x.
    Este wrapper codifica candle_x e funde a representação no mesmo d_model.
    """

    def __init__(self, base_embedding, d_model, candle_input_dim=5,
                 candle_encoder_type="mlp", candle_hidden_dim=64,
                 candle_dropout=0.1):
        super().__init__()
        self.base_embedding = base_embedding
        self.d_model = d_model
        self.candle_x = None
        self.candle_encoder = CandleEncoder(
            candle_input_dim=candle_input_dim,
            d_model=d_model,
            hidden_dim=candle_hidden_dim,
            dropout=candle_dropout,
            encoder_type=candle_encoder_type,
        )
        self.fusion = nn.Linear(2 * d_model, d_model)

    def set_candle_x(self, candle_x):
        self.candle_x = candle_x

    def clear_candle_x(self):
        self.candle_x = None

    def forward(self, x):
        x_emb = self.base_embedding(x)
        candle_x = self.candle_x
        if candle_x is None:
            raise ValueError("Candle Encoder Fusion ativo, mas candle_x não foi informado.")

        if candle_x.dim() != 4:
            raise ValueError(
                "candle_x deve ter shape (batch, seq_len, channels, candle_features). "
                f"Recebido: {tuple(candle_x.shape)}"
            )

        batch, seq_len, channels = x.shape
        if candle_x.shape[:3] != (batch, seq_len, channels):
            raise ValueError(
                "candle_x deve alinhar com x nas dimensões batch, seq_len e channels. "
                f"x={(batch, seq_len, channels)}, candle_x={tuple(candle_x.shape[:3])}"
            )

        candle_emb = self.candle_encoder(candle_x)  # (B, L, N, D)

        if x_emb.dim() == 3:
            # Embedding agregado: (B, L, D). Agrega candle entre canais.
            candle_emb = candle_emb.mean(dim=2)
            return self.fusion(torch.cat([x_emb, candle_emb], dim=-1))

        if x_emb.dim() == 4:
            # Embedding channel-independent: (B, N, L, D).
            candle_emb = candle_emb.permute(0, 2, 1, 3)
            return self.fusion(torch.cat([x_emb, candle_emb], dim=-1))

        raise ValueError(f"Shape inesperado do embedding base: {tuple(x_emb.shape)}")


class CandleFusionModelWrapper(nn.Module):
    """Wrapper genérico para habilitar Candle Encoder Fusion em modelos existentes."""

    def __init__(self, model, d_model, candle_input_dim=5,
                 candle_encoder_type="mlp", candle_hidden_dim=64,
                 candle_dropout=0.1, loss_name="mse", loss_kwargs=None):
        super().__init__()
        if not hasattr(model, "embedding"):
            raise ValueError("O modelo precisa possuir atributo .embedding para usar CandleFusionModelWrapper.")

        self.model = model
        self.forecast_model_name = f"{model.__class__.__name__}CandleFusion"
        self.loss_fn = get_loss(loss_name, **(loss_kwargs or {}))
        self.model.embedding = CandleEmbeddingWrapper(
            base_embedding=model.embedding,
            d_model=d_model,
            candle_input_dim=candle_input_dim,
            candle_encoder_type=candle_encoder_type,
            candle_hidden_dim=candle_hidden_dim,
            candle_dropout=candle_dropout,
        )

    def forward(self, x, y=None, return_loss=False, candle_x=None):
        self.model.embedding.set_candle_x(candle_x)
        try:
            output = self.model(x, y=None, return_loss=False)
        finally:
            self.model.embedding.clear_candle_x()

        if return_loss and y is not None:
            loss = self.loss_fn(output, y[:, -output.size(1):, :])
            return output, loss
        return output

    def get_loss(self, pred, target):
        return self.loss_fn(pred, target)
