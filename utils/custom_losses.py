import torch
import torch.nn as nn


class DilateLoss(nn.Module):
    """
    DILATE loss: shape loss via Soft-DTW + temporal distortion loss.

    Referência:
    Le Guen & Thome, "Shape and Time Distortion Loss for Training Deep Time Series Forecasting Models",
    NeurIPS 2019.

    Aceita tensores:
      - [batch, horizon]
      - [batch, horizon, channels]
      - ou formatos em que a dimensão temporal seja a penúltima.
    """

    def __init__(self, alpha=0.5, gamma=0.01):
        super().__init__()

        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha deve estar entre 0 e 1.")

        if gamma <= 0:
            raise ValueError("gamma deve ser positivo.")

        self.alpha = alpha
        self.gamma = gamma

    def _prepare(self, x):
        """
        Converte:
          [B, T]       -> [B, T, 1]
          [B, T, C]    -> [B, T, C]
          [..., T, C]  -> [N, T, C]
        """
        if x.dim() == 2:
            x = x.unsqueeze(-1)

        if x.dim() < 3:
            raise ValueError(
                "DilateLoss espera pred/target com ao menos 2 dimensões."
            )

        t = x.size(-2)
        c = x.size(-1)
        return x.reshape(-1, t, c)

    def _pairwise_distances(self, pred, target):
        """
        pred:   [B, T, C]
        target: [B, T, C]

        Retorna:
          D: [B, T, T]
        """
        diff = pred.unsqueeze(2) - target.unsqueeze(1)
        return torch.sum(diff * diff, dim=-1)

    def _soft_dtw(self, D):
        """
        Versão mais segura para autograd.

        Evita:
          - operações in-place em R[:, i, j];
          - uso direto de inf na programação dinâmica;
          - NaNs em logsumexp/gradiente.

        D: [B, T, T]
        Retorna:
          soft-DTW por amostra: [B]
        """
        batch_size, t, _ = D.shape
        device = D.device
        dtype = D.dtype

        huge_val = torch.tensor(1e8, device=device, dtype=dtype)

        previous_row = torch.full(
            (batch_size, t + 1),
            huge_val,
            device=device,
            dtype=dtype,
        )
        previous_row[:, 0] = 0.0

        rows = [previous_row]

        for i in range(1, t + 1):
            current_row = [huge_val.expand(batch_size)]

            for j in range(1, t + 1):
                r0 = -rows[i - 1][:, j - 1] / self.gamma  # diagonal
                r1 = -rows[i - 1][:, j] / self.gamma      # cima
                r2 = -current_row[j - 1] / self.gamma     # esquerda

                r = torch.stack([r0, r1, r2], dim=-1)
                softmin = -self.gamma * torch.logsumexp(r, dim=-1)

                value = D[:, i - 1, j - 1] + softmin
                current_row.append(value)

            rows.append(torch.stack(current_row, dim=1))

        return rows[-1][:, -1]

    def forward(self, pred, target):
        pred = self._prepare(pred)
        target = self._prepare(target)

        if pred.shape != target.shape:
            raise ValueError(
                f"pred e target devem ter o mesmo shape após preparação: "
                f"{pred.shape} != {target.shape}"
            )

        t = pred.size(1)

        # Matriz de custo entre previsão e alvo.
        D = self._pairwise_distances(pred, target)  # [B, T, T]

        # Termo de forma: Soft-DTW.
        soft_dtw = self._soft_dtw(D)  # [B]
        loss_shape = soft_dtw.mean()

        # Matriz de alinhamento suave.
        # Usar soft_dtw.sum() evita dividir o gradiente pelo batch size.
        alignment = torch.autograd.grad(
            soft_dtw.sum(),
            D,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]  # [B, T, T]

        # Matriz de penalização temporal Omega.
        idx = torch.arange(t, device=pred.device, dtype=pred.dtype)
        omega = (idx[:, None] - idx[None, :]).pow(2)  # [T, T]

        # Termo temporal da DILATE.
        loss_temporal = torch.sum(
            alignment * omega.unsqueeze(0),
            dim=(1, 2),
        ).mean() / (t * t)

        return self.alpha * loss_shape + (1.0 - self.alpha) * loss_temporal


def _str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y", "sim", "s"}:
        return True
    if value in {"false", "0", "no", "n", "nao", "não"}:
        return False
    raise ValueError(f"Valor booleano inválido: {value}")


def add_loss_arguments(parser):
    """Adiciona ao parser os argumentos imputáveis via .sh para configurar losses."""
    loss_group = parser.add_argument_group("loss")
    loss_group.add_argument(
        "--loss",
        "--loss_name",
        dest="loss_name",
        type=str,
        default="mse",
        choices=["mse", "mae", "dilate"],
        help="Função de perda usada no treinamento.",
    )
    loss_group.add_argument(
        "--dilate_alpha",
        "--dilate-alpha",
        dest="dilate_alpha",
        type=float,
        default=0.5,
        help="Peso da componente shape da DILATE. A temporal recebe 1-alpha.",
    )
    loss_group.add_argument(
        "--dilate_gamma",
        "--dilate-gamma",
        dest="dilate_gamma",
        type=float,
        default=0.01,
        help="Parâmetro de suavização do Soft-DTW usado pela DILATE.",
    )
    return parser


def get_loss_kwargs_from_args(args):
    """Extrai dos args apenas os parâmetros necessários para instanciar a loss."""
    return {
        "alpha": getattr(args, "dilate_alpha", 0.5),
        "gamma": getattr(args, "dilate_gamma", 0.01),
    }


def get_loss(loss_name="mse", **loss_kwargs):
    loss_name = loss_name.lower()
    if loss_name == "dilate":
        return DilateLoss(
            alpha=loss_kwargs.get("alpha", 0.5),
            gamma=loss_kwargs.get("gamma", 0.01),
        )
    if loss_name == "mae":
        return nn.L1Loss()
    return nn.MSELoss()
