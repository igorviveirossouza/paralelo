import torch
import torch.nn as nn


class DilateLoss(nn.Module):
    """DILATE: shape loss via Soft-DTW + temporal distortion loss.

    Referência: Le Guen & Thome, NeurIPS 2019.

    Espera tensores no formato:
      - [batch, horizon]
      - [batch, horizon, channels]
      - ou qualquer formato em que a dimensão temporal seja a penúltima.
    """

    def __init__(self, alpha=0.5, gamma=0.01, normalize=True):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.normalize = normalize

    def _prepare(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # [B, T, 1]
        if x.dim() < 3:
            raise ValueError("DilateLoss espera pred/target com ao menos 2 dimensões.")

        # Mantém a penúltima dimensão como tempo e achata batch/canais extras.
        t = x.size(-2)
        x = x.reshape(-1, t, x.size(-1))
        return x

    def _pairwise_distances(self, pred, target):
        # [N, T, D] x [N, T, D] -> [N, T, T]
        diff = pred.unsqueeze(2) - target.unsqueeze(1)
        return torch.sum(diff * diff, dim=-1)

    def _soft_dtw(self, D):
        n, t, _ = D.shape
        inf = torch.tensor(float("inf"), device=D.device, dtype=D.dtype)
        R = torch.full((n, t + 2, t + 2), inf, device=D.device, dtype=D.dtype)
        R[:, 0, 0] = 0.0

        for i in range(1, t + 1):
            for j in range(1, t + 1):
                r0 = -R[:, i - 1, j - 1] / self.gamma
                r1 = -R[:, i - 1, j] / self.gamma
                r2 = -R[:, i, j - 1] / self.gamma
                rmax = torch.maximum(torch.maximum(r0, r1), r2)
                softmin = -self.gamma * (
                    torch.log(
                        torch.exp(r0 - rmax)
                        + torch.exp(r1 - rmax)
                        + torch.exp(r2 - rmax)
                    )
                    + rmax
                )
                R[:, i, j] = D[:, i - 1, j - 1] + softmin

        return R[:, t, t]

    def forward(self, pred, target):
        pred = self._prepare(pred)
        target = self._prepare(target)

        if pred.shape != target.shape:
            raise ValueError(
                f"pred e target devem ter o mesmo shape após preparação: "
                f"{pred.shape} != {target.shape}"
            )

        t = pred.size(1)
        D = self._pairwise_distances(pred, target)
        D.requires_grad_(True)

        soft_dtw = self._soft_dtw(D)
        loss_shape = soft_dtw.mean()

        alignment = torch.autograd.grad(
            loss_shape,
            D,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        idx = torch.arange(t, device=pred.device, dtype=pred.dtype)
        omega = (idx[:, None] - idx[None, :]).pow(2)
        if self.normalize and t > 1:
            omega = omega / ((t - 1) ** 2)

        loss_temporal = torch.sum(alignment * omega.unsqueeze(0), dim=(1, 2)).mean()

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
    loss_group.add_argument(
        "--dilate_normalize",
        "--dilate-normalize",
        dest="dilate_normalize",
        type=_str2bool,
        default=True,
        help="Normaliza a penalização temporal da DILATE pelo horizonte.",
    )
    return parser


def get_loss_kwargs_from_args(args):
    """Extrai dos args apenas os parâmetros necessários para instanciar a loss."""
    return {
        "alpha": getattr(args, "dilate_alpha", 0.5),
        "gamma": getattr(args, "dilate_gamma", 0.01),
        "normalize": getattr(args, "dilate_normalize", True),
    }


def get_loss(loss_name="mse", **loss_kwargs):
    loss_name = loss_name.lower()
    if loss_name == "dilate":
        return DilateLoss(
            alpha=loss_kwargs.get("alpha", 0.5),
            gamma=loss_kwargs.get("gamma", 0.01),
            normalize=loss_kwargs.get("normalize", True),
        )
    if loss_name == "mae":
        return nn.L1Loss()
    return nn.MSELoss()
