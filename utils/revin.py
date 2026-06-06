import torch
import torch.nn as nn


class RevIN(nn.Module):
    """
    RevIN simples para tensores (B, L, N).

    Normaliza cada amostra e canal usando a dimensão temporal e permite
    reverter a escala antes do cálculo da loss.
    """
    def __init__(self, num_features, eps=1e-5, affine=False):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        if affine:
            self.weight = nn.Parameter(torch.ones(1, 1, num_features))
            self.bias = nn.Parameter(torch.zeros(1, 1, num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self._cached_mean = None
        self._cached_std = None

    def forward(self, x, mode):
        if x.dim() != 3:
            raise ValueError(f"RevIN espera tensor (B, L, N), recebido {tuple(x.shape)}")

        if mode == "norm":
            self._cached_mean = x.mean(dim=1, keepdim=True).detach()
            var = x.var(dim=1, keepdim=True, unbiased=False)
            self._cached_std = torch.sqrt(var + self.eps).detach()
            x = (x - self._cached_mean) / self._cached_std
            if self.affine:
                x = x * self.weight + self.bias
            return x

        if mode == "denorm":
            if self._cached_mean is None or self._cached_std is None:
                raise RuntimeError("RevIN denorm chamado antes de norm.")
            if self.affine:
                x = (x - self.bias) / (self.weight + self.eps)
            return x * self._cached_std + self._cached_mean

        raise ValueError("mode deve ser 'norm' ou 'denorm'.")
