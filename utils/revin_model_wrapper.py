import torch.nn as nn
from utils.custom_losses import get_loss
from utils.revin import RevIN


class RevINModelWrapper(nn.Module):
    def __init__(self, model, enc_in, loss_name='mse', loss_kwargs=None, affine=False):
        super().__init__()
        self.model = model
        self.forecast_model_name = model.__class__.__name__
        self.revin = RevIN(enc_in, affine=affine)
        self.loss_fn = get_loss(loss_name, **(loss_kwargs or {}))

    def forward(self, x, y=None, return_loss=False, candle_x=None):
        x_norm = self.revin(x, mode='norm')
        output = self.model(x_norm, y=None, return_loss=False, candle_x=candle_x)
        output = self.revin(output, mode='denorm')
        if return_loss and y is not None:
            loss = self.loss_fn(output, y[:, -output.size(1):, :])
            return output, loss
        return output

    def get_loss(self, pred, target):
        return self.loss_fn(pred, target)
