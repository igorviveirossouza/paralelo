import torch.nn as nn
from utils.custom_losses import get_loss
from utils.revin import RevIN


class RevINModelWrapper(nn.Module):
    def __init__(self, model, enc_in, loss_name='mse', loss_kwargs=None,revin_kwargs=None):
        super().__init__()
        self.model = model
        self.revin = RevIN(enc_in,**revin_kwargs)
        self.loss_fn = get_loss(loss_name, **(loss_kwargs or {}))

    def forward(self, x, y=None, return_loss=False):
        x_norm = self.revin(x, mode='norm')
        output = self.model(x_norm, y=None, return_loss=False)
        output = self.revin(output, mode='denorm')
        if return_loss and y is not None:
            loss = self.loss_fn(output, y[:, -output.size(1):, :])
            return output, loss
        return output

    def get_loss(self, pred, target):
        return self.loss_fn(pred, target)
