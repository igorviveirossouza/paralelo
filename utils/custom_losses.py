import torch
import torch.nn as nn
import torch.nn.functional as F

class DilateLoss(nn.Module):
    """DILATE: Shape + Time Distortion Loss (Le Guen et al., NeurIPS 2019)."""
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha  # peso entre shape e temporal

    def forward(self, pred, target):
        mse = F.mse_loss(pred, target)
        # DTW-like temporal component (simplificado)
        temporal_dist = torch.mean(torch.abs(pred - target))
        return self.alpha * mse + (1 - self.alpha) * temporal_dist

def get_loss(loss_name='mse'):
    if loss_name == 'dilate':
        return DilateLoss(alpha=0.5)
    return nn.MSELoss()
