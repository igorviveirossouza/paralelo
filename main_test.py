import torch
from datasets.data_loader import TimeSeriesDataset
from torch.utils.data import DataLoader
from models.attention_solo import AttentionSolo

# Configuração
seq_len = 96
pred_len = 96
batch_size = 32

# Dataset
dataset = TimeSeriesDataset(
    root_path='data/',          # ajuste para seu caminho
    data_path='ETTh1.csv',      # exemplo
    flag='train',
    size=[seq_len, 0, pred_len],
    features='M'
)

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Modelo
model = AttentionSolo(seq_len=seq_len, pred_len=pred_len, enc_in=7)  # ETTh1 tem 7 vars
model = model.cuda() if torch.cuda.is_available() else model

# Teste forward
for batch_x, batch_y in dataloader:
    batch_x = batch_x.cuda() if torch.cuda.is_available() else batch_x
    batch_y = batch_y.cuda() if torch.cuda.is_available() else batch_y
    
    pred, loss = model(batch_x, batch_y, return_loss=True)
    print(f"Pred shape: {pred.shape} | Loss: {loss.item():.6f}")
    break
