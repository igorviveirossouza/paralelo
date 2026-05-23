import torch
from datasets.data_loader import TimeSeriesDataset
from torch.utils.data import DataLoader
from models.attention_solo import AttentionSolo

# Configuração
seq_len = 96
pred_len = 96
batch_size = 8

# Dataset - usando dados B3 disponíveis
dataset = TimeSeriesDataset(
    root_path='/home/workdir/attachments',  
    data_path='b3_daily_financeiro_ohlcv.csv',
    flag='train',
    size=[seq_len, 0, pred_len],
    features='M'
)

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# Modelo
model = AttentionSolo(seq_len=seq_len, pred_len=pred_len, enc_in=7)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Teste forward
for i, (batch_x, batch_y) in enumerate(dataloader):
    batch_x = batch_x.to(device)
    batch_y = batch_y.to(device)
    
    pred, loss = model(batch_x, batch_y, return_loss=True)
    print(f"Batch {i} | Pred shape: {pred.shape} | Loss: {loss.item():.6f}")
    break