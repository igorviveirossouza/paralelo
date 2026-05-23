import torch
import argparse
from datasets.data_loader import TimeSeriesDataset
from torch.utils.data import DataLoader
from models.attention_solo import AttentionSolo

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='b3_daily_financeiro.csv')
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--pred_len', type=int, default=48)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--features', type=str, default='M')
    parser.add_argument('--loss_name', type=str,default='mse')
    args = parser.parse_args()

    print(f"Configuração:")
    print(f"  Dataset: {args.data_path}")
    print(f"  seq_len: {args.seq_len} | pred_len: {args.pred_len}")
    print(f"  batch_size: {args.batch_size} | features: {args.features}")

    # Dataset
    dataset = TimeSeriesDataset(
        root_path='/sonic_home/igor.viveiros/paralelo/data',
        data_path=args.data_path,
        flag='train',
        size=[args.seq_len, 0, args.pred_len],
        features=args.features
    )

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Detectar número de features automaticamente
    sample_x, _ = dataset[0]
    enc_in = sample_x.shape[1]
    print(f"  Features detectadas: {enc_in}")

    # Modelo
    model = AttentionSolo(seq_len=args.seq_len, pred_len=args.pred_len, enc_in=enc_in,loss_name=loss_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"  Device: {device}")

    # Teste forward
    for i, (batch_x, batch_y) in enumerate(dataloader):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        pred, loss = model(batch_x, batch_y, return_loss=True)
        print(f"Batch {i} | Pred shape: {pred.shape} | Loss: {loss.item():.6f}")
        break

    print("✅ Teste concluído com sucesso!")

if __name__ == "__main__":
    main()