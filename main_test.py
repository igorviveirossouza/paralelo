import torch
import argparse
import os
from datasets.data_loader import TimeSeriesDataset
from torch.utils.data import DataLoader
from models.attention_solo import AttentionSolo
from trainer.training_loop import Trainer
from forecaster.rolling_forecast import run_one_step_rolling_forecast

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='b3_daily_financeiro.csv')
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--pred_len', type=int, default=48)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--features', type=str, default='M')
    parser.add_argument('--loss_name', type=str,default='mse')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--output_dir', type=str, default='previsoes')
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
    model = AttentionSolo(
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        enc_in=enc_in,
        loss_name=args.loss_name
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"  Device: {device}")
    print(f"  Meta columns preservadas: {dataset.meta_columns}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = Trainer(model=model, optimizer=optimizer, device=device)
    for epoch in range(args.epochs):
        train_loss = trainer.train_one_epoch(dataloader)
        print(f"Epoch {epoch + 1}/{args.epochs} | Train loss: {train_loss:.6f}")

    # Teste forward
    for i, (batch_x, batch_y) in enumerate(dataloader):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        pred, loss = model(batch_x, batch_y, return_loss=True)
        print(f"Batch {i} | Pred shape: {pred.shape} | Loss: {loss.item():.6f}")
        break

    os.makedirs(args.output_dir, exist_ok=True)
    run_one_step_rolling_forecast(model, dataset, output_dir=args.output_dir)
    print(f"✅ Rolling forecast salvo em: {args.output_dir}")

if __name__ == "__main__":
    main()
