
#%%
import torch
import argparse
import os
from loader.data_loader import TimeSeriesDataset
from torch.utils.data import DataLoader
from models.attention_solo import AttentionSolo
from trainer.training_loop import Trainer
from forecaster.rolling_forecast import run_one_step_rolling_forecast
from pathlib import Path



def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_de_dados', type=str, default='b3_daily_financeiro.csv')
    parser.add_argument('--lookback', type=int, default=96)
    parser.add_argument('--pred_len', type=int, default=24)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--loss_name', type=str,default='mse')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--output_dir', type=str, default='previsoes')
    args = parser.parse_args()

    print(f"Configuração:")
    print(f"  Dataset: {args.base_de_dados}")
    print(f"  lookback: {args.lookback} | pred_len: {args.pred_len}")
    print(f"  batch_size: {args.batch_size} | epochs:{args.epochs}")
    print(f"Loss: {args.loss_name}")

    # Dataset

    BASE_DIR = Path(__file__).resolve().parents[0]

    dataset = TimeSeriesDataset(
        data_path=f"{BASE_DIR}/data/{args.base_de_dados}",
        lookback=args.lookback,
        pred_len=args.pred_len,
        cols=None,
    )

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"DataLoader criado com {len(dataloader)} batches\n")

    # Teste de um batch
    for i, (batch_x, batch_y) in enumerate(dataloader):
        print(f"Batch {i+1} shapes:")
        print(f"  batch_x: {batch_x.shape}")   # Deve ser (B, T, N)
        print(f"  batch_y: {batch_y.shape}")   # (B, label_len+lookback, N)
        break  # só o primeiro batch

    print("\n✅ Pipeline DataLoader funcionando em formato multivariado (B x T x N)")

#%%  Modelo

    sample_x, _ = dataset[0]
    enc_in = sample_x.shape[1]
    print(f"  Features detectadas: {enc_in}")

    # Modelo
    model = AttentionSolo(
        lookback=args.lookback,
        pred_len=args.pred_len,
        enc_in=enc_in,
        loss_name=args.loss_name
    )
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
    print("\n✅ Carregamento em GPU realizado com sucesso")
#%% Treinamento

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = Trainer(model=model, optimizer=optimizer, device=device)
    for epoch in range(args.epochs):
        train_loss = trainer.train_one_epoch(dataloader)
        print(f"Epoch {epoch + 1}/{args.epochs} | Train loss: {train_loss:.6f}")


    os.makedirs(args.output_dir, exist_ok=True)
    run_one_step_rolling_forecast(model, dataset, output_dir=args.output_dir)
    print(f"✅ Rolling forecast salvo em: {args.output_dir}")

if __name__ == "__main__":
    main()
# %%
