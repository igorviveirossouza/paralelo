# %%
import torch
import argparse
import os
from pathlib import Path

from loader.data_loader import TimeSeriesDataset
from torch.utils.data import DataLoader

from models.attention_solo import AttentionSolo
from models.attention_solo_naive import AttentionSoloNaive
from models.attention_solo_channel_independent import AttentionSoloChannelIndependent
from trainer.training_loop import Trainer
from forecaster.rolling_forecast import run_one_step_rolling_forecast


MODEL_REGISTRY = {
    "AttentionSoloNaive": AttentionSoloNaive,
    "AttentionSolo": AttentionSolo,
    "AttentionSoloChannelIndependent": AttentionSoloChannelIndependent,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_de_dados', type=str, default='b3_daily_financeiro.csv')
    parser.add_argument('--cols', type=str, default=None, help="None para multivariate, ou nome do ticker")
    parser.add_argument('--lookback', type=int, default=96)
    parser.add_argument('--pred_len', type=int, default=24)
    parser.add_argument('--test_ratio', type=float, default=0.2)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--loss_name', type=str, default='mse')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--output_dir', type=str, default='previsoes')
    parser.add_argument('--extra_dirs', type=str, nargs='*', default=[])
    parser.add_argument(
        '--model_name',
        type=str,
        default='AttentionSoloNaive',
        choices=list(MODEL_REGISTRY.keys()),
        help='Modelo a ser treinado/executado'
    )
    args = parser.parse_args()

    print(f"Configuração:")
    print(f"  Base de dados: {args.base_de_dados}")
    print(f"  Modelo: {args.model_name}")
    print(f"  lookback: {args.lookback} | pred_len: {args.pred_len}")
    print(f"  test_ratio: {args.test_ratio} | batch_size: {args.batch_size}")
    print(f"  epochs: {args.epochs} | Loss: {args.loss_name}")
    print(f"  cols: {args.cols if args.cols else 'Multivariate'}\n")

    # ====================== RESOLUÇÃO DO CAMINHO ======================
    possible_paths = [
        args.base_de_dados,                                           # caminho direto
        f"data/{args.base_de_dados}",                                 # pasta data (principal)
        str(Path("data") / args.base_de_dados),
        f"attachments/{args.base_de_dados}",                          # fallback
        str(Path("/home/workdir/attachments") / args.base_de_dados),
    ]

    data_path = None
    for p in possible_paths:
        if os.path.exists(p):
            data_path = p
            print(f"✅ Arquivo encontrado em: {data_path}")
            break

    if data_path is None:
        raise FileNotFoundError(
            f"Arquivo '{args.base_de_dados}' não encontrado.\n"
            f"Procurei em:\n" + "\n".join(possible_paths)
        )

    # ====================== DATASETS ======================
    train_dataset = TimeSeriesDataset(
        data_path=data_path,
        lookback=args.lookback,
        pred_len=args.pred_len,
        stride=1,
        cols=args.cols,
        train=True,
        test_ratio=args.test_ratio
    )

    test_dataset = TimeSeriesDataset(
        data_path=data_path,
        lookback=args.lookback,
        pred_len=args.pred_len,
        stride=1,
        cols=args.cols,
        train=False,
        test_ratio=args.test_ratio
    )

    # DataLoader para treino
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"\nDataLoader de treino criado com {len(train_loader)} batches\n")

    # ====================== MODELO ======================
    sample_x, _ = train_dataset[0]
    enc_in = sample_x.shape[1]
    print(f"Features detectadas: {enc_in}")

    model_class = MODEL_REGISTRY[args.model_name]
    model = model_class(
        lookback=args.lookback,
        pred_len=args.pred_len,
        enc_in=enc_in,
        loss_name=args.loss_name
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Modelo carregado: {model.__class__.__name__}")
    print(f"Modelo carregado no device: {device}")

    # ====================== TREINAMENTO ======================
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = Trainer(model=model, optimizer=optimizer, device=device)

    print("\nIniciando treinamento...")
    for epoch in range(args.epochs):
        train_loss = trainer.train_one_epoch(train_loader)
        if epoch % 5 == 0 or epoch == len(args.epochs)-1:
            print(f"Epoch {epoch + 1}/{args.epochs} | Train loss: {train_loss:.6f}")

    # ====================== ROLLING FORECAST (APENAS TESTE) ======================
    print("\nIniciando Rolling Forecast no conjunto de TESTE (fora da amostra)...")
    
    forecast_dir = run_one_step_rolling_forecast(
        model=model,
        dataset=test_dataset,
        output_dir=args.output_dir,
        dataset_name=args.base_de_dados,
        extra_dirs=args.extra_dirs
    )

    print(f"\n✅ Pipeline concluído! Previsões fora da amostra salvas em: {forecast_dir}")

if __name__ == "__main__":
    main()
