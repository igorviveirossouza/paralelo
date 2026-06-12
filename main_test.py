# %%
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / f"matplotlib-{os.getuid()}"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import torch
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from loader.data_loader import TimeSeriesDataset
from torch.utils.data import DataLoader

from models.attention_solo import AttentionSolo
from models.attention_solo_naive import AttentionSoloNaive
from models.attention_solo_channel_independent import AttentionSoloChannelIndependent
from models.attention_solo_channel_independent_shared_specific import AttentionSoloChannelIndependentSharedSpecific
from models.attention_solo_channel_independent_sharedINspecific import AttentionSoloChannelIndependentSharedINSpecific
from models.transformer import TransformerChannelIndependent
from models.transformer_shared_specific import TransformerChannelIndependentSharedSpecific
from models.transformer_sharedINspecific import TransformerChannelIndependentSharedINSpecific
from trainer.training_loop import Trainer
from forecaster.rolling_forecast import run_one_step_rolling_forecast
from utils.custom_losses import add_loss_arguments, get_loss_kwargs_from_args
from utils.embeddings import add_embedding_arguments, get_embedding_kwargs_from_args
from utils.revin_model_wrapper import RevINModelWrapper
from utils.candle_fusion_wrapper import CandleFusionModelWrapper

MODEL_REGISTRY = {
    "AttentionSoloNaive": AttentionSoloNaive,
    "AttentionSolo": AttentionSolo,
    "AttentionSoloChannelIndependent": AttentionSoloChannelIndependent,
    "AttentionSoloChannelIndependentSharedSpecific": AttentionSoloChannelIndependentSharedSpecific,
    "AttentionSoloChannelIndependentShrINSpec": AttentionSoloChannelIndependentSharedINSpecific,
    "Transformer": TransformerChannelIndependent,
    "TransformerSpecific": TransformerChannelIndependentSharedSpecific,
    "TransformerShrINSpec": TransformerChannelIndependentSharedINSpecific,
}


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y", "sim"}:
        return True
    if value in {"false", "0", "no", "n", "nao", "não"}:
        return False
    raise argparse.ArgumentTypeError("Valor booleano inválido.")


def add_candle_arguments(parser):
    candle_group = parser.add_argument_group("candle_encoder")
    candle_group.add_argument(
        "--use_candle_encoder",
        type=str2bool,
        default=False,
        help="Ativa Candle Encoder Fusion: true/false.",
    )
    candle_group.add_argument(
        "--candle_encoder_type",
        type=str,
        default="mlp",
        choices=["linear", "mlp"],
        help="Arquitetura do Candle Encoder.",
    )
    candle_group.add_argument(
        "--candle_hidden_dim",
        type=int,
        default=64,
        help="Dimensão oculta do Candle Encoder MLP.",
    )
    candle_group.add_argument(
        "--candle_dropout",
        type=float,
        default=0.1,
        help="Dropout do Candle Encoder.",
    )
    candle_group.add_argument(
        "--candle_feature_mode",
        type=str,
        default="ohlcv_relative",
        choices=["ohlcv_relative", "raw"],
        help="Como preparar OHLCV antes do encoder.",
    )
    candle_group.add_argument(
        "--candle_cols",
        type=str,
        nargs="*",
        default=None,
        help="Colunas usadas no modo raw. Default: abertura maxima minima data volume.",
    )
    return parser


def salvar_relatorio_loss_treino(train_losses, output_dir):
    if not train_losses:
        return None, None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    loss_df = pd.DataFrame({
        "epoch": range(1, len(train_losses) + 1),
        "train_loss": train_losses,
    })

    csv_path = output_dir / "train_loss.csv"
    png_path = output_dir / "train_loss.png"

    loss_df.to_csv(csv_path, index=False)

    plt.figure(figsize=(10, 4))
    plt.plot(loss_df["epoch"], loss_df["train_loss"], marker="o", linewidth=1.8)
    plt.title("Loss de treino por época")
    plt.xlabel("Época")
    plt.ylabel("Train loss")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close()

    return csv_path, png_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_de_dados', type=str, default='b3_daily_financeiro.csv')
    parser.add_argument('--cols', type=str, default=None, help="None para multivariate, ou nome do ticker")
    parser.add_argument('--lookback', type=int, default=96)
    parser.add_argument('--pred_len', type=int, default=24)
    parser.add_argument('--test_ratio', type=float, default=0.2)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--output_dir', type=str, default='previsoes')
    parser.add_argument('--extra_dirs', type=str, nargs='*', default=[])
    parser.add_argument('--revin', type=str2bool, default=False, help='Ativa RevIN: true/false')
    parser.add_argument("--revin_affine", type=str2bool, default=False)
    parser.add_argument(
        '--model_name',
        type=str,
        default='AttentionSoloNaive',
        choices=list(MODEL_REGISTRY.keys()),
        help='Modelo a ser treinado/executado'
    )
    add_loss_arguments(parser)
    add_embedding_arguments(parser)
    add_candle_arguments(parser)
    args = parser.parse_args()

    print(f"Configuração:")
    print(f"  Base de dados: {args.base_de_dados}")
    print(f"  Modelo: {args.model_name}")
    print(f"  Embedding: {args.embedding_type}")
    print(f"  RevIN: {args.revin}")
    print(f"  RevIN affine: {args.revin_affine}")
    print(f"  Candle Encoder Fusion: {args.use_candle_encoder}")
    if args.use_candle_encoder:
        print(f"  Candle Encoder type: {args.candle_encoder_type}")
        print(f"  Candle feature mode: {args.candle_feature_mode}")
    print(f"  lookback: {args.lookback} | pred_len: {args.pred_len}")
    print(f"  test_ratio: {args.test_ratio} | batch_size: {args.batch_size}")
    print(f"  epochs: {args.epochs} | Loss: {args.loss_name}")
    print(f"  cols: {args.cols if args.cols else 'Multivariate'}\n")

    possible_paths = [
        args.base_de_dados,
        f"data/{args.base_de_dados}",
        str(Path("data") / args.base_de_dados),
        f"attachments/{args.base_de_dados}",
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

    dataset_kwargs = dict(
        data_path=data_path,
        lookback=args.lookback,
        pred_len=args.pred_len,
        stride=1,
        cols=args.cols,
        test_ratio=args.test_ratio,
        use_candle_encoder=args.use_candle_encoder,
        candle_cols=args.candle_cols,
        candle_feature_mode=args.candle_feature_mode,
    )

    train_dataset = TimeSeriesDataset(
        train=True,
        **dataset_kwargs,
    )

    test_dataset = TimeSeriesDataset(
        train=False,
        **dataset_kwargs,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"\nDataLoader de treino criado com {len(train_loader)} batches\n")

    sample = train_dataset[0]
    sample_x = sample[0]
    enc_in = sample_x.shape[1]
    print(f"Features detectadas: {enc_in}")

    candle_input_dim = None
    if args.use_candle_encoder:
        candle_input_dim = len(train_dataset.candle_feature_names)
        print(f"Features do candle detectadas: {candle_input_dim} | {train_dataset.candle_feature_names}")

    loss_kwargs = get_loss_kwargs_from_args(args)
    model_class = MODEL_REGISTRY[args.model_name]

    model = model_class(
        lookback=args.lookback,
        pred_len=args.pred_len,
        enc_in=enc_in,
        loss_name=args.loss_name,
        loss_kwargs=loss_kwargs,
        embedding_kwargs=get_embedding_kwargs_from_args(args)
    )

    if args.use_candle_encoder:
        model = CandleFusionModelWrapper(
            model=model,
            d_model=getattr(model, "d_model", 32),
            candle_input_dim=candle_input_dim,
            candle_encoder_type=args.candle_encoder_type,
            candle_hidden_dim=args.candle_hidden_dim,
            candle_dropout=args.candle_dropout,
            loss_name=args.loss_name,
            loss_kwargs=loss_kwargs,
        )

    if args.revin:
        model = RevINModelWrapper(
            model=model,
            enc_in=enc_in,
            loss_name=args.loss_name,
            loss_kwargs=loss_kwargs,
            affine=args.revin_affine
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Modelo carregado: {model.__class__.__name__}")
    print(f"Modelo de forecast: {getattr(model, 'forecast_model_name', model.__class__.__name__)}")
    print(f"Modelo carregado no device: {device}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = Trainer(model=model, optimizer=optimizer, device=device)

    train_losses = []
    print("\nIniciando treinamento...")
    for epoch in range(args.epochs):
        train_loss = trainer.train_one_epoch(train_loader)
        train_losses.append(train_loss)

        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            print(f"Epoch {epoch + 1}/{args.epochs} | Train loss: {train_loss:.6f}")

    print("\nIniciando Rolling Forecast no conjunto de TESTE (fora da amostra)...")
    forecast_dir = run_one_step_rolling_forecast(
        model=model,
        dataset=test_dataset,
        output_dir=args.output_dir,
        dataset_name=args.base_de_dados,
        extra_dirs=args.extra_dirs
    )

    loss_csv, loss_png = salvar_relatorio_loss_treino(train_losses, forecast_dir)
    if loss_csv is not None:
        print(f"✅ Histórico da loss de treino salvo em: {loss_csv}")
        print(f"✅ Gráfico da loss de treino salvo em: {loss_png}")

    print(f"\n✅ Pipeline concluído! Previsões fora da amostra salvas em: {forecast_dir}")

if __name__ == "__main__":
    main()
