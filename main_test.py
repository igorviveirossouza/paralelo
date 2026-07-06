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
from models.timexer_ohlcv import TimeXerOHLCV
from models.master import MASTER
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
    "TimeXerOHLCV": TimeXerOHLCV,
    "MASTER": MASTER,
}

DIRECT_CANDLE_MODEL_NAMES = {"TimeXerOHLCV", "MASTER"}
REQUIRED_CANDLE_MODEL_NAMES = {"TimeXerOHLCV"}


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y", "sim"}:
        return True
    if value in {"false", "0", "no", "n", "nao", "não"}:
        return False
    raise argparse.ArgumentTypeError("Valor booleano inválido.")


def resolve_input_file(file_name):
    path = Path(file_name)
    if path.is_absolute():
        candidates = [path]
    else:
        candidates = [
            path,
            Path("data") / file_name,
            Path("attachments") / file_name,
            Path("/home/workdir/attachments") / file_name,
        ]

    for candidate in candidates:
        if candidate.exists():
            print(f"✅ Arquivo encontrado em: {candidate}")
            return str(candidate)

    raise FileNotFoundError(
        f"Arquivo '{file_name}' não encontrado. Procurei em:\n"
        + "\n".join(str(candidate) for candidate in candidates)
    )


def _preview_list(values, max_items=12):
    values = list(values or [])
    if len(values) <= max_items:
        return values
    return values[:max_items] + [f"... +{len(values) - max_items}"]


def add_candle_arguments(parser):
    candle_group = parser.add_argument_group("candle_encoder")
    candle_group.add_argument(
        "--use_candle_encoder",
        type=str2bool,
        default=False,
        help="Ativa Candle Encoder Fusion nos modelos usuais. Modelos diretos usam OHLCV sem wrapper.",
    )
    candle_group.add_argument(
        "--candle_encoder_type",
        type=str,
        default="mlp",
        choices=["linear", "mlp"],
        help="Arquitetura do Candle Encoder Fusion.",
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
        help="Como preparar OHLCV antes do encoder/modelo direto.",
    )
    candle_group.add_argument(
        "--candle_cols",
        type=str,
        nargs="*",
        default=None,
        help="Colunas usadas no modo raw. Default: abertura maxima minima data volume.",
    )
    return parser


def add_market_arguments(parser):
    market_group = parser.add_argument_group("market_features")
    market_group.add_argument(
        "--use_market_features",
        type=str2bool,
        default=None,
        help="Ativa features globais de mercado. Default: True para MASTER, False para os demais.",
    )
    market_group.add_argument(
        "--market_feature_files",
        type=str,
        nargs="*",
        default=["indices.csv"],
        help="Arquivos em data/ com features globais. Ex.: indices.csv FEATURES_1.csv FEATURES_2.csv.",
    )
    market_group.add_argument(
        "--market_feature_mode",
        type=str,
        default="master",
        choices=["master", "raw"],
        help="master cria retornos/rolling stats de Close/Volume; raw usa colunas numéricas diretamente.",
    )
    market_group.add_argument(
        "--market_windows",
        type=int,
        nargs="*",
        default=[5, 10, 20, 30, 60],
        help="Janelas usadas no modo master.",
    )
    market_group.add_argument(
        "--market_date_col",
        type=str,
        default=None,
        help="Coluna de data nos arquivos de mercado. Default: detecta date_pregao/date/datetime.",
    )
    return parser


def add_stock_factor_arguments(parser):
    factor_group = parser.add_argument_group("stock_factors")
    factor_group.add_argument(
        "--use_stock_factors",
        type=str2bool,
        default=False,
        help="Cria fatores Alpha158-like por papel a partir de OHLCV e adiciona ao src do MASTER.",
    )
    factor_group.add_argument(
        "--ohlcv_feature_file",
        type=str,
        default=None,
        help="Arquivo OHLCV usado para fatores/candles quando a base-alvo é retorno/log-retorno. Ex.: b3_daily_tfb_ohlcv.csv.",
    )
    factor_group.add_argument(
        "--stock_factor_mode",
        type=str,
        default="alpha158",
        choices=["alpha158"],
        help="Modo de construção dos fatores por papel.",
    )
    factor_group.add_argument(
        "--stock_factor_windows",
        type=int,
        nargs="*",
        default=[5, 10, 20, 30, 60],
        help="Janelas usadas nos fatores Alpha158.",
    )
    factor_group.add_argument(
        "--stock_factor_normalize",
        type=str2bool,
        default=True,
        help="Aplica RobustZScoreNorm com estatísticas do treino e clip [-3, 3].",
    )
    return parser


def add_timexer_arguments(parser):
    timexer_group = parser.add_argument_group("timexer_ohlcv")
    timexer_group.add_argument("--timexer_patch_len", type=int, default=16)
    timexer_group.add_argument("--timexer_patch_stride", type=int, default=None)
    timexer_group.add_argument("--timexer_num_layers", type=int, default=1)
    timexer_group.add_argument("--timexer_dim_feedforward", type=int, default=None)
    return parser


def add_master_arguments(parser):
    master_group = parser.add_argument_group("master")
    master_group.add_argument("--master_d_model", type=int, default=64)
    master_group.add_argument("--master_t_nhead", type=int, default=4)
    master_group.add_argument("--master_s_nhead", type=int, default=2)
    master_group.add_argument("--master_dropout", type=float, default=0.3)
    master_group.add_argument("--master_beta", type=float, default=5.0)
    master_group.add_argument(
        "--master_target_mode",
        type=str,
        default="returns_cumulative",
        choices=["returns_cumulative", "log_returns_cumulative", "last"],
        help="Como reduzir y[:, 1:h] para o alvo escalar do MASTER.",
    )
    return parser


def salvar_relatorio_loss_treino(train_losses, output_dir):
    if not train_losses:
        return None, None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    loss_df = pd.DataFrame({"epoch": range(1, len(train_losses) + 1), "train_loss": train_losses})
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
    parser.add_argument('--base_de_dados', type=str, default='b3_daily_tfb.csv')
    parser.add_argument('--cols', type=str, default=None, help="None para multivariate, ou nome do ticker")
    parser.add_argument('--lookback', type=int, default=96)
    parser.add_argument('--pred_len', type=int, default=24)
    parser.add_argument('--test_ratio', type=float, default=0.2)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--output_dir', type=str, default='previsoes')
    parser.add_argument('--extra_dirs', type=str, nargs='*', default=[])
    parser.add_argument('--duplicate_policy', type=str, default='error', choices=['error', 'last'], help='Tratamento de duplicatas (date, cols): error ou last.')
    parser.add_argument('--revin', type=str2bool, default=False, help='Ativa RevIN: true/false')
    parser.add_argument("--revin_affine", type=str2bool, default=False)
    parser.add_argument('--model_name', type=str, default='AttentionSoloNaive', choices=list(MODEL_REGISTRY.keys()))
    add_loss_arguments(parser)
    add_embedding_arguments(parser)
    add_candle_arguments(parser)
    add_market_arguments(parser)
    add_stock_factor_arguments(parser)
    add_timexer_arguments(parser)
    add_master_arguments(parser)
    args = parser.parse_args()

    if args.use_stock_factors and args.model_name != "MASTER":
        raise ValueError("--use_stock_factors está implementado apenas para --model_name MASTER.")

    uses_direct_candle_model = args.model_name in DIRECT_CANDLE_MODEL_NAMES
    requires_candle = args.model_name in REQUIRED_CANDLE_MODEL_NAMES
    dataset_uses_candle = args.use_candle_encoder or requires_candle or args.use_stock_factors
    apply_candle_fusion = args.use_candle_encoder and not uses_direct_candle_model
    pass_candle_directly = dataset_uses_candle and uses_direct_candle_model

    if args.use_market_features is None:
        dataset_uses_market = args.model_name == "MASTER"
    else:
        dataset_uses_market = args.use_market_features

    print("Configuração:")
    print(f"  Base de dados: {args.base_de_dados}")
    print(f"  Modelo: {args.model_name}")
    print(f"  Embedding: {args.embedding_type}")
    print(f"  Duplicate policy: {args.duplicate_policy}")
    print(f"  RevIN: {args.revin}")
    print(f"  RevIN affine: {args.revin_affine}")
    print(f"  Candle Encoder Fusion: {apply_candle_fusion}")
    print(f"  OHLCV direto no modelo: {pass_candle_directly and args.use_candle_encoder}")
    print(f"  Stock factors no src: {args.use_stock_factors}")
    print(f"  Features globais de mercado: {dataset_uses_market}")
    if dataset_uses_candle and args.ohlcv_feature_file:
        print(f"  Fonte OHLCV/fatores: {args.ohlcv_feature_file}")
    if dataset_uses_candle and args.use_candle_encoder:
        print(f"  Candle feature mode: {args.candle_feature_mode}")
        if apply_candle_fusion:
            print(f"  Candle Encoder type: {args.candle_encoder_type}")
    if args.use_stock_factors:
        print(f"  Stock factor mode: {args.stock_factor_mode}")
        print(f"  Stock factor windows: {args.stock_factor_windows}")
        print(f"  Stock factor normalize: {args.stock_factor_normalize}")
    if dataset_uses_market:
        print(f"  Market files: {args.market_feature_files}")
        print(f"  Market feature mode: {args.market_feature_mode}")
        print(f"  Market windows: {args.market_windows}")
    if args.model_name == "MASTER":
        print(
            "  MASTER: "
            f"d_model={args.master_d_model} | "
            f"t_nhead={args.master_t_nhead} | "
            f"s_nhead={args.master_s_nhead} | "
            f"beta={args.master_beta} | "
            f"target_mode={args.master_target_mode}"
        )
    print(f"  lookback: {args.lookback} | pred_len/horizonte: {args.pred_len}")
    print(f"  test_ratio: {args.test_ratio} | batch_size: {args.batch_size}")
    print(f"  epochs: {args.epochs} | Loss: {args.loss_name}")
    print(f"  cols: {args.cols if args.cols else 'Multivariate'}\n")

    data_path = resolve_input_file(args.base_de_dados)

    ohlcv_feature_path = None
    if dataset_uses_candle and args.ohlcv_feature_file:
        ohlcv_feature_path = resolve_input_file(args.ohlcv_feature_file)

    market_feature_paths = []
    if dataset_uses_market:
        if not args.market_feature_files:
            raise ValueError("Features de mercado ativas, mas nenhum arquivo foi informado em --market_feature_files.")
        market_feature_paths = [resolve_input_file(file_name) for file_name in args.market_feature_files]

    dataset_kwargs = dict(
        data_path=data_path,
        lookback=args.lookback,
        pred_len=args.pred_len,
        stride=1,
        cols=args.cols,
        test_ratio=args.test_ratio,
        duplicate_policy=args.duplicate_policy,
        use_candle_encoder=args.use_candle_encoder,
        candle_cols=args.candle_cols,
        candle_feature_mode=args.candle_feature_mode,
        feature_source_path=ohlcv_feature_path,
        use_market_features=dataset_uses_market,
        market_feature_files=market_feature_paths,
        market_feature_mode=args.market_feature_mode,
        market_windows=args.market_windows,
        market_date_col=args.market_date_col,
        use_stock_factors=args.use_stock_factors,
        stock_factor_mode=args.stock_factor_mode,
        stock_factor_windows=args.stock_factor_windows,
        stock_factor_normalize=args.stock_factor_normalize,
    )

    train_dataset = TimeSeriesDataset(train=True, **dataset_kwargs)
    test_dataset = TimeSeriesDataset(train=False, **dataset_kwargs)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"\nDataLoader de treino criado com {len(train_loader)} batches\n")

    sample = train_dataset[0]
    sample_x = sample[0]
    enc_in = sample_x.shape[1]
    print(f"Features detectadas: {enc_in}")

    candle_input_dim = None
    if dataset_uses_candle:
        candle_input_dim = len(train_dataset.candle_feature_names)
        print(f"Features por papel no src detectadas: {candle_input_dim}")
        print(f"Nomes src extras: {_preview_list(train_dataset.candle_feature_names)}")

    market_input_dim = None
    if dataset_uses_market:
        market_input_dim = len(train_dataset.market_feature_names)
        print(f"Features de mercado detectadas: {market_input_dim}")
        print(f"Nomes market features: {_preview_list(train_dataset.market_feature_names)}")

    loss_kwargs = get_loss_kwargs_from_args(args)
    model_class = MODEL_REGISTRY[args.model_name]

    model_kwargs = dict(
        lookback=args.lookback,
        pred_len=args.pred_len,
        enc_in=enc_in,
        loss_name=args.loss_name,
        loss_kwargs=loss_kwargs,
        embedding_kwargs=get_embedding_kwargs_from_args(args),
    )

    if args.model_name == "TimeXerOHLCV":
        model_kwargs.update(
            candle_input_dim=candle_input_dim,
            patch_len=args.timexer_patch_len,
            patch_stride=args.timexer_patch_stride,
            num_layers=args.timexer_num_layers,
            dim_feedforward=args.timexer_dim_feedforward,
        )

    if args.model_name == "MASTER":
        model_kwargs.update(
            d_model=args.master_d_model,
            t_nhead=args.master_t_nhead,
            s_nhead=args.master_s_nhead,
            dropout=args.master_dropout,
            beta=args.master_beta,
            candle_input_dim=candle_input_dim,
            use_candle_features=pass_candle_directly,
            market_input_dim=market_input_dim,
            use_market_features=dataset_uses_market,
            master_target_mode=args.master_target_mode,
        )

    model = model_class(**model_kwargs)

    if apply_candle_fusion:
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
            affine=args.revin_affine,
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
        extra_dirs=args.extra_dirs,
        model_name=args.model_name,
    )

    loss_csv, loss_png = salvar_relatorio_loss_treino(train_losses, forecast_dir)
    if loss_csv is not None:
        print(f"✅ Histórico da loss de treino salvo em: {loss_csv}")
        print(f"✅ Gráfico da loss de treino salvo em: {loss_png}")

    print(f"\n✅ Pipeline concluído! Previsões fora da amostra salvas em: {forecast_dir}")

if __name__ == "__main__":
    main()
