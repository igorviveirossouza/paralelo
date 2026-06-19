from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_EXCLUDE_COLUMNS = {
    "step",
    "date",
    "data",
    "cols",
    "h",
    "horizon",
    "janela",
    "window",
    "origin_step",
    "target_step",
    "origin_pos",
    "target_pos",
}


def _read_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path).loc[:, lambda df: ~df.columns.str.startswith("Unnamed")]


def _origin_step_from_window(wide: pd.DataFrame, steps: pd.Series) -> int:
    """Usa a origem salva pelo rolling forecast; cai no cálculo antigo se não existir."""
    if "origin_step" not in wide.columns:
        return int(steps.min()) - 1

    origin_steps = pd.to_numeric(wide["origin_step"], errors="raise").astype(int)
    unique_origins = origin_steps.dropna().unique()
    if len(unique_origins) != 1:
        raise ValueError("Coluna 'origin_step' deve ser constante dentro de cada janela.")
    return int(unique_origins[0])


def load_prediction_windows(
    pred_dir: str | Path,
    *,
    step_col: str = "step",
    horizon: int | None = None,
    file_glob: str = "*.csv",
) -> pd.DataFrame:
    """Carrega previsões no formato h x papel.

    Cada arquivo deve representar uma janela de previsão. As linhas são os horizontes
    futuros e as colunas são os papéis. A coluna ``step`` deve indicar o rótulo real
    do alvo previsto em cada horizonte, compatível com ``date`` na base de preços.
    Se a coluna ``origin_step`` existir, ela será usada como data de origem da
    previsão. Caso contrário, mantém o comportamento antigo: ``min(step) - 1``.
    """
    pred_dir = Path(pred_dir)
    files = sorted(pred_dir.glob(file_glob))
    if not files:
        raise FileNotFoundError(f"Nenhum arquivo encontrado em {pred_dir} com padrão {file_glob!r}.")

    frames: list[pd.DataFrame] = []
    for file_path in files:
        wide = _read_csv(file_path)
        if step_col not in wide.columns:
            raise ValueError(f"Arquivo {file_path} não contém coluna {step_col!r}.")

        if horizon is not None:
            wide = wide.iloc[:horizon].copy()

        steps = pd.to_numeric(wide[step_col], errors="raise").astype(int)
        origin_step = _origin_step_from_window(wide, steps)

        asset_cols = [c for c in wide.columns if c not in DEFAULT_EXCLUDE_COLUMNS]
        if not asset_cols:
            raise ValueError(f"Arquivo {file_path} não contém colunas de papéis.")

        tmp = wide[asset_cols].copy()
        tmp["h"] = np.arange(1, len(tmp) + 1)
        tmp["target_step"] = steps.values
        tmp["origin_step"] = origin_step
        tmp["janela"] = file_path.stem

        long = tmp.melt(
            id_vars=["janela", "origin_step", "target_step", "h"],
            var_name="papel",
            value_name="y_pred",
        )
        frames.append(long)

    return pd.concat(frames, ignore_index=True)


def load_price_data(
    price_path: str | Path,
    *,
    step_col: str = "date",
    asset_col: str = "cols",
    price_col: str = "data",
) -> pd.DataFrame:
    """Carrega preços em formato longo: step, papel, preço."""
    prices = _read_csv(price_path)
    required = {step_col, asset_col, price_col}
    missing = required - set(prices.columns)
    if missing:
        raise ValueError(f"Preço sem colunas obrigatórias: {sorted(missing)}")

    out = prices[[step_col, asset_col, price_col]].copy()
    out = out.rename(columns={step_col: "step", asset_col: "papel", price_col: "price"})
    out["step"] = pd.to_numeric(out["step"], errors="raise").astype(int)
    out["price"] = pd.to_numeric(out["price"], errors="coerce")
    return out.dropna(subset=["price"])


def build_signals(
    predictions: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    model_output: str = "returns",
    rebalance_k: int = 5,
    returns_mode: str = "step",
) -> pd.DataFrame:
    """Converte previsões para retorno previsto até k e junta retorno realizado."""
    if rebalance_k < 1:
        raise ValueError("rebalance_k deve ser >= 1.")

    max_h = int(predictions["h"].max())
    if rebalance_k > max_h:
        raise ValueError(f"rebalance_k={rebalance_k} maior que horizonte disponível={max_h}.")

    model_output = model_output.lower()
    returns_mode = returns_mode.lower()

    base = prices.rename(columns={"step": "origin_step", "price": "origin_price"})
    fut = prices.rename(columns={"step": "target_step", "price": "future_price"})

    if model_output == "returns":
        if returns_mode == "step":
            pred_k = (
                predictions[predictions["h"].between(1, rebalance_k)]
                .assign(gross_pred=lambda x: 1.0 + x["y_pred"].astype(float))
                .groupby(["janela", "origin_step", "papel"], as_index=False)["gross_pred"]
                .prod()
            )
            pred_k["pred_ret_k"] = pred_k["gross_pred"] - 1.0
        elif returns_mode == "cumulative":
            pred_k = predictions[predictions["h"] == rebalance_k].copy()
            pred_k["pred_ret_k"] = pred_k["y_pred"].astype(float)
        else:
            raise ValueError("returns_mode deve ser 'step' ou 'cumulative'.")
        pred_k = pred_k[["janela", "origin_step", "papel", "pred_ret_k"]]

    elif model_output == "prices":
        pred_k = predictions[predictions["h"] == rebalance_k].copy()
        pred_k = pred_k.merge(base, on=["origin_step", "papel"], how="left")
        pred_k["pred_ret_k"] = pred_k["y_pred"].astype(float) / pred_k["origin_price"] - 1.0
        pred_k = pred_k[["janela", "origin_step", "papel", "pred_ret_k"]]

    else:
        raise ValueError("model_output deve ser 'returns' ou 'prices'.")

    realized = predictions[predictions["h"] == rebalance_k][
        ["janela", "origin_step", "target_step", "papel"]
    ].merge(base, on=["origin_step", "papel"], how="left").merge(
        fut, on=["target_step", "papel"], how="left"
    )
    realized["real_ret_k"] = realized["future_price"] / realized["origin_price"] - 1.0

    signals = pred_k.merge(
        realized[["janela", "origin_step", "target_step", "papel", "real_ret_k"]],
        on=["janela", "origin_step", "papel"],
        how="inner",
    )
    return signals.dropna(subset=["pred_ret_k", "real_ret_k"])


def simulate_top_j_strategy(
    signals: pd.DataFrame,
    *,
    rebalance_k: int = 5,
    max_assets: int = 5,
    only_positive_pred: bool = True,
    annual_rf: float = 0.043,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Simula carteira long-only equally weighted com top-j papéis."""
    if max_assets < 1:
        raise ValueError("max_assets deve ser >= 1.")

    origins = sorted(signals["origin_step"].unique())
    rebalance_origins = set(origins[::rebalance_k])
    df = signals[signals["origin_step"].isin(rebalance_origins)].copy()

    portfolio_rows: list[dict] = []
    selected_rows: list[pd.DataFrame] = []
    ic_rows: list[dict] = []

    for origin_step, g0 in df.groupby("origin_step", sort=True):
        g = g0.copy()

        if g["pred_ret_k"].nunique() > 1 and g["real_ret_k"].nunique() > 1:
            ic = g["pred_ret_k"].corr(g["real_ret_k"], method="spearman")
        else:
            ic = np.nan

        ic_rows.append({"origin_step": origin_step, "spearman_ic": ic})

        if only_positive_pred:
            g = g[g["pred_ret_k"] > 0]

        g = g.sort_values("pred_ret_k", ascending=False).head(max_assets)

        if len(g) == 0:
            port_ret = 0.0
            precision_positive = np.nan
        else:
            g = g.assign(weight=1.0 / len(g))
            port_ret = float((g["weight"] * g["real_ret_k"]).sum())
            precision_positive = float((g["real_ret_k"] > 0).mean())
            selected_rows.append(g)

        portfolio_rows.append(
            {
                "origin_step": origin_step,
                "target_step": int(g0["target_step"].iloc[0]),
                "n_assets": int(len(g)),
                "portfolio_ret_k": port_ret,
                "precision_positive": precision_positive,
            }
        )

    portfolio = pd.DataFrame(portfolio_rows).sort_values("origin_step")
    selected = pd.concat(selected_rows, ignore_index=True) if selected_rows else pd.DataFrame()
    ic_df = pd.DataFrame(ic_rows).sort_values("origin_step")

    if len(portfolio) == 0:
        metrics = {}
        return portfolio, selected, ic_df, metrics

    portfolio["equity"] = (1.0 + portfolio["portfolio_ret_k"]).cumprod()
    portfolio["drawdown"] = portfolio["equity"] / portfolio["equity"].cummax() - 1.0

    periods_per_year = 252.0 / rebalance_k
    n_periods = len(portfolio)
    total_return = float(portfolio["equity"].iloc[-1] - 1.0)
    annual_return = float((1.0 + total_return) ** (periods_per_year / n_periods) - 1.0)
    annual_vol = float(portfolio["portfolio_ret_k"].std(ddof=1) * np.sqrt(periods_per_year))
    sharpe = float((annual_return - annual_rf) / annual_vol) if annual_vol > 0 else np.nan

    ic_std = ic_df["spearman_ic"].std(ddof=1)
    metrics = {
        "total_return": total_return,
        "annual_return": annual_return,
        "annual_vol": annual_vol,
        "sharpe": sharpe,
        "max_drawdown": float(portfolio["drawdown"].min()),
        "mean_precision_positive": float(portfolio["precision_positive"].mean()),
        "mean_spearman_ic": float(ic_df["spearman_ic"].mean()),
        "icir": float(ic_df["spearman_ic"].mean() / ic_std) if ic_std and ic_std > 0 else np.nan,
        "mean_n_assets": float(portfolio["n_assets"].mean()),
        "n_periods": int(n_periods),
    }
    return portfolio, selected, ic_df, metrics


def run_backtest(
    *,
    pred_dir: str | Path,
    price_path: str | Path,
    output_dir: str | Path = "simulacoes",
    model_output: str = "returns",
    rebalance_k: int = 5,
    max_assets: int = 5,
    horizon: int | None = 24,
    only_positive_pred: bool = True,
    returns_mode: str = "step",
    annual_rf: float = 0.043,
    run_name: str | None = None,
) -> dict:
    predictions = load_prediction_windows(pred_dir, horizon=horizon)
    prices = load_price_data(price_path)
    signals = build_signals(
        predictions,
        prices,
        model_output=model_output,
        rebalance_k=rebalance_k,
        returns_mode=returns_mode,
    )
    portfolio, selected, ic_df, metrics = simulate_top_j_strategy(
        signals,
        rebalance_k=rebalance_k,
        max_assets=max_assets,
        only_positive_pred=only_positive_pred,
        annual_rf=annual_rf,
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if run_name is None:
        run_name = f"ranking_{model_output}_k{rebalance_k}_top{max_assets}_{stamp}"
    out_dir = Path(output_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    signals.to_csv(out_dir / "sinais.csv", index=False)
    portfolio.to_csv(out_dir / "carteira.csv", index=False)
    selected.to_csv(out_dir / "selecionados.csv", index=False)
    ic_df.to_csv(out_dir / "ic.csv", index=False)

    params = {
        "pred_dir": str(pred_dir),
        "price_path": str(price_path),
        "model_output": model_output,
        "rebalance_k": rebalance_k,
        "max_assets": max_assets,
        "horizon": horizon,
        "only_positive_pred": only_positive_pred,
        "returns_mode": returns_mode,
        "annual_rf": annual_rf,
    }
    with open(out_dir / "metricas.json", "w", encoding="utf-8") as f:
        json.dump({"params": params, "metrics": metrics}, f, indent=2, ensure_ascii=False)

    return {"output_dir": str(out_dir), "metrics": metrics}


def _bool_arg(value: str) -> bool:
    value = value.lower().strip()
    if value in {"1", "true", "sim", "yes", "y"}:
        return True
    if value in {"0", "false", "nao", "não", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Use true/false.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backtest de carteira top-j a partir de previsões h x papel."
    )
    parser.add_argument("--pred_dir", required=True, help="Pasta com janela_*.csv.")
    parser.add_argument("--price_path", required=True, help="CSV de preços realizados em formato longo.")
    parser.add_argument("--output_dir", default="simulacoes", help="Pasta raiz para salvar simulações.")
    parser.add_argument("--model_output", choices=["returns", "prices"], required=True)
    parser.add_argument("--rebalance_k", type=int, default=5)
    parser.add_argument("--max_assets", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--only_positive_pred", type=_bool_arg, default=True)
    parser.add_argument("--returns_mode", choices=["step", "cumulative"], default="step")
    parser.add_argument("--annual_rf", type=float, default=0.043)
    parser.add_argument("--run_name", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_backtest(**vars(args))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
