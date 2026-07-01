from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from estrategias.ranking_backtest import (
    load_prediction_windows,
    load_price_data,
    simulate_top_j_strategy,
)


def _portfolio_metrics(
    portfolio: pd.DataFrame,
    ic_df: pd.DataFrame,
    *,
    rebalance_k: int,
    annual_rf: float,
) -> dict:
    if len(portfolio) == 0:
        return {}

    portfolio["equity"] = (1.0 + portfolio["portfolio_ret_k"]).cumprod()
    portfolio["drawdown"] = portfolio["equity"] / portfolio["equity"].cummax() - 1.0

    periods_per_year = 252.0 / rebalance_k
    n_periods = len(portfolio)
    total_return = float(portfolio["equity"].iloc[-1] - 1.0)
    annual_return = float((1.0 + total_return) ** (periods_per_year / n_periods) - 1.0)
    annual_vol = float(portfolio["portfolio_ret_k"].std(ddof=1) * np.sqrt(periods_per_year))
    sharpe = float((annual_return - annual_rf) / annual_vol) if annual_vol > 0 else np.nan

    ic_std = ic_df["spearman_ic"].std(ddof=1) if "spearman_ic" in ic_df else np.nan
    ic_mean = ic_df["spearman_ic"].mean() if "spearman_ic" in ic_df else np.nan

    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "annual_vol": annual_vol,
        "sharpe": sharpe,
        "max_drawdown": float(portfolio["drawdown"].min()),
        "mean_precision_positive": float(portfolio["precision_positive"].mean()),
        "mean_spearman_ic": float(ic_mean),
        "icir": float(ic_mean / ic_std) if pd.notna(ic_std) and ic_std > 0 else np.nan,
        "mean_n_assets": float(portfolio["n_assets"].mean()),
        "n_periods": int(n_periods),
    }


def build_realized_from_reference(
    *,
    reference_pred_dir: str | Path,
    prices: pd.DataFrame,
    rebalance_k: int,
    horizon: int | None,
) -> pd.DataFrame:
    """Usa as mesmas janelas/horizontes das previsões neurais como calendário do benchmark."""
    reference = load_prediction_windows(reference_pred_dir, horizon=horizon)
    ref_k = reference[reference["h"] == rebalance_k][
        ["janela", "origin_step", "target_step", "papel"]
    ].drop_duplicates()

    if len(ref_k) == 0:
        raise ValueError(f"Nenhuma janela com h={rebalance_k} em {reference_pred_dir}.")

    base = prices.rename(columns={"step": "origin_step", "price": "origin_price"})
    fut = prices.rename(columns={"step": "target_step", "price": "future_price"})

    realized = ref_k.merge(base, on=["origin_step", "papel"], how="left").merge(
        fut, on=["target_step", "papel"], how="left"
    )
    realized["real_ret_k"] = realized["future_price"] / realized["origin_price"] - 1.0
    return realized.dropna(subset=["origin_price", "future_price", "real_ret_k"])


def build_momentum_signals(
    *,
    realized: pd.DataFrame,
    prices: pd.DataFrame,
    rebalance_k: int,
) -> pd.DataFrame:
    """Momentum simples: retorno passado de k dias usado como previsão para os próximos k dias."""
    past = prices.rename(columns={"step": "past_step", "price": "past_price"})

    signals = realized.copy()
    signals["past_step"] = signals["origin_step"] - rebalance_k
    signals = signals.merge(past, on=["past_step", "papel"], how="left")
    signals["pred_ret_k"] = signals["origin_price"] / signals["past_price"] - 1.0

    return signals[
        ["janela", "origin_step", "target_step", "papel", "pred_ret_k", "real_ret_k"]
    ].dropna(subset=["pred_ret_k", "real_ret_k"])


def build_random_signals(
    *,
    realized: pd.DataFrame,
    random_seed: int,
) -> pd.DataFrame:
    """Atribui scores aleatórios; a carteira top-j será sorteada via ranking desses scores."""
    rng = np.random.default_rng(random_seed)
    signals = realized[["janela", "origin_step", "target_step", "papel", "real_ret_k"]].copy()
    signals["pred_ret_k"] = rng.random(len(signals))
    return signals[["janela", "origin_step", "target_step", "papel", "pred_ret_k", "real_ret_k"]]


def simulate_equal_weight(
    *,
    realized: pd.DataFrame,
    rebalance_k: int,
    annual_rf: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Compra todos os papéis disponíveis com pesos iguais em cada rebalanceamento."""
    origins = sorted(realized["origin_step"].unique())
    rebalance_origins = set(origins[::rebalance_k])
    df = realized[realized["origin_step"].isin(rebalance_origins)].copy()

    portfolio_rows: list[dict] = []
    selected_rows: list[pd.DataFrame] = []
    ic_rows: list[dict] = []

    for origin_step, g in df.groupby("origin_step", sort=True):
        g = g.dropna(subset=["real_ret_k"]).copy()
        ic_rows.append({"origin_step": origin_step, "spearman_ic": np.nan})

        if len(g) == 0:
            port_ret = 0.0
            precision_positive = np.nan
        else:
            g = g.assign(weight=1.0 / len(g), pred_ret_k=np.nan)
            port_ret = float((g["weight"] * g["real_ret_k"]).sum())
            precision_positive = float((g["real_ret_k"] > 0).mean())
            selected_rows.append(g)

        portfolio_rows.append(
            {
                "origin_step": origin_step,
                "target_step": int(g["target_step"].iloc[0]) if len(g) else np.nan,
                "n_assets": int(len(g)),
                "portfolio_ret_k": port_ret,
                "precision_positive": precision_positive,
            }
        )

    portfolio = pd.DataFrame(portfolio_rows).sort_values("origin_step")
    selected = pd.concat(selected_rows, ignore_index=True) if selected_rows else pd.DataFrame()
    ic_df = pd.DataFrame(ic_rows).sort_values("origin_step")
    metrics = _portfolio_metrics(portfolio, ic_df, rebalance_k=rebalance_k, annual_rf=annual_rf)
    return portfolio, selected, ic_df, metrics


def _save_outputs(
    *,
    out_dir: Path,
    signals: pd.DataFrame,
    portfolio: pd.DataFrame,
    selected: pd.DataFrame,
    ic_df: pd.DataFrame,
    params: dict,
    metrics: dict,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    signals.to_csv(out_dir / "sinais.csv", index=False)
    portfolio.to_csv(out_dir / "carteira.csv", index=False)
    selected.to_csv(out_dir / "selecionados.csv", index=False)
    ic_df.to_csv(out_dir / "ic.csv", index=False)

    with open(out_dir / "metricas.json", "w", encoding="utf-8") as f:
        json.dump({"params": params, "metrics": metrics}, f, indent=2, ensure_ascii=False)

    return {"output_dir": str(out_dir), "metrics": metrics}


def run_benchmark(
    *,
    reference_pred_dir: str | Path,
    price_path: str | Path,
    output_dir: str | Path,
    strategy: str,
    rebalance_k: int,
    max_assets: int,
    horizon: int | None,
    only_positive_pred: bool,
    annual_rf: float,
    random_seed: int,
    run_name: str | None,
) -> dict:
    strategy = strategy.lower()
    prices = load_price_data(price_path)
    realized = build_realized_from_reference(
        reference_pred_dir=reference_pred_dir,
        prices=prices,
        rebalance_k=rebalance_k,
        horizon=horizon,
    )

    signals: pd.DataFrame
    if strategy == "momentum":
        signals = build_momentum_signals(realized=realized, prices=prices, rebalance_k=rebalance_k)
        portfolio, selected, ic_df, metrics = simulate_top_j_strategy(
            signals,
            rebalance_k=rebalance_k,
            max_assets=max_assets,
            only_positive_pred=only_positive_pred,
            annual_rf=annual_rf,
        )
    elif strategy == "random_topj":
        signals = build_random_signals(realized=realized, random_seed=random_seed)
        portfolio, selected, ic_df, metrics = simulate_top_j_strategy(
            signals,
            rebalance_k=rebalance_k,
            max_assets=max_assets,
            only_positive_pred=False,
            annual_rf=annual_rf,
        )
    elif strategy == "equal_weight":
        signals = realized[["janela", "origin_step", "target_step", "papel", "real_ret_k"]].copy()
        signals["pred_ret_k"] = np.nan
        signals = signals[["janela", "origin_step", "target_step", "papel", "pred_ret_k", "real_ret_k"]]
        portfolio, selected, ic_df, metrics = simulate_equal_weight(
            realized=realized,
            rebalance_k=rebalance_k,
            annual_rf=annual_rf,
        )
    else:
        raise ValueError("strategy deve ser 'momentum', 'random_topj' ou 'equal_weight'.")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if run_name is None:
        if strategy == "random_topj":
            run_name = f"{strategy}_k{rebalance_k}_top{max_assets}_seed{random_seed}_{stamp}"
        else:
            run_name = f"{strategy}_k{rebalance_k}_top{max_assets}_{stamp}"

    params = {
        "strategy": strategy,
        "reference_pred_dir": str(reference_pred_dir),
        "price_path": str(price_path),
        "rebalance_k": rebalance_k,
        "max_assets": max_assets,
        "horizon": horizon,
        "only_positive_pred": only_positive_pred if strategy == "momentum" else False,
        "annual_rf": annual_rf,
        "random_seed": random_seed if strategy == "random_topj" else None,
    }

    return _save_outputs(
        out_dir=Path(output_dir) / run_name,
        signals=signals,
        portfolio=portfolio,
        selected=selected,
        ic_df=ic_df,
        params=params,
        metrics=metrics,
    )


def _bool_arg(value: str) -> bool:
    value = value.lower().strip()
    if value in {"1", "true", "sim", "yes", "y"}:
        return True
    if value in {"0", "false", "nao", "não", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Use true/false.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmarks de carteira: momentum, random top-j e equal-weight.")
    parser.add_argument("--reference_pred_dir", required=True, help="Pasta com janela_*.csv para ancorar janelas.")
    parser.add_argument("--price_path", required=True, help="CSV de preços realizados em formato longo.")
    parser.add_argument("--output_dir", required=True, help="Pasta raiz da estratégia.")
    parser.add_argument("--strategy", choices=["momentum", "random_topj", "equal_weight"], required=True)
    parser.add_argument("--rebalance_k", type=int, required=True)
    parser.add_argument("--max_assets", type=int, default=9)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--only_positive_pred", type=_bool_arg, default=True)
    parser.add_argument("--annual_rf", type=float, default=0.043)
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--run_name", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_benchmark(**vars(args))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
