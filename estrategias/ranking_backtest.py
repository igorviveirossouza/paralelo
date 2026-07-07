from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

EXCLUDE_COLS = {
    "step", "date", "data", "cols", "h", "horizon", "janela", "window",
    "origin_step", "target_step", "origin_pos", "target_pos",
}


def _read_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path).loc[:, lambda df: ~df.columns.str.startswith("Unnamed")]


def _norm_steps(values: Any) -> pd.Series:
    s = pd.Series(values)
    txt = s.astype(str).str.strip()
    valid = s.notna() & txt.ne("")
    num = pd.to_numeric(txt, errors="coerce")
    if bool(num[valid].notna().all()):
        out = pd.Series(pd.NA, index=s.index, dtype="Int64")
        out.loc[valid] = num.loc[valid].round().astype("Int64")
        return out
    parsed = pd.to_datetime(txt.where(valid), errors="coerce")
    if bool(parsed[valid].notna().all()):
        return parsed.dt.strftime("%Y-%m-%d")
    return txt


def _sort_key(values: Any) -> pd.Series:
    labels = _norm_steps(values)
    valid = labels.notna()
    num = pd.to_numeric(labels, errors="coerce")
    if bool(num[valid].notna().all()):
        return num.astype(float)
    parsed = pd.to_datetime(labels, errors="coerce")
    if bool(parsed[valid].notna().all()):
        return pd.Series(parsed.view("int64"), index=labels.index, dtype="float64")
    return labels.astype(str)


def _ordered_steps(values: Any) -> list[Any]:
    labels = _norm_steps(values)
    tmp = pd.DataFrame({"step": labels}).dropna().drop_duplicates("step")
    tmp["_sort"] = _sort_key(tmp["step"])
    return tmp.sort_values("_sort", kind="mergesort")["step"].tolist()


def _origin_from_window(wide: pd.DataFrame, steps: pd.Series) -> Any:
    if "origin_step" in wide.columns:
        origins = _norm_steps(wide["origin_step"]).dropna().unique()
        if len(origins) != 1:
            raise ValueError("origin_step deve ser constante dentro de cada janela.")
        return origins[0]
    num = pd.to_numeric(steps, errors="coerce")
    if num.notna().all():
        return int(num.min()) - 1
    raise ValueError("Arquivo sem origin_step e step não numérico; regere as previsões.")


def load_prediction_windows(pred_dir: str | Path, *, step_col: str = "step", horizon: int | None = None,
                            file_glob: str = "janela_*.csv") -> pd.DataFrame:
    pred_dir = Path(pred_dir)
    files = sorted(pred_dir.glob(file_glob))
    if not files:
        raise FileNotFoundError(f"Nenhum arquivo encontrado em {pred_dir} com padrão {file_glob!r}.")
    frames: list[pd.DataFrame] = []
    for file_path in files:
        wide = _read_csv(file_path)
        if step_col not in wide.columns:
            raise ValueError(f"Arquivo {file_path} não contém coluna {step_col!r}.")
        if "h" in wide.columns:
            wide = wide.copy()
            wide["h"] = pd.to_numeric(wide["h"], errors="raise").astype(int)
            if horizon is not None:
                wide = wide[wide["h"] <= horizon].copy()
        elif horizon is not None:
            wide = wide.iloc[:horizon].copy()
        if wide.empty:
            continue
        steps = _norm_steps(wide[step_col])
        origin_step = _origin_from_window(wide, steps)
        asset_cols = [c for c in wide.columns if c not in EXCLUDE_COLS]
        if not asset_cols:
            raise ValueError(f"Arquivo {file_path} não contém colunas de papéis.")
        tmp = wide[asset_cols].copy()
        tmp["h"] = wide["h"].values if "h" in wide.columns else np.arange(1, len(tmp) + 1)
        tmp["target_step"] = steps.values
        tmp["origin_step"] = origin_step
        tmp["janela"] = file_path.stem
        id_vars = ["janela", "origin_step", "target_step", "h"]
        for col in ["origin_pos", "target_pos"]:
            if col in wide.columns:
                tmp[col] = pd.to_numeric(wide[col], errors="raise").astype(int).values
                id_vars.append(col)
        frames.append(tmp.melt(id_vars=id_vars, var_name="papel", value_name="y_pred"))
    if not frames:
        raise ValueError(f"Nenhuma previsão válida encontrada em {pred_dir}.")
    return pd.concat(frames, ignore_index=True)


def load_price_data(price_path: str | Path, *, step_col: str = "date", asset_col: str = "cols",
                    price_col: str = "data") -> pd.DataFrame:
    prices = _read_csv(price_path)
    missing = {step_col, asset_col, price_col} - set(prices.columns)
    if missing:
        raise ValueError(f"Preço sem colunas obrigatórias: {sorted(missing)}")
    out = prices[[step_col, asset_col, price_col]].copy()
    out = out.rename(columns={step_col: "step", asset_col: "papel", price_col: "price"})
    out["step"] = _norm_steps(out["step"])
    out["papel"] = out["papel"].astype(str)
    out["price"] = pd.to_numeric(out["price"], errors="coerce")
    out = out.dropna(subset=["step", "price"])
    if out.duplicated(["step", "papel"], keep=False).any():
        ex = out.loc[out.duplicated(["step", "papel"], keep=False), ["step", "papel"]].value_counts().head(20)
        raise ValueError(f"Preço com duplicatas em (step, papel):\n{ex.to_string()}")
    order = pd.DataFrame({"step": _ordered_steps(out["step"])})
    order["step_pos"] = np.arange(len(order), dtype=int)
    return out.merge(order, on="step", how="left").sort_values(["papel", "step_pos"]).reset_index(drop=True)


def _validate_calendar(pred: pd.DataFrame, prices: pd.DataFrame, rebalance_k: int) -> pd.DataFrame:
    pos_map = prices[["step", "step_pos"]].drop_duplicates("step")
    cols = ["janela", "origin_step", "target_step", "h"] + [c for c in ["origin_pos", "target_pos"] if c in pred.columns]
    chk = pred[cols].drop_duplicates().copy()
    chk = chk.merge(pos_map.rename(columns={"step": "origin_step", "step_pos": "origin_step_pos"}), on="origin_step", how="left")
    chk = chk.merge(pos_map.rename(columns={"step": "target_step", "step_pos": "target_step_pos"}), on="target_step", how="left")
    missing = chk[chk[["origin_step_pos", "target_step_pos"]].isna().any(axis=1)]
    if not missing.empty:
        raise ValueError("origin_step/target_step sem correspondência em preços. Exemplos:\n" + missing.head(12).to_string(index=False))
    chk["holding_period"] = chk["target_step_pos"] - chk["origin_step_pos"]
    bad_h = chk[chk["holding_period"] != chk["h"]]
    if not bad_h.empty:
        raise ValueError(
            "Calendário inconsistente: target_step não está h pregões após origin_step. "
            "Provável ordenação lexicográfica. Exemplos:\n" + bad_h.head(12).to_string(index=False)
        )
    if "origin_pos" in chk.columns:
        bad = chk[chk["origin_pos"] != chk["origin_step_pos"]]
        if not bad.empty:
            raise ValueError("origin_pos não bate com posição cronológica dos preços. Regere as previsões. Exemplos:\n" + bad.head(12).to_string(index=False))
    if "target_pos" in chk.columns:
        bad = chk[chk["target_pos"] != chk["target_step_pos"]]
        if not bad.empty:
            raise ValueError("target_pos não bate com posição cronológica dos preços. Regere as previsões. Exemplos:\n" + bad.head(12).to_string(index=False))
    k = chk[chk["h"] == rebalance_k][["janela", "origin_step", "target_step", "origin_step_pos", "target_step_pos"]].drop_duplicates()
    bad_k = k[k["target_step_pos"] - k["origin_step_pos"] != rebalance_k]
    if not bad_k.empty:
        raise ValueError(f"rebalance_k={rebalance_k} não bate com target-origin. Exemplos:\n" + bad_k.head(12).to_string(index=False))
    return k


def build_signals(predictions: pd.DataFrame, prices: pd.DataFrame, *, model_output: str = "returns",
                  rebalance_k: int = 5, returns_mode: str = "step") -> pd.DataFrame:
    if rebalance_k < 1:
        raise ValueError("rebalance_k deve ser >= 1.")
    if rebalance_k > int(predictions["h"].max()):
        raise ValueError(f"rebalance_k={rebalance_k} maior que horizonte disponível={int(predictions['h'].max())}.")
    model_output = model_output.lower()
    returns_mode = returns_mode.lower()
    calendar_k = _validate_calendar(predictions, prices, rebalance_k)
    base = prices.rename(columns={"step": "origin_step", "price": "origin_price", "step_pos": "origin_pos"})
    fut = prices.rename(columns={"step": "target_step", "price": "future_price", "step_pos": "target_pos"})

    if model_output == "returns":
        if returns_mode == "step":
            pred_k = (predictions[predictions["h"].between(1, rebalance_k)]
                      .assign(gross_pred=lambda x: 1.0 + x["y_pred"].astype(float))
                      .groupby(["janela", "origin_step", "papel"], as_index=False)["gross_pred"].prod())
            pred_k["pred_ret_k"] = pred_k["gross_pred"] - 1.0
        elif returns_mode == "cumulative":
            pred_k = predictions[predictions["h"] == rebalance_k].copy()
            pred_k["pred_ret_k"] = pred_k["y_pred"].astype(float)
        else:
            raise ValueError("returns_mode deve ser 'step' ou 'cumulative'.")
        pred_k = pred_k[["janela", "origin_step", "papel", "pred_ret_k"]]
    elif model_output == "log_returns":
        if returns_mode == "step":
            pred_k = (predictions[predictions["h"].between(1, rebalance_k)]
                      .assign(log_pred=lambda x: x["y_pred"].astype(float))
                      .groupby(["janela", "origin_step", "papel"], as_index=False)["log_pred"].sum())
            pred_k["pred_ret_k"] = np.exp(pred_k["log_pred"]) - 1.0
        elif returns_mode == "cumulative":
            pred_k = predictions[predictions["h"] == rebalance_k].copy()
            pred_k["pred_ret_k"] = np.exp(pred_k["y_pred"].astype(float)) - 1.0
        else:
            raise ValueError("returns_mode deve ser 'step' ou 'cumulative'.")
        pred_k = pred_k[["janela", "origin_step", "papel", "pred_ret_k"]]
    elif model_output == "prices":
        pred_k = predictions[predictions["h"] == rebalance_k].copy()
        pred_k = pred_k.merge(base[["origin_step", "papel", "origin_price"]], on=["origin_step", "papel"], how="left")
        pred_k["pred_ret_k"] = pred_k["y_pred"].astype(float) / pred_k["origin_price"] - 1.0
        pred_k = pred_k[["janela", "origin_step", "papel", "pred_ret_k"]]
    else:
        raise ValueError("model_output deve ser 'returns', 'log_returns' ou 'prices'.")

    realized = predictions[predictions["h"] == rebalance_k][["janela", "origin_step", "target_step", "papel"]]
    realized = realized.merge(calendar_k, on=["janela", "origin_step", "target_step"], how="left")
    realized = realized.merge(base[["origin_step", "papel", "origin_price"]], on=["origin_step", "papel"], how="left")
    realized = realized.merge(fut[["target_step", "papel", "future_price"]], on=["target_step", "papel"], how="left")
    realized["real_ret_k"] = realized["future_price"] / realized["origin_price"] - 1.0
    signals = pred_k.merge(
        realized[["janela", "origin_step", "target_step", "papel", "origin_step_pos", "target_step_pos", "real_ret_k"]],
        on=["janela", "origin_step", "papel"], how="inner",
    )
    signals = signals.rename(columns={"origin_step_pos": "origin_pos", "target_step_pos": "target_pos"})
    return signals.dropna(subset=["pred_ret_k", "real_ret_k"])


def simulate_top_j_strategy(signals: pd.DataFrame, *, rebalance_k: int = 5, max_assets: int = 5,
                            only_positive_pred: bool = True, annual_rf: float = 0.043):
    if max_assets < 1:
        raise ValueError("max_assets deve ser >= 1.")
    sort_col = "origin_pos" if "origin_pos" in signals.columns else "origin_step"
    origins = signals[[sort_col]].drop_duplicates().sort_values(sort_col)[sort_col].tolist()
    df = signals[signals[sort_col].isin(set(origins[::rebalance_k]))].copy()
    portfolio_rows, selected_rows, ic_rows = [], [], []
    for _, g0 in df.groupby(sort_col, sort=True):
        g = g0.copy()
        origin_step, target_step = g0["origin_step"].iloc[0], g0["target_step"].iloc[0]
        origin_pos, target_pos = int(g0["origin_pos"].iloc[0]), int(g0["target_pos"].iloc[0])
        ic = g["pred_ret_k"].corr(g["real_ret_k"], method="spearman") if g["pred_ret_k"].nunique() > 1 and g["real_ret_k"].nunique() > 1 else np.nan
        ic_rows.append({"origin_step": origin_step, "origin_pos": origin_pos, "spearman_ic": ic})
        if only_positive_pred:
            g = g[g["pred_ret_k"] > 0]
        g = g.sort_values("pred_ret_k", ascending=False).head(max_assets)
        if g.empty:
            port_ret, precision_positive = 0.0, np.nan
        else:
            g = g.assign(weight=1.0 / len(g))
            port_ret = float((g["weight"] * g["real_ret_k"]).sum())
            precision_positive = float((g["real_ret_k"] > 0).mean())
            selected_rows.append(g)
        portfolio_rows.append({
            "origin_step": origin_step, "target_step": target_step,
            "origin_pos": origin_pos, "target_pos": target_pos,
            "holding_period": int(target_pos - origin_pos),
            "n_assets": int(len(g)), "portfolio_ret_k": port_ret,
            "precision_positive": precision_positive,
        })
    portfolio = pd.DataFrame(portfolio_rows).sort_values("origin_pos") if portfolio_rows else pd.DataFrame()
    selected = pd.concat(selected_rows, ignore_index=True) if selected_rows else pd.DataFrame()
    ic_df = pd.DataFrame(ic_rows).sort_values("origin_pos") if ic_rows else pd.DataFrame()
    if portfolio.empty:
        return portfolio, selected, ic_df, {}
    bad = portfolio[portfolio["holding_period"] != rebalance_k]
    if not bad.empty:
        raise ValueError("Carteira com holding_period diferente de rebalance_k:\n" + bad.head(10).to_string(index=False))
    portfolio["equity"] = (1.0 + portfolio["portfolio_ret_k"]).cumprod()
    portfolio["drawdown"] = portfolio["equity"] / portfolio["equity"].cummax() - 1.0
    periods_per_year = 252.0 / rebalance_k
    n_periods = len(portfolio)
    total_return = float(portfolio["equity"].iloc[-1] - 1.0)
    annual_return = float((1.0 + total_return) ** (periods_per_year / n_periods) - 1.0)
    annual_vol = float(portfolio["portfolio_ret_k"].std(ddof=1) * np.sqrt(periods_per_year))
    ic_std = ic_df["spearman_ic"].std(ddof=1)
    metrics = {
        "total_return": total_return,
        "annual_return": annual_return,
        "annual_vol": annual_vol,
        "sharpe": float((annual_return - annual_rf) / annual_vol) if annual_vol > 0 else np.nan,
        "max_drawdown": float(portfolio["drawdown"].min()),
        "mean_precision_positive": float(portfolio["precision_positive"].mean()),
        "mean_spearman_ic": float(ic_df["spearman_ic"].mean()),
        "icir": float(ic_df["spearman_ic"].mean() / ic_std) if ic_std and ic_std > 0 else np.nan,
        "mean_n_assets": float(portfolio["n_assets"].mean()),
        "n_periods": int(n_periods),
    }
    return portfolio, selected, ic_df, metrics


def run_backtest(**kwargs) -> dict:
    pred_dir = kwargs.pop("pred_dir")
    price_path = kwargs.pop("price_path")
    output_dir = kwargs.pop("output_dir", "simulacoes")
    run_name = kwargs.pop("run_name", None)
    predictions = load_prediction_windows(pred_dir, horizon=kwargs.get("horizon", 24))
    prices = load_price_data(price_path)
    signals = build_signals(predictions, prices, model_output=kwargs.get("model_output", "returns"),
                            rebalance_k=kwargs.get("rebalance_k", 5), returns_mode=kwargs.get("returns_mode", "step"))
    portfolio, selected, ic_df, metrics = simulate_top_j_strategy(
        signals, rebalance_k=kwargs.get("rebalance_k", 5), max_assets=kwargs.get("max_assets", 5),
        only_positive_pred=kwargs.get("only_positive_pred", True), annual_rf=kwargs.get("annual_rf", 0.043)
    )
    if run_name is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"ranking_{kwargs.get('model_output', 'returns')}_k{kwargs.get('rebalance_k', 5)}_top{kwargs.get('max_assets', 5)}_{stamp}"
    out_dir = Path(output_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    signals.to_csv(out_dir / "sinais.csv", index=False)
    portfolio.to_csv(out_dir / "carteira.csv", index=False)
    selected.to_csv(out_dir / "selecionados.csv", index=False)
    ic_df.to_csv(out_dir / "ic.csv", index=False)
    params = {"pred_dir": str(pred_dir), "price_path": str(price_path), **kwargs}
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
    parser = argparse.ArgumentParser(description="Backtest de carteira top-j a partir de previsões h x papel.")
    parser.add_argument("--pred_dir", required=True)
    parser.add_argument("--price_path", required=True)
    parser.add_argument("--output_dir", default="simulacoes")
    parser.add_argument("--model_output", choices=["returns", "log_returns", "prices"], required=True)
    parser.add_argument("--rebalance_k", type=int, default=5)
    parser.add_argument("--max_assets", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--only_positive_pred", type=_bool_arg, default=True)
    parser.add_argument("--returns_mode", choices=["step", "cumulative"], default="step")
    parser.add_argument("--annual_rf", type=float, default=0.043)
    parser.add_argument("--run_name", default=None)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run_backtest(**vars(parse_args())), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
