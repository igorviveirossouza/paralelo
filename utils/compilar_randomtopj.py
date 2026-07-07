from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

METRICAS = [
    "mean_spearman_ic",
    "mean_precision_positive",
    "mean_precision_negative",
    "auc_alta",
    "auc_queda",
    "auc_alta_media_janela",
    "auc_queda_media_janela",
    "total_return",
    "annual_return",
    "annual_vol",
    "sharpe",
    "max_drawdown",
    "icir",
    "mean_n_assets",
    "n_periods",
]


def _safe_float(value: Any) -> float:
    if value is None:
        return np.nan
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    try:
        return float(str(value).replace(",", "."))
    except Exception:
        return np.nan


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def _first_dict(*values: Any) -> dict[str, Any]:
    for value in values:
        if isinstance(value, dict):
            return value
    return {}


def _extract_metrics(payload: dict[str, Any]) -> dict[str, float]:
    source = _first_dict(
        payload.get("metrics"),
        payload.get("metricas"),
        payload.get("results"),
        payload.get("resultados"),
        payload.get("performance"),
    ) or payload
    return {k: _safe_float(source.get(k)) for k in METRICAS if k in source}


def _extract_params(payload: dict[str, Any]) -> dict[str, Any]:
    return _first_dict(
        payload.get("params"),
        payload.get("config"),
        payload.get("args"),
        payload.get("hyperparameters"),
    )


def _find_int(text: str, patterns: list[str]) -> int | float:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    return np.nan


def _param_int(params: dict[str, Any], keys: list[str]) -> int | float:
    for key in keys:
        if key in params and params[key] is not None:
            value = _safe_float(params[key])
            if pd.notna(value):
                return int(value)
    return np.nan


def _param_str(params: dict[str, Any], keys: list[str]) -> str | float:
    for key in keys:
        value = params.get(key)
        if value not in (None, ""):
            return str(value)
    return np.nan


def _tipo_dado(text: str, params: dict[str, Any]) -> str | float:
    explicit = _param_str(params, ["tipo_dado", "dataset_plot", "model_output", "tipo_saida", "returns_mode"])
    if isinstance(explicit, str):
        low = explicit.lower()
        if "log" in low:
            return "log-retornos"
        if "return" in low or "retorno" in low:
            return "retornos simples"
        if "price" in low or "preco" in low or "preço" in low:
            return "preços"
    low = text.lower()
    if "log_ret" in low or "log-return" in low or "log_returns" in low:
        return "log-retornos"
    if "retornos_simples" in low or re.search(r"__returns\b", low) or "/returns/" in low:
        return "retornos simples"
    if "prices" in low or "precos" in low or "preços" in low:
        return "preços"
    return np.nan


def _metadata(path: Path, root: Path, payload: dict[str, Any]) -> dict[str, Any]:
    params = _extract_params(payload)
    rel = path.relative_to(root)
    rel_text = "/".join(rel.parts)

    lookback = _param_int(params, ["lookback", "lookback_window", "seq_len", "input_len", "window_size"])
    if pd.isna(lookback):
        lookback = _find_int(rel_text, [r"lookback[_-]?(\d+)", r"lb[_-]?(\d+)", r"seq[_-]?len[_-]?(\d+)"])

    pred_len = _param_int(params, ["pred_len", "horizon", "horizonte", "prediction_length", "forecast_horizon"])
    if pd.isna(pred_len):
        pred_len = _find_int(rel_text, [r"pred[_-]?len[_-]?(\d+)", r"horizon[_-]?(\d+)", r"h[_-]?(\d+)", r"pred[_-]?(\d+)"])

    janela_trading = _param_int(params, ["janela_trading", "rebalance_k", "trading_window", "k", "window"])
    if pd.isna(janela_trading):
        matches = re.findall(r"(?:^|[/_-])k[_-]?(\d+)(?:$|[/_-])", rel_text, flags=re.IGNORECASE)
        janela_trading = int(matches[-1]) if matches else pred_len

    top_j = _param_int(params, ["top_j", "topj", "max_assets", "j", "top"])
    if pd.isna(top_j):
        top_j = _find_int(rel_text, [r"top[_-]?j[_-]?(\d+)", r"top[_-]?(\d+)", r"j[_-]?(\d+)"])

    return {
        "grupo": "RandomTopJ",
        "modelo": "RandomTopJ",
        "tipo_dado": _tipo_dado(rel_text, params),
        "lookback": lookback,
        "pred_len": pred_len,
        "janela_trading": janela_trading,
        "top_j": top_j,
        "run": path.parent.name,
        "json_path": str(path),
    }


def compilar_randomtopj(root: str | Path) -> pd.DataFrame:
    root = Path(root)
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*.json")):
        payload = _read_json(path)
        if payload is None:
            continue
        metrics = _extract_metrics(payload)
        if not metrics:
            continue
        row = _metadata(path, root, payload)
        row.update(metrics)
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    for col in ["lookback", "pred_len", "janela_trading", "top_j"]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    for col in METRICAS:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values(["tipo_dado", "lookback", "pred_len", "janela_trading", "top_j", "run"]).reset_index(drop=True)


def resumir_randomtopj(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    group_cols = ["tipo_dado", "lookback", "pred_len", "janela_trading", "top_j"]
    metric_cols = [c for c in METRICAS if c in df.columns]
    rows = []
    for keys, part in df.groupby(group_cols, dropna=False):
        base = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        base["n"] = len(part)
        for metric in metric_cols:
            s = part[metric].dropna()
            if s.empty:
                continue
            base[f"{metric}_mean"] = s.mean()
            base[f"{metric}_median"] = s.median()
            base[f"{metric}_std"] = s.std(ddof=1) if len(s) > 1 else 0.0
            base[f"{metric}_min"] = s.min()
            base[f"{metric}_max"] = s.max()
            base[f"{metric}_q05"] = s.quantile(0.05)
            base[f"{metric}_q95"] = s.quantile(0.95)
        rows.append(base)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compila JSONs de RandomTopJ em bases longas e resumos.")
    parser.add_argument("--root", default="simulacoes/RandomTopJ", help="Pasta com os JSONs RandomTopJ.")
    parser.add_argument("--out", default="simulacoes/comparativo_global_master_tfb", help="Pasta de saída.")
    args = parser.parse_args()

    root = Path(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = compilar_randomtopj(root)
    resumo = resumir_randomtopj(df)

    df.to_csv(out / "randomtopj_metricas_long.csv", index=False)
    resumo.to_csv(out / "randomtopj_resumo_por_configuracao.csv", index=False)

    print(f"RandomTopJ: {len(df)} simulações compiladas")
    print(f"Saídas: {out / 'randomtopj_metricas_long.csv'} e {out / 'randomtopj_resumo_por_configuracao.csv'}")


if __name__ == "__main__":
    main()
