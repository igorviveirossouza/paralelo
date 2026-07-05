from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from utils.auc_direcional import calcular_auc_sinais
except Exception:  # pragma: no cover
    try:
        from auc_direcional import calcular_auc_sinais
    except Exception:  # pragma: no cover
        calcular_auc_sinais = None


METRICAS_INTERESSE = [
    "mean_spearman_ic",
    "mean_precision_positive",
    "mean_precision_negative",
    "auc_alta",
    "auc_queda",
    "auc_alta_media_janela",
    "auc_queda_media_janela",
]

METRICAS_COMPLEMENTARES = [
    "total_return",
    "annual_return",
    "annual_vol",
    "sharpe",
    "max_drawdown",
    "icir",
    "mean_n_assets",
    "n_periods",
]

TODAS_METRICAS = METRICAS_INTERESSE + METRICAS_COMPLEMENTARES

DEFAULT_GROUPS = {
    "MASTER": "simulacoes/master_tfb_experimento",
    "TFB": "simulacoes/tfb_multi_lb_predlen_carteiras",
    "RandomTopJ": "simulacoes/RandoTopj",
    "Benchmark": "simulacoes",
}

BENCHMARK_RE = re.compile(r"momentum|equal\s*_?weights?|equalweights?", re.IGNORECASE)


@dataclass(frozen=True)
class GrupoBusca:
    nome: str
    root: Path
    benchmark_only: bool = False


def _safe_json_load(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _as_number(value: Any) -> float | int | np.nan:
    if value is None:
        return np.nan
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    try:
        return float(str(value).replace(",", "."))
    except Exception:
        return np.nan


def _first_dict(*candidates: Any) -> dict[str, Any]:
    for candidate in candidates:
        if isinstance(candidate, dict):
            return candidate
    return {}


def _extract_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    nested = _first_dict(
        payload.get("metrics"),
        payload.get("metricas"),
        payload.get("results"),
        payload.get("resultados"),
        payload.get("performance"),
    )
    source = nested or payload
    return {k: _as_number(source.get(k)) for k in TODAS_METRICAS if k in source}


def _extract_params(payload: dict[str, Any]) -> dict[str, Any]:
    params = _first_dict(
        payload.get("params"),
        payload.get("config"),
        payload.get("args"),
        payload.get("hyperparameters"),
    )
    if params:
        return params
    return {k: v for k, v in payload.items() if k not in {"metrics", "metricas", "results", "resultados"}}


def _find_int_in_text(text: str, patterns: list[str]) -> int | float:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    return np.nan


def _first_param_int(params: dict[str, Any], keys: list[str]) -> int | float:
    for key in keys:
        if key in params and params[key] is not None:
            value = _as_number(params[key])
            if pd.notna(value):
                return int(value)
    return np.nan


def _first_param_str(params: dict[str, Any], keys: list[str]) -> str | float:
    for key in keys:
        value = params.get(key)
        if value not in (None, ""):
            return str(value)
    return np.nan


def _guess_modelo(rel_parts: tuple[str, ...], params: dict[str, Any], grupo: str) -> str:
    value = _first_param_str(params, ["model", "modelo", "model_name", "model_name_tfb", "model_id"])
    if isinstance(value, str):
        return value

    candidates = [p for p in rel_parts[:-1] if not re.search(r"lookback|pred_len|k\d+|top\d+", p, re.I)]
    if grupo == "TFB" and len(candidates) >= 2:
        return candidates[1]
    if grupo == "MASTER":
        for part in candidates:
            if "master" in part.lower():
                return part
        return "MASTER"
    if candidates:
        return candidates[-1]
    return grupo


def _guess_dataset(rel_parts: tuple[str, ...], params: dict[str, Any]) -> str | float:
    value = _first_param_str(params, ["dataset", "data", "data_name", "data_name_tfb", "price_dataset"])
    if isinstance(value, str):
        return Path(value).stem
    if rel_parts:
        return rel_parts[0]
    return np.nan


def _metadata_from_path(json_path: Path, root: Path, grupo: str, params: dict[str, Any]) -> dict[str, Any]:
    try:
        rel = json_path.relative_to(root)
    except ValueError:
        rel = json_path
    rel_parts = rel.parts
    rel_text = "/".join(rel_parts)

    lookback = _first_param_int(params, ["lookback", "lookback_window", "seq_len", "input_len", "window_size"])
    if pd.isna(lookback):
        lookback = _find_int_in_text(rel_text, [r"lookback[_-]?(\d+)", r"lb[_-]?(\d+)", r"seq[_-]?len[_-]?(\d+)"])

    pred_len = _first_param_int(params, ["pred_len", "horizon", "horizonte", "prediction_length", "forecast_horizon"])
    if pd.isna(pred_len):
        pred_len = _find_int_in_text(rel_text, [r"pred[_-]?len[_-]?(\d+)", r"horizon[_-]?(\d+)", r"pred[_-]?(\d+)"])

    janela_trading = _first_param_int(params, ["rebalance_k", "janela_trading", "trading_window", "k", "window"])
    if pd.isna(janela_trading):
        matches = re.findall(r"(?:^|[/_-])k[_-]?(\d+)(?:$|[/_-])", rel_text, flags=re.IGNORECASE)
        janela_trading = int(matches[-1]) if matches else np.nan

    max_assets = _first_param_int(params, ["max_assets", "top_j", "topj", "j", "top"])
    if pd.isna(max_assets):
        max_assets = _find_int_in_text(rel_text, [r"top[_-]?j[_-]?(\d+)", r"top[_-]?(\d+)"])

    return {
        "grupo": grupo,
        "dataset": _guess_dataset(rel_parts, params),
        "modelo": _guess_modelo(rel_parts, params, grupo),
        "lookback": lookback,
        "pred_len": pred_len,
        "janela_trading": janela_trading,
        "horizonte_comparavel": janela_trading,
        "max_assets": max_assets,
        "model_output": _first_param_str(params, ["model_output", "tipo_saida", "output_type"]),
        "returns_mode": _first_param_str(params, ["returns_mode"]),
        "run": json_path.parent.name,
        "json_path": str(json_path),
    }


def _auc_from_sinais(json_path: Path, root: Path) -> dict[str, Any]:
    if calcular_auc_sinais is None:
        return {}
    sinais_path = json_path.parent / "sinais.csv"
    if not sinais_path.exists():
        return {}
    try:
        auc = calcular_auc_sinais(sinais_path, root)
    except Exception:
        return {}
    if not auc:
        return {}
    return {k: auc.get(k) for k in METRICAS_INTERESSE if k.startswith("auc_") and k in auc}


def _valid_result(metrics: dict[str, Any]) -> bool:
    return any(k in metrics for k in TODAS_METRICAS)


def _scan_group(grupo: GrupoBusca) -> list[dict[str, Any]]:
    root = grupo.root
    if not root.exists():
        return []

    rows: list[dict[str, Any]] = []
    for json_path in sorted(root.rglob("*.json")):
        if grupo.benchmark_only and not BENCHMARK_RE.search(str(json_path)):
            continue

        payload = _safe_json_load(json_path)
        if payload is None:
            continue

        metrics = _extract_metrics(payload)
        if not _valid_result(metrics):
            continue

        params = _extract_params(payload)
        auc_metrics = _auc_from_sinais(json_path, root)
        for key, value in auc_metrics.items():
            metrics.setdefault(key, value)

        row = _metadata_from_path(json_path, root, grupo.nome, params)
        row.update({k: metrics.get(k, np.nan) for k in TODAS_METRICAS})
        rows.append(row)
    return rows


def carregar_metricas(grupos: dict[str, str] | None = None, base_dir: str | Path = ".") -> pd.DataFrame:
    base_dir = Path(base_dir)
    grupos = grupos or DEFAULT_GROUPS

    rows: list[dict[str, Any]] = []
    for nome, root_str in grupos.items():
        root = base_dir / root_str
        benchmark_only = nome.lower().startswith("benchmark")
        rows.extend(_scan_group(GrupoBusca(nome=nome, root=root, benchmark_only=benchmark_only)))

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    for col in ["lookback", "pred_len", "janela_trading", "horizonte_comparavel", "max_assets"]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    for col in TODAS_METRICAS:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values(["grupo", "dataset", "modelo", "lookback", "pred_len", "janela_trading", "run"]).reset_index(drop=True)


def resumo_por_modelo(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    group_cols = ["grupo", "dataset", "modelo", "janela_trading"]
    metric_cols = [c for c in METRICAS_INTERESSE if c in df.columns]
    out = (
        df.groupby(group_cols, dropna=False)[metric_cols]
        .agg(["mean", "median", "std", "count"])
        .reset_index()
    )
    out.columns = ["_".join([x for x in col if x]) if isinstance(col, tuple) else col for col in out.columns]
    return out


def top_configs(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    id_cols = [
        "grupo",
        "dataset",
        "modelo",
        "lookback",
        "pred_len",
        "janela_trading",
        "max_assets",
        "run",
        "json_path",
    ]
    rows = []
    for metric in METRICAS_INTERESSE:
        if metric not in df.columns:
            continue
        tmp = df.dropna(subset=[metric]).copy()
        if tmp.empty:
            continue
        tmp = tmp.sort_values(metric, ascending=False).head(n)
        tmp.insert(0, "metrica", metric)
        tmp.insert(1, "valor", tmp[metric])
        rows.append(tmp[["metrica", "valor", *[c for c in id_cols if c in tmp.columns]]])
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def ranking_global(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    for metric in METRICAS_INTERESSE:
        if metric in out.columns:
            out[f"rank_{metric}"] = out[metric].rank(ascending=False, method="min")
    rank_cols = [c for c in out.columns if c.startswith("rank_")]
    if rank_cols:
        out["rank_medio_metricas_interesse"] = out[rank_cols].mean(axis=1, skipna=True)
        out = out.sort_values("rank_medio_metricas_interesse")
    return out


def comparar_global(
    base_dir: str | Path = ".",
    output_dir: str | Path = "simulacoes/comparativo_global_master_tfb",
    grupos: dict[str, str] | None = None,
    top_n: int = 20,
) -> dict[str, pd.DataFrame]:
    base_dir = Path(base_dir)
    output_dir = base_dir / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    metricas = carregar_metricas(grupos=grupos, base_dir=base_dir)
    resumo = resumo_por_modelo(metricas)
    tops = top_configs(metricas, n=top_n)
    ranking = ranking_global(metricas)

    metricas.to_csv(output_dir / "metricas_global_long.csv", index=False)
    resumo.to_csv(output_dir / "resumo_por_modelo.csv", index=False)
    tops.to_csv(output_dir / "top_configs_por_metrica.csv", index=False)
    ranking.to_csv(output_dir / "ranking_global.csv", index=False)

    cols_interesse = [
        "grupo",
        "dataset",
        "modelo",
        "lookback",
        "pred_len",
        "janela_trading",
        "max_assets",
        *[c for c in METRICAS_INTERESSE if c in metricas.columns],
        "json_path",
    ]
    tabela = metricas[[c for c in cols_interesse if c in metricas.columns]].copy() if not metricas.empty else pd.DataFrame()
    tabela.to_csv(output_dir / "tabela_metricas_interesse.csv", index=False)

    return {
        "metricas": metricas,
        "resumo": resumo,
        "top_configs": tops,
        "ranking": ranking,
        "tabela_metricas_interesse": tabela,
    }


def _parse_group_arg(values: list[str] | None) -> dict[str, str] | None:
    if not values:
        return None
    grupos = {}
    for item in values:
        if "=" not in item:
            raise ValueError("Use --group Nome=caminho/da/pasta")
        nome, path = item.split("=", 1)
        grupos[nome] = path
    return grupos


def main() -> None:
    parser = argparse.ArgumentParser(description="Compara metricas globais de MASTER, TFB, RandomTopJ e benchmarks.")
    parser.add_argument("--base_dir", default=".", help="Raiz do repositório paralelo.")
    parser.add_argument("--output_dir", default="simulacoes/comparativo_global_master_tfb")
    parser.add_argument("--top_n", type=int, default=20)
    parser.add_argument("--group", action="append", help="Grupo=caminho. Pode repetir.")
    args = parser.parse_args()

    dfs = comparar_global(
        base_dir=args.base_dir,
        output_dir=args.output_dir,
        grupos=_parse_group_arg(args.group),
        top_n=args.top_n,
    )
    print(f"Linhas carregadas: {len(dfs['metricas'])}")
    print(f"Saídas salvas em: {Path(args.base_dir) / args.output_dir}")


if __name__ == "__main__":
    main()
