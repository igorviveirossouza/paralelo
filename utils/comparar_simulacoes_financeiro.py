from __future__ import annotations

import argparse
import json
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_STATS_ORDER = [
    "total_return",
    "annual_return",
    "annual_vol",
    "sharpe",
    "max_drawdown",
    "mean_precision_positive",
    "mean_spearman_ic",
    "icir",
    "mean_n_assets",
    "n_periods",
]


def _safe_int_from_token(value: str, prefix: str) -> int | None:
    match = re.search(rf"{re.escape(prefix)}_?(\d+)", value)
    return int(match.group(1)) if match else None


def _infer_metadata(json_path: Path, root: Path, payload: dict[str, Any]) -> dict[str, Any]:
    rel_parts = json_path.relative_to(root).parts
    params = payload.get("params", {}) or {}

    tipo_serie = rel_parts[0] if len(rel_parts) >= 1 else "desconhecido"
    modelo = rel_parts[1] if len(rel_parts) >= 2 else "modelo_desconhecido"
    lookback_dir = rel_parts[2] if len(rel_parts) >= 3 else "lookback_desconhecido"
    janela_dir = rel_parts[3] if len(rel_parts) >= 4 else json_path.parent.name

    lookback = _safe_int_from_token(lookback_dir, "lookback")
    janela_previsao = _safe_int_from_token(janela_dir, "k")

    if janela_previsao is None:
        janela_previsao = params.get("rebalance_k")

    coluna = f"{tipo_serie}__{modelo}__lookback_{lookback if lookback is not None else 'NA'}"

    return {
        "tipo_serie": tipo_serie,
        "modelo": modelo,
        "lookback": lookback,
        "janela_previsao": int(janela_previsao),
        "coluna": coluna,
    }


def collect_metrics(root: str | Path) -> pd.DataFrame:
    root = Path(root)
    json_files = sorted(root.glob("**/metricas.json"))
    rows: list[dict[str, Any]] = []

    for json_path in json_files:
        with open(json_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        metrics = payload.get("metrics", {}) or {}
        if not metrics:
            continue

        meta = _infer_metadata(json_path, root, payload)
        for stat, value in metrics.items():
            rows.append({**meta, "estatistica": stat, "valor": value, "arquivo": str(json_path)})

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def build_comparison_table(metrics_long: pd.DataFrame) -> pd.DataFrame:
    if metrics_long.empty:
        return metrics_long

    stats_seen = list(OrderedDict.fromkeys(metrics_long["estatistica"].tolist()))
    stats_order = [s for s in DEFAULT_STATS_ORDER if s in stats_seen]
    stats_order.extend([s for s in stats_seen if s not in stats_order])

    table = metrics_long.pivot_table(
        index=["janela_previsao", "estatistica"],
        columns="coluna",
        values="valor",
        aggfunc="first",
    ).reset_index()

    table["estatistica"] = pd.Categorical(table["estatistica"], categories=stats_order, ordered=True)
    table = table.sort_values(["janela_previsao", "estatistica"]).reset_index(drop=True)
    table["estatistica"] = table["estatistica"].astype(str)

    model_cols = sorted([c for c in table.columns if c not in {"janela_previsao", "estatistica"}])
    return table[["janela_previsao", "estatistica", *model_cols]]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Une metricas.json de simulacoes financeiras em uma tabela comparativa."
    )
    parser.add_argument("--root", default="simulacoes/financeiro", help="Raiz das simulações.")
    parser.add_argument(
        "--output",
        default="simulacoes/financeiro/comparativo_metricas.csv",
        help="CSV de saída.",
    )
    parser.add_argument(
        "--long_output",
        default=None,
        help="Opcional: salva também a base longa antes do pivot.",
    )
    args = parser.parse_args()

    metrics_long = collect_metrics(args.root)
    table = build_comparison_table(metrics_long)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)

    if args.long_output:
        long_output = Path(args.long_output)
        long_output.parent.mkdir(parents=True, exist_ok=True)
        metrics_long.to_csv(long_output, index=False)

    print(f"Comparativo salvo em: {output}")
    print(f"Shape: {table.shape}")


if __name__ == "__main__":
    main()
