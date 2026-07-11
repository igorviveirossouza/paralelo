from __future__ import annotations

import argparse
import json
import re
import sys
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from estrategias.acerto_negativos import calcular_acerto_negativos_sinais


DEFAULT_STATS_ORDER = [
    "total_return",
    "annual_return",
    "annual_vol",
    "sharpe",
    "max_drawdown",
    "mean_precision_positive",
    "taxa_acerto_negativos",
    "mean_precision_negative",
    "mean_spearman_ic",
    "icir",
    "mean_n_assets",
    "n_periods",
    "n_pred_negativos",
    "n_acertos_negativos",
    "n_janelas_com_negativos",
]


def _safe_int_from_token(value: str, prefix: str) -> int | None:
    match = re.search(rf"{re.escape(prefix)}_?(\d+)", value)
    return int(match.group(1)) if match else None


def _first_int_from_parts(parts: tuple[str, ...], prefix: str) -> int | None:
    for part in parts:
        value = _safe_int_from_token(part, prefix)
        if value is not None:
            return value
    return None


def _last_int_from_parts(parts: tuple[str, ...], prefix: str) -> int | None:
    for part in reversed(parts):
        value = _safe_int_from_token(part, prefix)
        if value is not None:
            return value
    return None


def _infer_metadata(
    json_path: Path,
    root: Path,
    payload: dict[str, Any],
    include_pred_len: bool = False,
) -> dict[str, Any]:
    rel_parts = json_path.relative_to(root).parts
    params = payload.get("params", {}) or {}

    tipo_serie = rel_parts[0] if len(rel_parts) >= 1 else "desconhecido"
    modelo = rel_parts[1] if len(rel_parts) >= 2 else "modelo_desconhecido"

    lookback = _first_int_from_parts(rel_parts, "lookback")
    pred_len = _first_int_from_parts(rel_parts, "pred_len")

    # Mecanismo antigo: root/tipo/modelo/lookback_X/k_Y/metricas.json
    # Mecanismo novo:  root/tipo/modelo/lookback_X/pred_len_Y/k_Z/metricas.json
    janela_previsao = _last_int_from_parts(rel_parts, "k")
    if janela_previsao is None:
        janela_previsao = params.get("rebalance_k")

    coluna = f"{tipo_serie}__{modelo}__lookback_{lookback if lookback is not None else 'NA'}"
    if include_pred_len:
        coluna = f"{coluna}__pred_len_{pred_len if pred_len is not None else 'NA'}"

    meta = {
        "tipo_serie": tipo_serie,
        "modelo": modelo,
        "lookback": lookback,
        "janela_previsao": int(janela_previsao),
        "coluna": coluna,
    }

    if include_pred_len:
        meta["pred_len"] = pred_len

    return meta


def _collect_one_metrics(task: tuple[str, str, bool]) -> list[dict[str, Any]]:
    json_path_str, root_str, include_pred_len = task
    json_path = Path(json_path_str)
    root = Path(root_str)

    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    metrics = payload.get("metrics", {}) or {}
    if not metrics:
        return []

    neg_metrics = calcular_acerto_negativos_sinais(json_path.parent / "sinais.csv")
    metrics = {**metrics, **neg_metrics}
    meta = _infer_metadata(json_path, root, payload, include_pred_len=include_pred_len)

    return [
        {**meta, "estatistica": stat, "valor": value, "arquivo": str(json_path)}
        for stat, value in metrics.items()
    ]


def collect_metrics(
    root: str | Path,
    include_pred_len: bool = False,
    workers: int = 1,
) -> pd.DataFrame:
    root = Path(root)
    json_files = sorted(root.glob("**/metricas.json"))
    if not json_files:
        return pd.DataFrame()

    workers = max(1, int(workers))
    tasks = [(str(path), str(root), include_pred_len) for path in json_files]

    if workers == 1:
        chunks = [_collect_one_metrics(task) for task in tasks]
    else:
        max_workers = min(workers, len(tasks))
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            chunks = list(executor.map(_collect_one_metrics, tasks, chunksize=1))

    rows = [row for chunk in chunks for row in chunk]
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
    parser.add_argument(
        "--pred_len",
        action="store_true",
        help="Inclui pred_len como dimensão nas colunas comparativas.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Número de processos usados para ler e calcular as métricas.",
    )
    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers deve ser maior ou igual a 1.")

    metrics_long = collect_metrics(
        args.root,
        include_pred_len=args.pred_len,
        workers=args.workers,
    )
    table = build_comparison_table(metrics_long)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)

    if args.long_output:
        long_output = Path(args.long_output)
        long_output.parent.mkdir(parents=True, exist_ok=True)
        metrics_long.to_csv(long_output, index=False)

    print(f"Comparativo salvo em: {output}")
    print(f"Workers: {args.workers}")
    print(f"Shape: {table.shape}")


if __name__ == "__main__":
    main()
