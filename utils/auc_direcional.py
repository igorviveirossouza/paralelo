from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


def auc_rank(y_true, score) -> float:
    """AUC binária por ranks, sem depender de sklearn."""
    y = pd.Series(y_true).astype(float)
    s = pd.Series(score).astype(float)
    valid = y.notna() & s.notna()
    y = y[valid].astype(int)
    s = s[valid]

    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return np.nan

    ranks = s.rank(method="average")
    rank_sum_pos = float(ranks[y == 1].sum())
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _int_from_parts(parts: tuple[str, ...], prefix: str) -> int | float:
    for part in parts:
        match = re.search(rf"{re.escape(prefix)}_?(\d+)", part)
        if match:
            return int(match.group(1))
    return np.nan


def _last_int_from_parts(parts: tuple[str, ...], prefix: str) -> int | float:
    for part in reversed(parts):
        match = re.search(rf"{re.escape(prefix)}_?(\d+)", part)
        if match:
            return int(match.group(1))
    return np.nan


def metadata_sinais(sinais_path: Path, root: Path) -> dict:
    parts = sinais_path.relative_to(root).parts
    dataset = parts[0] if len(parts) > 0 else "desconhecido"
    modelo = parts[1] if len(parts) > 1 else "modelo_desconhecido"

    return {
        "dataset": dataset,
        "modelo": modelo,
        "lookback": _int_from_parts(parts, "lookback"),
        "pred_len": _int_from_parts(parts, "pred_len"),
        "janela_trading": _last_int_from_parts(parts, "k"),
        "sinais_path": str(sinais_path),
    }


def calcular_auc_sinais(sinais_path: str | Path, root: str | Path) -> dict | None:
    sinais_path = Path(sinais_path)
    root = Path(root)
    sinais = pd.read_csv(sinais_path)

    if not {"pred_ret_k", "real_ret_k"}.issubset(sinais.columns):
        return None

    pred = pd.to_numeric(sinais["pred_ret_k"], errors="coerce")
    real = pd.to_numeric(sinais["real_ret_k"], errors="coerce")

    meta = metadata_sinais(sinais_path, root)
    auc_alta = auc_rank(real > 0, pred)
    auc_queda = auc_rank(real < 0, -pred)

    auc_alta_janelas = []
    auc_queda_janelas = []
    if "origin_step" in sinais.columns:
        tmp = sinais.assign(_pred=pred, _real=real)
        for _, g in tmp.groupby("origin_step"):
            auc_alta_janelas.append(auc_rank(g["_real"] > 0, g["_pred"]))
            auc_queda_janelas.append(auc_rank(g["_real"] < 0, -g["_pred"]))

    return {
        **meta,
        "auc_alta": auc_alta,
        "auc_queda": auc_queda,
        "auc_alta_media_janela": float(np.nanmean(auc_alta_janelas)) if auc_alta_janelas else np.nan,
        "auc_queda_media_janela": float(np.nanmean(auc_queda_janelas)) if auc_queda_janelas else np.nan,
        "n_observacoes": int((pred.notna() & real.notna()).sum()),
        "n_janelas_auc": int(pd.Series(auc_alta_janelas).notna().sum()) if auc_alta_janelas else 0,
    }


def calcular_auc_root(root: str | Path) -> pd.DataFrame:
    root = Path(root)
    rows = []
    for sinais_path in sorted(root.rglob("sinais.csv")):
        row = calcular_auc_sinais(sinais_path, root)
        if row is not None:
            rows.append(row)

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .sort_values(["dataset", "modelo", "lookback", "pred_len", "janela_trading"])
        .reset_index(drop=True)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Calcula AUC direcional por configuração.")
    parser.add_argument("--root", required=True, help="Raiz das simulações.")
    parser.add_argument("--output_csv", required=True, help="CSV de saída.")
    args = parser.parse_args()

    auc = calcular_auc_root(args.root)
    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    auc.to_csv(output, index=False)
    print(f"AUC direcional salva em {output} com {len(auc)} linhas.")


if __name__ == "__main__":
    main()
