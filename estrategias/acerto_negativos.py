from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def calcular_acerto_negativos_sinais(sinais_path: str | Path) -> dict:
    """Calcula taxa de acerto das previsões negativas a partir de um sinais.csv.

    A métrica principal é:
      taxa_acerto_negativos = P(real_ret_k < 0 | pred_ret_k < 0)

    Também calcula a média por janela:
      mean_precision_negative = média_t P(real_ret_k < 0 | pred_ret_k < 0, janela=t)
    """
    sinais_path = Path(sinais_path)

    if not sinais_path.exists():
        return {
            "taxa_acerto_negativos": np.nan,
            "mean_precision_negative": np.nan,
            "n_pred_negativos": 0,
            "n_acertos_negativos": 0,
            "n_janelas_com_negativos": 0,
        }

    sinais = pd.read_csv(sinais_path)
    required = {"pred_ret_k", "real_ret_k"}
    missing = required - set(sinais.columns)
    if missing:
        raise ValueError(f"{sinais_path} sem colunas obrigatórias: {sorted(missing)}")

    pred = pd.to_numeric(sinais["pred_ret_k"], errors="coerce")
    real = pd.to_numeric(sinais["real_ret_k"], errors="coerce")

    valid = pred.notna() & real.notna()
    pred_neg = valid & (pred < 0)
    acerto_neg = pred_neg & (real < 0)

    n_pred_neg = int(pred_neg.sum())
    n_acertos_neg = int(acerto_neg.sum())
    taxa_global = float(n_acertos_neg / n_pred_neg) if n_pred_neg > 0 else np.nan

    tmp = sinais.loc[valid, ["pred_ret_k", "real_ret_k"]].copy()
    if "origin_step" in sinais.columns:
        tmp["origin_step"] = sinais.loc[valid, "origin_step"].values
    else:
        tmp["origin_step"] = 0

    por_janela = []
    for _, g in tmp.groupby("origin_step", sort=True):
        g_neg = g[g["pred_ret_k"] < 0]
        if len(g_neg) > 0:
            por_janela.append(float((g_neg["real_ret_k"] < 0).mean()))

    mean_precision_negative = float(np.mean(por_janela)) if por_janela else np.nan

    return {
        "taxa_acerto_negativos": taxa_global,
        "mean_precision_negative": mean_precision_negative,
        "n_pred_negativos": n_pred_neg,
        "n_acertos_negativos": n_acertos_neg,
        "n_janelas_com_negativos": int(len(por_janela)),
    }


def coletar_acerto_negativos(root: str | Path) -> pd.DataFrame:
    root = Path(root)
    rows: list[dict] = []

    for sinais_path in sorted(root.rglob("sinais.csv")):
        metrics = calcular_acerto_negativos_sinais(sinais_path)
        rows.append(
            {
                "output_dir": str(sinais_path.parent),
                "sinais_path": str(sinais_path),
                **metrics,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "output_dir",
                "sinais_path",
                "taxa_acerto_negativos",
                "mean_precision_negative",
                "n_pred_negativos",
                "n_acertos_negativos",
                "n_janelas_com_negativos",
            ]
        )

    return pd.DataFrame(rows).sort_values("output_dir")


def mesclar_no_resumo(
    *,
    resumo_csv: str | Path,
    acerto_df: pd.DataFrame,
    output_csv: str | Path | None = None,
) -> Path:
    resumo_csv = Path(resumo_csv)
    output_csv = Path(output_csv) if output_csv else resumo_csv

    resumo = pd.read_csv(resumo_csv)
    if "output_dir" not in resumo.columns:
        raise ValueError(f"{resumo_csv} não contém coluna 'output_dir'.")

    cols = [
        "output_dir",
        "taxa_acerto_negativos",
        "mean_precision_negative",
        "n_pred_negativos",
        "n_acertos_negativos",
        "n_janelas_com_negativos",
    ]
    acerto_df = acerto_df[cols].drop_duplicates("output_dir")

    drop_cols = [c for c in cols if c != "output_dir" and c in resumo.columns]
    if drop_cols:
        resumo = resumo.drop(columns=drop_cols)

    out = resumo.merge(acerto_df, on="output_dir", how="left")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    return output_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Calcula taxa de acerto das previsões negativas.")
    parser.add_argument("--root", required=True, help="Pasta raiz onde procurar sinais.csv.")
    parser.add_argument("--output_csv", required=True, help="CSV com as estatísticas de negativos.")
    parser.add_argument("--merge_csv", default=None, help="CSV de resumo a ser enriquecido com as estatísticas.")
    parser.add_argument("--merged_output_csv", default=None, help="Saída do CSV enriquecido. Se omitido, sobrescreve merge_csv.")
    args = parser.parse_args()

    df = coletar_acerto_negativos(args.root)

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Acerto de negativos salvo em {output_csv} com {len(df)} linhas.")

    if args.merge_csv:
        merged_path = mesclar_no_resumo(
            resumo_csv=args.merge_csv,
            acerto_df=df,
            output_csv=args.merged_output_csv,
        )
        print(f"Resumo enriquecido salvo em {merged_path}.")


if __name__ == "__main__":
    main()
