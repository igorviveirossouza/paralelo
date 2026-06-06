"""
Cria uma versão em número índice do dataset B3 daily financeiro.

Para cada papel em `cols`, a série escolhida é dividida pelo primeiro
valor válido observado. Assim, cada papel começa em 1.0 e os pontos
seguintes representam a variação acumulada desde esse início.

Entrada esperada, por padrão:
    data/b3_daily_financeiro.csv

Saída padrão:
    data/b3_daily_financeiro_indice.csv

Exemplo:
    python scripts/criar_b3_daily_financeiro_indice.py

    python scripts/criar_b3_daily_financeiro_indice.py \
        --input data/b3_daily_financeiro.csv \
        --output data/b3_daily_financeiro_indice.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def criar_numero_indice(
    df: pd.DataFrame,
    grupo_col: str = "cols",
    valor_col: str = "data",
    data_col: str = "date",
    novo_valor_col: str | None = None,
) -> pd.DataFrame:
    """Transforma cada série em número índice com primeiro valor válido = 1."""
    colunas_necessarias = {grupo_col, valor_col}
    faltantes = colunas_necessarias.difference(df.columns)
    if faltantes:
        raise ValueError(f"Colunas ausentes no dataset: {sorted(faltantes)}")

    df = df.copy()

    if data_col in df.columns:
        df = df.sort_values([grupo_col, data_col]).reset_index(drop=True)
    else:
        df = df.sort_values([grupo_col]).reset_index(drop=True)

    bases = df.groupby(grupo_col, sort=False)[valor_col].transform(
        lambda serie: serie.dropna().iloc[0] if not serie.dropna().empty else pd.NA
    )

    if (bases == 0).any():
        papeis_zero = sorted(df.loc[bases == 0, grupo_col].unique())
        raise ValueError(
            "Há séries com primeiro valor válido igual a zero, impossibilitando o índice: "
            f"{papeis_zero}"
        )

    destino = novo_valor_col or valor_col
    df[destino] = df[valor_col] / bases

    return df


dataset = "b3_daily_financeiro.csv"

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cria dataset B3 daily financeiro em número índice."
    )
    parser.add_argument(
        "--input",
        default=f"data/{dataset}",
        help="Caminho do CSV original.",
    )
    parser.add_argument(
        "--output",
        default=f"data/{dataset}",
        help="Caminho do CSV transformado.",
    )
    parser.add_argument(
        "--group-col",
        default="cols",
        help="Coluna que identifica cada papel/série.",
    )
    parser.add_argument(
        "--value-col",
        default="data",
        help="Coluna numérica a transformar em índice.",
    )
    parser.add_argument(
        "--date-col",
        default="date",
        help="Coluna temporal usada para ordenar cada série.",
    )
    parser.add_argument(
        "--new-value-col",
        default=None,
        help="Se informado, salva o índice em nova coluna; senão substitui value-col.",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    df = pd.read_csv(input_path)
    df_indice = criar_numero_indice(
        df=df,
        grupo_col=args.group_col,
        valor_col=args.value_col,
        data_col=args.date_col,
        novo_valor_col=args.new_value_col,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_indice.to_csv(output_path, index=False)

    print(f"Dataset salvo em: {output_path}")
    print(f"Formato: {df_indice.shape[0]} linhas x {df_indice.shape[1]} colunas")
    print(f"Séries transformadas: {df_indice[args.group_col].nunique()}")


if __name__ == "__main__":
    main()
