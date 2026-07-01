from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _flatten(prefix: str, data: dict) -> dict:
    out: dict = {}
    for key, value in data.items():
        name = f"{prefix}_{key}" if prefix else key
        if isinstance(value, dict):
            out.update(_flatten(name, value))
        else:
            out[name] = value
    return out


def collect_metrics(root: str | Path) -> pd.DataFrame:
    root = Path(root)
    rows: list[dict] = []

    for path in sorted(root.rglob("metricas.json")):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        params = data.get("params", {})
        metrics = data.get("metrics", {})
        row = {
            "output_dir": str(path.parent),
            "metricas_path": str(path),
            **_flatten("", params),
            **_flatten("", metrics),
        }
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    sort_cols = [c for c in ["strategy", "rebalance_k", "max_assets", "horizon", "random_seed", "output_dir"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols)

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Resume arquivos metricas.json em um CSV único.")
    parser.add_argument("--root", required=True, help="Pasta raiz onde procurar metricas.json.")
    parser.add_argument("--output_csv", required=True, help="Arquivo CSV de saída.")
    args = parser.parse_args()

    df = collect_metrics(args.root)
    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Resumo salvo em {out} com {len(df)} linhas.")


if __name__ == "__main__":
    main()
