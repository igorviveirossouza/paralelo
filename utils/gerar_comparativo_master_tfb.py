from __future__ import annotations

import argparse
from pathlib import Path

try:
    from utils.comparativo_metricas_global import comparar_global
    from utils.compilar_randomtopj import compilar_randomtopj, resumir_randomtopj
except Exception:  # pragma: no cover
    from comparativo_metricas_global import comparar_global
    from compilar_randomtopj import compilar_randomtopj, resumir_randomtopj


def gerar_resultados(
    base_dir: str | Path = ".",
    output_dir: str | Path = "simulacoes/comparativo_global_master_tfb",
    random_root: str | Path = "simulacoes/RandomTopJ",
    top_n: int = 20,
) -> dict:
    base_dir = Path(base_dir)
    out = base_dir / output_dir
    out.mkdir(parents=True, exist_ok=True)

    dfs = comparar_global(base_dir=base_dir, output_dir=output_dir, top_n=top_n)

    random_path = base_dir / random_root
    random_long = compilar_randomtopj(random_path)
    random_resumo = resumir_randomtopj(random_long)
    random_long.to_csv(out / "randomtopj_metricas_long.csv", index=False)
    random_resumo.to_csv(out / "randomtopj_resumo_por_configuracao.csv", index=False)

    dfs["randomtopj_metricas_long"] = random_long
    dfs["randomtopj_resumo_por_configuracao"] = random_resumo
    return dfs


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera todo o pacote de resultados MASTER/TFB/RandomTopJ.")
    parser.add_argument("--base_dir", default=".")
    parser.add_argument("--output_dir", default="simulacoes/comparativo_global_master_tfb")
    parser.add_argument("--random_root", default="simulacoes/RandomTopJ")
    parser.add_argument("--top_n", type=int, default=20)
    args = parser.parse_args()
    dfs = gerar_resultados(
        base_dir=args.base_dir,
        output_dir=args.output_dir,
        random_root=args.random_root,
        top_n=args.top_n,
    )
    print(f"metricas_global_long: {len(dfs['metricas'])} linhas")
    print(f"randomtopj_metricas_long: {len(dfs['randomtopj_metricas_long'])} linhas")
    print(f"Saídas em: {Path(args.base_dir) / args.output_dir}")


if __name__ == "__main__":
    main()
