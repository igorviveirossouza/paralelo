
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd


SAMPLE_RE = re.compile(r"csv_sample_(\d+)_inference_data\.csv$")
IDX_RE = re.compile(r"_idx(\d+)_")
RESERVED_COLS = {"step", "date"}


@dataclass(frozen=True)
class PredictionFile:
    path: Path
    sample_idx: int


def _unique_preserve_order(values: Sequence[object]) -> list[str]:
    """Valores únicos preservando ordem de primeira aparição."""
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if pd.isna(value):
            continue
        s = str(value).strip()
        if not s:
            continue
        if s.lower() == "label":
            continue
        if s in RESERVED_COLS:
            continue
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def infer_original_length(original_dataset_path: str | Path) -> int:
    """
    Infere o comprimento temporal real do dataset original.

    Suporta:
      - TFB long: colunas ['date', 'data', 'cols']
      - wide: uma linha por timestep
    """
    path = Path(original_dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset original não encontrado: {path}")

    df = pd.read_csv(path)

    if "cols" in df.columns:
        work = df[df["cols"].astype(str).str.lower() != "label"].copy()
        if "date" in work.columns:
            return int(work.groupby("cols", sort=False)["date"].nunique().max())
        return int(work["cols"].value_counts(sort=False).max())

    return int(len(df))


def infer_target_columns(original_dataset_path: str | Path, n_value_cols: int) -> list[str]:
    """
    Lê os nomes reais das séries a partir do arquivo original.

    Para CSV TFB long, usa os valores únicos de `cols`, preservando a ordem
    de primeira aparição no arquivo. Isso replica a lógica usada pelo loader
    do TFB para formar as colunas wide.
    """
    path = Path(original_dataset_path)
    header = pd.read_csv(path, nrows=0)

    if "cols" in header.columns:
        cols_series = pd.read_csv(path, usecols=["cols"])["cols"]
        cols = _unique_preserve_order(cols_series.tolist())
    else:
        cols = [
            str(c).strip()
            for c in header.columns
            if str(c).strip()
            and str(c).strip() not in RESERVED_COLS
            and str(c).strip().lower() != "label"
        ]

    if len(cols) != n_value_cols:
        raise ValueError(
            "Número de colunas incompatível. "
            f"O dataset original forneceu {len(cols)} nomes: {cols}. "
            f"O CSV de previsão tem {n_value_cols} colunas de valores. "
            "Verifique se --original-dataset é exatamente o CSV usado para gerar essas previsões."
        )

    if len(set(cols)) != len(cols):
        raise ValueError(f"Nomes duplicados inferidos no dataset original: {cols}")

    return cols


def infer_original_dates(original_dataset_path: str | Path) -> Optional[list[str]]:
    """Retorna datas únicas ordenadas, se o dataset original tiver coluna date."""
    df = pd.read_csv(original_dataset_path)
    if "date" not in df.columns:
        return None
    dates = pd.Series(pd.to_datetime(df["date"].unique())).sort_values()
    return dates.dt.strftime("%Y-%m-%d").tolist()


def extract_sample_idx(path: Path) -> int:
    """
    Extrai o índice da janela pelo nome do arquivo.
    Prioridade:
      1) csv_sample_<idx>_inference_data.csv
      2) _idx<idx>_
    """
    name = path.name

    m = SAMPLE_RE.search(name)
    if m:
        return int(m.group(1))

    m = IDX_RE.search(name)
    if m:
        return int(m.group(1))

    raise ValueError(f"Não consegui extrair sample_idx do arquivo: {name}")


def list_prediction_files(pred_dir: str | Path, pattern: str) -> list[PredictionFile]:
    pred_dir = Path(pred_dir)
    if not pred_dir.exists():
        raise FileNotFoundError(f"Diretório de previsões não encontrado: {pred_dir}")
    if not pred_dir.is_dir():
        raise NotADirectoryError(f"pred_dir não é diretório: {pred_dir}")

    files: list[PredictionFile] = []
    for path in pred_dir.glob(pattern):
        if path.is_file():
            files.append(PredictionFile(path=path, sample_idx=extract_sample_idx(path)))

    if not files:
        raise FileNotFoundError(f"Nenhum arquivo encontrado em {pred_dir} com pattern={pattern!r}")

    files.sort(key=lambda x: x.sample_idx)
    return files


def normalize_prediction_frame(df: pd.DataFrame, target_columns: list[str]) -> pd.DataFrame:
    """
    Remove colunas auxiliares antigas e renomeia apenas as colunas de valores.
    Saída parcial: colunas dos ativos, ainda sem step.
    """
    df = df.copy()

    for col in list(df.columns):
        if str(col) in RESERVED_COLS:
            df = df.drop(columns=[col])

    if len(df.columns) != len(target_columns):
        raise ValueError(
            f"CSV possui {len(df.columns)} colunas de valores, "
            f"mas o dataset original forneceu {len(target_columns)} nomes."
        )

    df.columns = target_columns
    return df


def build_output_filename(sample_idx: int, template: str) -> str:
    """
    Exemplo:
      janela_{sample_idx:06d}.csv -> janela_000000.csv
    """
    try:
        name = template.format(sample_idx=sample_idx, i=sample_idx)
    except Exception as exc:
        raise ValueError(
            f"Template inválido: {template!r}. Use algo como 'janela_{{sample_idx:06d}}.csv'."
        ) from exc

    if not name.endswith(".csv"):
        name += ".csv"
    return name


def add_step_to_prediction_files(
    pred_dir: str | Path,
    original_dataset_path: str | Path,
    pred_len: int,
    lookback: int,
    *,
    stride: int = 1,
    step_offset: int = 1,
    first_step: Optional[int] = None,
    pattern: str = "*csv_sample_*_inference_data.csv",
    overwrite: bool = False,
    output_dir: str | Path | None = None,
    add_date: bool = False,
    strict_pred_len: bool = True,
    output_name_template: str = "janela_{sample_idx:06d}.csv",
) -> pd.DataFrame:
    """
    Processa todos os CSVs de previsão.

    O step interno é zero-based:
        first_step_zero_based = original_len - pred_len - (n_files - 1) * stride

    Por padrão, o step salvo é 1-based:
        step_visivel = step_zero_based + 1

    Para forçar o primeiro step exatamente, use --first-step.
    """
    if pred_len <= 0:
        raise ValueError("pred_len deve ser > 0.")
    if lookback <= 0:
        raise ValueError("lookback deve ser > 0.")
    if stride <= 0:
        raise ValueError("stride deve ser > 0.")

    pred_files = list_prediction_files(pred_dir, pattern)
    original_len = infer_original_length(original_dataset_path)
    n_files = len(pred_files)

    sample_indices = [pf.sample_idx for pf in pred_files]
    expected = list(range(min(sample_indices), max(sample_indices) + 1))
    if sample_indices != expected:
        raise ValueError(
            "Os sample_idx não são contíguos. Pode haver arquivos faltantes. "
            f"Primeiros: {sample_indices[:10]}; últimos: {sample_indices[-10:]}"
        )
    if min(sample_indices) != 0:
        raise ValueError(f"Esperava sample_idx iniciando em 0, mas iniciou em {min(sample_indices)}.")

    first_step_zero_based = original_len - pred_len - (n_files - 1) * stride
    if first_step_zero_based < 0:
        raise ValueError(
            "Configuração inconsistente: first_step_zero_based ficou negativo. "
            f"original_len={original_len}, pred_len={pred_len}, n_files={n_files}, stride={stride}."
        )

    first_input_start = first_step_zero_based - lookback
    if first_input_start < 0:
        raise ValueError(
            "Configuração inconsistente: a primeira janela não teria lookback suficiente. "
            f"first_step_zero_based={first_step_zero_based}, lookback={lookback}."
        )

    first_df = pd.read_csv(pred_files[0].path)
    value_cols = [c for c in first_df.columns if str(c) not in RESERVED_COLS]
    target_columns = infer_target_columns(original_dataset_path, len(value_cols))

    dates = infer_original_dates(original_dataset_path) if add_date else None

    if overwrite:
        out_dir = Path(pred_dir)
    else:
        if output_dir is None:
            output_dir = Path(pred_dir).with_name(Path(pred_dir).name + "_with_step")
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for pf in pred_files:
        df = pd.read_csv(pf.path)

        if strict_pred_len and len(df) != pred_len:
            raise ValueError(f"{pf.path.name}: esperado pred_len={pred_len} linhas, mas veio {len(df)}.")

        start_zero = first_step_zero_based + pf.sample_idx * stride
        zero_steps = list(range(start_zero, start_zero + len(df)))

        if zero_steps[-1] >= original_len:
            raise ValueError(
                f"{pf.path.name}: step zero-based final {zero_steps[-1]} excede original_len-1={original_len - 1}."
            )

        df = normalize_prediction_frame(df, target_columns)

        if add_date:
            if dates is None:
                raise ValueError("add_date=True, mas o dataset original não tem coluna date.")
            df["date"] = [dates[s] for s in zero_steps]

        if first_step is None:
            visible_steps = [s + step_offset for s in zero_steps]
        else:
            start_visible = first_step + pf.sample_idx * stride
            visible_steps = list(range(start_visible, start_visible + len(df)))

        df["step"] = visible_steps

        final_cols = target_columns + (["date"] if add_date else []) + ["step"]
        df = df[final_cols]

        output_name = build_output_filename(pf.sample_idx, output_name_template)
        save_path = out_dir / output_name
        df.to_csv(save_path, index=False)

        rows.append(
            {
                "source_file": pf.path.name,
                "output_file": output_name,
                "sample_idx": pf.sample_idx,
                "start_step": visible_steps[0],
                "end_step": visible_steps[-1],
                "rows": len(df),
                "saved_to": str(save_path),
            }
        )

    summary = pd.DataFrame(rows)
    summary.attrs["target_columns"] = target_columns
    summary.attrs["original_len"] = original_len
    summary.attrs["first_step_zero_based"] = first_step_zero_based
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Corrige CSVs de previsão do TFB: nomes reais, step final e janela_000000.csv."
    )
    parser.add_argument("--pred-dir", required=True, help="Pasta com os CSVs de previsão.")
    parser.add_argument("--original-dataset", required=True, help="CSV original do dataset TFB.")
    parser.add_argument("--pred-len", type=int, required=True, help="Horizonte de previsão.")
    parser.add_argument("--lookback", type=int, required=True, help="Janela de entrada/seq_len.")
    parser.add_argument("--stride", type=int, default=1, help="Deslocamento entre janelas.")
    parser.add_argument("--step-offset", type=int, default=1, help="1 para step 1-based; 0 para zero-based.")
    parser.add_argument("--first-step", type=int, default=None, help="Força o primeiro step do csv_sample_0.")
    parser.add_argument(
        "--pattern",
        default="*csv_sample_*_inference_data.csv",
        help="Glob dos arquivos de previsão.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Pasta de saída. Se omitida, cria <pred-dir>_with_step.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Grava na própria pasta de entrada, mas com nomes janela_*.csv.",
    )
    parser.add_argument(
        "--output-name-template",
        default="janela_{sample_idx:06d}.csv",
        help="Template de saída. Padrão: janela_{sample_idx:06d}.csv",
    )
    parser.add_argument(
        "--add-date",
        action="store_true",
        help="Também adiciona date antes de step, se o original tiver date.",
    )
    parser.add_argument(
        "--no-strict-pred-len",
        action="store_true",
        help="Não exige que cada CSV tenha exatamente pred_len linhas.",
    )

    args = parser.parse_args()

    summary = add_step_to_prediction_files(
        pred_dir=args.pred_dir,
        original_dataset_path=args.original_dataset,
        pred_len=args.pred_len,
        lookback=args.lookback,
        stride=args.stride,
        step_offset=args.step_offset,
        first_step=args.first_step,
        pattern=args.pattern,
        overwrite=args.overwrite,
        output_dir=args.output_dir,
        add_date=args.add_date,
        strict_pred_len=not args.no_strict_pred_len,
        output_name_template=args.output_name_template,
    )

    print(summary.head().to_string(index=False))
    print("...")
    print(summary.tail().to_string(index=False))
    print(f"\nArquivos processados: {len(summary)}")
    print(f"Primeiro step: {summary['start_step'].min()}")
    print(f"Último step: {summary['end_step'].max()}")
    print("Colunas inferidas:", ",".join(summary.attrs["target_columns"]))


if __name__ == "__main__":
    main()
