import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch


def _sanitize_path_part(value):
    """Converte valores em nomes seguros para subpastas."""
    value = Path(str(value)).stem
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._-")
    return value or "sem_nome"


def _resolve_model_name(model):
    """Recupera o nome do modelo base, ignorando wrappers."""
    if hasattr(model, "base_model_name"):
        return model.base_model_name

    inner_model = getattr(model, "model", None)
    if inner_model is not None:
        return _resolve_model_name(inner_model)

    return getattr(model, "forecast_model_name", model.__class__.__name__)


def run_one_step_rolling_forecast(
    model,
    dataset,
    output_dir="previsoes",
    batch_size=1,
    dataset_name=None,
    run_date=None,
    extra_dirs=None,
    model_name=None,
):
    """
    Rolling forecast SOMENTE PREVISÕES (sem valores true).
    'step' reflete posição real na série original.

    Salva os arquivos em:
    output_dir/dataset_name/model_class_name/janela_*.csv
    """
    dataset_name = dataset_name or getattr(dataset, "dataset_name", None) or "dataset"
    model_name = model_name or _resolve_model_name(model)
    run_date = run_date or datetime.now().strftime("%Y-%m-%d")

    extra_dirs = extra_dirs or []

    final_output_dir = (
        Path(output_dir)
        / _sanitize_path_part(dataset_name)
    )

    for paths in extra_dirs:
        final_output_dir = final_output_dir / _sanitize_path_part(paths)

    final_output_dir = final_output_dir / _sanitize_path_part(model_name)

    final_output_dir.mkdir(parents=True, exist_ok=True)

    device = next(model.parameters()).device
    model.eval()

    with torch.no_grad():
        for idx in range(len(dataset)):
            sample = dataset[idx]
            if len(sample) == 2:
                seq_x, _ = sample
                seq_candle = None
            elif len(sample) == 3:
                seq_x, _, seq_candle = sample
            else:
                raise ValueError(f"Amostra inesperada com {len(sample)} elementos")

            global_start = dataset.indices[idx]

            batch_x = seq_x.unsqueeze(0).to(device)
            forward_kwargs = {}
            if seq_candle is not None:
                forward_kwargs["candle_x"] = seq_candle.unsqueeze(0).to(device)

            pred = model(batch_x, **forward_kwargs)
            if isinstance(pred, tuple):
                pred = pred[0]
            pred_np = pred.squeeze(0).cpu().numpy()

            pred_df = pd.DataFrame(pred_np, columns=dataset.feature_columns)

            real_start_step = global_start + dataset.lookback
            pred_df["step"] = range(real_start_step, real_start_step + len(pred_df))

            out_file = final_output_dir / f"janela_{idx:06d}.csv"
            pred_df.to_csv(out_file, index=False)
            if idx % 50 == 0 or idx == len(dataset) - 1:
                print(f"Gerada previsão {idx + 1}/{len(dataset)}")

    print(f"✅ Rolling forecast concluído! Arquivos salvos em: {final_output_dir}")
    return final_output_dir
