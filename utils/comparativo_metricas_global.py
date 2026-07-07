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
    "RandomTopJ": "simulacoes/RandomTopJ",
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
            obj = json.load(f)
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def _as_number(value: Any) -> float:
    if value is None:
        return np.nan
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    try:
        return float(str(value).replace(",", "."))
    except Exception:
        return np.nan


def _first_dict(*values: Any) -> dict[str, Any]:
    for value in values:
        if isinstance(value, dict):
            return value
    return {}


def _extract_metrics(payload: dict[str, Any]) -> dict[str, float]:
    source = _first_dict(payload.get("metrics"), payload.get("metricas"), payload.get("results"), payload.get("resultados"), payload.get("performance")) or payload
    return {k: _as_number(source.get(k)) for k in TODAS_METRICAS if k in source}


def _extract_params(payload: dict[str, Any]) -> dict[str, Any]:
    params = _first_dict(payload.get("params"), payload.get("config"), payload.get("args"), payload.get("hyperparameters"))
    if params:
        return params
    return {k: v for k, v in payload.items() if k not in {"metrics", "metricas", "results", "resultados"}}


def _find_int(text: str, patterns: list[str]) -> int | float:
    for pattern in patterns:
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if m:
            return int(m.group(1))
    return np.nan


def _param_int(params: dict[str, Any], keys: list[str]) -> int | float:
    for key in keys:
        if key in params and params[key] is not None:
            value = _as_number(params[key])
            if pd.notna(value):
                return int(value)
    return np.nan


def _param_str(params: dict[str, Any], keys: list[str]) -> str | float:
    for key in keys:
        value = params.get(key)
        if value not in (None, ""):
            return str(value)
    return np.nan


def _guess_modelo(rel_parts: tuple[str, ...], params: dict[str, Any], grupo: str) -> str:
    value = _param_str(params, ["model", "modelo", "model_name", "model_name_tfb", "model_id"])
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
    return candidates[-1] if candidates else grupo


def _guess_dataset(rel_parts: tuple[str, ...], params: dict[str, Any]) -> str | float:
    value = _param_str(params, ["dataset", "data", "data_name", "data_name_tfb", "price_dataset"])
    if isinstance(value, str):
        return Path(value).stem
    return rel_parts[0] if rel_parts else np.nan


def _metadata_from_path(json_path: Path, root: Path, grupo: str, params: dict[str, Any]) -> dict[str, Any]:
    try:
        rel = json_path.relative_to(root)
    except ValueError:
        rel = json_path
    parts = rel.parts
    text = "/".join(parts)
    lookback = _param_int(params, ["lookback", "lookback_window", "seq_len", "input_len", "window_size"])
    if pd.isna(lookback):
        lookback = _find_int(text, [r"lookback[_-]?(\d+)", r"lb[_-]?(\d+)", r"seq[_-]?len[_-]?(\d+)"])
    pred_len = _param_int(params, ["pred_len", "horizon", "horizonte", "prediction_length", "forecast_horizon"])
    if pd.isna(pred_len):
        pred_len = _find_int(text, [r"pred[_-]?len[_-]?(\d+)", r"horizon[_-]?(\d+)", r"__h(\d+)", r"pred[_-]?(\d+)"])
    janela = _param_int(params, ["rebalance_k", "janela_trading", "trading_window", "k", "window"])
    if pd.isna(janela):
        m = re.findall(r"(?:^|[/_-])k[_-]?(\d+)(?:$|[/_-])", text, flags=re.I)
        janela = int(m[-1]) if m else np.nan
    max_assets = _param_int(params, ["max_assets", "top_j", "topj", "j", "top"])
    if pd.isna(max_assets):
        max_assets = _find_int(text, [r"top[_-]?j[_-]?(\d+)", r"top[_-]?(\d+)"])
    return {
        "grupo": grupo,
        "dataset": _guess_dataset(parts, params),
        "modelo": _guess_modelo(parts, params, grupo),
        "lookback": lookback,
        "pred_len": pred_len,
        "janela_trading": janela,
        "horizonte_comparavel": janela,
        "max_assets": max_assets,
        "model_output": _param_str(params, ["model_output", "tipo_saida", "output_type"]),
        "returns_mode": _param_str(params, ["returns_mode"]),
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
    return {k: auc.get(k) for k in METRICAS_INTERESSE if k.startswith("auc_") and k in auc} if auc else {}


def _negative_precision_from_sinais(json_path: Path) -> dict[str, Any]:
    sinais_path = json_path.parent / "sinais.csv"
    if not sinais_path.exists():
        return {}
    try:
        sinais = pd.read_csv(sinais_path)
    except Exception:
        return {}
    if not {"pred_ret_k", "real_ret_k"}.issubset(sinais.columns):
        return {}
    pred = pd.to_numeric(sinais["pred_ret_k"], errors="coerce")
    real = pd.to_numeric(sinais["real_ret_k"], errors="coerce")
    valid = pred.notna() & real.notna()
    if not valid.any():
        return {"mean_precision_negative": np.nan}
    tmp = pd.DataFrame({"pred_ret_k": pred[valid], "real_ret_k": real[valid]})
    tmp["origin_step"] = sinais.loc[valid, "origin_step"].values if "origin_step" in sinais.columns else 0
    vals = []
    for _, g in tmp.groupby("origin_step", sort=True):
        gneg = g[g["pred_ret_k"] < 0]
        if len(gneg):
            vals.append(float((gneg["real_ret_k"] < 0).mean()))
    return {"mean_precision_negative": float(np.mean(vals)) if vals else np.nan}


def _validar_carteira(json_path: Path, janela_trading: Any) -> tuple[bool, str]:
    carteira_path = json_path.parent / "carteira.csv"
    if not carteira_path.exists():
        return True, "sem_carteira_para_validar"
    try:
        carteira = pd.read_csv(carteira_path)
    except Exception as exc:
        return False, f"erro_lendo_carteira: {exc}"
    if carteira.empty:
        return False, "carteira_vazia"
    k = _as_number(janela_trading)
    if pd.notna(k):
        if {"origin_pos", "target_pos"}.issubset(carteira.columns):
            hold = pd.to_numeric(carteira["target_pos"], errors="coerce") - pd.to_numeric(carteira["origin_pos"], errors="coerce")
            if hold.notna().any() and not (hold.dropna() == int(k)).all():
                return False, "holding_period_pos_diferente_de_k"
        elif {"origin_step", "target_step"}.issubset(carteira.columns):
            ori = pd.to_numeric(carteira["origin_step"], errors="coerce")
            tar = pd.to_numeric(carteira["target_step"], errors="coerce")
            hold = tar - ori
            if hold.notna().any() and not (hold.dropna() == int(k)).all():
                return False, "holding_period_step_diferente_de_k"
    if "portfolio_ret_k" in carteira.columns:
        ret = pd.to_numeric(carteira["portfolio_ret_k"], errors="coerce")
        if (ret.abs() > 1.0).any():
            return False, "retorno_periodo_abs_maior_que_100pct"
    return True, "ok"


def _valid_result(metrics: dict[str, Any]) -> bool:
    return any(k in metrics for k in TODAS_METRICAS)


def _scan_group(grupo: GrupoBusca) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    root = grupo.root
    if not root.exists():
        return [], []
    rows: list[dict[str, Any]] = []
    excluidos: list[dict[str, Any]] = []
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
        row = _metadata_from_path(json_path, root, grupo.nome, params)
        ok, motivo = _validar_carteira(json_path, row.get("janela_trading"))
        row["resultado_valido"] = bool(ok)
        row["motivo_validacao"] = motivo
        if not ok:
            excluidos.append({**row, **{k: metrics.get(k, np.nan) for k in TODAS_METRICAS}})
            continue
        for key, value in _auc_from_sinais(json_path, root).items():
            metrics.setdefault(key, value)
        for key, value in _negative_precision_from_sinais(json_path).items():
            if key not in metrics or pd.isna(metrics.get(key, np.nan)):
                metrics[key] = value
        row.update({k: metrics.get(k, np.nan) for k in TODAS_METRICAS})
        rows.append(row)
    return rows, excluidos


def carregar_metricas(grupos: dict[str, str] | None = None, base_dir: str | Path = ".") -> tuple[pd.DataFrame, pd.DataFrame]:
    base_dir = Path(base_dir)
    grupos = grupos or DEFAULT_GROUPS
    rows: list[dict[str, Any]] = []
    excluidos: list[dict[str, Any]] = []
    for nome, root_str in grupos.items():
        group_rows, group_excluidos = _scan_group(GrupoBusca(nome=nome, root=base_dir / root_str, benchmark_only=nome.lower().startswith("benchmark")))
        rows.extend(group_rows)
        excluidos.extend(group_excluidos)
    df = pd.DataFrame(rows)
    excl = pd.DataFrame(excluidos)
    for frame in [df, excl]:
        if frame.empty:
            continue
        for col in ["lookback", "pred_len", "janela_trading", "horizonte_comparavel", "max_assets"]:
            if col in frame:
                frame[col] = pd.to_numeric(frame[col], errors="coerce").astype("Int64")
        for col in TODAS_METRICAS:
            if col in frame:
                frame[col] = pd.to_numeric(frame[col], errors="coerce")
    if not df.empty:
        df = df.sort_values(["grupo", "dataset", "modelo", "lookback", "pred_len", "janela_trading", "run"]).reset_index(drop=True)
    return df, excl.reset_index(drop=True)


def resumo_por_modelo(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    group_cols = ["grupo", "dataset", "modelo", "janela_trading"]
    metric_cols = [c for c in TODAS_METRICAS if c in df.columns]
    out = df.groupby(group_cols, dropna=False)[metric_cols].agg(["mean", "median", "std", "count"]).reset_index()
    out.columns = ["_".join([x for x in col if x]) if isinstance(col, tuple) else col for col in out.columns]
    return out


def top_configs(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    id_cols = ["grupo", "dataset", "modelo", "lookback", "pred_len", "janela_trading", "max_assets", "model_output", "run", "json_path"]
    rows = []
    for metric in METRICAS_INTERESSE:
        if metric not in df.columns:
            continue
        tmp = df.dropna(subset=[metric]).sort_values(metric, ascending=False).head(n).copy()
        if tmp.empty:
            continue
        tmp.insert(0, "metrica", metric)
        tmp.insert(1, "valor", tmp[metric])
        rows.append(tmp[["metrica", "valor", *[c for c in id_cols if c in tmp.columns]]])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


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


def analise_k1_ic(df: pd.DataFrame, n: int = 30) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty or "mean_spearman_ic" not in df.columns:
        return pd.DataFrame(), pd.DataFrame()
    k1 = df[df["janela_trading"].astype("Int64") == 1].copy()
    if k1.empty:
        return pd.DataFrame(), pd.DataFrame()
    top = k1.dropna(subset=["mean_spearman_ic"]).sort_values("mean_spearman_ic", ascending=False).head(n)
    group_cols = ["grupo", "dataset", "modelo"]
    metric_cols = [c for c in ["mean_spearman_ic", "icir", "mean_precision_positive", "mean_precision_negative", "auc_alta", "auc_queda"] if c in k1.columns]
    resumo = k1.groupby(group_cols, dropna=False)[metric_cols].agg(["mean", "median", "std", "count"]).reset_index()
    resumo.columns = ["_".join([x for x in col if x]) if isinstance(col, tuple) else col for col in resumo.columns]
    resumo = resumo.sort_values("mean_spearman_ic_mean", ascending=False) if "mean_spearman_ic_mean" in resumo.columns else resumo
    return resumo, top


def comparar_global(base_dir: str | Path = ".", output_dir: str | Path = "simulacoes/comparativo_global_master_tfb",
                    grupos: dict[str, str] | None = None, top_n: int = 20) -> dict[str, pd.DataFrame]:
    base_dir = Path(base_dir)
    output_dir = base_dir / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    metricas, excluidos = carregar_metricas(grupos=grupos, base_dir=base_dir)
    resumo = resumo_por_modelo(metricas)
    tops = top_configs(metricas, n=top_n)
    ranking = ranking_global(metricas)
    k1_resumo, k1_top = analise_k1_ic(metricas, n=top_n)
    cols_interesse = ["grupo", "dataset", "modelo", "lookback", "pred_len", "janela_trading", "max_assets", "model_output", *[c for c in METRICAS_INTERESSE if c in metricas.columns], "json_path"]
    tabela = metricas[[c for c in cols_interesse if c in metricas.columns]].copy() if not metricas.empty else pd.DataFrame()
    metricas.to_csv(output_dir / "metricas_global_long.csv", index=False)
    resumo.to_csv(output_dir / "resumo_por_modelo.csv", index=False)
    tops.to_csv(output_dir / "top_configs_por_metrica.csv", index=False)
    ranking.to_csv(output_dir / "ranking_global.csv", index=False)
    tabela.to_csv(output_dir / "tabela_metricas_interesse.csv", index=False)
    k1_resumo.to_csv(output_dir / "analise_k1_ic_modelos.csv", index=False)
    k1_top.to_csv(output_dir / "top_k1_ic_configuracoes.csv", index=False)
    excluidos.to_csv(output_dir / "resultados_excluidos_validacao.csv", index=False)
    return {"metricas": metricas, "resumo": resumo, "top_configs": tops, "ranking": ranking, "tabela_metricas_interesse": tabela, "analise_k1_ic_modelos": k1_resumo, "top_k1_ic_configuracoes": k1_top, "resultados_excluidos_validacao": excluidos}


def _parse_group_arg(values: list[str] | None) -> dict[str, str] | None:
    if not values:
        return None
    out = {}
    for item in values:
        if "=" not in item:
            raise ValueError("Use --group Nome=caminho/da/pasta")
        k, v = item.split("=", 1)
        out[k] = v
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Compara métricas globais de MASTER, TFB, RandomTopJ e benchmarks.")
    parser.add_argument("--base_dir", default=".")
    parser.add_argument("--output_dir", default="simulacoes/comparativo_global_master_tfb")
    parser.add_argument("--top_n", type=int, default=20)
    parser.add_argument("--group", action="append")
    args = parser.parse_args()
    dfs = comparar_global(base_dir=args.base_dir, output_dir=args.output_dir, grupos=_parse_group_arg(args.group), top_n=args.top_n)
    print(f"Linhas válidas carregadas: {len(dfs['metricas'])}")
    print(f"Linhas excluídas por validação: {len(dfs['resultados_excluidos_validacao'])}")
    print(f"Saídas salvas em: {Path(args.base_dir) / args.output_dir}")


if __name__ == "__main__":
    main()
