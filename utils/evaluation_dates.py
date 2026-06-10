import numpy as np
import pandas as pd


DATA_FINAL_OBSERVADA_PADRAO = "2025-06-06"


def anexar_datas_por_step(
    df,
    step_col="step",
    date_col="date",
    data_final_observada=DATA_FINAL_OBSERVADA_PADRAO,
    freq="B",
):
    """
    Cria uma coluna de datas alinhada ao step da avaliacao.

    A data maxima do conjunto plotado recebe data_final_observada.
    Steps anteriores recebem datas retrocedendo pela frequencia definida.

    Isso permite plotar janelas de teste diferentes, pois o alinhamento usa
    min/max step efetivos do dataframe recebido, nao um intervalo fixo.

    Parametros
    ----------
    df : pandas.DataFrame
        Base com uma coluna numerica de step.
    step_col : str
        Nome da coluna de step.
    date_col : str
        Nome da coluna de data a criar/substituir.
    data_final_observada : str ou datetime-like
        Data associada ao maior step observado no dataframe.
    freq : str
        Frequencia do calendario. Use 'B' para dias uteis.
    """
    out = df.copy()
    out[step_col] = pd.to_numeric(out[step_col], errors="coerce")
    out = out.dropna(subset=[step_col])
    out[step_col] = out[step_col].astype(int)

    if out.empty:
        out[date_col] = pd.NaT
        return out

    data_final = pd.Timestamp(data_final_observada)
    steps = np.sort(out[step_col].unique())
    n_steps = len(steps)

    datas = pd.date_range(end=data_final, periods=n_steps, freq=freq)
    mapa_datas = dict(zip(steps, datas))

    out[date_col] = out[step_col].map(mapa_datas)
    return out


def ordenar_e_anexar_datas(
    df,
    group_cols=None,
    step_col="step",
    date_col="date",
    data_final_observada=DATA_FINAL_OBSERVADA_PADRAO,
    freq="B",
):
    """
    Aplica anexar_datas_por_step globalmente ou por grupos.

    Use group_cols quando cada modelo/janela tiver seu proprio conjunto de teste.
    Exemplo: group_cols=['modelo'] ou ['modelo', 'papel'].
    """
    if not group_cols:
        return anexar_datas_por_step(
            df,
            step_col=step_col,
            date_col=date_col,
            data_final_observada=data_final_observada,
            freq=freq,
        )

    return (
        df.groupby(group_cols, group_keys=False, observed=True)
        .apply(
            lambda g: anexar_datas_por_step(
                g,
                step_col=step_col,
                date_col=date_col,
                data_final_observada=data_final_observada,
                freq=freq,
            )
        )
        .reset_index(drop=True)
    )
