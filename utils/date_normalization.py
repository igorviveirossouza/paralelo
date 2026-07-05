import re

import pandas as pd


def normalize_date_labels(values):
    """Normaliza rótulos de data sem colapsar índices numéricos.

    Regras:
    - valores numéricos são preservados como string canônica;
    - strings puramente numéricas também são preservadas;
    - apenas strings claramente datadas, como YYYY-MM-DD ou DD/MM/YYYY,
      são convertidas para YYYY-MM-DD.

    Isso evita que pandas.to_datetime interprete inteiros como nanossegundos
    desde 1970 e transforme todos os pregões em 1970-01-01.
    """
    series = pd.Series(values)

    def normalize_one(value):
        if pd.isna(value):
            return value

        text = str(value).strip()
        if text == "":
            return text

        # Preserve índices de pregão e datas compactas como 20200102.
        if re.fullmatch(r"\d+(\.0)?", text):
            return str(int(float(text)))

        # Só converte strings que carregam separadores típicos de data.
        if any(sep in text for sep in ["-", "/", ":"]):
            parsed = pd.to_datetime(text, errors="coerce")
            if pd.notna(parsed):
                return parsed.strftime("%Y-%m-%d")

        return text

    return [normalize_one(value) for value in series]
