from __future__ import annotations

import re

import pandas as pd

from loader.data_loader import TimeSeriesDataset


def normalize_dates_chronological(values):
    series = pd.Series(values)

    def normalize_one(value):
        if pd.isna(value):
            return value
        text = str(value).strip()
        if text == "":
            return text
        if re.fullmatch(r"\d+(\.0)?", text):
            return f"{int(float(text)):012d}"
        if any(sep in text for sep in ["-", "/", ":"]):
            parsed = pd.to_datetime(text, errors="coerce")
            if pd.notna(parsed):
                return parsed.strftime("%Y-%m-%d")
        return text

    return [normalize_one(value) for value in series]


TimeSeriesDataset._normalize_dates = staticmethod(normalize_dates_chronological)

from main_test import main  # noqa: E402


if __name__ == "__main__":
    main()
