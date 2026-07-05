import runpy

from loader.data_loader import TimeSeriesDataset
from utils.date_normalization import normalize_date_labels


TimeSeriesDataset._normalize_dates = staticmethod(normalize_date_labels)


runpy.run_module("main_test", run_name="__main__")
