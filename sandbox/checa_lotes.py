#%%
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from loader.data_loader import TimeSeriesDataset


base_de_dados ="b3_daily_financeiro.csv"
DATA_DIR = f"{BASE_DIR}/data/{base_de_dados}"

dataset = TimeSeriesDataset(
    data_path=DATA_DIR,
    lookback=96,
    pred_len=7,
    cols="ITSA4",
)

print("BASE_DIR:", BASE_DIR)
print("DATA_DIR:", DATA_DIR)
print("Total de amostras no dataset:", len(dataset))

for i in [0, 8, 50, 100, len(dataset)-5, len(dataset)-1]:
    x, y = dataset[i]
    print(f"idx={i:4d} | x.shape={x.shape} | y.shape={y.shape}")

# Para rodar no slurm:
#   srun -p gorgonas_dev bash -lc "cd /sonic_home/igor.viveiros/paralelo && source /sonic_home/igor.viveiros/py310/bin/activate && python sandbox/checa_lotes.py"
# %%
