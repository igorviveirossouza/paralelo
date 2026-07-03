from __future__ import annotations

import json
from pathlib import Path

NOTEBOOK_PATH = Path("utils/comparativo_metricas_multiplas_TFB.ipynb")
CELL_ID = "auc_direcional"

CELL_SOURCE = [
    "# AUC direcional por dataset, modelo, lookback, pred_len e janela de trading (k).\n",
    "# Usa os sinais.csv gerados pelas simulações.\n",
    "\n",
    "from pathlib import Path\n",
    "from utils.auc_direcional import calcular_auc_root\n",
    "\n",
    "root_auc = Path(ROOT)\n",
    "auc_direcional = calcular_auc_root(root_auc)\n",
    "\n",
    "auc_output = root_auc / \"auc_direcional_por_config.csv\"\n",
    "auc_direcional.to_csv(auc_output, index=False)\n",
    "\n",
    "print(f\"AUC direcional salva em: {auc_output}\")\n",
    "auc_direcional\n",
]


def main() -> None:
    if not NOTEBOOK_PATH.exists():
        raise FileNotFoundError(f"Notebook não encontrado: {NOTEBOOK_PATH}")

    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)

    cells = nb.setdefault("cells", [])
    cells = [cell for cell in cells if cell.get("id") != CELL_ID]

    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "id": CELL_ID,
            "metadata": {},
            "outputs": [],
            "source": CELL_SOURCE,
        }
    )
    nb["cells"] = cells

    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
        f.write("\n")

    print(f"Célula AUC adicionada ao final de {NOTEBOOK_PATH}.")


if __name__ == "__main__":
    main()
