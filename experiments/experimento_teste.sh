#!/bin/bash
#SBATCH -p gorgonas
#SBATCH --time=14:00:00
#SBATCH --cpus-per-task=32
#SBATCH --output=/sonic_home/igor.viveiros/logs/tioms-context-window-%j.out

set -euo pipefail

source /sonic_home/igor.viveiros/py310/bin/activate
PYTHON_BIN=/sonic_home/igor.viveiros/py310/bin/python

cd /sonic_home/igor.viveiros/paralelo || exit 1

"$PYTHON_BIN" ./main_test.py \
