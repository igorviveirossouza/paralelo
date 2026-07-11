#!/bin/bash
#SBATCH -p gorgonas
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --array=0-107%16
#SBATCH --mem=4G
#SBATCH --time=08:00:00
#SBATCH --job-name=master_backtest
#SBATCH --output=/sonic_home/igor.viveiros/paralelo/logs/master-backtest-%A_%a.out
#SBATCH --error=/sonic_home/igor.viveiros/paralelo/logs/master-backtest-%A_%a.err

set -euo pipefail

PARALELO_ROOT="${PARALELO_ROOT:-/sonic_home/igor.viveiros/paralelo}"
cd "$PARALELO_ROOT"
mkdir -p logs

export RUN_ESTIMATION=false
export RUN_BACKTEST=true

# Dentro de um job já alocado, as diretivas #SBATCH do script chamado são apenas comentários.
bash experimentos/carteiras_master_array.sh
