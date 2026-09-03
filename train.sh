#!/usr/bin/env bash
# Single-job launcher so each model can run on its own machine (--gpu_idx 0).
# Usage:  bash train.sh <job>
#   jobs: mutag_gsat  mutag_smgnn  sst2p_gsat  sst2p_dir  sst2p_smgnn
cd "$(dirname "$(readlink -f "$0")")"          # repo root, wherever it is cloned
# activate a virtualenv if VENV points at one, otherwise use the current interpreter
[ -n "${VENV:-}" ] && [ -f "$VENV/bin/activate" ] && source "$VENV/bin/activate"
export PATH="$PWD:$PATH"  # bare goodtg -> clean-repo wrapper

M=final_configs/MUTAG/basis/no_shift
S=final_configs/SST2Planted/basis/no_shift

case "$1" in
  mutag_gsat)   CFG=$M/GSAT.yaml;        SEEDS=1/2/3/4/5 ;;
  mutag_smgnn)  CFG=$M/SMGNN_sec6.yaml;  SEEDS=1/2/3/4/5 ;;
  sst2p_gsat)   CFG=$S/GSAT.yaml;        SEEDS=1/2/3/4/5 ;;
  sst2p_dir)    CFG=$S/DIR_K10pct.yaml;  SEEDS=1/2/4/5   ;;   # drop seed 3 (paper D.3)
  sst2p_smgnn)  CFG=$S/SMGNN_sec6.yaml;  SEEDS=1/2/3/4   ;;   # drop seed 5 (paper D.3)
  *) echo "unknown job: $1"; echo "jobs: mutag_gsat mutag_smgnn sst2p_gsat sst2p_dir sst2p_smgnn"; exit 1 ;;
esac

echo "=== $1 : $CFG  seeds=$SEEDS ==="
goodtg --config_path "$CFG" --seeds "$SEEDS" --task train \
       --backbone ACR2 --gpu_idx 0 \
       >> logs/${1}_train.log 2>&1
echo "=== $1 DONE -> logs/${1}_train.log ==="
