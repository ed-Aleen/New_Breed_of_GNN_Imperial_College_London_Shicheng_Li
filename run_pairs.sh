#!/bin/bash
# Paired qualitative panels for every cell.  Usage: bash run_pairs.sh <gpu> [seed]
# Selection is pre-registered: the first 8 id_val graphs, same graphs in both
# arms (paired by graph identity), never chosen by appearance.
set -u
cd "$(dirname "$0")"
GPU=${1:?gpu_idx}; SEED=${2:-1}; N=${3:-8}; RANK=${4:-first}
OUT=/vol/bitbucket/sl8025/gnn_deg_expl/Thesis/figures/pairs
[ "$RANK" != first ] && OUT="${OUT}_$RANK"
mkdir -p "$OUT"
run() {  # run <dataset_dir> <model_yaml> <thr> <extra...>
  echo "===== $1 / $2 (seed $SEED)"
  python plot_pairs.py --config_path "final_configs/$1/basis/no_shift/$2.yaml" \
    --occ_config "${2}_OCC.yaml" --seeds "$SEED" --task test --ratios "$3" \
    --backbone ACR2 --gpu_idx "$GPU" --n "$N" --rank "$RANK" --out "$OUT" 2>&1 \
    | grep -aE "#R#|Traceback|Error:" | sed 's/\x1b\[[0-9;]*m//g'
}
run BAColorGVIsol SMGNN_sec6 0.5
run BAColorGVIsol GSAT       0.5
run MUTAG         GSAT       0.9
run MUTAG         SMGNN_sec6 0.5
run MNIST         GSAT       0.9
run MNIST         SMGNN_sec6 0.5
run SST2Planted   GSAT       0.9
run SST2Planted   SMGNN_sec6 0.5
