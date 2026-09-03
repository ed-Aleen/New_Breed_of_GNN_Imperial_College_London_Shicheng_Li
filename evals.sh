#!/bin/bash
# E4 evaluation launcher (both arms: metrics + definition-level certificate).
#   bash evals.sh <gpu_idx> <part>      part = A | B | C | all
# Same DONE-marker resume logic as overnight.sh: finished jobs are skipped,
# so rerunning continues where it stopped.
#   A = MUTAG-SMGNN + RBGV both cells      (light)
#   B = MNIST both cells                   (needs --numsamples_budget 500)
#   C = SST2P both cells
# Each cell runs: metrics baseline, metrics OCC, certificate baseline,
# certificate OCC. MUTAG-GSAT metrics are already done (markers present).
cd "$(dirname "$0")"
mkdir -p logs_occ
GPU=${1:?gpu_idx}
PART=${2:-all}

case $PART in
  A) CELLS="mutag_gsat mutag_smgnn rbgv_gsat rbgv_smgnn" ;;
  B) CELLS="mnist_gsat mnist_smgnn" ;;
  C) CELLS="sst2p_gsat sst2p_smgnn" ;;
  all) CELLS="mutag_gsat mutag_smgnn rbgv_gsat rbgv_smgnn mnist_gsat mnist_smgnn sst2p_gsat sst2p_smgnn" ;;
  *) echo "unknown part $PART"; exit 1 ;;
esac

for c in $CELLS; do
  for j in "certb_$c" "certo_$c" "evalb_$c" "evalo_$c"; do
    if [ -e "logs_occ/DONE_$j" ]; then
      echo "skip $j (done)"; continue
    fi
    echo "$(date +%H:%M) start $j"
    if bash run_occ.sh "$GPU" "$j" > "logs_occ/$j.log" 2>&1; then
      touch "logs_occ/DONE_$j"; echo "$(date +%H:%M) done  $j"
    else
      echo "$(date +%H:%M) FAIL  $j -- see logs_occ/$j.log"
    fi
  done
done
echo "PART $PART FINISHED $(date)"
