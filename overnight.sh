#!/bin/bash
# Overnight E4/OCC orchestrator.
#   bash overnight.sh "<gpu list>"      e.g.  bash overnight.sh "0 1 2"
# Groups (cell = StageA -> StageB kept adjacent) are distributed round-robin
# over the given GPUs; each lane runs its groups SEQUENTIALLY (no GPU pile-up).
# A finished job leaves logs_occ/DONE_<job>; rerunning this script skips those,
# so it resumes wherever the night ended.
cd "$(dirname "$0")"
mkdir -p logs_occ
read -ra GPUS <<< "${1:-0}"
N=${#GPUS[@]}
# Optional 2nd arg: partition for running on SEPARATE MACHINES (shared /vol
# filesystem carries the DONE markers, so parts never redo each other's work):
#   A = MUTAG both cells + RBGV both cells (+ leftover mnist_smgnn occ)
#   B = RBGV-SMGNN baseline + all MNIST
#   C = all SST2P
PART=${2:-all}

GROUPS_LIST=(
  "occB_mutag_gsat evalb_mutag_gsat evalo_mutag_gsat"
  "occA_mutag_smgnn occB_mutag_smgnn"
  "occA_rbgv_gsat occB_rbgv_gsat"
  "base_rbgv_smgnn"
  "occA_rbgv_smgnn occB_rbgv_smgnn"
  "base_mnist_gsat"
  "occA_mnist_gsat occB_mnist_gsat"
  "base_mnist_smgnn"
  "occA_mnist_smgnn occB_mnist_smgnn"
  "base_sst2p_gsat"
  "occA_sst2p_gsat occB_sst2p_gsat"
  "base_sst2p_smgnn"
  "occA_sst2p_smgnn occB_sst2p_smgnn"
)

run_lane() {
  local gpu=$1; shift
  for grp in "$@"; do
    for job in $grp; do
      if [ -e "logs_occ/DONE_$job" ]; then
        echo "[lane $gpu] skip $job (done)"; continue
      fi
      echo "[lane $gpu] $(date +%H:%M) start $job"
      if bash run_occ.sh "$gpu" "$job" > "logs_occ/$job.log" 2>&1; then
        touch "logs_occ/DONE_$job"
        echo "[lane $gpu] $(date +%H:%M) done  $job"
      else
        echo "[lane $gpu] $(date +%H:%M) FAIL  $job (lane continues; deps of this cell will fail too)"
      fi
    done
  done
}

case $PART in
  all) IDX=$(seq 0 12) ;;
  A)   IDX="0 1 2 4 8" ;;
  B)   IDX="3 5 6 7" ;;
  C)   IDX="9 11 10 12" ;;
  *) echo "unknown part $PART (use A|B|C or omit)"; exit 1 ;;
esac

declare -a LANE_GROUPS
n_i=0
for i in $IDX; do
  lane=$((n_i % N))
  LANE_GROUPS[$lane]="${LANE_GROUPS[$lane]}|${GROUPS_LIST[$i]}"
  n_i=$((n_i + 1))
done

PIDS=()
for lane in $(seq 0 $((N - 1))); do
  IFS='|' read -ra grps <<< "${LANE_GROUPS[$lane]#|}"
  run_lane "${GPUS[$lane]}" "${grps[@]}" &
  PIDS+=($!)
done
wait "${PIDS[@]}"
echo "ALL LANES FINISHED $(date)"
