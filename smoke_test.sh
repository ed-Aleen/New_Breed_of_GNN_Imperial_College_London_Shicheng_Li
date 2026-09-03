#!/usr/bin/env bash
# Smoke test: build + train each config for 1 seed / 6 epochs into an ISOLATED
# checkpoint dir (--save_tag smoketest), so it cannot touch real checkpoints.
# Verifies: config loads, dataset loads, model builds (ACR2 + norms), training
# steps run, checkpoint saves. Run BEFORE the overnight jobs.
# Usage:  bash smoke_test.sh
cd "$(dirname "$(readlink -f "$0")")"          # repo root, wherever it is cloned
# activate a virtualenv if VENV points at one, otherwise use the current interpreter
[ -n "${VENV:-}" ] && [ -f "$VENV/bin/activate" ] && source "$VENV/bin/activate"
export PATH="$PWD:$PATH"  # bare goodtg -> clean-repo wrapper

M=final_configs/MUTAG/basis/no_shift
S=final_configs/SST2Planted/basis/no_shift

JOBS=(
  "mutag_gsat   $M/GSAT.yaml"
  "mutag_smgnn  $M/SMGNN_sec6.yaml"
  "sst2p_gsat   $S/GSAT.yaml"
  "sst2p_dir    $S/DIR_K10pct.yaml"
  "sst2p_smgnn  $S/SMGNN_sec6.yaml"
)

mkdir -p logs
PASS=0; FAIL=0
for entry in "${JOBS[@]}"; do
  name=$(echo $entry | awk '{print $1}')
  cfg=$(echo  $entry | awk '{print $2}')
  echo "----- SMOKE: $name -----"
  goodtg --config_path "$cfg" --seeds 1 --task train \
         --backbone ACR2 --gpu_idx 0 \
         --max_epoch 6 --save_tag smoketest \
         > logs/smoke_${name}.log 2>&1
  rc=$?
  if [ $rc -eq 0 ]; then echo "  [PASS] $name"; PASS=$((PASS+1))
  else echo "  [FAIL rc=$rc] $name  -> see logs/smoke_${name}.log"; FAIL=$((FAIL+1)); fi
done

echo "============================================="
echo "SMOKE RESULT: $PASS passed, $FAIL failed"
[ $FAIL -eq 0 ] && echo "All good -> safe to launch overnight." \
                || echo "Fix failures before overnight run."
# Clean up the isolated smoke checkpoints
rm -rf storage/checkpoints/round1/*/*/*/*/smoketest 2>/dev/null
