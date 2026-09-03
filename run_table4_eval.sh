#!/usr/bin/env bash
# =============================================================================
# Reproduce Table 4 (NATURAL training, Section 6) for the overnight checkpoints.
# Columns: Test Acc | AUCROC | EST | Fid- | RFid-
#   - Test Acc / AUCROC : --task test           (AUCROC only prints for MNIST = has GT)
#   - EST               : metric "suff_cause"   (Extension Sufficiency Test)
#   - Fid-              : metric "fidm"
#   - RFid-             : metric "rfidm"
#
# Paper-alignment decisions (validated by smoke tests):
#   Threshold rule per model (Paper D.5), strictly aligned:
#   * GSAT  : raw scores, threshold 0.9   -> --ratios 0.9
#             (RBGV GSAT is the documented exception at 0.5 — RBGV already done.)
#   * SMGNN : threshold 0.5, with INSTANCE-WISE min-max normalisation applied
#             ONLY on MNIST & MUTAG (paper: "this happened for MNISTsp and MUTAG").
#             RBGV/SST2P SMGNN keep raw scores at 0.5 (Fig 20). The min-max is now
#             enabled in basic_pipeline.py (gated on SMGNN + dataset MNIST/MUTAG).
#   * DIR   : threshold 0.5 + topK from config ood_param (K), applied automatically.
#   * NO --pretrain : these are naturally-trained ckpts (--pretrain degenerate is
#     for the ATTACK models / Table 3).
#   * Metric split = id_val ; budget = 50 (matches the existing RBGV run).
#   * MNIST/SST2P: --numsamples_budget caps #graphs to avoid the RAM watchdog
#     SIGTERM (compute_metric materialises #graphs x budget Data objects at once).
#     Verified peak RSS ~5.8 GB at N=300/budget=30; N=500/budget=50 stays safe.
#
# NOTE (SMGNN MNIST/MUTAG): even with min-max, our trained SMGNN under-reports EST
#   vs paper (MUTAG seed1 ~0.45 vs 0.75) because the explanation head collapsed to
#   near-constant attention — this is the known training finding, NOT a settings bug.
#
# SST2P SMGNN: training crashed overnight; only seed 1 checkpoint exists.
#
# Usage:  bash run_table4_eval.sh <gpu_idx> [job]
#   job=1 : MUTAG (GSAT+SMGNN)       <- run on gpu36
#   job=2 : MNIST (GSAT+SMGNN)       <- run on gpu36
#   job=3 : SST2P (GSAT+DIR)         <- run on gpu04
#   job=all (default): run everything sequentially
# =============================================================================
set -u
cd /vol/bitbucket/sl8025/gnn_deg_expl_clean
source /vol/bitbucket/sl8025/gsat_venv/bin/activate
export PATH="/vol/bitbucket/sl8025/gnn_deg_expl_clean:$PATH"  # bare goodtg -> clean-repo wrapper

GPU="${1:-0}"
JOB="${2:-all}"
BUDGET=50
CFGROOT=final_configs
LOGDIR=logs/table4_eval
mkdir -p "$LOGDIR"

# run_combo <tag> <config> <seeds> <ratio> <numsamples|0> [extra args...]
run_combo () {
  local tag="$1" cfg="$2" seeds="$3" ratio="$4" nsamp="$5"; shift 5
  local extra=("$@")
  local nsflag=()
  [ "$nsamp" != "0" ] && nsflag=(--numsamples_budget "$nsamp")

  echo "==================================================================="
  echo "### $tag   (config=$cfg seeds=$seeds thr=$ratio nsamp=$nsamp)"
  echo "==================================================================="

  # Pass A: Test Acc + AUCROC
  goodtg --config_path "$cfg" --seeds "$seeds" --task test \
         --backbone ACR2 --gpu_idx "$GPU" "${extra[@]}" \
         > "$LOGDIR/${tag}_acc.log" 2>&1
  echo "  [acc]  $(grep -E '^(ID_TEST|TEST) ' "$LOGDIR/${tag}_acc.log" | head -2 | tr '\n' ' ')"
  grep -iE 'aucroc' "$LOGDIR/${tag}_acc.log" | tail -1 | sed 's/^/  [auc]  /'

  # Pass B: EST / Fid- / RFid-
  goodtg --config_path "$cfg" --seeds "$seeds" --task eval_metric \
         --metrics "suff_cause/fidm/rfidm" --splits id_val --ratios "$ratio" \
         --expval_budget "$BUDGET" "${nsflag[@]}" \
         --backbone ACR2 --gpu_idx "$GPU" "${extra[@]}" \
         > "$LOGDIR/${tag}_metric.log" 2>&1
  grep -E '(suff_cause|fidm|rfidm) rejection ' "$LOGDIR/${tag}_metric.log" \
       | sed 's/^/  [rej]  /'
  echo
}

# ---- Job 1: MUTAG (GSAT + SMGNN, seeds 1-5) ----
if [[ "$JOB" == "1" || "$JOB" == "all" ]]; then
  run_combo MUTAG_GSAT  "$CFGROOT/MUTAG/basis/no_shift/GSAT.yaml"       1/2/3/4/5 0.9 0
  run_combo MUTAG_SMGNN "$CFGROOT/MUTAG/basis/no_shift/SMGNN_sec6.yaml" 1/2/3/4/5 0.5 0
fi

# ---- Job 2: MNIST (GSAT + SMGNN, seeds 1-5; capped to 500 graphs for RAM) ----
if [[ "$JOB" == "2" || "$JOB" == "all" ]]; then
  run_combo MNIST_GSAT  "$CFGROOT/MNIST/basis/no_shift/GSAT.yaml"       1/2/3/4/5 0.9 500
  run_combo MNIST_SMGNN "$CFGROOT/MNIST/basis/no_shift/SMGNN_sec6.yaml" 1/2/3/4/5 0.5 500
fi

# ---- Job 3: SST2P (GSAT 1-5; DIR 1/2/4/5 drop seed3) ----
if [[ "$JOB" == "3" || "$JOB" == "all" ]]; then
  run_combo SST2P_GSAT "$CFGROOT/SST2Planted/basis/no_shift/GSAT.yaml"       1/2/3/4/5 0.9 500
  run_combo SST2P_DIR  "$CFGROOT/SST2Planted/basis/no_shift/DIR_K10pct.yaml" 1/2/4/5   0.5 500
fi

echo "############ DONE. Per-combo logs in $LOGDIR/ ############"
echo "Summary (rejection = RejRatio; EST=suff_cause, Fid-=fidm, RFid-=rfidm):"
grep -E '(suff_cause|fidm|rfidm) rejection ' "$LOGDIR"/*_metric.log
