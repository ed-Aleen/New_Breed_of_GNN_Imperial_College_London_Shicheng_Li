#!/bin/bash
# Set GOODTG=${GOODTG:-goodtg} to force imports from this checkout when another copy
# of the package is installed in the environment.
# E4 / OCC launcher (clean repo). Usage:
#   bash run_occ.sh <gpu_idx> <job>
# Long jobs: run under nohup, e.g.
#   mkdir -p logs_occ
#   nohup bash run_occ.sh 0 occA_mutag_gsat > logs_occ/occA_mutag_gsat.log 2>&1 &
#
# Jobs (in the order they should be run):
#   prereg           E-OCC0 ceilings, MUTAG+RBGV (zero training; run FIRST and
#                    archive the printed numbers before any training)
#   check            channel-equivalence certificate, all 8 cells (zero training)
#   smokeA           3-epoch Stage-A smoke on MUTAG-GSAT seed 1
#   base_<cell>      missing same-protocol baselines (mnist_gsat topup s4-5,
#                    rbgv_smgnn, mnist_smgnn, sst2p_gsat, sst2p_smgnn)
#   occA_<cell>      OCC Stage A (calibrate+freeze classifier), 5 seeds
#   occB_<cell>      OCC Stage B (extractor vs frozen g0), 5 seeds
#   evalb_<cell>     baseline metrics  (test + EST/Fid-/RFid-/Nec/Suf)
#   evalo_<cell>     OCC metrics       (same protocol, --save_tag occ)
#   certb_<cell>     baseline certificate (rho, I(O;yhat), |R|, band) -- the
#   certo_<cell>     OCC certificate       DEFINITION-level test, zero training
# Cells: mutag_gsat mutag_smgnn rbgv_gsat rbgv_smgnn mnist_gsat mnist_smgnn
#        sst2p_gsat sst2p_smgnn
# Known-bad seeds (report, do not silently drop): MUTAG-SMGNN baseline s2/s4
# dead (HEALTH_REPORT); SST2P-SMGNN baseline historically unstable on s3/s5.
set -e
cd "$(dirname "$0")"
GPU=${1:?gpu_idx}
JOB=${2:?job}
SEEDS=1/2/3/4/5

cfg() {  # cfg <cell> <occ?>  -> config path
  local d m
  case ${1%_*} in
    mutag) d=MUTAG ;; rbgv) d=BAColorGVIsol ;; mnist) d=MNIST ;; sst2p) d=SST2Planted ;;
  esac
  case ${1#*_} in
    gsat) m=GSAT ;; smgnn) m=SMGNN_sec6 ;;
  esac
  [ "$2" = occ ] && m=${m}_OCC
  echo "final_configs/$d/basis/no_shift/$m.yaml"
}
ratio() {  # D.5 protocol threshold
  case $1 in
    rbgv_gsat|*_smgnn) echo 0.5 ;;
    *_gsat) echo 0.9 ;;
  esac
}
membudget() {
  case $1 in
    mnist_*|sst2p_*) echo "--numsamples_budget 500" ;;
    *) echo "" ;;
  esac
}

case $JOB in
  prereg)
    python occ/occ_ceiling.py --config_path "$(cfg mutag_gsat occ)" --seeds 1 --task test --backbone ACR2 --gpu_idx "$GPU"
    python occ/occ_ceiling.py --config_path "$(cfg rbgv_gsat occ)" --seeds 1 --task test --backbone ACR2 --gpu_idx "$GPU"
    ;;
  check)
    for c in mutag_gsat mutag_smgnn rbgv_gsat rbgv_smgnn mnist_gsat mnist_smgnn sst2p_gsat sst2p_smgnn; do
      python occ/occ_channel_check.py --config_path "$(cfg "$c" occ)" --seeds 1 --task test --backbone ACR2 --gpu_idx "$GPU"
    done
    ;;
  smokeA)
    python occ/occ_stage_a.py --config_path "$(cfg mutag_gsat occ)" --seeds 1 --task train \
      --backbone ACR2 --gpu_idx "$GPU" --save_tag occsmoke --occ_epochs 3
    ;;
  base_mnist_gsat)  ${GOODTG:-goodtg} --config_path "$(cfg mnist_gsat)"  --seeds 4/5    --task train --backbone ACR2 --gpu_idx "$GPU" ;;
  base_rbgv_smgnn)  ${GOODTG:-goodtg} --config_path "$(cfg rbgv_smgnn)"  --seeds $SEEDS --task train --backbone ACR2 --gpu_idx "$GPU" ;;
  base_mnist_smgnn) ${GOODTG:-goodtg} --config_path "$(cfg mnist_smgnn)" --seeds $SEEDS --task train --backbone ACR2 --gpu_idx "$GPU" ;;
  base_sst2p_gsat)  ${GOODTG:-goodtg} --config_path "$(cfg sst2p_gsat)"  --seeds $SEEDS --task train --backbone ACR2 --gpu_idx "$GPU" ;;
  base_sst2p_smgnn) ${GOODTG:-goodtg} --config_path "$(cfg sst2p_smgnn)" --seeds $SEEDS --task train --backbone ACR2 --gpu_idx "$GPU" ;;
  occA_*)
    C=${JOB#occA_}
    python occ/occ_stage_a.py --config_path "$(cfg "$C" occ)" --seeds $SEEDS --task train \
      --backbone ACR2 --gpu_idx "$GPU" --save_tag occ
    ;;
  occB_*)
    C=${JOB#occB_}
    ${GOODTG:-goodtg} --config_path "$(cfg "$C" occ)" --seeds $SEEDS --task train \
      --backbone ACR2 --gpu_idx "$GPU" --save_tag occ
    ;;
  certb_*|certo_*)
    C=${JOB#cert?_}
    if [ "${JOB%%_*}" = certo ]; then CFG=$(cfg "$C" occ); TAG="--save_tag occ"; else CFG=$(cfg "$C"); TAG=""; fi
    python occ/occ_certificate.py --config_path "$CFG" --seeds $SEEDS --task test \
      --splits id_val --ratios "$(ratio "$C")" --backbone ACR2 --gpu_idx "$GPU" $TAG
    ;;
  evalb_*|evalo_*)
    C=${JOB#eval?_}
    if [ "${JOB%%_*}" = evalo ]; then CFG=$(cfg "$C" occ); TAG="--save_tag occ"; else CFG=$(cfg "$C"); TAG=""; fi
    ${GOODTG:-goodtg} --config_path "$CFG" --seeds $SEEDS --task test --backbone ACR2 --gpu_idx "$GPU" $TAG
    ${GOODTG:-goodtg} --config_path "$CFG" --seeds $SEEDS --task eval_metric \
      --metrics "suff_cause/fidm/rfidm/nec/suff" --splits id_val \
      --ratios "$(ratio "$C")" --expval_budget 50 $(membudget "$C") \
      --backbone ACR2 --gpu_idx "$GPU" $TAG
    ;;
  *) echo "unknown job $JOB"; exit 1 ;;
esac
