# Occurrence-Calibrated Classifiers for Self-Explainable GNNs

Code for the MSc Individual Project *Why Self-Explainable GNNs Produce Degenerate Explanations,
and How to Stop Them*, Department of Computing, Imperial College London.

**Shicheng Li** · `shicheng.li25@imperial.ac.uk`

---

## What this repository provides

A self-explainable GNN selects a subgraph and predicts from it alone, so the selected subgraph is
supposed to be the evidence. It need not be. Because the classifier is trained jointly with the
extractor, it is free to learn any codebook it likes, and a subgraph that occurs in *every* input
carries the predicted label just as well as one that carries evidence. The explanation is then a
register, not a reason, and accuracy is untouched.

Existing work establishes that this happens and supplies a metric that detects it. This repository
supplies the remedy and the certificate.

**The remedy is to confiscate the codebook.** The classifier is calibrated on random displays drawn
independently of the extractor, at the granularity of Weisfeiler-Leman colour classes, and then
frozen; the extractor is trained afterwards against that fixed decoder. The colour-class granularity
is not a design preference. A message-passing extractor can only ever display unions of colour
classes, so calibrating at that granularity covers exactly the family of displays the extractor can
reach.

**Three things follow, and this repository measures all three.** A universally present register
decodes to the label marginal, so displaying it buys no accuracy. Accuracy becomes purchasable only
with evidence, and the ceiling is a contingency-table quantity computable *before any training*. The
residual smuggling channel is confined to counting statistics.

On the released checkpoints, with accuracy preserved (0.998 → 0.998), the calibrated extractor
selects ground-truth evidence more accurately than the jointly trained baseline on **82% of the
graphs**, and the universally present, label-irrelevant register that enters the baseline
explanation in half of the graphs enters the calibrated explanation in none of them.

The diagnostics run on **frozen checkpoints with zero training and zero backpropagation**: a
five-tuple certificate for the degeneration index, a pre-registered accuracy ceiling, a display
channel equivalence check, and gradient-support and barrier probes for the underlying mechanism.

---

## Installation

```bash
python -m venv venv && source venv/bin/activate
pip install -e .          # registers the goodtg entry point
```

Config paths are passed without the `configs/` prefix. If another copy of the package is installed
in the same environment, set `GOODTG` to a wrapper that pins imports to this checkout.

## Reproducing the pipeline

Eight cells: `{MUTAG, MNISTsp, RBGV, SST2P} × {GSAT, SMGNN}`, five seeds each. Every stage is a job
of `run_occ.sh <gpu_idx> <job>`.

```bash
bash run_occ.sh 0 prereg              # accuracy ceiling from the contingency table (no training)
bash run_occ.sh 0 check               # certify the display channel against true subgraph re-encoding
bash run_occ.sh 0 occA_mutag_gsat     # stage A: calibrate the classifier on class-level random displays, freeze
bash run_occ.sh 0 occB_mutag_gsat     # stage B: train the extractor against the frozen decoder
bash run_occ.sh 0 evalo_mutag_gsat    # EST, Fid-, RFid-, Nec, Suf
bash run_occ.sh 0 certo_mutag_gsat    # five-tuple certificate for the degeneration index
```

Replace `occ*`/`evalo`/`certo` with `base_*`/`evalb`/`certb` for the jointly trained baseline, and
`mutag_gsat` with any of `mutag_smgnn mnist_gsat mnist_smgnn rbgv_gsat rbgv_smgnn sst2p_gsat
sst2p_smgnn`.

Paired qualitative panels, same graphs in both arms, paired by graph identity:

```bash
bash run_pairs.sh 0 1 8               # all eight cells, seed 1, eight graphs each
```

## Mechanism probes (frozen checkpoints, no training)

| Script | Measures |
|---|---|
| `occ/occ_certificate.py` | degeneration index, display geometry, confidence band, codebook |
| `occ/occ_ceiling.py` | pre-registered evidence-availability ceiling with a permutation null |
| `occ/occ_channel_check.py` | masked forward pass against true subgraph re-encoding |
| `probes/freeze_probe.py` | input-gradient support of the display, layer by layer |
| `probes/barrier_scan.py` | switching barrier between realised and alternative colour classes |
| `probes/wl_colors.py`, `probes/shape_check.py` | colour refinement and reachable display family |

## Layout

| Path | Contents |
|---|---|
| `occ/` | calibration stage and the zero-training diagnostics |
| `probes/` | mechanism probes on frozen checkpoints |
| `figures/` | paired explanation panels and the mechanism figures |
| `data/` | dataset acquisition |
| `configs/final_configs/` | one directory per dataset; `*_OCC.yaml` selects the calibrated pipeline |
| `GOOD/networks/models/` | GSAT, SMGNN, DIR, GIN backbones |
| `GOOD/ood_algorithms/algorithms/` | per-model training objectives |
| `GOOD/kernel/pipelines/` | training loop, display generation, faithfulness metrics |
| `GOOD/utils/occ.py` | colour refinement, class-level mask sampling, subgraph re-encoding |
| `PORTS.md` | every deviation from the upstream code, with its reason |

## Datasets

Implementations are in `GOOD/data/good_datasets`. MUTAG and SST2P are fetched by
`data/get_mutag_data.sh` and `data/get_sst2_data.sh`; MNIST75sp superpixels are extracted by
`scripts/extract_mnist_superpixels.py` and placed with `data/get_mnist_data.sh`; RBGV is generated
on first use.

## Built on

This repository extends the code released with Azzolin et al., *GNN Explanations that do not Explain
and How to find Them*, ICLR 2026 ([paper](https://arxiv.org/abs/2601.20815),
[repo](https://github.com/steveazzolin/gnn_deg_expl)), which in turn builds on
[GOOD](https://github.com/divelab/GOOD). The calibration design is transplanted from Jethani et al.,
*Have We Learned to Explain?*, AISTATS 2021 ([REAL-X](https://arxiv.org/abs/2103.01890)); the
colour-class granularity, the certificate, and the pre-registered ceiling are contributed here. The
upstream README is preserved as `README.upstream.md`.
