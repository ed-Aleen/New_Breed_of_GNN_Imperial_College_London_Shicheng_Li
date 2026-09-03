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

On the released checkpoints, at matched display size and with accuracy preserved (0.998 → 0.998),
the label-irrelevant register enters the baseline explanation in half of the graphs and enters the
calibrated model's explanation in none of them; ground-truth precision rises from 0.835 to 1.000.

The diagnostics run on **frozen checkpoints with zero training and zero backpropagation**: a
five-tuple certificate for the degeneration index, a pre-registered accuracy ceiling, a display
channel equivalence check, and gradient-support and barrier probes for the underlying mechanism.

---

## Installation

```bash
python -m venv venv && source venv/bin/activate
pip install -e .          # registers the goodtg entry point
```

Use `./goodtg` rather than the bare `goodtg` console script; the wrapper pins imports to this
checkout. Config paths are passed without the `configs/` prefix.

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
sst2p_smgnn`. `overnight.sh` and `evals.sh` sequence the full campaign across several machines.

Paired qualitative panels, same graphs in both arms, paired by graph identity:

```bash
bash run_pairs.sh 0 1 8               # all eight cells, seed 1, eight graphs each
```

## Mechanism probes (frozen checkpoints, no training)

| Script | Measures |
|---|---|
| `occ_certificate.py` | degeneration index, display geometry, confidence band, codebook |
| `occ_ceiling.py` | pre-registered evidence-availability ceiling with a permutation null |
| `occ_channel_check.py` | masked forward pass against true subgraph re-encoding |
| `freeze_probe.py` | input-gradient support of the display, layer by layer |
| `barrier_scan.py` | switching barrier between realised and alternative colour classes |
| `wl_colors.py`, `shape_check.py` | colour refinement and reachable display family |

## Layout

| Path | Contents |
|---|---|
| `GOOD/networks/models/` | GSAT, SMGNN, DIR, GIN backbones |
| `GOOD/ood_algorithms/algorithms/` | per-model training objectives |
| `GOOD/kernel/pipelines/` | training loop, display generation, faithfulness metrics |
| `GOOD/utils/occ.py` | colour refinement, class-level mask sampling, subgraph re-encoding |
| `configs/final_configs/` | one directory per dataset; `*_OCC.yaml` selects the calibrated pipeline |
| `PORTS.md` | every deviation from the upstream code, with its reason |

## Datasets

Implementations are in `GOOD/data/good_datasets`. MUTAG and SST2P are fetched by
`get_mutag_data.sh` and `get_sst2_data.sh`; MNIST75sp superpixels are extracted by
`scripts/extract_mnist_superpixels.py`; RBGV is generated on first use.

## Built on

This repository extends the code released with Azzolin et al., *GNN Explanations that do not Explain
and How to find Them*, ICLR 2026 ([paper](https://arxiv.org/abs/2601.20815),
[repo](https://github.com/steveazzolin/gnn_deg_expl)), which in turn builds on
[GOOD](https://github.com/divelab/GOOD). The calibration design is transplanted from Jethani et al.,
*Have We Learned to Explain?*, AISTATS 2021 ([REAL-X](https://arxiv.org/abs/2103.01890)); the
colour-class granularity, the certificate, and the pre-registered ceiling are contributed here. The
upstream README is preserved as `README.upstream.md`.
