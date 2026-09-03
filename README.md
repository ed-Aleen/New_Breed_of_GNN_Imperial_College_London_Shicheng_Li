# When Explanations Do Not Explain

### Understanding and Mitigating Explanation Degeneration in Self-Explainable GNNs

MSc Individual Project, Department of Computing, Imperial College London.
**Shicheng Li** · `shicheng.li25@imperial.ac.uk`

---

A self-explainable GNN selects a subgraph and predicts from it alone, so that subgraph is supposed
to be the evidence. It need not be. A subgraph occurring in *every* input carries the predicted label
just as well as one carrying evidence, so the explanation can become a register rather than a reason
while accuracy stays perfect. Prior work establishes that this happens and detects it. This project
explains **why** it happens and supplies a **remedy**.

## Why explanations degenerate

**The objective never pays for faithfulness.** Every published training objective is a functional of
the prediction behaviour and the score field. Whether the displayed content occurs in the input is
not an argument of any term, so the most faithful and the strictly degenerate strategy can score
identically.

**The extractor picks from a menu fixed before training.** A message-passing scorer assigns equal
scores to nodes of equal Weisfeiler-Leman colour, so any display is a union of colour classes. That
menu is enumerable by one pass of colour refinement, independently of the weights, and the
architecture does not say which item to pick.

**Once picked, the choice sticks.** The classifier specialises on the current display while the
backbone receives exactly zero gradient on the unselected classes, and the stationary point is held
by a barrier whose existence is decided by a sign measurable at any checkpoint.

When the displayed types are near-universal — a contingency-table condition checkable before
training — the explanation carries only the code of the selection act. A five-tuple certificate
decides this on any checkpoint with **zero training and zero backpropagation**.

## How to stop it

The freedom being exploited is the classifier's: trained jointly, it can learn any codebook. **The
remedy is to confiscate it.** The classifier is calibrated on random displays drawn independently of
the extractor, at colour-class granularity, then frozen; the extractor is trained afterwards against
that fixed decoder. The granularity is forced by the confinement result above, making the calibration
support coincide with the displays the extractor can reach.

A universally present register now decodes to the label marginal, so displaying it buys no accuracy.
Accuracy becomes purchasable only with evidence, up to a ceiling computable before any training, and
the residual channel is confined to counting statistics.

On the released checkpoints, with accuracy preserved (0.998 → 0.998), the calibrated extractor
selects ground-truth evidence more accurately than the jointly trained baseline on **82% of graphs**,
and the label-irrelevant register that enters the baseline explanation in half of the graphs enters
the calibrated explanation in none of them.

## Installation

```bash
python -m venv venv && source venv/bin/activate
pip install -e .          # registers the goodtg entry point
```

Config paths omit the `configs/` prefix. If another copy of the package is installed in the
environment, set `GOODTG` to a wrapper pinning imports to this checkout.

## Reproducing

Eight cells, `{MUTAG, MNISTsp, RBGV, SST2P} × {GSAT, SMGNN}`, five seeds each.

```bash
bash run_occ.sh 0 prereg             # accuracy ceiling from the contingency table (no training)
bash run_occ.sh 0 check              # certify the display channel against subgraph re-encoding
bash run_occ.sh 0 occA_mutag_gsat    # calibrate the classifier, freeze it
bash run_occ.sh 0 occB_mutag_gsat    # train the extractor against the frozen decoder
bash run_occ.sh 0 evalo_mutag_gsat   # EST, Fid-, RFid-, Nec, Suf
bash run_occ.sh 0 certo_mutag_gsat   # degeneration certificate
bash run_pairs.sh 0 1 8              # paired explanation panels, both arms, same graphs
```

Use `base_*`, `evalb_*`, `certb_*` for the jointly trained baseline, and `mutag_smgnn mnist_gsat
mnist_smgnn rbgv_gsat rbgv_smgnn sst2p_gsat sst2p_smgnn` for the other cells.

## Layout

| Path | Contents |
|---|---|
| `occ/` | calibration stage; certificate, ceiling, channel and calibration checks |
| `probes/` | colour refinement, reachable displays, gradient support, switching barrier |
| `figures/` | paired explanation panels and the mechanism figures |
| `data/`, `configs/` | dataset acquisition; one config directory per dataset, `*_OCC.yaml` selects the calibrated pipeline |
| `GOOD/` | model implementations, training loop, faithfulness metrics, `utils/occ.py` |
| `PORTS.md` | every deviation from the upstream code, with its reason |

Dataset implementations are in `GOOD/data/good_datasets`; MNIST75sp superpixels come from
`scripts/extract_mnist_superpixels.py` and RBGV is generated on first use.

## Built on

The code released with Azzolin et al., *GNN Explanations that do not Explain and How to find Them*,
ICLR 2026 ([paper](https://arxiv.org/abs/2601.20815),
[repo](https://github.com/steveazzolin/gnn_deg_expl)), itself built on
[GOOD](https://github.com/divelab/GOOD). The calibration design is transplanted from Jethani et al.,
*Have We Learned to Explain?*, AISTATS 2021 ([REAL-X](https://arxiv.org/abs/2103.01890)); the
colour-class granularity, the certificate and the pre-registered ceiling are contributed here.
`README.upstream.md` keeps the upstream text.
