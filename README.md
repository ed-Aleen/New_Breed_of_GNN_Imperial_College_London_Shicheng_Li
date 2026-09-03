# When Explanations Do Not Explain

### Understanding and Mitigating Explanation Degeneration in Self-Explainable GNNs

MSc Individual Project, Department of Computing, Imperial College London.
**Shicheng Li** · `shicheng.li25@imperial.ac.uk`

---

A self-explainable GNN selects a subgraph and predicts from it alone, so that subgraph is supposed to
be the evidence. It need not be. A subgraph occurring in *every* input carries the predicted label
just as well as one carrying evidence, so the explanation can become a register rather than a reason
while accuracy stays perfect. Prior work establishes that this happens and detects it. This project
defines what it is, explains **why** it happens, and supplies a **remedy**.

## What degeneration is

The classifier reads the selected subgraph and nothing else, so the display determines the prediction
by construction — for a faithful model and a degenerate one alike. That is why *the explanation
predicts the label* has no discriminating power, and why necessity tests rank a register highest of
all: deleting the model's only input does change the prediction.

The discriminating question concerns the other channel. Write `O` for the occurrence profile: for
each content the extractor displays, whether that content actually occurs in the input. Degeneration
is

```
rho  =  H(y_hat | O) / H(y_hat)  ∈ [0, 1]
```

the fraction of the prediction that survives knowing the occurrence profile. `rho = 1` is strict
degeneration: every bit the explanation carries comes from the act of selecting, none from the
evidence being there. It is a property of the trained model and the distribution, not of an instance.

## Why it happens

**1. The objective is blind.** Every published training objective — classification loss plus a score
or size regulariser — is a functional of the prediction behaviour and the score field. The occurrence
profile is not an argument of any term, so two strategies, one displaying a class marker and one a
universal token, can score *identically*. Nothing pays for faithfulness, and the mechanisms below
meet no opposition.

**2. The extractor is confined to a menu fixed before training.** A message-passing scorer gives
equal scores to nodes of equal Weisfeiler-Leman colour, so any display is a union of colour classes.
That menu is finite, enumerable by one pass of colour refinement, and independent of the weights.
The converse is just as tight: any score field constant on colour classes is realisable, including
one that reverses any preference order you care to name. **Confinement fixes the menu; it does not
say which item to pick.** The choice is left to training, which is the first source of seed variance.

**3. Once picked, the choice sticks.** The classifier specialises on the current display while the
backbone receives exactly zero gradient on the unselected classes, so switching means feeding the
classifier content it has never seen. The resulting barrier is concrete: its existence is decided by
the sign of a quantity measurable on any checkpoint with two forward passes, and its height is capped
by the sparsity toll rather than by the fit term.

**4. The verdict, and where it does not apply.** When the displayed types are near-universal — a
contingency-table condition computable *before training* — the occurrence profile is constant, so
`rho = 1` while accuracy is untouched. A five-tuple certificate settles this on any checkpoint with
**zero training and zero backpropagation**. That condition is a genuine premise: a watermark
counterexample shows no architecture-and-objective-only argument can exist, and the cells where it
fails are exactly those the literature reports as non-degenerate. The chain predicts its own
silences.

## How to stop it

Step 1 rules out the objective as a lever and step 2 rules out the menu. What remains is the
classifier's freedom: trained jointly it can learn any codebook, which is what makes a meaningless
display decodable. **The remedy is to confiscate it.** The classifier is calibrated on random
displays drawn independently of the extractor, at colour-class granularity, then frozen; the
extractor is trained afterwards against that fixed decoder. Step 2 forces the granularity, making
the calibration support coincide with the displays the extractor can reach.

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
bash run_occ.sh 0 certo_mutag_gsat   # the five-tuple certificate
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
