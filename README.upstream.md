# GNN Explanations that do not Explain and How to find Them

Official repo for the ICLR 2026 paper [*GNN Explanations that do not Explain and How to find Them*](https://arxiv.org/pdf/2601.20815).

⚠️This repo is still under maintainance. Do not esistate to write to the corresponding author for any inquiry⚠️

## Installation

Regarding the installation of the library -- which is required in order to run the code with the `goodtg` functionality -- please refer to [GOOD](https://github.com/divelab/GOOD) or to our previous [project](https://github.com/steveazzolin/beyond-topo-segnns/).

## Model Implementations

Models implementation can be found here `GOOD\networks\models`.
Basic implementations of classifiers and shared modules are available in `GOOD\networks\models\Classifiers.py`.

## Training Details

The file `GOOD\ood_algorithms\algorithms\BaseOOD.py` contains the basic training logic for each model. Then, for specific training protocols of each architecture, please refer to the corresponding file in the same folder. Note that configurations of additional models not tested in this work may be present, as they are inhereted from the original GOOD implementation.

In this work, the following models are considered:
 - GIN (ERM - Empirical Risk Minimization)
 - GSAT
 - DIR
 - SMGNN

## Configurations files

Configuration files and hyper-parameter details for each experiment are available in `configs/final_configs/`.

## Datasets

Dataset implementations are provided in `GOOD\data\good_datasets`.

For generating MNIST75sp the MNIST dataset, please refer to the [original paper](https://github.com/bknyaz/graph_attention_pool/tree/master/scripts). We included in our codebase the file `scripts\extract_mnist_superpixels.py` for ease of reproduction.

## Checkpoints

Checkpoints will be made available soon.

## Experiments

Please refer to `scripts\eval.sh` for examples of how to run the code for each model and dataset.

The `--task` argument regulates the behaviour of the code, and can be set as follows:

- `test`: Evaluate the model
- `train`: Train the model
- `plot_explanations`: Plot examples of explanations
- `eval_metric`: Evaluate a faithfulness metric. The metric to test has to be specified in the `metrics` argument (e.g. `--metrics rfidm/rfidp/suff_cause/suff/nec/counter_fid`). See `scripts\faithfulness_ablation_expval_budget.sh` for an example.
