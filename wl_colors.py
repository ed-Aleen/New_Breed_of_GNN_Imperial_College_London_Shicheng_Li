"""
E1 / step 1 of the experiment pipeline (EXPERIMENT_DESIGN.md):
architecture-matched colour refinement, computed BEFORE any training.

For each graph, computes the depth-L WL colours in two variants:
  - vanilla : new_c(v) = hash(c(v), multiset of neighbour colours)
  - acr     : new_c(v) = hash(c(v), multiset of neighbour colours,
                              multiset of ALL colours in the graph)
    The 'acr' variant matches the ACRConv update
        V(x) + A(sum over in-neighbours) + R(global add-pool broadcast)
    (GOOD/networks/models/BaseGNN.py:398-...), which is why vanilla WL
    would report FALSE tie-violations on this backbone (design clause E1-1).

Outputs (per dataset):
  * per-graph node->colour arrays for every requested split  -> .pt file
    (reused later by shape_check.py / certificate.py)
  * the E1-a table row: |X_L|, iota per attribute, tie-pair coverage,
    class-multiplicity stats
  * sanity check: acr colours refine vanilla colours (must be 100%)

Usage (inside gsat_venv, repo root):
  python wl_colors.py --config_path final_configs/MUTAG/basis/no_shift/GSAT.yaml
  python wl_colors.py --config_path final_configs/BAColorGVIsol/basis/no_shift/GSAT.yaml
  python wl_colors.py --config_path final_configs/MNIST/basis/no_shift/GSAT.yaml --decimals 4
  python wl_colors.py --config_path final_configs/SST2Planted/basis/no_shift/GSAT.yaml --decimals 4

Notes:
  --rounds defaults to config.model.model_layer (=2 for all main cells).
  --decimals: rounding for continuous node features before hashing
              (None = exact bytes; use 4 for MNISTsp/SST2P, and expect the
              near-unique warning there — exact-tie analysis is only
              meaningful on discrete-attribute datasets, see design E1-2).
"""
import argparse
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import torch


# --------------------------------------------------------------------------
# dataset loading through the repo's own config system (same data/splits as
# training, which is what pi_c statistics later must match)
# --------------------------------------------------------------------------
def load_repo_dataset(config_path):
    from GOOD.utils.args import args_parser
    from GOOD import config_summoner
    from GOOD.data import load_dataset

    argv_backup = sys.argv
    sys.argv = [
        "wl_colors.py",
        "--config_path", config_path,
        "--seeds", "1",
        "--task", "test",
        "--gpu_idx", "0",
    ]
    try:
        args = args_parser()
        config = config_summoner(args)
    finally:
        sys.argv = argv_backup
    dataset = load_dataset(config.dataset.dataset_name, config)
    return dataset, config


# --------------------------------------------------------------------------
# colour refinement
# --------------------------------------------------------------------------
def attr_key(x_row, decimals):
    a = x_row.astype(np.float64)
    if decimals is not None:
        a = np.round(a, decimals)
    return a.tobytes()


def initial_colours(graphs, decimals):
    """Global (cross-graph) initial colouring by node attributes."""
    table = {}
    init = []
    for g in graphs:
        x = g["x"]
        cols = np.empty(x.shape[0], dtype=np.int64)
        for v in range(x.shape[0]):
            k = attr_key(x[v], decimals)
            if k not in table:
                table[k] = len(table)
            cols[v] = table[k]
        init.append(cols)
    return init, table


def refine(graphs, init_cols, rounds, enrich):
    """One shared colour table across all graphs (cross-graph comparability)."""
    cols = [c.copy() for c in init_cols]
    for _ in range(rounds):
        table = {}
        new_cols = []
        for g, c in zip(graphs, cols):
            n = len(c)
            # in-neighbours: message flows src -> dst (PyG propagate convention)
            nb = [[] for _ in range(n)]
            ei = g["edge_index"]
            for e in range(ei.shape[1]):
                nb[ei[1, e]].append(c[ei[0, e]])
            if enrich == "acr":
                glob = tuple(sorted(Counter(c.tolist()).items()))
            else:
                glob = ()
            nc = np.empty(n, dtype=np.int64)
            for v in range(n):
                key = (int(c[v]), tuple(sorted(nb[v])), glob)
                if key not in table:
                    table[key] = len(table)
                nc[v] = table[key]
            new_cols.append(nc)
        cols = new_cols
    return cols


def refines(fine, coarse):
    """Check: every 'fine' class sits inside one 'coarse' class."""
    mapping = {}
    for f, cvec in zip(fine, coarse):
        for fc, cc in zip(f.tolist(), cvec.tolist()):
            if fc in mapping and mapping[fc] != cc:
                return False
            mapping[fc] = cc
    return True


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_path", required=True)
    ap.add_argument("--rounds", type=int, default=None,
                    help="default: config.model.model_layer")
    ap.add_argument("--decimals", type=int, default=None,
                    help="rounding for continuous features (None = exact)")
    ap.add_argument("--splits", default="train/id_val/id_test",
                    help="'/'-separated; missing splits are skipped")
    ap.add_argument("--out", default="storage/wl_colors")
    args = ap.parse_args()

    dataset, config = load_repo_dataset(args.config_path)
    rounds = args.rounds if args.rounds is not None else int(config.model.model_layer)
    name = config.dataset.dataset_name
    print(f"#IN# dataset={name}  rounds(L)={rounds}  decimals={args.decimals}")

    # ---- flatten requested splits to plain numpy graphs -------------------
    graphs, split_of, y_all = [], [], []
    used_splits = []
    for sp in args.splits.split("/"):
        if not isinstance(dataset, dict) or sp not in dataset or dataset[sp] is None:
            continue
        used_splits.append(sp)
        for data in dataset[sp]:
            graphs.append({
                "x": data.x.detach().cpu().numpy(),
                "edge_index": data.edge_index.detach().cpu().numpy(),
            })
            split_of.append(sp)
            y = data.y
            y_all.append(int(y.view(-1)[0].item()))
    print(f"#IN# splits used: {used_splits}  |graphs|={len(graphs)}")
    assert graphs, "no graphs loaded — check --splits against this dataset"

    init, attr_table = initial_colours(graphs, args.decimals)
    n_attr = len(attr_table)
    n_nodes = sum(len(c) for c in init)
    uniq_frac = n_attr / max(n_nodes, 1)
    print(f"#IN# initial attribute classes: {n_attr}  ({100*uniq_frac:.2f}% of nodes)")
    if uniq_frac > 0.5:
        print("#W#  near-unique initial colours -> exact-tie analysis is "
              "vacuous on this dataset; use the metric version (design E1-2).")

    out = {"meta": {"dataset": name, "rounds": rounds,
                    "decimals": args.decimals, "splits": used_splits},
           "y": np.array(y_all), "split_of": split_of}

    vanilla = refine(graphs, init, rounds, "none")
    acr = refine(graphs, init, rounds, "acr")
    assert refines(acr, vanilla), "acr colours must refine vanilla — bug"
    print("#IN# sanity: acr refines vanilla  [OK]")

    for tag, cols in (("vanilla", vanilla), ("acr", acr)):
        all_final = np.concatenate(cols)
        all_init = np.concatenate(init)
        n_col = len(np.unique(all_final))

        # iota(a): how many final colours per initial attribute class
        iota = defaultdict(set)
        for fc, ic in zip(all_final.tolist(), all_init.tolist()):
            iota[ic].add(fc)
        iota_counts = {a: len(s) for a, s in iota.items()}
        iota_max = max(iota_counts.values())
        n_split_attrs = sum(1 for v in iota_counts.values() if v >= 2)

        # multiplicity of colour classes within graphs
        frac_tie_graphs = np.mean([
            1.0 if Counter(c.tolist()).most_common(1)[0][1] >= 2 else 0.0
            for c in cols])
        max_mult = max(max(Counter(c.tolist()).values()) for c in cols)

        print(f"#R# [{tag:7s}] |X_L|={n_col:6d}  max_iota={iota_max:4d}  "
              f"attrs_split={n_split_attrs}/{n_attr}  "
              f"graphs_with_tie_pair={100*frac_tie_graphs:5.1f}%  "
              f"max_class_mult={max_mult}")
        out[tag] = {"colors": cols, "n_colors": n_col,
                    "iota": iota_counts, "iota_max": iota_max,
                    "frac_tie_graphs": frac_tie_graphs}

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(
        args.out, f"{name}_L{rounds}_d{args.decimals}.pt")
    torch.save(out, path)
    print(f"#IN# saved -> {path}")

    # E1-a markdown row
    a = out["acr"]
    print("\n#R# E1-a row (acr-enriched):")
    print(f"| {name} | L={rounds} | |X_L|={a['n_colors']} "
          f"| max iota={a['iota_max']} "
          f"| tie-graphs={100*a['frac_tie_graphs']:.1f}% |")


if __name__ == "__main__":
    main()
