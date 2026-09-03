r"""OCC Stage-A calibration sanity (E4 clause 2, the deferred lookup check).

Builds the evidence-bit LOOKUP predictor from the pre-registered b0 table
(storage/occ_prereg/<dataset>_prereg.pt, radius 1): under the SAME mu0
displays as occ_stage_a.py's calibration eval, a display's features are
  e1 = exists an INTACT evidence ball voting label 1
  e0 = exists an INTACT evidence ball voting label 0
  d  = clipped count difference n1 - n0
(a ball of radius 1 is intact iff the node and all its neighbours are
displayed). Lookup tables over (e0,e1) [4 cells] and d [7 cells] are fit on
train displays (Laplace +1) and scored on id_val displays.

Verdict guide: g0's calib NLL <= lookup NLL  -> g0 is at/above the
registered-granularity reference: PASS. g0 >> lookup -> undertrained.
Also reports the mu0-conditional evidence availability (how often the true
label's evidence survives Bern(r0) class masking) -- the bridge between the
existence ceiling (evidence in G) and Stage A (evidence in D).

Usage:
  python occ_calib_check.py --config_path final_configs/MUTAG/basis/no_shift/GSAT_OCC.yaml \
      --seeds 1 --task test --backbone ACR2 --gpu_idx 0
Extras: --occ_r0 0.5 --train_draws 4
"""
import math
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime

import torch


def pop_flag(argv, name, default):
    if name in argv:
        i = argv.index(name)
        v = argv[i + 1]
        del argv[i:i + 2]
        return v
    return default


R0 = float(pop_flag(sys.argv, "--occ_r0", "0.5"))
DRAWS = int(pop_flag(sys.argv, "--train_draws", "4"))
PREREG = pop_flag(sys.argv, "--prereg", "storage/occ_prereg")

from GOOD import config_summoner                            # noqa: E402
from GOOD.utils.args import args_parser                     # noqa: E402
from GOOD.utils.loader import initialize_model_dataset      # noqa: E402
from GOOD.utils.logger import load_logger                   # noqa: E402
from GOOD.utils.occ import (node_class_keys, initial_colour_keys,  # noqa: E402
                            refine_colour_keys, sample_class_mask)


def display_features(data, disp, votes_local):
    """Per-graph (e0, e1, n1-n0) from intact radius-1 evidence balls."""
    src, dst = data.edge_index[0], data.edge_index[1]
    viol = torch.zeros(data.x.shape[0], dtype=torch.long)
    viol.index_add_(0, dst, (~disp[src]).long())
    intact = disp & (viol == 0)
    nb = int(data.batch.max()) + 1
    n1 = torch.zeros(nb, dtype=torch.long)
    n0 = torch.zeros(nb, dtype=torch.long)
    v1 = intact & (votes_local == 1)
    v0 = intact & (votes_local == 0)
    n1.index_add_(0, data.batch, v1.long())
    n0.index_add_(0, data.batch, v0.long())
    return n0, n1


def cells(n0, n1):
    e = list(zip((n0 > 0).tolist(), (n1 > 0).tolist()))
    d = torch.clamp(n1 - n0, -3, 3).tolist()
    return e, d


def main():
    args = args_parser()
    t0 = datetime.now()
    args.random_seed = int(args.seeds.split("/")[0])
    args.exp_round = args.random_seed
    config = config_summoner(args)
    load_logger(config)
    _, loader = initialize_model_dataset(config)
    name = config.dataset.dataset_name
    rounds = int(config.model.model_layer)

    pre = torch.load(os.path.join(PREREG, f"{name}_prereg.pt"),
                     weights_only=False)
    b0 = pre["radii"][1]["b0"]
    b0_single = {c[0]: y for c, y in b0.items() if len(c) == 1}
    print(f"#IN# {name}: b0 radius-1 single-content votes: {len(b0_single)}")

    def votes_for(data):
        batch = data.batch
        local = refine_colour_keys(initial_colour_keys(data.x),
                                   data.edge_index, batch,
                                   rounds=1, enrich=False)
        return torch.tensor([b0_single.get(int(k), -1)
                             for k in local], dtype=torch.long)

    # fit lookup on train displays
    tabE = defaultdict(Counter)
    tabD = defaultdict(Counter)
    ycnt = Counter()
    gen = torch.Generator().manual_seed(1)
    for _ in range(DRAWS):
        for data in loader["train"]:
            keys = node_class_keys(data, "class", rounds)
            disp = sample_class_mask(keys, data.batch, R0, gen) > 0.5
            n0, n1 = display_features(data, disp, votes_for(data))
            e, d = cells(n0, n1)
            for gi, y in enumerate(data.y.view(-1).tolist()):
                y = int(y)
                tabE[e[gi]][y] += 1
                tabD[d[gi]][y] += 1
                ycnt[y] += 1

    labels = sorted(ycnt)
    py = {y: ycnt[y] / sum(ycnt.values()) for y in labels}
    hy = -sum(p * math.log(p) for p in py.values())

    def nll_of(tab, cell, y):
        cc = tab[cell]
        tot = sum(cc.values()) + len(labels)
        return -math.log((cc[y] + 1) / tot)

    # score on id_val displays (same generator protocol as calib_nll: seed 0)
    gen = torch.Generator().manual_seed(0)
    sE = sD = 0.0
    n = 0
    avail = 0
    for data in loader["id_val"]:
        keys = node_class_keys(data, "class", rounds)
        disp = sample_class_mask(keys, data.batch, R0, gen) > 0.5
        n0, n1 = display_features(data, disp, votes_for(data))
        e, d = cells(n0, n1)
        for gi, y in enumerate(data.y.view(-1).tolist()):
            y = int(y)
            sE += nll_of(tabE, e[gi], y)
            sD += nll_of(tabD, d[gi], y)
            avail += int((n1[gi] > 0) if y == 1 else (n0[gi] > 0))
            n += 1
    print(f"#R# | calib-ref | {name} | r0={R0} | H(Y)={hy:.4f} "
          f"| lookup NLL: evidence-bits={sE/n:.4f} count-diff={sD/n:.4f} "
          f"| mu0 evidence availability (true-label intact ball shown)="
          f"{avail/n:.4f} |")
    print("#IN# verdict: PASS Stage A if g0 calib NLL <= the lookup NLL "
          "(g0 sees strictly more than these features).")
    print(f"#IN# done in {datetime.now() - t0}")


if __name__ == "__main__":
    main()
