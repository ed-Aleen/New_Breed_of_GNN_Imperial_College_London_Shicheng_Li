r"""OCC channel-equivalence certificate (zero training).

Certifies, per cell, that the display channel used everywhere in OCC
(masked forward with fix_mask_aggregation: true, hard {0,1} class masks)
equals TRUE subgraph re-encoding (node deletion + Data rebuild), and
quantifies the aggregation leak of the shipped channel (fix off) that
E4 clause 1 warns about.

Predictions: sum-pool cells (MUTAG/RBGV/MNIST) max|dp| == 0 with fix on and
> 0 with fix off; SST2P (weighted MEAN pool divides by total N, not sum of
kept weights) keeps a documented denominator deviation even with fix on.

Usage:
  python occ_channel_check.py --config_path final_configs/MUTAG/basis/no_shift/GSAT_OCC.yaml \
      --seeds 1 --task test --backbone ACR2 --gpu_idx 0
Extras: --max_graphs 200
"""
import sys
from datetime import datetime

import numpy as np
import torch


def pop_flag(argv, name, default):
    if name in argv:
        i = argv.index(name)
        v = argv[i + 1]
        del argv[i:i + 2]
        return v
    return default


MAX_GRAPHS = int(pop_flag(sys.argv, "--max_graphs", "200"))

import os as _os, sys as _sys                                  # noqa: E402
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from GOOD import config_summoner                            # noqa: E402
from GOOD.utils.args import args_parser                     # noqa: E402
from GOOD.utils.loader import initialize_model_dataset      # noqa: E402
from GOOD.ood_algorithms.ood_manager import load_ood_alg    # noqa: E402
from GOOD.utils.logger import load_logger                   # noqa: E402
from GOOD.utils.occ import (node_class_keys, detect_granularity,  # noqa: E402
                            sample_class_mask, induced_subbatch)
from GOOD.utils.masked_forward import masked_logits, probs_from_logits  # noqa: E402


def main():
    args = args_parser()
    t0 = datetime.now()
    for i, seed in enumerate(args.seeds.split("/")):
        seed = int(seed)
        args.random_seed = seed
        args.exp_round = seed
        config = config_summoner(args)
        if i == 0:
            load_logger(config)
        torch.manual_seed(seed)
        model, loader = initialize_model_dataset(config)
        ood_algorithm = load_ood_alg(config.ood.ood_alg, config)
        dev = config.device
        model.to(dev)
        model.eval()
        name, mdl = config.dataset.dataset_name, config.model.model_name
        gran = detect_granularity(next(iter(loader["id_val"])).x)
        rounds = int(config.model.model_layer)
        gen = torch.Generator().manual_seed(0)

        d_on, d_off = 0.0, 0.0
        n = 0
        with torch.no_grad():
            for data in loader["id_val"]:
                if n >= MAX_GRAPHS:
                    break
                data = data.to(dev)
                keys = node_class_keys(data, gran, rounds)
                att = sample_class_mask(keys, data.batch, 0.5, gen).view(-1, 1)
                sub = induced_subbatch(data, att.view(-1) > 0.5)
                ones = torch.ones(sub.x.shape[0], 1, device=dev)

                model.config.fix_mask_aggregation = True
                p_mask = probs_from_logits(
                    masked_logits(model, ood_algorithm, data, att))
                p_sub = probs_from_logits(
                    masked_logits(model, ood_algorithm, sub, ones))
                d_on = max(d_on, float((p_mask - p_sub).abs().max()))

                model.config.fix_mask_aggregation = False
                p_leak = probs_from_logits(
                    masked_logits(model, ood_algorithm, data, att))
                d_off = max(d_off, float((p_leak - p_sub).abs().max()))
                model.config.fix_mask_aggregation = True
                n += int(data.y.view(-1).shape[0])
        print(f"#R# | channel | {name}-{mdl} | seed {seed} | graphs={n} "
              f"| max|dp| fix-ON vs true-subgraph = {d_on:.3e} "
              f"| fix-OFF (shipped leak) = {d_off:.3e} |")
    print(f"\n#IN# done in {datetime.now() - t0}")


if __name__ == "__main__":
    main()
