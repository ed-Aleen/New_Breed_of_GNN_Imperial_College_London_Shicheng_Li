r"""OCC Stage A (EVAL-X / REAL-X Eq. 5 transplanted to SE-GNNs): calibrate the
classifier on selection-INDEPENDENT random class-level displays, then freeze.

Trains ONLY the classifier-side parameters (gnn_clf + classifierS when the
cell has a separate classifier encoder; gnn + classifierS for 0clf cells such
as RBGV-GSAT) with cross-entropy on displays D ~ mu0 = Bern(r0) over realised
broadcast-enriched colour classes (discrete cells) or nodes (continuous
cells, honest fallback -- printed). Display channel = the repo's masked
forward with `fix_mask_aggregation: true` (config assert); its equivalence to
true subgraph re-encoding is certified by occ_channel_check.py per cell.

Saves the best-calibration state to <ckpt_dir>/occ_g0.ckpt. ALWAYS pass
--save_tag occ so nothing collides with baseline checkpoints.

Usage (from the clean repo root; script-dir import wins over the venv):
  python occ_stage_a.py --config_path final_configs/MUTAG/basis/no_shift/GSAT_OCC.yaml \
      --seeds 1/2/3/4/5 --task train --backbone ACR2 --gpu_idx 0 --save_tag occ
Extras (stripped before the repo parser):
  --occ_r0 0.5   --occ_epochs -1 (-1 = config train.max_epoch)
  --occ_granularity auto|class|node   --occ_rounds -1 (-1 = model layers)
"""
import json
import os
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


R0 = float(pop_flag(sys.argv, "--occ_r0", "0.5"))
EPOCHS = int(pop_flag(sys.argv, "--occ_epochs", "-1"))
GRAN = pop_flag(sys.argv, "--occ_granularity", "auto")
ROUNDS = int(pop_flag(sys.argv, "--occ_rounds", "-1"))

import os as _os, sys as _sys                                  # noqa: E402
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from GOOD import config_summoner                            # noqa: E402
from GOOD.utils.args import args_parser                     # noqa: E402
from GOOD.utils.loader import initialize_model_dataset      # noqa: E402
from GOOD.ood_algorithms.ood_manager import load_ood_alg    # noqa: E402
from GOOD.utils.logger import load_logger                   # noqa: E402
from GOOD.utils.occ import (node_class_keys, detect_granularity,  # noqa: E402
                            sample_class_mask)
from GOOD.utils.masked_forward import masked_logits, nll_from_logits   # noqa: E402


def classifier_prefixes(model):
    return ["gnn_clf", "classifierS"] if getattr(model, "gnn_clf", None) \
        else ["gnn", "classifierS"]


def named_modules_for(model, prefixes):
    mods = []
    for pref in prefixes:
        m = model
        for part in pref.split("."):
            m = getattr(m, part)
        mods.append(m)
    return mods


def calib_nll(model, ood_algorithm, loader, gran, rounds, dev):
    """Holdout calibration NLL under fixed-seed mu0 displays."""
    gen = torch.Generator().manual_seed(0)
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for data in loader:
            data = data.to(dev)
            keys = node_class_keys(data, gran, rounds)
            att = sample_class_mask(keys, data.batch, R0, gen).view(-1, 1)
            logits = masked_logits(model, ood_algorithm, data, att)
            ys = data.y.view(-1)
            tot += nll_from_logits(logits, ys).item() * len(ys)
            n += len(ys)
    return tot / max(n, 1)


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
        assert bool(getattr(config, "fix_mask_aggregation", False)), \
            "OCC requires fix_mask_aggregation: true (use the *_OCC.yaml)"
        assert config.save_tag, "pass --save_tag occ (checkpoint isolation)"
        torch.manual_seed(seed)
        np.random.seed(seed)

        model, loader = initialize_model_dataset(config)
        ood_algorithm = load_ood_alg(config.ood.ood_alg, config)
        dev = config.device
        model.to(dev)

        prefixes = classifier_prefixes(model)
        params = [p for n, p in model.named_parameters()
                  if any(n.startswith(pref + ".") for pref in prefixes)]
        assert params, f"no classifier-side params under {prefixes}"
        opt = torch.optim.Adam(params, lr=config.train.lr,
                               weight_decay=config.train.weight_decay)

        first = next(iter(loader["train"]))
        gran = GRAN if GRAN != "auto" else detect_granularity(first.x)
        rounds = ROUNDS if ROUNDS >= 0 else int(config.model.model_layer)
        epochs = EPOCHS if EPOCHS > 0 else int(config.train.max_epoch)
        name, mdl = config.dataset.dataset_name, config.model.model_name
        print(f"#IN# OCC-A {name}-{mdl} seed {seed}: granularity={gran} "
              f"(auto-detect={detect_granularity(first.x)}), r0={R0}, "
              f"rounds={rounds}, epochs={epochs}, frozen_prefixes={prefixes}")

        train_mods = named_modules_for(model, prefixes)
        gen = torch.Generator().manual_seed(seed * 7919 + 13)
        best = (float("inf"), None, -1)
        nll_curve = []
        for ep in range(epochs):
            model.eval()
            for m in train_mods:
                m.train()
            for data in loader["train"]:
                data = data.to(dev)
                keys = node_class_keys(data, gran, rounds)
                att = sample_class_mask(keys, data.batch, R0, gen).view(-1, 1)
                logits = masked_logits(model, ood_algorithm, data, att)
                loss = nll_from_logits(logits, data.y.view(-1))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            nll = calib_nll(model, ood_algorithm, loader["id_val"],
                            gran, rounds, dev)
            nll_curve.append(nll)
            if nll < best[0]:
                sd = {k: v.detach().cpu().clone()
                      for k, v in model.state_dict().items()
                      if any(k.startswith(p + ".") for p in prefixes)}
                best = (nll, sd, ep)
            if ep % 10 == 0 or ep == epochs - 1:
                print(f"#IN#   epoch {ep}: id_val calib NLL={nll:.4f} "
                      f"(best {best[0]:.4f} @ {best[2]})")

        # majority-class NLL reference (calibration floor sanity)
        ys = torch.cat([d.y.view(-1) for d in loader["train"]]).long()
        pv = torch.bincount(ys).float()
        pv = pv[pv > 0] / pv.sum()
        h_y = float(-(pv * pv.log()).sum())  # label-blind floor
        pmaj = float(pv.max())
        out = os.path.join(config.ckpt_dir, "occ_g0.ckpt")
        os.makedirs(config.ckpt_dir, exist_ok=True)
        torch.save({"state_dict": best[1],
                    "meta": {"frozen_prefixes": prefixes, "r0": R0,
                             "granularity": gran, "rounds": rounds,
                             "best_nll": best[0], "best_epoch": best[2],
                             "nll_curve": nll_curve,
                             "majority_nll": float(-np.log(pmaj)),
                             "marginal_entropy": h_y,
                             "dataset": name, "model": mdl, "seed": seed}},
                   out)
        print(f"#R# | OCC-A | {name}-{mdl} | seed {seed} | gran={gran} r0={R0} "
              f"| calib NLL best={best[0]:.4f} @ep{best[2]} "
              f"| label-blind floor H(Y)={h_y:.4f} | saved {out}")
        with open(os.path.join(config.ckpt_dir, "occ_g0_meta.json"), "w") as f:
            json.dump({"best_nll": best[0], "best_epoch": best[2],
                       "nll_curve": nll_curve, "granularity": gran,
                       "r0": R0}, f)
    print(f"\n#IN# done in {datetime.now() - t0}")


if __name__ == "__main__":
    main()
