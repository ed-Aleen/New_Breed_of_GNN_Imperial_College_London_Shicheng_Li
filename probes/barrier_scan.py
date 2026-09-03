"""
E3.3 of the experiment pipeline (EXPERIMENT_DESIGN.md): the WALL — the
empirical analogue of A2R Fig. 2(a), instantiated as SWITCH_BARRIER.md's
switch path.

For a trained checkpoint, sweeps the display level t of an alternative
attribute type beta from 0 to 1 (protocol display kept ON, undisplayed
beta-attr nodes get attention weight t) and records the fit objective:

  curve F (theta frozen)   : J_fit(t) with the trained classifier
                             -> the unfamiliarity wall (U_beta's integral)
  curve R (theta retrained): J_fit(t) after re-fitting ONLY the classifier
                             head at each t (Adam, from the trained state)

Honesty: the retrained inner player is RESTRICTED to the head, so curve R
upper-bounds the true best-response landscape J*(t); hence
F(t) - R(t) LOWER-bounds the specialization gain Sp at t.
NLL is computed from probabilities with a 1e-9 clamp — fully saturated
checkpoints (p pinned at 1.0 in float32) flatten both curves; prefer cells
with nonzero eval loss (RBGV-SMGNN, RBGV-DIR seed 3, MUTAG-SMGNN).

Usage (single seed per run recommended):
  python barrier_scan.py --config_path final_configs/BAColorGVIsol/basis/no_shift/SMGNN_sec6.yaml \
      --seeds 1 --task test --ratios 0.5 --splits train --backbone ACR2 --gpu_idx 0
Extras (stripped before the repo parser): --tpoints 11 --max_graphs 400
  --retrain_steps 200 --retrain_lr 1e-3 --retrain_attr -1 (auto = max wall)
  --out storage/barrier
"""

import os as _os, sys as _sys                                  # noqa: E402
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import copy
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


TPOINTS = int(pop_flag(sys.argv, "--tpoints", "11"))
MAX_GRAPHS = int(pop_flag(sys.argv, "--max_graphs", "400"))
RETRAIN_STEPS = int(pop_flag(sys.argv, "--retrain_steps", "200"))
RETRAIN_LR = float(pop_flag(sys.argv, "--retrain_lr", "1e-3"))
RETRAIN_ATTR = int(pop_flag(sys.argv, "--retrain_attr", "-1"))
OUT = pop_flag(sys.argv, "--out", "storage/barrier")

from GOOD import config_summoner                        # noqa: E402
from GOOD.utils.args import args_parser                 # noqa: E402
from GOOD.utils.loader import initialize_model_dataset  # noqa: E402
from GOOD.ood_algorithms.ood_manager import load_ood_alg   # noqa: E402
from GOOD.kernel.pipeline_manager import load_pipeline     # noqa: E402
from GOOD.utils.logger import load_logger               # noqa: E402
from torch_geometric.data import Batch                  # noqa: E402

from GOOD.utils.masked_forward import masked_logits, nll_from_logits  # noqa: E402


def build_att(items, t, beta, dev):
    atts = []
    for it in items:
        a = it["mask"].astype(np.float32).copy()
        if beta is not None:
            idx = it["cand"].get(beta)
            if idx is not None and len(idx):
                a[idx] = t
        atts.append(a)
    return torch.tensor(np.concatenate(atts), device=dev).view(-1, 1)


def eval_curve(model, ood_algorithm, batches, beta, ts, dev):
    vals = []
    with torch.no_grad():
        for t in ts:
            tot, n = 0.0, 0
            for graphs_b, batch, ys, items in batches:
                att = build_att(items, t, beta, dev)
                logits = masked_logits(model, ood_algorithm, batch, att)
                tot += nll_from_logits(logits, ys).item() * len(ys)
                n += len(ys)
            vals.append(tot / max(n, 1))
    return vals


def main():
    args = args_parser()
    splits = args.splits.split("/") if args.splits else ["train"]
    thrs = [float(r) for r in args.ratios.split("/")] if args.ratios else [0.5]
    split, thr = splits[0], thrs[0]
    os.makedirs(OUT, exist_ok=True)
    ts = np.linspace(0.0, 1.0, TPOINTS).tolist()
    t0 = datetime.now()

    for i, seed in enumerate(args.seeds.split("/")):
        seed = int(seed)
        args.random_seed = seed
        args.exp_round = seed
        config = config_summoner(args)
        config["task"] = "test"
        config["load_split"] = ""
        if i == 0:
            load_logger(config)
        model, loader = initialize_model_dataset(config)
        ood_algorithm = load_ood_alg(config.ood.ood_alg, config)
        pipeline = load_pipeline(config.pipeline, config.task, model, loader,
                                 ood_algorithm, config)
        pipeline.load_task(load_param=True, load_split="id")
        model.eval()
        name, mdl = config.dataset.dataset_name, config.model.model_name
        dev = config.device

        samples, _, _ = pipeline.generate_binary_explanations(
            is_weight=True, thrs=[thr], splits=[split], convert_to_nx=False,
            is_node_expl=not config.ood.extra_param[0])
        graphs = samples[split][thr][:MAX_GRAPHS]

        # precompute per-graph mask + per-attr undisplayed candidates
        attrs_seen = set()
        items, keep = [], []
        for g in graphs:
            mask = g.node_mask.cpu().numpy().astype(bool)
            if mask.sum() == 0:
                continue
            x = g.x.cpu().numpy()
            av = x.argmax(1) if x.ndim > 1 else x
            cand = {}
            for a in sorted(set(av.tolist())):
                cand[a] = np.where((av == a) & (~mask))[0]
                if len(cand[a]):
                    attrs_seen.add(a)
            items.append({"mask": mask, "cand": cand,
                          "y": int(g.y.view(-1)[0])})
            keep.append(g)
        attrs_seen = sorted(attrs_seen)
        print(f"#IN# {name}-{mdl} seed {seed}: {len(keep)} graphs, "
              f"alternative attrs: {attrs_seen}")

        # fixed mini-batches (att rebuilt per t)
        BS = 64
        batches = []
        for s in range(0, len(keep), BS):
            gs = keep[s:s + BS]
            its = items[s:s + BS]
            batch = Batch.from_data_list(gs).to(dev)
            ys = torch.tensor([it["y"] for it in its], device=dev)
            batches.append((gs, batch, ys, its))

        # ---- frozen-theta curves for every attr ------------------------
        frozen = {}
        for a in attrs_seen:
            frozen[a] = eval_curve(model, ood_algorithm, batches, a, ts, dev)
            wall = max(frozen[a]) - frozen[a][0]
            print(f"#R#   frozen beta=attr{a}: J(0)={frozen[a][0]:.4f} "
                  f"max={max(frozen[a]):.4f} wall={wall:+.4f} "
                  f"J(1)={frozen[a][-1]:.4f}")

        # ---- retrained-head curve for the chosen attr ------------------
        beta = RETRAIN_ATTR if RETRAIN_ATTR >= 0 else max(
            attrs_seen, key=lambda a: max(frozen[a]) - frozen[a][0])
        head = [(n, p) for n, p in model.named_parameters()
                if "classifier" in n.lower()]
        assert head, "no classifier-head parameters found"
        theta0 = {n: p.detach().clone() for n, p in head}
        retrained = []
        for t in ts:
            with torch.no_grad():  # reset head to trained state
                for n, p in head:
                    p.copy_(theta0[n])
            opt = torch.optim.Adam([p for _, p in head], lr=RETRAIN_LR)
            for step in range(RETRAIN_STEPS):
                _, batch, ys, its = batches[step % len(batches)]
                att = build_att(its, t, beta, dev)
                logits = masked_logits(model, ood_algorithm, batch, att)
                loss = nll_from_logits(logits, ys)
                opt.zero_grad()
                loss.backward()
                opt.step()
            retrained.append(eval_curve(model, ood_algorithm, batches,
                                        beta, [t], dev)[0])
        with torch.no_grad():
            for n, p in head:
                p.copy_(theta0[n])

        f = frozen[beta]
        sp = [fv - rv for fv, rv in zip(f, retrained)]
        print(f"#R# retrained beta=attr{beta}: "
              f"wall_frozen={max(f)-f[0]:+.4f} "
              f"wall_retrained={max(retrained)-retrained[0]:+.4f} "
              f"max Sp-lower-bound={max(sp):+.4f} at t="
              f"{ts[int(np.argmax(sp))]:.1f}")

        torch.save({"meta": {"dataset": name, "model": mdl, "seed": seed,
                             "split": split, "thr": thr, "beta": beta,
                             "retrain_steps": RETRAIN_STEPS},
                    "ts": ts, "frozen": frozen, "retrained": retrained,
                    "sp_lower": sp},
                   os.path.join(OUT, f"{name}_{mdl}_seed{seed}_{split}.pt"))

    print(f"\n#IN# done in {datetime.now() - t0}")


if __name__ == "__main__":
    main()
