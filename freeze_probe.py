"""
E3.2 of the experiment pipeline (EXPERIMENT_DESIGN.md): the FREEZING leg of
interlocking — "the classifier only trains on what the extractor selects".

For each graph, takes the protocol display mask (D.5 path, same as the paper
metrics), puts the CLASSIFIER-PASS loss gradient on the node input features,
and compares per-node gradient norms between selected and unselected nodes.

Predictions (chain-A B1/B3, implementation-level form):
  * SMGNN ({0,1} weights): unselected nodes receive EXACTLY ZERO gradient
    (frac_zero -> 100%) — the backbone is frozen on unselected types;
  * GSAT ({r,1} weights): unselected nodes receive attenuated-but-nonzero
    gradient; attenuation increases monotonically as the score drops
    (per-layer gating makes it stronger than the pooling-position s_v ratio —
    reported as-is, two-pass caveat applies);
  * DIR is excluded (no shared backbone — separate CausalAttNet).

Usage (same flags as goodtg + extras stripped before the repo parser):
  python freeze_probe.py --config_path final_configs/MUTAG/basis/no_shift/SMGNN_sec6.yaml \
      --seeds 1/3/4/5 --task test --ratios 0.5 --splits id_val --backbone ACR2 --gpu_idx 0
Extras: --max_graphs 300  --out storage/freeze  --att_mode binary|raw
  binary (default) = protocol display mask {0,1}; raw = the model's own score
  channel as the gate (the training-time form; for SMGNN scores are ~{0,1} so
  raw == binary, for GSAT raw is soft with a nonzero floor -> frac_zero ~ 0).

Halo law (B3', corrected registration): set_masks gates only the per-layer
V(x*s) term and the weighted pooling; message() never multiplies the mask
(PyG 2.6 path, finding_mask_does_not_cut_aggregation). Hence the input-feature
gradient of a zero-gated node is EXACTLY zero iff no positive-gated node lies
within L hops (L = #conv layers). The halo check below verifies this
prediction node-by-node.
"""
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


MAX_GRAPHS = int(pop_flag(sys.argv, "--max_graphs", "300"))
OUT = pop_flag(sys.argv, "--out", "storage/freeze")
ATT_MODE = pop_flag(sys.argv, "--att_mode", "binary")
assert ATT_MODE in ("binary", "raw"), ATT_MODE


def l_hop_reach(edge_index, n, seeds, L):
    """Boolean mask: nodes within L hops of any seed node (seeds included).
    edge_index is the PyG [2, E] array (undirected graphs carry both
    directions, so a plain forward BFS is correct)."""
    from collections import deque
    adj = [[] for _ in range(n)]
    for a, b in zip(edge_index[0].tolist(), edge_index[1].tolist()):
        adj[a].append(b)
    dist = np.full(n, -1, dtype=np.int64)
    q = deque()
    for v in np.where(seeds)[0]:
        dist[int(v)] = 0
        q.append(int(v))
    while q:
        u = q.popleft()
        if dist[u] >= L:
            continue
        for w in adj[u]:
            if dist[w] < 0:
                dist[w] = dist[u] + 1
                q.append(w)
    return dist >= 0

from GOOD import config_summoner                        # noqa: E402
from GOOD.utils.args import args_parser                 # noqa: E402
from GOOD.utils.loader import initialize_model_dataset  # noqa: E402
from GOOD.ood_algorithms.ood_manager import load_ood_alg   # noqa: E402
from GOOD.kernel.pipeline_manager import load_pipeline     # noqa: E402
from GOOD.utils.logger import load_logger               # noqa: E402
from torch_geometric.data import Batch                  # noqa: E402

from masked_forward import masked_logits, nll_from_logits, sanity_check  # noqa: E402


def main():
    args = args_parser()
    splits = args.splits.split("/") if args.splits else ["id_val"]
    thrs = [float(r) for r in args.ratios.split("/")] if args.ratios else [0.5]
    split, thr = splits[0], thrs[0]
    os.makedirs(OUT, exist_ok=True)
    t0 = datetime.now()
    rows = []

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
        assert mdl != "DIR", "DIR excluded from the freezing leg (no shared backbone)"

        samples, _, _ = pipeline.generate_binary_explanations(
            is_weight=True, thrs=[thr], splits=[split], convert_to_nx=False,
            is_node_expl=not config.ood.extra_param[0])
        graphs = samples[split][thr]

        L = int(config.model.model_layer)
        g_sel, g_unsel, scores_all, grads_all = [], [], [], []
        n_zero_unsel, n_unsel_nodes = 0, 0
        n_outh = n_outh_zero = n_inh = n_inh_zero = 0
        n_used = n_dead = 0
        dead_losses, live_losses = [], []
        col_stats = {}
        checked = False
        for g in graphs[:MAX_GRAPHS]:
            mask = g.node_mask.cpu().numpy().astype(bool)
            if mask.sum() == 0 or mask.all():
                continue  # need both populations in the graph
            y = int(g.y.view(-1)[0])
            batch = Batch.from_data_list([g]).to(dev)
            batch.x = batch.x.detach().clone().requires_grad_(True)
            if ATT_MODE == "raw":
                att_np = g.node_expl.detach().cpu().numpy().astype(
                    np.float32).reshape(-1)
            else:
                att_np = mask.astype(np.float32)
            att = torch.tensor(att_np, device=dev).view(-1, 1)
            if not checked:  # helper reproduces the repo's predict path
                err = sanity_check(model, ood_algorithm, batch, att)
                print(f"#IN# masked_logits sanity vs predict_from_subgraph: "
                      f"max |dp| = {err:.2e}")
                checked = True
            logits = masked_logits(model, ood_algorithm, batch, att)
            loss = nll_from_logits(
                logits, torch.tensor([y], device=dev))
            model.zero_grad(set_to_none=True)
            loss.backward()
            if batch.x.grad is None:
                continue
            gn = batch.x.grad.norm(dim=1).detach().cpu().numpy()
            n_used += 1
            g_sel.extend(gn[mask].tolist())
            g_unsel.extend(gn[~mask].tolist())
            n_zero_unsel += int((gn[~mask] == 0.0).sum())
            n_unsel_nodes += int((~mask).sum())
            # graph-level loss saturation: p pinned at 1.0 in float32 kills
            # the WHOLE graph's gradient (selected included) -- flag those and
            # keep the halo-law stats on LIVE graphs only
            dead = bool((gn == 0.0).all())
            lv = float(loss.item())
            (dead_losses if dead else live_losses).append(lv)
            if dead:
                n_dead += 1
                continue
            # halo-law check (B3'): predicted-zero = zero-gated nodes with no
            # positive-gated node within L hops
            in_halo = l_hop_reach(g.edge_index.cpu().numpy(), len(att_np),
                                  att_np > 0, L)
            zero = gn == 0.0
            out_h = ~in_halo & ~mask
            in_h = in_halo & ~mask
            n_outh += int(out_h.sum())
            n_outh_zero += int((zero & out_h).sum())
            n_inh += int(in_h.sum())
            n_inh_zero += int((zero & in_h).sum())
            # residual in-halo zeros broken down by node colour (one-hot argmax)
            av = g.x.cpu().numpy().argmax(1)
            for c in np.unique(av[in_h]).tolist():
                st = col_stats.setdefault(int(c), [0, 0])
                cc = in_h & (av == c)
                st[0] += int((zero & cc).sum())
                st[1] += int(cc.sum())
            scores_all.extend(g.node_expl.cpu().numpy().tolist())
            grads_all.extend(gn.tolist())

        g_sel, g_unsel = np.array(g_sel), np.array(g_unsel)
        scores_all, grads_all = np.array(scores_all), np.array(grads_all)
        mean_sel = float(g_sel.mean()) if len(g_sel) else float("nan")
        mean_unsel = float(g_unsel.mean()) if len(g_unsel) else float("nan")
        ratio = mean_unsel / mean_sel if mean_sel > 0 else float("nan")
        frac_zero = n_zero_unsel / max(n_unsel_nodes, 1)

        # gradient norm binned by score decile (monotone attenuation check)
        bins = np.quantile(scores_all, np.linspace(0, 1, 6)) \
            if len(scores_all) else None
        binned = []
        if bins is not None:
            for lo, hi in zip(bins[:-1], bins[1:]):
                sel = (scores_all >= lo) & (scores_all <= hi)
                if sel.sum():
                    binned.append((float(lo), float(hi),
                                   float(grads_all[sel].mean())))

        row = (f"| {name}-{mdl} | {ATT_MODE} | seed {seed} | graphs={n_used} "
               f"| grad(sel)={mean_sel:.3e} grad(unsel)={mean_unsel:.3e} "
               f"ratio={ratio:.4f} | unsel exactly-zero: "
               f"{n_zero_unsel}/{n_unsel_nodes} ({100*frac_zero:.1f}%) |")
        print("#R# " + row)
        import math
        print(f"#R#   dead graphs (all-zero grad): {n_dead}/{n_used} "
              f"(max dead-loss={max(dead_losses, default=0.0):.2e}, "
              f"min live-loss={min(live_losses, default=math.nan):.2e})")
        print(f"#R#   halo(L={L}, LIVE graphs) check: P(grad=0 | outside halo)="
              f"{100*n_outh_zero/max(n_outh,1):.1f}% ({n_outh_zero}/{n_outh})"
              f"; P(grad=0 | in halo, unsel)="
              f"{100*n_inh_zero/max(n_inh,1):.1f}% ({n_inh_zero}/{n_inh})")
        if col_stats:
            tops = sorted(col_stats.items(), key=lambda kv: -kv[1][1])[:5]
            print("#R#   live in-halo unsel zeros by colour: " + "; ".join(
                f"attr{c}: {100*z/max(t,1):.0f}% ({z}/{t})"
                for c, (z, t) in tops))
        if binned:
            print("#R#   grad-by-score bins: " + "; ".join(
                f"[{lo:.2f},{hi:.2f}]:{m:.2e}" for lo, hi, m in binned))
        rows.append(row)

        torch.save({"meta": {"dataset": name, "model": mdl, "seed": seed,
                             "split": split, "thr": thr,
                             "att_mode": ATT_MODE, "layers": L},
                    "grad_sel": g_sel, "grad_unsel": g_unsel,
                    "scores": scores_all, "grads": grads_all,
                    "frac_zero_unsel": frac_zero, "ratio": ratio,
                    "halo": {"outh": n_outh, "outh_zero": n_outh_zero,
                             "inh": n_inh, "inh_zero": n_inh_zero,
                             "n_dead": n_dead, "n_used": n_used,
                             "dead_losses": dead_losses,
                             "live_losses": live_losses,
                             "col_stats": col_stats}},
                   os.path.join(
                       OUT,
                       f"{name}_{mdl}_seed{seed}_{split}_{ATT_MODE}.pt"))

    print(f"\n#IN# done in {datetime.now() - t0}")
    print("#R# ===== E3.2 freezing table =====")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
