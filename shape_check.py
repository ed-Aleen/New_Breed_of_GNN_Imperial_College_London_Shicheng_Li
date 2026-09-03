"""
E2 (+ E1 step 2) of the experiment pipeline (EXPERIMENT_DESIGN.md):

Given trained checkpoints, extract explanations through the repo's own
D.5 protocol path (Pipeline.generate_binary_explanations: SMGNN min-max
rescue, DIR top-K, correct-prediction filtering) and check, per graph:

  [E1-2] tie deviation : max score spread within each ACR-enriched WL
                         colour class of multiplicity >= 2
                         (Prop 1(i): should be ~ float precision)
  [E2-1] shape         : is R a union of colour classes?
                         (Prop 1(ii): violation rate ~ 0 for threshold
                          readers; DIR violations expected at top-K ties)
  [E2-2] menu-dish     : which content type each seed's extractor selects
                         (data for the menu-dish figure + Lambda for
                          certificate.py)

Usage (inside gsat_venv, repo root) — same flags as goodtg plus extras:
  python shape_check.py --config_path final_configs/MUTAG/basis/no_shift/GSAT.yaml \
      --seeds 1/2/3/4/5 --task test --ratios 0.9 --splits id_val --backbone ACR2 --gpu_idx 0
  python shape_check.py --config_path final_configs/BAColorGVIsol/basis/no_shift/GSAT.yaml \
      --seeds 1/2/3/4/5 --task test --ratios 0.5 --splits id_val --backbone ACR2 --gpu_idx 0
  python shape_check.py --config_path final_configs/MUTAG/basis/no_shift/SMGNN_sec6.yaml \
      --seeds 1/2/3/4/5 --task test --ratios 0.5 --splits id_val --backbone ACR2 --gpu_idx 0
Extra flags (stripped before the repo parser sees them):
  --decimals N   feature rounding for colour refinement (continuous cells)
  --out DIR      output dir (default storage/expl_shapes)
"""
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np
import torch

from wl_colors import initial_colours, refine


# ---- strip our extra flags before the repo's strict parser runs ----------
def pop_flag(argv, name, default):
    if name in argv:
        i = argv.index(name)
        val = argv[i + 1]
        del argv[i:i + 2]
        return val
    return default


EXTRA_DECIMALS = pop_flag(sys.argv, "--decimals", None)
EXTRA_DECIMALS = int(EXTRA_DECIMALS) if EXTRA_DECIMALS is not None else None
EXTRA_OUT = pop_flag(sys.argv, "--out", "storage/expl_shapes")

from GOOD import config_summoner                       # noqa: E402
from GOOD.utils.args import args_parser                # noqa: E402
from GOOD.utils.loader import initialize_model_dataset # noqa: E402
from GOOD.ood_algorithms.ood_manager import load_ood_alg  # noqa: E402
from GOOD.kernel.pipeline_manager import load_pipeline    # noqa: E402
from GOOD.utils.logger import load_logger              # noqa: E402


def graph_colors(x, edge_index, rounds, decimals):
    g = {"x": x, "edge_index": edge_index}
    init, _ = initial_colours([g], decimals)
    return refine([g], init, rounds, "acr")[0]


def content_key(x, edge_index, mask):
    """Canonical proxy for [[R]]: (|R|, attr multiset, in-R degree seq).
    Exact for single-node contents; good canonical proxy for small R."""
    idx = np.where(mask)[0]
    attrs = []
    for v in idx:
        row = x[v]
        attrs.append(int(row.argmax()) if row.ndim else int(row))
    sel = set(idx.tolist())
    deg = Counter()
    for e in range(edge_index.shape[1]):
        s, d = int(edge_index[0, e]), int(edge_index[1, e])
        if s in sel and d in sel:
            deg[d] += 1
    degs = sorted(deg.get(v, 0) for v in idx)
    return (len(idx), tuple(sorted(attrs)), tuple(degs))


def main():
    args = args_parser()
    splits = args.splits.split("/") if args.splits else ["id_val"]
    thrs = [float(r) for r in args.ratios.split("/")] if args.ratios else [0.5]

    os.makedirs(EXTRA_OUT, exist_ok=True)
    startTime = datetime.now()
    summary_rows = []

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

        samples, _, _ = pipeline.generate_binary_explanations(
            is_weight=True, thrs=thrs, splits=splits, convert_to_nx=False,
            is_node_expl=not config.ood.extra_param[0])

        rounds = int(config.model.model_layer)
        name = config.dataset.dataset_name
        mdl = config.model.model_name

        for split in splits:
            for thr in thrs:
                graphs = samples[split][thr]
                n_cls_checked = 0        # colour classes with mult >= 2
                n_cls_violated = 0
                n_graphs_violated = 0
                tie_devs = []            # within-class score spread
                near_thr_viol = 0        # violations w/ spread < 10*|gap to thr|
                dishes = Counter()
                per_graph = []

                for g in graphs:
                    x = g.x.detach().cpu().numpy()
                    ei = g.edge_index.detach().cpu().numpy()
                    expl = g.node_expl.detach().cpu().numpy()
                    mask = g.node_mask.detach().cpu().numpy().astype(bool)
                    cols = graph_colors(x, ei, rounds, EXTRA_DECIMALS)

                    g_viol = 0
                    for c, cnt in Counter(cols.tolist()).items():
                        if cnt < 2:
                            continue
                        n_cls_checked += 1
                        sel = mask[cols == c]
                        sc = expl[cols == c]
                        tie_devs.append(float(sc.max() - sc.min()))
                        if 0 < sel.sum() < cnt:
                            n_cls_violated += 1
                            g_viol += 1
                            spread = sc.max() - sc.min()
                            gap = np.abs(sc - thr).min()
                            if spread < 10 * max(gap, 1e-12) or spread < 1e-5:
                                near_thr_viol += 1
                    if g_viol:
                        n_graphs_violated += 1
                    ck = content_key(x, ei, mask)
                    dishes[ck] += 1
                    per_graph.append({
                        "y": int(g.y.view(-1)[0]), "y_pred": int(g.y_pred),
                        "expl": expl, "mask": mask, "colors": cols,
                        "n_viol": g_viol, "content": ck})

                viol_rate = n_cls_violated / max(n_cls_checked, 1)
                tie_max = max(tie_devs) if tie_devs else float("nan")
                tie_p99 = (float(np.quantile(tie_devs, 0.99))
                           if tie_devs else float("nan"))
                top = dishes.most_common(5)
                row = (f"| {name}-{mdl} | seed {seed} | {split} thr={thr} "
                       f"| graphs={len(graphs)} | tie_dev max={tie_max:.2e} "
                       f"p99={tie_p99:.2e} | viol {n_cls_violated}/{n_cls_checked} "
                       f"({100*viol_rate:.2f}%) graphs {n_graphs_violated} "
                       f"| near-thr {near_thr_viol} |")
                print("#R# " + row)
                print("#R#   top dishes: " +
                      "; ".join(f"{k} x{v}" for k, v in top))
                summary_rows.append(row)

                torch.save(
                    {"meta": {"dataset": name, "model": mdl, "seed": seed,
                              "split": split, "thr": thr, "rounds": rounds,
                              "decimals": EXTRA_DECIMALS},
                     "per_graph": per_graph, "dishes": dict(dishes),
                     "summary": {"viol_rate": viol_rate, "tie_max": tie_max,
                                 "tie_p99": tie_p99,
                                 "n_cls_checked": n_cls_checked,
                                 "near_thr_viol": near_thr_viol}},
                    os.path.join(EXTRA_OUT,
                                 f"{name}_{mdl}_seed{seed}_{split}_thr{thr}.pt"))

    print(f"\n#IN# done in {datetime.now() - startTime}")
    print("#R# ===== E2-a table =====")
    for r in summary_rows:
        print(r)


if __name__ == "__main__":
    main()
