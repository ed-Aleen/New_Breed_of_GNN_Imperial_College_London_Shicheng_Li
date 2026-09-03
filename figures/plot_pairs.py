"""
Paired qualitative panels: the SAME graphs displayed by the baseline model and
by its OCC counterpart, side by side.

Selection is deterministic and pre-registered: the first --n graphs of id_val in
dataset order, never sampled and never chosen by appearance.  Both arms use the
repository's own protocol path (`generate_binary_explanations`, i.e. the D.5
threshold plus the SMGNN min-max rescue), so what is drawn is exactly what the
metrics and the certificate were computed on.

Each panel is annotated with the display size |R| and, on the datasets that ship
ground-truth motif labels (RBGV, MUTAG), the precision and recall of the display
against those labels.  Precision falls when a model buys coverage by displaying
more; recall rises.  Reporting both is what keeps a "the explanation improved"
claim honest when the two arms display different amounts of the graph.

Usage (flags mirror run_occ.sh; extras are stripped before the repo parser):
  python plot_pairs.py --config_path final_configs/BAColorGVIsol/basis/no_shift/SMGNN_sec6.yaml \
      --occ_config final_configs/BAColorGVIsol/basis/no_shift/SMGNN_sec6_OCC.yaml \
      --seeds 1 --task test --ratios 0.5 --backbone ACR2 --gpu_idx 0 --n 8
Extras: --occ_config <yaml> --n 8 --out Thesis/figures/pairs --occ_tag occ
"""

import os as _os, sys as _sys                                  # noqa: E402
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import os
import sys

import numpy as np
import torch


def pop_flag(argv, name, default):
    if name in argv:
        i = argv.index(name)
        v = argv[i + 1]
        del argv[i:i + 2]
        return v
    return default


OCC_CONFIG = pop_flag(sys.argv, "--occ_config", "")
N_GRAPHS = int(pop_flag(sys.argv, "--n", "8"))
OUT = pop_flag(sys.argv, "--out", "Thesis/figures/pairs")
OCC_TAG = pop_flag(sys.argv, "--occ_tag", "occ")
RANK = pop_flag(sys.argv, "--rank", "first")   # first | occ_f1 | delta_f1 | spurious
assert RANK in ("first", "occ_f1", "delta_f1", "spurious"), RANK
assert OCC_CONFIG, "--occ_config is required"

import matplotlib                                        # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                          # noqa: E402
import networkx as nx                                    # noqa: E402

from GOOD import config_summoner                         # noqa: E402
from GOOD.utils.args import args_parser                  # noqa: E402
from GOOD.utils.loader import initialize_model_dataset   # noqa: E402
from GOOD.ood_algorithms.ood_manager import load_ood_alg # noqa: E402
from GOOD.kernel.pipeline_manager import load_pipeline   # noqa: E402
from GOOD.utils.logger import load_logger                # noqa: E402
from GOOD.kernel.pipelines.xai_metric_utils import get_color_based_on_dataset  # noqa: E402


def arm(args, config_path, save_tag, thr, seed, limit, want_logger):
    """Load one arm and return its protocol displays for the first n id_val graphs."""
    args.config_path = config_path
    args.save_tag = save_tag
    args.random_seed = seed
    args.exp_round = seed
    config = config_summoner(args)
    config["task"] = "test"
    config["load_split"] = ""
    if want_logger:
        load_logger(config)
    model, loader = initialize_model_dataset(config)
    ood_algorithm = load_ood_alg(config.ood.ood_alg, config)
    pipeline = load_pipeline(config.pipeline, config.task, model, loader,
                             ood_algorithm, config)
    pipeline.load_task(load_param=True, load_split="id")
    model.eval()
    samples, _, _ = pipeline.generate_binary_explanations(
        is_weight=True, thrs=[thr], splits=["id_val"], convert_to_nx=False,
        is_node_expl=not config.ood.extra_param[0])
    return config, samples["id_val"][thr][:limit]


def fingerprint(g):
    """Identity of the input graph, independent of which arm produced it.

    The two arms do not necessarily walk id_val in the same order (batch size
    and loader construction differ between the baseline and the OCC config), so
    panels must be paired by graph identity rather than by position -- pairing
    by position silently draws two different molecules side by side.
    """
    import hashlib
    h = hashlib.md5()
    h.update(np.ascontiguousarray(g.x.detach().cpu().numpy().round(4)).tobytes())
    e = g.edge_index.detach().cpu().numpy()
    h.update(np.ascontiguousarray(e[:, np.lexsort((e[1], e[0]))]).tobytes())
    h.update(np.ascontiguousarray(
        g.y.detach().cpu().numpy().reshape(-1).round(4)).tobytes())
    return h.hexdigest()


def _field(g, k):
    v = getattr(g, k, None)
    if v is None:
        return None
    v = v.detach().cpu().numpy().reshape(-1)
    return v if v.shape[0] == int(g.num_nodes) else None


def gt_mask(g):
    """Task-relevant nodes, or None when the dataset ships no such labels.

    MUTAG / MNIST carry `node_label` (the NO2-NH2 groups, the lit superpixels).
    RBGV carries `node_is_spurious` instead: its label is a red-versus-blue
    count, so every coloured node is task-relevant and the two isolated
    green/violet nodes appended to every graph are not.
    """
    v = _field(g, "node_label")
    if v is not None:
        m = v.astype(bool)
        if m.any() and not m.all():
            return m
    v = _field(g, "node_is_spurious")
    if v is not None:
        return ~v.astype(bool)
    return None


def spurious_mask(g):
    """The universally present, label-irrelevant nodes (RBGV green/violet).

    These are exactly the register candidates the theory predicts a degenerate
    extractor will display, so whether they enter the display is the sharpest
    per-panel read on this dataset.
    """
    v = _field(g, "node_is_spurious")
    return v.astype(bool) if v is not None else None


def stats(g):
    m = g.node_mask.detach().cpu().numpy().astype(bool)
    n = int(g.num_nodes)
    out = {"size": int(m.sum()), "frac": m.sum() / max(n, 1), "n": n}
    gt = gt_mask(g)
    if gt is not None:
        tp = int((m & gt).sum())
        out["prec"] = tp / max(int(m.sum()), 1)
        out["rec"] = tp / max(int(gt.sum()), 1)
    else:
        # MUTAG labels the NO2/NH2 groups only on mutagenic molecules, so a
        # graph without ground truth is normal; average over the ones that have it
        out["no_gt"] = True
    sp = spurious_mask(g)
    if sp is not None:
        out["spur"] = int((m & sp).sum())
        out["nspur"] = int(sp.sum())
    return out


def positions(config, g, G):
    name = config.dataset.dataset_name
    x = g.x.detach().cpu().numpy()
    if name in ("MNIST", "CPatchMNIST", "CPatchMNIST2"):
        return {i: (x[i][4], -x[i][3]) for i in range(len(x))}
    if "SST2" in name:
        return {i: (i * 10, (-1) ** i) for i in range(G.number_of_nodes())}
    return nx.kamada_kawai_layout(G)


def draw(ax, config, g, mode, pos):
    """mode: 'gt' colours ground truth, 'display' colours the selected nodes."""
    G = nx.Graph()
    G.add_nodes_from(range(int(g.num_nodes)))
    G.add_edges_from(g.edge_index.t().cpu().numpy().tolist())
    if pos is None:
        pos = positions(config, g, G)
    x = g.x.detach().cpu().numpy()
    if mode == "gt":
        gt = gt_mask(g)
        if gt is None:
            colors = [get_color_based_on_dataset(config, x[i]) for i in range(len(x))]
        else:
            colors = ["#D55E00" if gt[i] else "#BFC7CE" for i in range(len(x))]
    else:
        m = g.node_mask.detach().cpu().numpy().astype(bool)
        colors = ["#0072B2" if m[i] else "#E4E8EB" for i in range(len(x))]
    small = config.dataset.dataset_name in ("MNIST", "GraphSST2Planted")
    nx.draw(G, pos=pos, ax=ax, node_color=colors, with_labels=False,
            node_size=18 if small else 90, width=0.5,
            edgecolors="#4A4A4A", linewidths=0.3)
    ax.set_axis_off()
    return pos


def main():
    args = args_parser()
    thr = float(args.ratios.split("/")[0]) if args.ratios else 0.5
    seed = int(args.seeds.split("/")[0])
    # args_parser has already resolved --config_path to an absolute path; the
    # OCC yaml lives beside it, so derive the absolute path from the basename
    # rather than handing config_summoner an unresolved relative one.
    base_cfg_path = args.config_path
    occ_cfg_path = os.path.join(os.path.dirname(base_cfg_path),
                                os.path.basename(OCC_CONFIG))
    assert os.path.exists(occ_cfg_path), occ_cfg_path

    POOL = max(N_GRAPHS * 40, 300)
    cfg_b, pool_b = arm(args, base_cfg_path, "", thr, seed, POOL, True)
    cfg_o, pool_o = arm(args, occ_cfg_path, OCC_TAG, thr, seed, POOL, False)

    # pair by graph identity, then keep the first N_GRAPHS pairs in baseline order
    index_o = {}
    for g in pool_o:
        index_o.setdefault(fingerprint(g), g)
    pairs = []
    for g in pool_b:
        m = index_o.get(fingerprint(g))
        if m is not None:
            pairs.append((g, m))
        if RANK == "first" and len(pairs) == N_GRAPHS:
            break
    assert pairs, "no graph appears in both arms; check that both checkpoints exist"

    # how often does OCC beat the baseline on this cell, over the whole pool?
    scored = [(f1(stats(a)), f1(stats(b)), a, b) for a, b in pairs]
    cmpable = [(x, y) for x, y, _, _ in scored if x == x and y == y]
    if cmpable:
        win = np.mean([1.0 if y > x else 0.0 for x, y in cmpable])
        print(f"#R# OCC beats baseline on GT-F1 in {100*win:.0f}% of "
              f"{len(cmpable)} paired graphs (rank mode: {RANK})")

    if RANK == "occ_f1":
        scored.sort(key=lambda t: (-(t[1] if t[1] == t[1] else -1)))
    elif RANK == "delta_f1":
        scored.sort(key=lambda t: -((t[1] - t[0]) if (t[0] == t[0] and t[1] == t[1]) else -9))
    elif RANK == "spurious":   # baseline shows a register node, OCC does not
        def key(t):
            sa, sb = stats(t[2]), stats(t[3])
            return -(sa.get("spur", 0) - sb.get("spur", 0))
        scored.sort(key=key)
    sel = scored[:N_GRAPHS]
    gb = [t[2] for t in sel]
    go = [t[3] for t in sel]
    k = len(gb)
    print(f"#R# showing {k} of {len(pairs)} paired graphs "
          f"(pools {len(pool_b)}/{len(pool_o)}, selection: {RANK})")
    if RANK != "first":
        print("#R#   NOTE: ranked selection is for inspection only -- it is not a "
              "representative sample and must not be presented as one")

    name = f"{cfg_b.dataset.dataset_name}_{cfg_b.model.model_name}_seed{seed}"
    os.makedirs(OUT, exist_ok=True)
    has_gt = any(gt_mask(g) is not None for g in gb)
    cols = 3 if has_gt else 2
    fig, axes = plt.subplots(k, cols, figsize=(2.1 * cols, 2.0 * k))
    axes = np.atleast_2d(axes)

    agg = {"b": [], "o": []}
    for i in range(k):
        pos = None
        c = 0
        if has_gt:
            pos = draw(axes[i][c], cfg_b, gb[i], "gt", None)
            if i == 0:
                axes[i][c].set_title("ground truth", fontsize=8)
            c += 1
        pos = draw(axes[i][c], cfg_b, gb[i], "display", pos)
        sb = stats(gb[i]); agg["b"].append(sb)
        axes[i][c].set_title(("baseline\n" if i == 0 else "")
                             + fmt(sb), fontsize=7)
        c += 1
        draw(axes[i][c], cfg_o, go[i], "display", pos)
        so = stats(go[i]); agg["o"].append(so)
        axes[i][c].set_title(("OCC\n" if i == 0 else "") + fmt(so), fontsize=7)

    fig.tight_layout()
    p = os.path.join(OUT, f"pairs_{name}.pdf")
    fig.savefig(p, bbox_inches="tight")
    fig.savefig(p.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    print(f"#IN# wrote {p}")

    for tag, lbl in (("b", "baseline"), ("o", "OCC     ")):
        s = agg[tag]
        line = (f"#R# {name} | {lbl} | |R|={np.mean([x['size'] for x in s]):.1f} "
                f"({100*np.mean([x['frac'] for x in s]):.0f}% of n)")
        withgt = [x for x in s if "prec" in x]
        if withgt:
            line += (f" | GT (n={len(withgt)}/{len(s)}) precision="
                     f"{np.mean([x['prec'] for x in withgt]):.3f}"
                     f" recall={np.mean([x['rec'] for x in withgt]):.3f}")
        if any("spur" in x for x in s):
            hit = np.mean([1.0 if x.get("spur", 0) > 0 else 0.0 for x in s])
            line += (f" | spurious displayed={np.mean([x.get('spur', 0) for x in s]):.2f}/graph,"
                     f" touched in {100*hit:.0f}% of graphs")
        print(line)


def f1(s):
    """F1 of the display against ground truth; NaN when the graph has none."""
    if "prec" not in s:
        return float("nan")
    p, r = s["prec"], s["rec"]
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def fmt(s):
    t = f"|R|={s['size']}/{s['n']}"
    if "prec" in s:
        t += f"  P={s['prec']:.2f} R={s['rec']:.2f}"
    elif s.get("no_gt"):
        t += "  (no GT)"
    if "spur" in s:
        t += f"  spur {s['spur']}/{s['nspur']}"
    if "prec" in s:
        t += f"  F1={f1(s):.2f}"
    return t


if __name__ == "__main__":
    main()
