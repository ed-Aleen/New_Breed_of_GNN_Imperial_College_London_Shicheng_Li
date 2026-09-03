"""
E3.1 + E3.4 of the experiment pipeline (EXPERIMENT_DESIGN.md):
the five-tuple persistence-degeneration certificate, per checkpoint:

    Cert = ( Lambda, {pi_c}, kappa(Lambda), {U_a}, H(yhat) )

plus the codebook cross-tab (predicted class x displayed content).

Protocol notes (all D'-consistent):
  * displays come from the repo's own generate_binary_explanations
    (SMGNN min-max rescue, DIR top-K, correct-prediction filtering);
  * Lambda is computed on NONEMPTY displays only (D' convention);
    empty-display rate is reported separately;
  * contents of size <= 4 use an exact canonical form (min over node
    permutations of (attrs, adjacency)); size >= 5 is bucketed LARGE and
    flagged auto-high-information (excluded from kappa, reported);
  * pi_c = fraction of split graphs containing c as an INDUCED attributed
    subgraph (size 1/2 exact count formulas; size 3/4 DFS with node cap);
  * U_a ("one more node of type a"): for graphs whose display is nonempty
    and that contain an undisplayed node of attr a, flip ONE such node on
    and measure the NLL change through predict_from_subgraph — the
    implementation-level unfamiliarity gap of SWITCH_BARRIER.md;
  * H(yhat) and Acc come from a full-split get_subgraph pass (no filter).

E4 port (clean repo): same instrument, run on BOTH arms to answer the
definition-level question "did OCC reduce DEGENERATION" (rho), not just
"did the faithfulness metrics improve" (EST). Adds two confound controls
the E4 comparison needs:

  * display geometry: mean |R|, mean |R|/n, empty-display rate -- a metric
    can fall because the explanation grew to the whole graph or shrank to
    nothing (the anchor's own "low EST != faithful" trap);
  * confidence band: P(p_clean in [0.40, 0.60]) -- basic_pipeline's EST
    rejection DISABLES its uncertainty penalty when the clean prediction is
    already uncertain (uncertain_to_penalize = uncertain_pert & ~uncertain_clean,
    line ~1142), so an arm whose predictions sit nearer 0.5 can score a lower
    rejection rate for a mechanical reason. Reported per arm; a large gap
    means the EST comparison needs the confident-subset caveat.

Usage (same flags as goodtg + extras, stripped before the repo parser):
  python occ_certificate.py --config_path final_configs/MUTAG/basis/no_shift/GSAT.yaml \
      --seeds 1/2/3/4/5 --task test --ratios 0.9 --splits id_val --backbone ACR2 --gpu_idx 0
  # OCC arm: use the _OCC.yaml AND --save_tag occ
Extras: --delta 0.05  --topM 10  --max_u_graphs 300  --out storage/certificates
"""
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime
from itertools import permutations

import numpy as np
import torch


def pop_flag(argv, name, default):
    if name in argv:
        i = argv.index(name)
        v = argv[i + 1]
        del argv[i:i + 2]
        return v
    return default


DELTA = float(pop_flag(sys.argv, "--delta", "0.05"))
TOPM = int(pop_flag(sys.argv, "--topM", "10"))
MAX_U_GRAPHS = int(pop_flag(sys.argv, "--max_u_graphs", "300"))
OUT = pop_flag(sys.argv, "--out", "storage/certificates")
DFS_CAP = 200000  # node-expansion cap for size-3/4 containment search

from GOOD import config_summoner                        # noqa: E402
from GOOD.utils.args import args_parser                 # noqa: E402
from GOOD.utils.loader import initialize_model_dataset  # noqa: E402
from GOOD.ood_algorithms.ood_manager import load_ood_alg   # noqa: E402
from GOOD.kernel.pipeline_manager import load_pipeline     # noqa: E402
from GOOD.utils.logger import load_logger               # noqa: E402
from torch_geometric.data import Batch                  # noqa: E402
from torch_geometric.loader import DataLoader           # noqa: E402


# ---------------- content canonical form (exact for n <= 4) ---------------
def canon_content(x, edge_index, mask):
    idx = np.where(mask)[0]
    n = len(idx)
    if n == 0:
        return ("EMPTY",)
    if n > 4:
        return ("LARGE", n)
    attrs = [int(x[v].argmax()) if x[v].ndim else int(x[v]) for v in idx]
    pos = {int(v): i for i, v in enumerate(idx)}
    adj = np.zeros((n, n), dtype=bool)
    for e in range(edge_index.shape[1]):
        s, d = int(edge_index[0, e]), int(edge_index[1, e])
        if s in pos and d in pos and s != d:
            adj[pos[s], pos[d]] = True
            adj[pos[d], pos[s]] = True
    best = None
    for p in permutations(range(n)):
        key = (tuple(attrs[p[i]] for i in range(n)),
               tuple(adj[p[i], p[j]] for i in range(n) for j in range(i + 1, n)))
        if best is None or key < best:
            best = key
    return (n, best[0], best[1])


# ---------------- induced containment: pi_c ------------------------------
def graph_summaries(gs):
    out = []
    for g in gs:
        x = g["x"]
        attrs = x.argmax(1) if x.ndim > 1 else x
        attrs = np.asarray(attrs, dtype=int)
        eset = set()
        for e in range(g["edge_index"].shape[1]):
            s, d = int(g["edge_index"][0, e]), int(g["edge_index"][1, e])
            if s != d:
                eset.add((min(s, d), max(s, d)))
        by_attr = defaultdict(list)
        for v, a in enumerate(attrs.tolist()):
            by_attr[a].append(v)
        out.append({"attrs": attrs, "eset": eset, "by_attr": by_attr,
                    "n": len(attrs)})
    return out


def contains(gsum, content):
    n, attrs, adjbits = content
    adj = np.zeros((n, n), dtype=bool)
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            adj[i, j] = adj[j, i] = adjbits[k]
            k += 1
    if n == 1:
        return len(gsum["by_attr"].get(attrs[0], [])) > 0
    if n == 2:
        a, b = attrs
        A, B = gsum["by_attr"].get(a, []), gsum["by_attr"].get(b, [])
        if not A or not B:
            return False
        want_edge = bool(adj[0, 1])
        pairs = ((min(u, v), max(u, v)) for u in A for v in B if u != v)
        if want_edge:
            return any(p in gsum["eset"] for p in pairs)
        n_pairs = (len(A) * len(B) - (len(A) if a == b else 0)) // (2 if a == b else 1)
        if n_pairs == 0:
            return False
        n_edges = sum(1 for (u, v) in gsum["eset"]
                      if {gsum["attrs"][u], gsum["attrs"][v]} == {a, b}
                      or (a == b == gsum["attrs"][u] == gsum["attrs"][v]))
        return n_edges < n_pairs
    # n = 3 or 4: DFS with cap
    order = sorted(range(n), key=lambda i: -adj[i].sum())
    budget = [DFS_CAP]

    def dfs(pos_i, assign):
        if budget[0] <= 0:
            return None  # undecided
        budget[0] -= 1
        if pos_i == n:
            return True
        pi = order[pos_i]
        for v in gsum["by_attr"].get(attrs[pi], []):
            if v in assign.values():
                continue
            ok = True
            for pj, u in assign.items():
                e = (min(u, v), max(u, v)) in gsum["eset"]
                if e != bool(adj[pi, pj]):
                    ok = False
                    break
            if ok:
                assign[pi] = v
                r = dfs(pos_i + 1, assign)
                if r:
                    return True
                if r is None:
                    return None
                del assign[pi]
        return False

    return dfs(0, {})


def hb(p):
    if p <= 0 or p >= 1:
        return 0.0
    return float(-p * np.log2(p) - (1 - p) * np.log2(1 - p))


def nll_of(probs_row, y, n_classes):
    p = probs_row.view(-1)
    if p.numel() == 1:  # single sigmoid logit
        py = p.item() if y == 1 else 1.0 - p.item()
    else:
        py = p[y].item()
    return -float(np.log(max(py, 1e-12)))


def logodds_of(probs_row, y):
    """True-class log-odds; saturation-robust companion to nll_of."""
    p = probs_row.view(-1)
    if p.numel() == 1:
        py = p.item() if y == 1 else 1.0 - p.item()
    else:
        py = p[y].item()
    py = min(max(py, 1e-9), 1.0 - 1e-9)
    return float(np.log(py / (1.0 - py)))


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

        # ---- pass 1: full-split predictions (no filtering) --------------
        ds = pipeline.get_local_dataset(split)
        raw = [{"x": d.x.cpu().numpy(), "edge_index": d.edge_index.cpu().numpy()}
               for d in ds]
        gsums = graph_summaries(raw)
        yhat, ytrue, pmax = [], [], []
        with torch.no_grad():
            for data in DataLoader(ds, batch_size=256, shuffle=False):
                data = data.to(dev)
                _, _, logits = model.get_subgraph(
                    data=data, edge_weight=None, ood_algorithm=ood_algorithm,
                    do_relabel=False)
                if logits.shape[-1] > 1:
                    pred = logits.argmax(1)
                    pr = logits.softmax(1).max(1).values
                else:
                    pred = (logits.view(-1) > 0.5).long()
                    pr = logits.view(-1).sigmoid()
                yhat.extend(pred.cpu().tolist())
                ytrue.extend(data.y.view(-1).cpu().long().tolist())
                pmax.extend(pr.cpu().tolist())
        yhat, ytrue = np.array(yhat), np.array(ytrue)
        pmax = np.array(pmax)
        # EST slack band: basic_pipeline treats a clean prediction inside
        # [0.40, 0.60] as already-uncertain and then SKIPS the forced-change
        # penalty for it -> arms differing here are not directly comparable.
        band = float(((pmax >= 0.40) & (pmax <= 0.60)).mean())
        acc = float((yhat == ytrue).mean())
        pj = np.bincount(yhat, minlength=int(max(yhat.max(), ytrue.max())) + 1)
        pj = pj / pj.sum()
        H_yhat = float(-(pj[pj > 0] * np.log2(pj[pj > 0])).sum())

        # ---- pass 2: protocol displays ----------------------------------
        samples, _, _ = pipeline.generate_binary_explanations(
            is_weight=True, thrs=[thr], splits=[split], convert_to_nx=False,
            is_node_expl=not config.ood.extra_param[0])
        graphs = samples[split][thr]
        contents, xtab = [], Counter()
        size_abs, size_rel = [], []
        for g in graphs:
            x = g.x.cpu().numpy()
            ei = g.edge_index.cpu().numpy()
            m = g.node_mask.cpu().numpy().astype(bool)
            size_abs.append(int(m.sum()))
            size_rel.append(float(m.sum()) / max(len(m), 1))
            c = canon_content(x, ei, m)
            contents.append(c)
            xtab[(int(g.y_pred), c if c[0] != "LARGE" else ("LARGE",))] += 1

        n_all = len(contents)
        nonempty = [c for c in contents if c[0] != "EMPTY"]
        empty_rate = 1.0 - len(nonempty) / max(n_all, 1)
        n_ne = max(len(nonempty), 1)
        # LARGE displays are size-buckets, not content types: they must NOT
        # aggregate frequency into Lambda (granularity artifact). They are
        # reported as large_share and trigger ABSTAIN when dominant.
        small = [c for c in nonempty if c[0] != "LARGE"]
        large_share = 1.0 - len(small) / n_ne
        freq = Counter(small)
        Lam = [c for c, k in freq.items() if k / n_ne >= DELTA]
        delta_eff = DELTA
        if not Lam:  # adaptive fallback: top-M small contents
            Lam = [c for c, _ in freq.most_common(TOPM)]
            delta_eff = (freq[Lam[-1]] / n_ne) if Lam else 0.0
        coverage = sum(freq[c] for c in Lam) / n_ne

        # ---- pi_c, kappa, and the per-graph occurrence picture -----------
        pis = {}
        Obits = np.zeros((len(gsums), len(Lam)), dtype=bool)
        for j, c in enumerate(Lam):
            for gi, gs in enumerate(gsums):
                Obits[gi, j] = bool(contains(gs, c))
            pis[c] = float(Obits[:, j].mean())
        kappa = sum(hb(p) for p in pis.values())

        # ---- direct plug-in MI of the occurrence picture -----------------
        def plug_mi(bits, labels):
            n = len(labels)
            pats = [tuple(b) for b in bits]
            pj, pp, pl = Counter(zip(pats, labels)), Counter(pats), Counter(labels)
            mi = sum((k / n) * np.log2(k * n / (pp[p] * pl[l]))
                     for (p, l), k in pj.items())
            return float(mi), len(pp)

        if len(Lam) > 0:
            mi_yhat, n_pat = plug_mi(Obits, yhat.tolist())
            mi_y, _ = plug_mi(Obits, ytrue.tolist())
        else:
            mi_yhat, mi_y, n_pat = 0.0, 0.0, 0
        rho_dir = 1.0 - mi_yhat / max(H_yhat, 1e-9)
        rho_bound = max(0.0, 1.0 - kappa / max(H_yhat, 1e-9))

        # ---- verdict -----------------------------------------------------
        # Asymmetry: MI is monotone under adding coordinates, so HIGH MI on
        # any sub-family certifies rho <= 1 - I/H regardless of coverage
        # (one-sided certificate); LOW MI on a partial family is inconclusive.
        if H_yhat < 0.1:
            verdict = "DEAD(D1a)"
        elif mi_yhat >= 0.5 * H_yhat:
            verdict = f"NON-DEG(certified) rho<={max(0.0, rho_dir):.2f}"
        elif large_share > 0.5:
            verdict = f"ABSTAIN(large={100*large_share:.0f}%)"
        else:
            verdict = f"rho_dir={rho_dir:.2f} bound>={rho_bound:.2f}"

        # ---- U_a: one-more-node-of-type-a gap ----------------------------
        n_cls = int(max(yhat.max(), ytrue.max())) + 1
        U, Um = defaultdict(list), defaultdict(list)
        rng = np.random.RandomState(seed)
        with torch.no_grad():
            for g in graphs[:MAX_U_GRAPHS]:
                m = g.node_mask.cpu().numpy().astype(bool)
                if m.sum() == 0:
                    continue
                x = g.x.cpu().numpy()
                attrs = x.argmax(1) if x.ndim > 1 else x
                y = int(g.y.view(-1)[0])
                batch = Batch.from_data_list([g]).to(dev)
                att0 = torch.tensor(m, dtype=torch.float, device=dev).view(-1, 1)
                p0 = model.predict_from_subgraph(
                    data=batch, edge_weight=None, edge_attn=None,
                    node_att=att0, ood_algorithm=ood_algorithm)
                nll0 = nll_of(p0[0], y, n_cls)
                lo0 = logodds_of(p0[0], y)
                for a in sorted(set(attrs.tolist())):
                    cand = np.where((attrs == a) & (~m))[0]
                    if len(cand) == 0:
                        continue
                    v = int(rng.choice(cand))
                    att1 = att0.clone()
                    att1[v] = 1.0
                    p1 = model.predict_from_subgraph(
                        data=batch, edge_weight=None, edge_attn=None,
                        node_att=att1, ood_algorithm=ood_algorithm)
                    U[a].append(nll_of(p1[0], y, n_cls) - nll0)
                    # margin drop: positive = wall (true-class odds fell)
                    Um[a].append(lo0 - logodds_of(p1[0], y))
        U_mean = {a: (float(np.mean(v)), len(v)) for a, v in U.items()}
        Um_mean = {a: float(np.mean(v)) for a, v in Um.items()}
        minU = min((u for u, _ in U_mean.values()), default=float("nan"))
        minUm = min(Um_mean.values(), default=float("nan"))

        # ---- report ------------------------------------------------------
        lam_str = "; ".join(
            f"{c}:pi={pis[c]:.3f}"
            for c in sorted(Lam, key=lambda c: -freq[c])[:6])
        u_str = " ".join(f"a{a}:{u:+.3f}(n={k})" for a, (u, k) in sorted(U_mean.items()))
        row = (f"| {name}-{mdl} | seed {seed} | Acc={acc:.3f} H(yhat)={H_yhat:.3f} "
               f"band={100*band:.0f}% "
               f"| |R|={np.mean(size_abs):.1f} ({100*np.mean(size_rel):.0f}% of n) "
               f"empty={100*empty_rate:.0f}% large={100*large_share:.0f}% "
               f"| |Lam|={len(Lam)} (dEff={delta_eff:.3f} cov={100*coverage:.0f}%) "
               f"| kappa={kappa:.3f} I(O;yhat)={mi_yhat:.3f} I(O;Y)={mi_y:.3f} "
               f"(pats={n_pat}) | minU={minU:+.3f} minUm={minUm:+.3f} "
               f"| {verdict} |")
        print("#R# " + row)
        print(f"#R#   Lambda: {lam_str}")
        print(f"#R#   U_a:    {u_str}")
        top_x = Counter()
        for (yp, c), k in xtab.items():
            top_x[(yp, c)] = k
        print("#R#   codebook (yhat -> top contents): " + "; ".join(
            f"y{yp}->{c} x{k}" for (yp, c), k in top_x.most_common(6)))
        rows.append(row)

        torch.save({"meta": {"dataset": name, "model": mdl, "seed": seed,
                             "split": split, "thr": thr, "delta": DELTA},
                    "acc": acc, "H_yhat": H_yhat, "empty_rate": empty_rate,
                    "band": band, "pmax": pmax,
                    "size_abs": np.array(size_abs),
                    "size_rel": np.array(size_rel),
                    "large_share": large_share, "Lambda": Lam, "pi": pis,
                    "kappa": kappa, "mi_yhat": mi_yhat, "mi_y": mi_y,
                    "n_pat": n_pat, "rho_dir": rho_dir, "rho_bound": rho_bound,
                    "verdict": verdict, "Obits": Obits, "yhat": yhat,
                    "ytrue": ytrue,
                    "U": {a: v for a, v in U.items()}, "U_mean": U_mean,
                    "Um_mean": Um_mean,
                    "freq": dict(freq), "xtab": dict(xtab)},
                   os.path.join(OUT, f"{name}_{mdl}_seed{seed}_{split}.pt"))

    print(f"\n#IN# done in {datetime.now() - t0}")
    print("#R# ===== E3-a certificate table =====")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
