r"""E-OCC0 pre-registration (zero training): the evidence-availability ceiling.

Theorem OCC-2: under OCC, Acc <= P[ exists content c in Sel(G): b0(c) = Y ].
This script computes, BEFORE any training, a pre-registered LOWER bound of
that ceiling by enumerating contents up to union size m (single colour
classes and pairs), estimating b0 from the TRAIN contingency table, and
scoring the existence probability on train / id_val / id_test.

Content identity uses LOCAL (unenriched) colour refinement at radius
r in {0,1,2}: contents must be cross-graph-comparable subgraph types (a
broadcast-enriched colour hashes the whole-graph summary into every node,
so keys almost never repeat across graphs -- the first run measured exactly
that artefact: MUTAG ceiling ~1%, RBGV contents kept = 0). The mu0 sampler
in occ_stage_a.py keeps the ENRICHED within-graph granularity (Lemma C);
only the cross-graph contingency table lives at local granularity.

Binary-OCCURRENCE semantics: the ceiling bounds strategies that display a
content type or not. The class-level display exposes multiplicities to a
sum-pool classifier, so the OCC-2' counting channel can legally exceed this
bound -- on count-defined labels (RBGV: y = [#red >= #blue]) expect exactly
that. Only meaningful for discrete cells (MUTAG, RBGV); continuous cells get
a printed refusal.

Usage:
  python occ_ceiling.py --config_path final_configs/MUTAG/basis/no_shift/GSAT_OCC.yaml \
      --seeds 1 --task test --backbone ACR2 --gpu_idx 0
Extras: --min_support 10 (drop contents seen in fewer train graphs)
        --pair_cap 60 (skip pair enumeration for graphs with more classes)
"""
import os
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime
from itertools import combinations

import torch


def pop_flag(argv, name, default):
    if name in argv:
        i = argv.index(name)
        v = argv[i + 1]
        del argv[i:i + 2]
        return v
    return default


MIN_SUP = int(pop_flag(sys.argv, "--min_support", "10"))
PAIR_CAP = int(pop_flag(sys.argv, "--pair_cap", "60"))
OUT = pop_flag(sys.argv, "--out", "storage/occ_prereg")

import os as _os, sys as _sys                                  # noqa: E402
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from GOOD import config_summoner                            # noqa: E402
from GOOD.utils.args import args_parser                     # noqa: E402
from GOOD.utils.loader import initialize_model_dataset      # noqa: E402
from GOOD.utils.logger import load_logger                   # noqa: E402
from GOOD.utils.occ import (initial_colour_keys, refine_colour_keys,  # noqa: E402
                            detect_granularity)


def graph_class_set(g, radius):
    """LOCAL content types: unenriched colour refinement, cross-graph stable."""
    batch = torch.zeros(g.x.shape[0], dtype=torch.long)
    keys = refine_colour_keys(initial_colour_keys(g.x), g.edge_index, batch,
                              rounds=radius, enrich=False)
    return set(keys.tolist())


def contents_of(classes):
    singles = [(c,) for c in sorted(classes)]
    if len(classes) <= PAIR_CAP:
        pairs = list(combinations(sorted(classes), 2))
    else:
        pairs = []
    return singles + pairs


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
    os.makedirs(OUT, exist_ok=True)

    first = loader["train"].dataset[0]
    if detect_granularity(first.x) != "class":
        print(f"#R# | ceiling | {name} | REFUSED: continuous features -- "
              "colour-class contents undefined; E-OCC2 stays qualitative here")
        return

    save = {}
    for radius in (0, 1, 2):
        cnt = defaultdict(Counter)
        cls_sets = {}
        ys = {}
        for split in ("train", "id_val", "id_test"):
            for gi, g in enumerate(loader[split].dataset):
                cs = graph_class_set(g, radius)
                cls_sets[(split, gi)] = cs
                ys[(split, gi)] = int(g.y.view(-1)[0])
                if split == "train":
                    for c in contents_of(cs):
                        cnt[c][ys[(split, gi)]] += 1

        def b0_from(label_of):
            c2 = defaultdict(Counter)
            for k in ys:
                if k[0] == "train":
                    for c in contents_of(cls_sets[k]):
                        c2[c][label_of[k]] += 1
            return {c: max(cc.items(), key=lambda kv: kv[1])[0]
                    for c, cc in c2.items() if sum(cc.values()) >= MIN_SUP}

        def ceil_split(b0map, split, m):
            keys = [k for k in ys if k[0] == split]
            hit = sum(1 for k in keys
                      if any(b0map.get(c) == ys[k]
                             for c in contents_of(cls_sets[k]) if len(c) <= m))
            return hit / max(len(keys), 1)

        b0 = {c: max(cc.items(), key=lambda kv: kv[1])[0]
              for c, cc in cnt.items() if sum(cc.values()) >= MIN_SUP}
        ytr = Counter(ys[k] for k in ys if k[0] == "train")
        maj = max(ytr.items(), key=lambda kv: kv[1])[0]
        pmaj = {s: sum(1 for k in ys if k[0] == s and ys[k] == maj) /
                   max(sum(1 for k in ys if k[0] == s), 1)
                for s in ("train", "id_val", "id_test")}

        # permutation null: same estimator on label-shuffled train -- controls
        # the existential-quantifier union inflation. ceiling ~ null at a
        # granularity means the number is vacuous there.
        tr_keys = [k for k in ys if k[0] == "train"]
        nulls = {1: [], 2: []}
        for perm in range(3):
            rng = random.Random(1000 + perm)
            lab = [ys[k] for k in tr_keys]
            rng.shuffle(lab)
            label_of = dict(zip(tr_keys, lab))
            b0p = b0_from(label_of)
            b0p_eval = {c: v for c, v in b0p.items()}
            for m in (1, 2):
                keys = [k for k in ys if k[0] == "id_val"]
                hit = sum(1 for k in keys
                          if any(b0p_eval.get(c) == ys[k]
                                 for c in contents_of(cls_sets[k])
                                 if len(c) <= m))
                nulls[m].append(hit / max(len(keys), 1))

        for m in (1, 2):
            row = [ceil_split(b0, sp, m)
                   for sp in ("train", "id_val", "id_test")]
            nl = nulls[m]
            print(f"#R# | ceiling | {name} | r={radius} m<={m} | "
                  f"train={row[0]:.4f} id_val={row[1]:.4f} "
                  f"id_test={row[2]:.4f} "
                  f"| NULL(idval)={sum(nl)/len(nl):.4f} "
                  f"(perms {min(nl):.3f}-{max(nl):.3f}) "
                  f"| majority: {pmaj['id_val']:.4f} "
                  f"| contents kept={len(b0)} (min_support={MIN_SUP}) |")
        save[radius] = {"b0": b0, "null_idval": nulls,
                        "counts": {c: dict(cc) for c, cc in cnt.items()}}
    print(f"#IN# note: binary-occurrence bound; the OCC-2' counting channel "
          f"(class display exposes multiplicities to sum-pool) may legally "
          f"exceed it on count-defined labels. Register a (r,m) cell only "
          f"where ceiling >> NULL; ceiling ~ NULL means the existential "
          f"union saturates at that granularity (vacuous).")
    torch.save({"radii": save, "min_support": MIN_SUP},
               os.path.join(OUT, f"{name}_prereg.pt"))
    print(f"#IN# saved {OUT}/{name}_prereg.pt in {datetime.now() - t0}")


if __name__ == "__main__":
    main()
