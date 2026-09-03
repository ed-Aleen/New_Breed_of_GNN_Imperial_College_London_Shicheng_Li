#!/usr/bin/env python3
"""Thesis figure from the E2 shape-check campaign.

Reads storage/expl_shapes/*.pt (written by shape_check.py) and produces:
  fig_confinement.pdf - per-cell count of violated colour classes
                        (explanation not a union of WL colour classes),
                        aggregated over seeds. Level-respecting readers
                        (GSAT/SMGNN) sit at exactly 0; the top-K reader (DIR)
                        shows the predicted tie-splitting violations.

Run from repo root:  python Thesis/figures/gen_fig_confinement.py
"""
import glob
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SHAPES = os.path.join(ROOT, "storage", "expl_shapes")

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10, "axes.labelsize": 10, "legend.fontsize": 8.5,
    "legend.frameon": False, "figure.dpi": 300, "savefig.dpi": 300,
    "savefig.bbox": "tight", "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.15,
})

C_LEVEL = "#0072B2"   # Okabe-Ito blue: level-respecting readers
C_TOPK = "#D55E00"    # vermillion: top-K reader

CELL_ORDER = [
    ("MUTAG", "GSAT", "MUTAG–GSAT"),
    ("MUTAG", "SMGNN", "MUTAG–SMGNN"),
    ("BAColorGVIsolated", "GSAT", "RBGV–GSAT"),
    ("BAColorGVIsolated", "SMGNN", "RBGV–SMGNN"),
    ("BAColorGVIsolated", "DIR", "RBGV–DIR (top-K)"),
]


def main():
    agg = defaultdict(lambda: [0, 0])  # (ds, mdl) -> [violated, checked]
    for f in sorted(glob.glob(os.path.join(SHAPES, "*.pt"))):
        d = torch.load(f, weights_only=False)
        m, s = d["meta"], d["summary"]
        key = (m["dataset"], m["model"])
        agg[key][0] += int(round(s["viol_rate"] * s["n_cls_checked"]))
        agg[key][1] += int(s["n_cls_checked"])

    labels, viol, checked, colors = [], [], [], []
    for ds, mdl, label in CELL_ORDER:
        if (ds, mdl) not in agg:
            continue
        v, n = agg[(ds, mdl)]
        labels.append(label)
        viol.append(v)
        checked.append(n)
        colors.append(C_TOPK if mdl == "DIR" else C_LEVEL)

    fig, ax = plt.subplots(figsize=(3.05, 2.4))
    y = np.arange(len(labels))[::-1]
    ax.barh(y, viol, color=colors, height=0.55, edgecolor="white")
    for yi, v, n in zip(y, viol, checked):
        note = f"{v} / {n:,}"
        if v > 0:
            note += "  (all at exact score ties)"
        ax.text(max(v, 0.05) + 0.08, yi, note, va="center", fontsize=8,
                color="#444444")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("violated colour classes (count)")
    ax.set_xlim(0, max(max(viol), 1) * 3.2)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=C_LEVEL,
                      label="level-respecting reader"),
        plt.Rectangle((0, 0), 1, 1, color=C_TOPK, label="top-K reader"),
    ]
    ax.legend(handles=handles, loc="lower right")
    fig.savefig(os.path.join(HERE, "fig_confinement.pdf"))
    fig.savefig(os.path.join(HERE, "fig_confinement.png"), dpi=300)
    print("saved fig_confinement.pdf")
    for label, v, n in zip(labels, viol, checked):
        print(f"  {label}: {v}/{n}")


if __name__ == "__main__":
    main()
