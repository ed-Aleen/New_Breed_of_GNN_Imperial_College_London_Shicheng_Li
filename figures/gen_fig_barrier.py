#!/usr/bin/env python3
"""Thesis figure from the E3.3 barrier campaign.

Reads storage/barrier/*.pt (written by barrier_scan.py) and produces
fig_barrier.pdf: for RBGV-SMGNN seed 1 and RBGV-DIR seed 3, the fit
objective J while the display level t of an alternative attribute is swept
from 0 to 1 with the trained classifier FROZEN (four attribute curves), vs
the same sweep after re-fitting only the classifier head (dashed).
Signal attributes (red/blue) raise steep walls; isolated context attributes
(green/violet) are flat; the head re-fit removes the wall almost entirely,
so the frozen-vs-refit gap lower-bounds the specialization gain Sp.

Run from repo root:  python Thesis/figures/gen_fig_barrier.py
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
BARR = os.path.join(ROOT, "storage", "barrier")

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10,
    "legend.fontsize": 8, "legend.frameon": False,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.15,
})

ATTR = {0: ("red", "#C0392B"), 1: ("blue", "#2471A3"),
        2: ("green (iso)", "#1E8449"), 3: ("violet (iso)", "#7D3C98")}
MARK = {0: "o", 1: "s", 2: "^", 3: "v"}


def draw(ax, fname, title, scale=1.0, unit="nats", sp_frac=0.5,
         ymax=None):
    d = torch.load(os.path.join(BARR, fname), weights_only=False)
    ts = np.array(d["ts"])
    for a in sorted(d["frozen"]):
        name, col = ATTR[a]
        ax.plot(ts, np.array(d["frozen"][a]) / scale, color=col,
                marker=MARK[a], ms=3.5, lw=1.6,
                label=rf"frozen, $\beta$={name}")
    beta = d["meta"]["beta"]
    r = np.array(d["retrained"]) / scale
    ax.plot(ts, r, color="#2C2C2C", linestyle="--", marker="o", mfc="none",
            ms=3.5, lw=1.4,
            label=rf"head re-fit, $\beta$={ATTR[beta][0]}")
    f1 = d["frozen"][beta][-1] / scale
    sp = d["sp_lower"][-1]
    ax.annotate("", xy=(1.0, r[-1]), xytext=(1.0, f1),
                arrowprops=dict(arrowstyle="->", color="#555555", lw=1.0))
    ax.annotate(rf"Sp $\geq$ {sp:,.0f} nats",
                xy=(0.985, r[-1] + sp_frac * (f1 - r[-1])),
                ha="right", va="center", fontsize=8.5, color="#333333")
    ax.set_xlabel(r"display level $t$ of alternative attribute $\beta$")
    ax.set_ylabel(f"fit objective $J$ ({unit})")
    ax.set_xlim(-0.03, 1.03)
    ax.set_title(title, fontsize=10)
    if ymax is not None:
        ax.set_ylim(top=ymax)
    ax.legend(loc="upper left", handlelength=1.5, labelspacing=0.2,
              fontsize=7.5)


def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.3, 2.8))
    draw(ax1, "BAColorGVIsolated_SMGNN_seed1_train.pt",
         "(a) RBGV–SMGNN (seed 1)", ymax=205)
    draw(ax2, "BAColorGVIsolated_DIR_seed3_train.pt",
         "(b) RBGV–DIR $K$=1% (seed 3)", scale=1e3,
         unit=r"$\times 10^{3}$ nats", sp_frac=0.60)
    fig.tight_layout(w_pad=2.0)
    fig.savefig(os.path.join(HERE, "fig_barrier.pdf"))
    fig.savefig(os.path.join(HERE, "fig_barrier.png"), dpi=300)
    print("saved fig_barrier.pdf")


if __name__ == "__main__":
    main()
