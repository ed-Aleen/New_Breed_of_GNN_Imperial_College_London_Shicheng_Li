#!/usr/bin/env python3
"""Thesis figures from the E3.1 certificate campaign.

Reads storage/certificates/*.pt (written by certificate.py v2) and produces:
  fig_occinfo.pdf   - occurrence-information strip plot across cells:
                      x = I(O; yhat) / H(yhat); degenerate cells cluster near 0,
                      the watermark (faithful-shortcut) control near 1.
  fig_codebook.pdf  - RBGV-DIR codebook cross-tabs (yhat x displayed node type)
                      for the four live seeds: three different diagonal-block
                      codebooks incl. one orientation flip.

Run from repo root:  python Thesis/figures/gen_fig_certificate.py
"""
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
CERTS = os.path.join(ROOT, "storage", "certificates")

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10,
    "legend.fontsize": 8.5, "legend.frameon": False,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.15,
})
FIG_FULL = (6.3, 3.0)

# Okabe-Ito
C_JUDGED = "#D55E00"   # vermillion: judged degenerate-regime
C_NONDEG = "#009E73"   # green: certified non-degenerate
C_ABST = "#8C8C8C"     # gray: abstain
C_DEAD = "#2C2C2C"     # dark: dead model

CELL_ORDER = [
    ("BAColorGVIsolated", "DIR", "RBGV–DIR (K=1%)"),
    ("BAColorGVIsolated", "SMGNN", "RBGV–SMGNN"),
    ("BAColorGVIsolated", "GSAT", "RBGV–GSAT"),
    ("MUTAG", "GSAT", "MUTAG–GSAT"),
    ("MUTAG", "SMGNN", "MUTAG–SMGNN"),
    ("BenzeneWatermarkBalanced", "DIR", "WM-balanced–DIR"),
    ("BenzeneWatermarkConfounded", "DIR", "WM-confounded–DIR"),
]


def load_all():
    rows = []
    for f in sorted(glob.glob(os.path.join(CERTS, "*.pt"))):
        d = torch.load(f, weights_only=False)
        if "mi_yhat" not in d:
            continue
        m = d["meta"]
        rows.append(d | {"dataset": m["dataset"], "model": m["model"],
                         "seed": m["seed"]})
    return rows


def verdict_class(d):
    if d["H_yhat"] < 0.1:
        return "DEAD"
    if d["mi_yhat"] >= 0.5 * d["H_yhat"]:
        return "NONDEG"
    if d["large_share"] > 0.5:
        return "ABSTAIN"
    return "JUDGED"


def fig_occinfo(rows):
    fig, ax = plt.subplots(figsize=FIG_FULL)
    yticks, ylabels = [], []
    style = {
        "JUDGED": dict(marker="o", color=C_JUDGED, mfc=C_JUDGED),
        "NONDEG": dict(marker="s", color=C_NONDEG, mfc=C_NONDEG),
        "ABSTAIN": dict(marker="o", color=C_ABST, mfc="none"),
        "DEAD": dict(marker="x", color=C_DEAD, mfc=C_DEAD),
    }
    for i, (ds, mdl, label) in enumerate(CELL_ORDER):
        y = len(CELL_ORDER) - 1 - i
        yticks.append(y)
        ylabels.append(label)
        cell = [r for r in rows if r["dataset"] == ds and r["model"] == mdl]
        for r in sorted(cell, key=lambda r: r["seed"]):
            v = verdict_class(r)
            x = 0.0 if v == "DEAD" else min(r["mi_yhat"] / max(r["H_yhat"], 1e-9), 1.02)
            jit = (r["seed"] - 3) * 0.09
            ax.plot([x], [y + jit], linestyle="none", markersize=5.5,
                    markeredgewidth=1.2, **style[v])
    # separator between natural cells and watermark controls
    ax.axhline(1.5, color="#BBBBBB", linewidth=0.8, linestyle=(0, (4, 3)))
    ax.axvline(0.5, color="#BBBBBB", linewidth=0.8, linestyle=":")
    ax.text(0.03, len(CELL_ORDER) - 0.45, "selection-coded\n(degenerate regime)",
            fontsize=8, color="#555555", style="italic", va="top")
    ax.text(0.99, len(CELL_ORDER) - 0.45, "occurrence-coded\n(faithful shortcut)",
            fontsize=8, color="#555555", style="italic", va="top", ha="right")
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.set_xlim(-0.04, 1.06)
    ax.set_ylim(-0.6, len(CELL_ORDER) - 0.2)
    ax.set_xlabel(r"occurrence information  $\hat I(O;\hat y)\,/\,\hat H(\hat y)$")
    handles = [
        plt.Line2D([], [], linestyle="none", markersize=5.5, markeredgewidth=1.2,
                   **style["JUDGED"], label="judged (degenerate regime)"),
        plt.Line2D([], [], linestyle="none", markersize=5.5, markeredgewidth=1.2,
                   **style["NONDEG"], label="certified non-degenerate"),
        plt.Line2D([], [], linestyle="none", markersize=5.5, markeredgewidth=1.2,
                   **style["ABSTAIN"], label="abstain (large display)"),
        plt.Line2D([], [], linestyle="none", markersize=5.5, markeredgewidth=1.2,
                   **style["DEAD"], label="dead model (H($\\hat y$)$\\approx$0)"),
    ]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(0.24, 0.42),
              handletextpad=0.4, borderaxespad=0.0)
    fig.savefig(os.path.join(HERE, "fig_occinfo.pdf"))
    fig.savefig(os.path.join(HERE, "fig_occinfo.png"), dpi=300)
    print("saved fig_occinfo.pdf")


ATTR_NAMES = ["red", "blue", "green\n(iso)", "violet\n(iso)"]
ATTR_TICK_COLORS = ["#C0392B", "#2471A3", "#1E8449", "#7D3C98"]


def fig_codebook(rows):
    seeds = [1, 3, 4, 5]  # live RBGV-DIR seeds (seed 2 is DEAD)
    cells = {r["seed"]: r for r in rows
             if r["dataset"] == "BAColorGVIsolated" and r["model"] == "DIR"}
    fig, axes = plt.subplots(1, len(seeds), figsize=(6.3, 1.9), sharey=True)
    for ax, s in zip(axes, seeds):
        d = cells[s]
        M = np.zeros((2, 4))
        other = 0
        for (yp, c), k in d["xtab"].items():
            if isinstance(c, tuple) and len(c) == 3 and c[0] == 1:
                M[int(yp), int(c[1][0])] += k
            else:
                other += k
        im = ax.imshow(M, cmap="Blues", aspect="auto",
                       vmin=0, vmax=max(M.max(), 1))
        for i in range(2):
            for j in range(4):
                if M[i, j] > 0:
                    ax.text(j, i, int(M[i, j]), ha="center", va="center",
                            fontsize=8,
                            color="white" if M[i, j] > 0.6 * M.max() else "#333333")
        ax.set_xticks(range(4))
        ax.set_xticklabels(ATTR_NAMES, fontsize=8)
        for tick, col in zip(ax.get_xticklabels(), ATTR_TICK_COLORS):
            tick.set_color(col)
        ax.set_yticks([0, 1])
        ax.set_yticklabels([r"$\hat y=0$", r"$\hat y=1$"])
        cov = 100.0 * M.sum() / max(M.sum() + other, 1)
        ax.set_title(f"seed {s}  ({cov:.0f}% single-node)", fontsize=9)
        ax.grid(False)
    fig.suptitle("RBGV–DIR displayed node type vs. prediction "
                 "(three codebooks, one orientation flip)", fontsize=10, y=1.06)
    fig.savefig(os.path.join(HERE, "fig_codebook.pdf"))
    fig.savefig(os.path.join(HERE, "fig_codebook.png"), dpi=300)
    print("saved fig_codebook.pdf")


def main():
    rows = load_all()
    assert rows, f"no certificate files under {CERTS}"
    print(f"loaded {len(rows)} certificate rows")
    fig_occinfo(rows)
    fig_codebook(rows)


if __name__ == "__main__":
    main()
