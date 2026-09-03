#!/usr/bin/env python3
"""Thesis figure from the E3.2 freezing campaign.

Reads storage/freeze/*_{binary,raw}.pt (written by freeze_probe.py) and
produces fig_freeze.pdf, two panels:
  (a) halo law: on live graphs (loss-saturated graphs excluded), the
      classifier-pass input gradient is EXACTLY zero outside the display's
      L-hop halo and essentially never zero inside it, in every cell;
  (b) gate dichotomy (MUTAG, saturation-free): under the binary protocol
      display both models attenuate hard; under its own trained soft gate
      GSAT barely attenuates (ratio 0.6-0.9) while SMGNN keeps freezing.

Run from repo root:  python Thesis/figures/gen_fig_freeze.py
"""
import glob
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
FREEZE = os.path.join(ROOT, "storage", "freeze")

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10,
    "legend.fontsize": 8, "legend.frameon": False,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.15,
})

C_OUT = "#0072B2"    # Okabe-Ito blue: outside halo (frozen)
C_IN = "#D55E00"     # vermillion: inside halo (live)
C_GSAT = "#009E73"   # green
C_SMGNN = "#CC79A7"  # pink

CELLS = [
    ("MUTAG", "GSAT", "MUTAG–GSAT"),
    ("MUTAG", "SMGNN", "MUTAG–SMGNN"),
    ("BAColorGVIsolated", "GSAT", "RBGV–GSAT"),
    ("BAColorGVIsolated", "SMGNN", "RBGV–SMGNN"),
]


def load(mode):
    out = {}
    for f in glob.glob(os.path.join(FREEZE, f"*_{mode}.pt")):
        d = torch.load(f, weights_only=False)
        m = d["meta"]
        out[(m["dataset"], m["model"], m["seed"])] = d
    return out


def panel_halo(ax, binary):
    agg = defaultdict(lambda: [0, 0, 0, 0])  # oz, o, iz, ih
    for (ds, mdl, _seed), d in binary.items():
        h = d["halo"]
        a = agg[(ds, mdl)]
        a[0] += h["outh_zero"]
        a[1] += h["outh"]
        a[2] += h["inh_zero"]
        a[3] += h["inh"]
    ys = np.arange(len(CELLS))[::-1]
    for y, (ds, mdl, label) in zip(ys, CELLS):
        oz, o, iz, ih = agg[(ds, mdl)]
        p_out = 100.0 * oz / max(o, 1)
        p_in = 100.0 * iz / max(ih, 1)
        ax.plot([p_in, p_out], [y, y], color="#BBBBBB", lw=1.0, zorder=1)
        ax.plot([p_out], [y], "o", color=C_OUT, ms=6, zorder=3)
        ax.plot([p_in], [y], "o", color=C_IN, ms=6, zorder=3)
        ax.annotate(f"{oz:,}/{o:,}", (p_out, y), textcoords="offset points",
                    xytext=(-4, 7), ha="right", fontsize=7, color=C_OUT)
        ax.annotate(f"{iz:,}/{ih:,}", (p_in, y), textcoords="offset points",
                    xytext=(4, 7), ha="left", fontsize=7, color=C_IN)
    ax.set_yticks(ys)
    ax.set_yticklabels([c[2] for c in CELLS], fontsize=9)
    ax.set_xlim(-6, 106)
    ax.set_ylim(-0.7, len(CELLS) + 0.15)
    ax.set_xlabel("unselected nodes with exactly zero gradient (%)")
    ax.text(100, 3.78, "outside $L$-hop halo", ha="right", va="center",
            fontsize=8, color=C_OUT)
    ax.text(0, 3.78, "inside halo", ha="left", va="center",
            fontsize=8, color=C_IN)
    ax.set_title("(a) gradient support = display halo", fontsize=10)


def panel_ratio(ax, binary, raw):
    rows = [("GSAT", C_GSAT, 1), ("SMGNN", C_SMGNN, 0)]
    for mdl, color, y in rows:
        for mode, data, mfc in (("binary", binary, color), ("raw", raw, "none")):
            xs = [d["ratio"] for (ds, m, _s), d in data.items()
                  if ds == "MUTAG" and m == mdl]
            jit = 0.14 if mode == "raw" else -0.14
            ax.plot(xs, [y + jit] * len(xs), "o", color=color, mfc=mfc,
                    ms=5.5, markeredgewidth=1.2, linestyle="none")
    ax.axvline(1.0, color="#BBBBBB", lw=0.8, linestyle=":")
    ax.text(1.0, 1.62, "no attenuation", fontsize=7.5, color="#777777",
            ha="right", rotation=0)
    ax.set_xscale("log")
    ax.set_xlim(0.012, 1.6)
    ax.set_yticks([1, 0])
    ax.set_yticklabels(["MUTAG–GSAT", "MUTAG–SMGNN"], fontsize=9)
    ax.set_ylim(-0.6, 1.75)
    ax.set_xlabel("gradient ratio  unselected / selected")
    h = [plt.Line2D([], [], marker="o", linestyle="none", color="#555555",
                    ms=5.5, label="binary protocol display"),
         plt.Line2D([], [], marker="o", linestyle="none", color="#555555",
                    mfc="none", ms=5.5, markeredgewidth=1.2,
                    label="own trained gate (raw)")]
    ax.legend(handles=h, loc="lower left")
    ax.set_title("(b) hard display freezes; GSAT's soft gate does not",
                 fontsize=10)


def main():
    binary, raw = load("binary"), load("raw")
    assert binary and raw, f"no freeze files under {FREEZE}"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.3, 2.7),
                                   gridspec_kw={"width_ratios": [1.15, 1]})
    panel_halo(ax1, binary)
    panel_ratio(ax2, binary, raw)
    fig.tight_layout(w_pad=2.0)
    fig.savefig(os.path.join(HERE, "fig_freeze.pdf"))
    fig.savefig(os.path.join(HERE, "fig_freeze.png"), dpi=300)
    print("saved fig_freeze.pdf")


if __name__ == "__main__":
    main()
