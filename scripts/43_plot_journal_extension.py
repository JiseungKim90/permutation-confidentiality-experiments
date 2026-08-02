"""Create the journal-extension summary figure from canonical JSON outputs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PAPER_ROOT = ROOT.parent


def main():
    with (ROOT / "outputs" / "mixed_query_collision_stress.json").open(
        encoding="utf-8"
    ) as handle:
        stress = json.load(handle)

    settings = stress["settings"]
    labels = ["{1}", "{1,2}", "{1,2,3}"]
    exact = np.asarray([row["exact"] for row in settings], dtype=float)
    ambiguous = np.asarray([row["ambiguous"] for row in settings], dtype=float)
    trials = float(stress["trials"])

    plt.rcParams.update({
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "legend.fontsize": 7,
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.55))

    x = np.arange(len(labels))
    axes[0].bar(x, exact / trials * 100, width=0.62,
                color="#3b78b5", label="Exact")
    axes[0].bar(x, ambiguous / trials * 100, width=0.62,
                bottom=exact / trials * 100, color="#b7bcc5", label="Ambiguous")
    axes[0].set_xticks(x, labels)
    axes[0].set_xlabel("Mixed-query coefficient set $\\Lambda$")
    axes[0].set_ylabel(r"Trials (\%)")
    axes[0].set_ylim(0, 105)
    axes[0].set_title("(a) Collision-heavy 8x8 layers")
    axes[0].legend(frameon=False, loc="center left")
    axes[0].grid(axis="y", color="#dddddd", linewidth=0.5)

    d = np.arange(1, 577)
    axes[1].plot(d, d + 1, color="#777777", linewidth=1.5,
                 label="Spectrum: $d+1$")
    axes[1].plot(d, 1 + 3 * d, color="#3b78b5", linewidth=1.5,
                 label="Bias anchor: $1+3d$")
    axes[1].plot(d, 1 + 4 * (d - 1), color="#b05a47", linewidth=1.5,
                 label="Bias-free: $4d-3$")
    axes[1].scatter([27, 27], [82, 105], color=["#3b78b5", "#b05a47"],
                    marker="o", s=18, zorder=3)
    axes[1].annotate("ResNet-20 conv1\n82 / 105 queries", xy=(27, 105),
                     xytext=(105, 350), arrowprops={"arrowstyle": "->", "lw": 0.7},
                     fontsize=7)
    axes[1].set_xlabel("Layer input dimension $d$")
    axes[1].set_ylabel("Queries")
    axes[1].set_xlim(0, 576)
    axes[1].set_ylim(0, 2400)
    axes[1].set_title("(b) Exact-recovery query cost")
    axes[1].legend(frameon=False, loc="upper left")
    axes[1].grid(color="#dddddd", linewidth=0.5)

    fig.tight_layout(w_pad=2.0)
    for suffix in ("pdf", "png"):
        fig.savefig(PAPER_ROOT / f"fig_journal_extension.{suffix}",
                    dpi=220, bbox_inches="tight")
    print(PAPER_ROOT / "fig_journal_extension.pdf")


if __name__ == "__main__":
    main()
