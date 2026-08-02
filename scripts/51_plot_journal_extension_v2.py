"""Plot deterministic first-layer and fresh-session orbit recovery.

This script reads only canonical JSON outputs produced by server experiments.
It does not recompute any attack result.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PAPER_ROOT = ROOT.parent


def load(name: str) -> dict:
    with (ROOT / "outputs" / name).open(encoding="utf-8") as handle:
        return json.load(handle)


def load_log(name: str) -> dict:
    with (ROOT / "outputs" / "logs" / name).open(encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    interpolation = load("interpolation_recovery.json")
    fresh4 = load("fresh_session_orbit_recovery_input15.json")
    fresh5 = load("fresh_session_orbit_recovery_input31.json")
    fresh8 = load("fresh_session_orbit_recovery_input255.json")
    fresh_tfhe = load_log("58_tfhe_fresh_session_orbit_recovery.json")

    stress = next(
        row for row in interpolation["collision_stress"] if row["shape"] == [8, 8]
    )
    trials = float(stress["trials"])
    exact = np.asarray(
        [stress["practical_exact"], stress["interpolation_exact"]], dtype=float
    )
    ambiguous = np.asarray([stress["practical_ambiguous"], 0.0], dtype=float)
    wrong = np.asarray(
        [stress["practical_wrong"], stress["interpolation_wrong"]], dtype=float
    )

    verified = np.asarray(
        [
            100.0 * fresh4["summary"]["all_trials_exact_layer_count"] / 21.0,
            100.0 * fresh5["summary"]["all_trials_exact_layer_count"] / 21.0,
            100.0 * fresh8["summary"]["all_trials_exact_layer_count"] / 21.0,
            100.0 * fresh8["summary"]["classifier_exact_trials"]
            / fresh8["summary"]["classifier_total_trials"],
            100.0 if fresh_tfhe["status"] == "exact_orbit" else 0.0,
        ]
    )

    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "legend.fontsize": 7,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.65))

    x = np.arange(2)
    axes[0].bar(x, exact / trials * 100, width=0.58, color="#3976b3", label="Exact")
    axes[0].bar(
        x,
        ambiguous / trials * 100,
        width=0.58,
        bottom=exact / trials * 100,
        color="#b7bcc5",
        label="Ambiguous",
    )
    axes[0].bar(
        x,
        wrong / trials * 100,
        width=0.58,
        bottom=(exact + ambiguous) / trials * 100,
        color="#b04a3f",
        label="Wrong",
    )
    axes[0].set_xticks(x, ["$\\{1,2,3\\}$\nsolver", "Interpolation\n$k=0,\\ldots,m$"])
    axes[0].set_ylabel("Trials (%)")
    axes[0].set_ylim(0, 105)
    axes[0].set_title("(a) Collision-heavy 8x8 layers")
    axes[0].legend(frameon=False, loc="lower right")
    axes[0].grid(axis="y", color="#dddddd", linewidth=0.5)
    axes[0].text(0, 95, "99 exact\n1 ambiguous", ha="center", va="top", fontsize=7)
    axes[0].text(1, 95, "100 exact", ha="center", va="top", fontsize=7)

    labels = [
        "Local conv\n4-bit",
        "Local conv\n5-bit",
        "Local conv\n8-bit",
        "Full FC\n8-bit",
        "TFHE operator\n8-bit",
    ]
    colors = ["#86b874", "#5f9e68", "#397b58", "#6f69a8", "#3976b3"]
    bx = np.arange(len(labels))
    axes[1].bar(bx, verified, width=0.62, color=colors)
    axes[1].set_xticks(bx, labels, fontsize=6.8)
    axes[1].set_ylabel("Verified cases (%)")
    axes[1].set_ylim(0, 108)
    axes[1].set_title("(b) Fresh-session layer-orbit recovery")
    axes[1].grid(axis="y", color="#dddddd", linewidth=0.5)
    denominators = ["3/21", "5/21", "15/21", "3/3", "1/1"]
    for pos, value, note in zip(bx, verified, denominators):
        axes[1].text(pos, value + 1.5, note, ha="center", va="bottom", fontsize=7)

    fig.tight_layout(w_pad=1.8)
    for suffix in ("pdf", "png"):
        fig.savefig(
            PAPER_ROOT / f"fig_journal_extension_v2.{suffix}",
            dpi=240,
            bbox_inches="tight",
        )
    print(PAPER_ROOT / "fig_journal_extension_v2.pdf")


if __name__ == "__main__":
    main()
