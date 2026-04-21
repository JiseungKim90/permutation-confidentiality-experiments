"""
Regenerate fig_phase_transition.pdf with fine-grained noise sweep data.

Sources:
  R56 CIFAR-10 (fine, N=100): outputs/logs/20b_fine_noise_r56_n100.log
  Imagenette ResNet-50:        outputs/logs/33_imagenet_tradeoff.log  (5-pt)
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── data ──────────────────────────────────────────────────────────────────────
# R56 CIFAR-10 fine sweep (N=100, 9 noise multipliers)
r56_nm  = [1, 10, 100, 200, 300, 500, 700, 1000, 5000]
r56_agr = [99.98, 99.84, 98.17, 95.95, 93.31, 86.69, 76.90, 51.84, 9.64]
r56_err = [0.00e+00, 1.33e-02, 1.06e-01, 1.82e-01, 2.67e-01,
           4.52e-01, 6.54e-01, 1.01e+00, 6.34e+00]
r56_acc = [82.22, 82.20, 82.18, 81.76, 80.83, 77.41, 70.49, 48.97, 10.00]

# Imagenette ResNet-50 (5 points — no fine-grained data collected)
inet_nm  = [1, 10, 100, 1000, 5000]
inet_agr = [100.0, 99.9, 99.2, 92.4, 62.0]
inet_err = [0.0, 1.29e-02, 9.44e-02, 1.45e+00, 8.12e+00]

# ── figure ────────────────────────────────────────────────────────────────────
fig, ax1 = plt.subplots(figsize=(5.5, 3.4))
ax2 = ax1.twinx()

# shaded "no operating point" band: atk_err < gamma (1/256 ≈ 0.004) is trivially broken
# Visually: region where atk_err ≥ 1/256 but pred_agr still acceptable
# Mark region between 300x and 1000x as the contested zone
ax1.axvspan(300, 1000, alpha=0.10, color="gray", label="_nolegend_")
ax1.text(550, 72, "no viable\noperating\npoint", fontsize=6.5,
         ha="center", va="center", color="gray", style="italic")

# Prediction agreement curves (left axis)
ax1.semilogx(r56_nm, r56_agr, "o-", color="#2166ac", lw=1.6, ms=4,
             label="R56 CIFAR-10 pred. agr. (fine)")
ax1.semilogx(inet_nm, inet_agr, "s--", color="#4dac26", lw=1.4, ms=4,
             label="R50 Imagenette pred. agr.")

# Attack error (right axis, log scale)
ax2.loglog(r56_nm, [max(e, 1e-4) for e in r56_err], "^-",
           color="#d6604d", lw=1.4, ms=3.5, label="R56 atk error")
ax2.loglog(inet_nm, [max(e, 1e-4) for e in inet_err], "^--",
           color="#b8860b", lw=1.2, ms=3.5, label="R50 atk error")

# 1/256 threshold line (gamma for 8-bit)
ax2.axhline(1/512, color="#d6604d", lw=0.8, ls=":", alpha=0.7)
ax2.text(1.2, 1/512 * 1.3, r"$\gamma/2=1/512$", fontsize=6,
         color="#d6604d", va="bottom")

# axes labels and formatting
ax1.set_xlabel("Noise multiplier", fontsize=9)
ax1.set_ylabel("Prediction agreement (%)", fontsize=9)
ax2.set_ylabel("Attack error ($\\ell_\\infty$)", fontsize=9)

ax1.set_ylim(0, 105)
ax1.set_xlim(0.8, 8000)
ax2.set_ylim(1e-4, 30)
ax1.xaxis.set_major_formatter(mticker.FuncFormatter(
    lambda x, _: f"{int(x):,}×" if x >= 1 else f"{x}×"
))

ax1.tick_params(axis="both", labelsize=7.5)
ax2.tick_params(axis="both", labelsize=7.5)

# combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2,
           fontsize=6.5, loc="lower left", framealpha=0.85, ncol=1)

plt.tight_layout(pad=0.4)

out_dir = os.path.join(os.path.dirname(__file__), "..",
                       "outputs", "figures")
os.makedirs(out_dir, exist_ok=True)
out_fig = os.path.join(out_dir, "fig_phase_transition.pdf")
plt.savefig(out_fig, dpi=300, bbox_inches="tight")
print(f"[done] saved {out_fig}")

# also save to Overleaf P050 Current/
overleaf = (r"C:\Users\CS615\Dropbox\앱\Overleaf"
            r"\P050-Impossibility of Permutation-Based Model Confidentiality"
            r"\Current\fig_phase_transition.pdf")
try:
    plt.savefig(overleaf, dpi=300, bbox_inches="tight")
    print(f"[done] saved {overleaf}")
except Exception as e:
    print(f"[warn] could not save to Overleaf: {e}", file=sys.stderr)
