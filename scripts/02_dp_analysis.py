"""
DP vacuousness analysis.
Produces: Table 1 (tab:eps0), Figure 1 (fig:dp_vacuousness)
Paper: Section 3.3 "The Shuffle DP Premise Cannot Hold"
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_FIG = os.path.join(os.path.dirname(__file__), "..", "outputs", "figures")
os.makedirs(OUT_FIG, exist_ok=True)

# ---- Table 1 (tab:eps0) ----
# Gaussian mechanism: eps_0 = Delta_f * sqrt(2 ln(1.25/delta_0)) / sigma
# With Delta_f = 1 (conservative), sigma = RLWE noise std
# delta_0 for Gaussian mechanism formula.
# Using 1e-10 (conservative, standard for crypto applications).
delta_0 = 1e-10
precisions = [
    ("8-bit",  256,    6.5e-4),
    ("12-bit", 4096,   4.1e-5),
    ("16-bit", 65536,  2.5e-6),
]

print("=== Table 1 (tab:eps0) ===")
print(f"{'Precision':>10s} {'p':>8s} {'noise_std':>12s} {'eps_0':>12s}")

eps0_values = []
for name, p, sigma in precisions:
    Delta_f = 1.0  # conservative
    eps_0 = Delta_f * np.sqrt(2 * np.log(1.25 / delta_0)) / sigma
    eps0_values.append(eps_0)
    print(f"{name:>10s} {p:>8d} {sigma:>12.2e} {eps_0:>12,.0f}")

# Threshold
n = 512
delta = 1e-5
threshold = np.log(n / (16 * np.log(2 / delta)))
print(f"\nThreshold (n={n}, delta={delta}): eps_0 <= {threshold:.2f}")
print(f"Bounded-support lower bound: eps_0 >= 2")

# ---- Figure 1 (fig:dp_vacuousness) ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# Left: eps_0 bar chart
names = [p[0] for p in precisions]
ax1.bar(names, eps0_values, color=["#4ECDC4", "#FF6B6B", "#45B7D1"])
ax1.axhline(y=1, color="green", linestyle="--", label="Strong privacy (eps=1)")
ax1.axhline(y=10, color="orange", linestyle="--", label="Vacuous (eps=10)")
ax1.set_ylabel("Per-query eps_0")
ax1.set_yscale("log")
ax1.set_title("Local DP parameter (Delta_f=1)")
ax1.legend(fontsize=8)

# Right: threshold diagram
n_values = np.arange(16, 1025)
thresholds = np.log(n_values / (16 * np.log(2 / delta)))
ax2.fill_between(n_values, 0, thresholds, alpha=0.2, color="blue", label="Amplification applies")
ax2.plot(n_values, thresholds, "b-", linewidth=2)
ax2.axhline(y=2, color="red", linestyle="--", linewidth=2, label="Bounded-support lower bound (eps_0>=2)")
ax2.axhline(y=threshold, color="blue", linestyle=":", alpha=0.5, label=f"n=512: threshold={threshold:.2f}")
ax2.set_xlabel("Output dimension n")
ax2.set_ylabel("Maximum eps_0 for amplification")
ax2.set_title("Shuffling amplification threshold")
ax2.legend(fontsize=8)
ax2.set_ylim(0, 4)

plt.tight_layout()
path = os.path.join(OUT_FIG, "fig_dp_vacuousness.png")
plt.savefig(path, dpi=200)
print("\nSaved outputs/figures/fig_dp_vacuousness.png")
