#!/usr/bin/env python3
"""
Reproduce hunter–prey training plots from log.txt only.

- Expects a file called 'log.txt' in the current directory.
- Produces three PDFs in ./figures:
    - values.pdf
    - nash_gap.pdf
    - hunter_right_prob.pdf

Curves are labeled as "vanilla" (no momentum) and "acceleration" (with momentum),
but plot titles do not mention these labels.
"""

import os
import re

import numpy as np
import matplotlib.pyplot as plt


def main():
    # ------------------------------------------------------------
    # 1) Load entire log
    # ------------------------------------------------------------
    with open("log.txt", "r") as f:
        text = f.read()

    # ------------------------------------------------------------
    # 2) Parse per-iteration lines: Iter k: V_hat= ... | Gap= ... |
    # ------------------------------------------------------------
    iter_pattern = re.compile(
        r"Iter\s+(\d+):\s+V_hat=\s*([0-9\.\-]+)\s*\|\s*Gap=\s*([0-9\.\-]+)\s*\|"
    )
    iter_matches = iter_pattern.findall(text)
    if not iter_matches:
        raise RuntimeError("No 'Iter ... V_hat= ... | Gap= ... |' lines found in log.txt.")

    # Convert to arrays
    V_hat_all = np.array([float(v) for _, v, _ in iter_matches], dtype=float)
    Gap_all = np.array([float(g) for _, _, g in iter_matches], dtype=float)

    # We have two runs: vanilla + acceleration
    K_total = len(V_hat_all)
    if K_total % 2 != 0:
        raise RuntimeError(f"Expected an even number of iterations (two runs), got {K_total}.")
    K_per_run = K_total // 2

    V_hat_vanilla = V_hat_all[:K_per_run]
    Gap_vanilla = Gap_all[:K_per_run]

    V_hat_accel = V_hat_all[K_per_run:]
    Gap_accel = Gap_all[K_per_run:]

    # ------------------------------------------------------------
    # 3) Parse hunter action probabilities to get P(Right)
    #    Lines look like:
    #    "   Hunter: S:0.17 U:0.17 D:0.25 L:0.17 R:0.23"
    # ------------------------------------------------------------
    hunter_pattern = re.compile(
        r"Hunter:\s*S:([0-9\.]+)\s*U:([0-9\.]+)\s*D:([0-9\.]+)\s*L:([0-9\.]+)\s*R:([0-9\.]+)"
    )
    hunter_matches = hunter_pattern.findall(text)
    if len(hunter_matches) != K_total:
        raise RuntimeError(
            f"Expected {K_total} 'Hunter:' lines (one per iteration), "
            f"but found {len(hunter_matches)}."
        )

    # Extract only the R (Right) probabilities
    R_all = np.array([float(r[-1]) for r in hunter_matches], dtype=float)
    R_vanilla = R_all[:K_per_run]
    R_accel = R_all[K_per_run:]

    # ------------------------------------------------------------
    # 4) Parse the true Nash value V*(start) from log
    #    We look for "V*(start) ≈ 17.551020" style text.
    # ------------------------------------------------------------
    vstar_match = re.search(r"V\*\(start\)\s*≈\s*([0-9\.]+)", text)
    if not vstar_match:
        raise RuntimeError("Could not find 'V*(start) ≈ ...' in log.txt.")
    V_star = float(vstar_match.group(1))

    # Best-response value = V* - gap
    BR_vanilla = V_star - Gap_vanilla
    BR_accel = V_star - Gap_accel

    # X-axis (outer iterations)
    iters_vanilla = np.arange(1, K_per_run + 1)
    iters_accel = np.arange(1, K_per_run + 1)

    # ------------------------------------------------------------
    # 5) Make output directory
    # ------------------------------------------------------------
    os.makedirs("figures", exist_ok=True)

    # ------------------------------------------------------------
    # 6) Plot 1: Values (V_hat and V_x,BR) + V* line
    # ------------------------------------------------------------
    plt.figure(figsize=(6, 4))
    plt.plot(
        iters_vanilla,
        V_hat_vanilla,
        label="V_hat (vanilla)",
        marker="o",
        markersize=3,
        alpha=0.7,
    )
    plt.plot(
        iters_vanilla,
        BR_vanilla,
        label="V_x,BR (vanilla)",
        marker="x",
        markersize=3,
        alpha=0.7,
    )
    plt.plot(
        iters_accel,
        V_hat_accel,
        label="V_hat (acceleration)",
        marker="s",
        markersize=3,
        alpha=0.7,
    )
    plt.plot(
        iters_accel,
        BR_accel,
        label="V_x,BR (acceleration)",
        marker="^",
        markersize=3,
        alpha=0.7,
    )

    plt.axhline(
        V_star,
        linestyle="--",
        linewidth=1,
        color="k",
        label="V* (tabular)",
    )
    plt.xlabel("Outer iteration k")
    plt.ylabel("Value at start")
    plt.title("Approx vs True vs Best-Response Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/values.pdf")
    plt.close()

    # ------------------------------------------------------------
    # 7) Plot 2: Nash gap (log scale on y-axis)
    # ------------------------------------------------------------
    # Clip gaps to be strictly positive for log scale (for plotting only)
    eps = 1e-8
    Gap_vanilla_plot = np.clip(Gap_vanilla, eps, None)
    Gap_accel_plot = np.clip(Gap_accel, eps, None)

    plt.figure(figsize=(6, 4))
    plt.plot(
        iters_vanilla,
        Gap_vanilla_plot,
        marker="s",
        markersize=3,
        alpha=0.7,
        label="Gap (vanilla)",
    )
    plt.plot(
        iters_accel,
        Gap_accel_plot,
        marker="d",
        markersize=3,
        alpha=0.7,
        label="Gap (acceleration)",
    )
    plt.xlabel("Outer iteration k")
    plt.ylabel(r"Nash gap $V^* - V_{x,\mathrm{BR}}$")
    plt.title("Distance to Nash")
    plt.yscale("log")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/nash_gap.pdf")
    plt.close()

    # ------------------------------------------------------------
    # 8) Plot 3: Hunter's Right probability at start
    # ------------------------------------------------------------
    plt.figure(figsize=(6, 4))
    plt.plot(
        iters_vanilla,
        R_vanilla,
        marker="d",
        markersize=3,
        alpha=0.7,
        label="P_right (vanilla)",
    )
    plt.plot(
        iters_accel,
        R_accel,
        marker="o",
        markersize=3,
        alpha=0.7,
        label="P_right (acceleration)",
    )
    plt.xlabel("Outer iteration k")
    plt.ylabel("P_hunter(action=Right | start)")
    plt.title("Hunter policy at start over outer iterations")
    plt.ylim(0.0, 1.0)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/hunter_right_prob.pdf")
    plt.close()

    print("Saved:")
    print("  figures/values.pdf")
    print("  figures/nash_gap.pdf")
    print("  figures/hunter_right_prob.pdf")


if __name__ == "__main__":
    main()
