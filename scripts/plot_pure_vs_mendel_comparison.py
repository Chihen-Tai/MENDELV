"""Pure MLIP vs MENDEL+MLIP comparison figure (3 panels).

Panel 1: Global force RMSE — pure MLIP on rMD17 ethanol and QO2Mol OOD
Panel 2: MENDEL site decomposition — heteroatom (reactive) vs C/H (spectator)
Panel 3: Route B — reactive-site weighted fine-tuning on salicylic acid

Usage:
  python scripts/plot_pure_vs_mendel_comparison.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = ROOT / "reports" / "figures" / "pure_vs_mendel_full_comparison.png"


def load(path: str) -> dict:
    return json.loads((ROOT / path).read_text())


def main() -> None:
    mace_eth       = load("reports/fg_force_mace_ethanol.json")
    ani2x_eth      = load("reports/fg_force_ani2x_ethanol.json")
    mace_qo2_bench = load("reports/mlip_qo2mol_mace_benchmark.json")
    ani2x_qo2_bench = load("reports/mlip_qo2mol_ani2x_benchmark.json")

    mace_eth_global  = mace_eth["global_force_rmse"]
    ani2x_eth_global = ani2x_eth["global_force_rmse"]
    mace_qo2_global  = mace_qo2_bench["force_rmse"]
    ani2x_qo2_global = ani2x_qo2_bench["force_rmse"]

    mace_eth_O  = mace_eth["per_element_force_rmse"]["O"]
    mace_eth_C  = mace_eth["per_element_force_rmse"]["C"]
    mace_eth_H  = mace_eth["per_element_force_rmse"]["H"]
    ani2x_eth_O = ani2x_eth["per_element_force_rmse"]["O"]
    ani2x_eth_C = ani2x_eth["per_element_force_rmse"]["C"]
    ani2x_eth_H = ani2x_eth["per_element_force_rmse"]["H"]

    # Route B salicylic acid per-group RMSE (from multimol experiment)
    mendel_reactive  = 0.2340
    uniform_reactive = 0.2461
    mendel_spectator  = 0.1511
    uniform_spectator = 0.1488
    mendel_global  = 0.2068
    uniform_global = 0.2148

    BLUE   = "#2E86AB"
    ORANGE = "#E84855"
    GREEN  = "#3BB273"
    GRAY   = "#8D99AE"
    LIGHT  = "#F4F6F9"

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.8))
    fig.patch.set_facecolor("white")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})

    w = 0.32

    # ── Panel 1: Global RMSE ─────────────────────────────────────────────────
    ax1 = axes[0]
    ax1.set_facecolor(LIGHT)
    groups = ["rMD17\nEthanol", "QO2Mol\nOOD"]
    x = np.arange(len(groups))
    b1 = ax1.bar(x - w/2, [mace_eth_global, mace_qo2_global], w,
                 label="MACE-OFF-small", color=BLUE, alpha=0.88, zorder=3)
    b2 = ax1.bar(x + w/2, [ani2x_eth_global, ani2x_qo2_global], w,
                 label="ANI-2x", color=ORANGE, alpha=0.88, zorder=3)

    for bar, val in zip(list(b1) + list(b2),
                        [mace_eth_global, mace_qo2_global,
                         ani2x_eth_global, ani2x_qo2_global]):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    for xi, (m, a) in enumerate([(mace_eth_global, ani2x_eth_global),
                                  (mace_qo2_global, ani2x_qo2_global)]):
        pct = 100 * (m - a) / m
        ax1.annotate(f"ANI-2x −{pct:.0f}%", xy=(xi, min(m, a) * 0.88),
                     fontsize=8.5, color=GREEN, ha="center", fontweight="bold")

    ax1.set_xticks(x)
    ax1.set_xticklabels(groups)
    ax1.set_ylabel("Force RMSE (eV/Å)")
    ax1.set_title("① Pure MLIP\nGlobal Force RMSE", fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.set_ylim(0, 0.58)
    ax1.yaxis.grid(True, color="white", linewidth=1.2, zorder=0)
    ax1.set_axisbelow(True)

    # ── Panel 2: MENDEL Decomposition (ethanol) ──────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor(LIGHT)
    labels = ["Global", "O\n(reactive)", "C", "H"]
    mace_vals  = [mace_eth_global,  mace_eth_O,  mace_eth_C,  mace_eth_H]
    ani2x_vals = [ani2x_eth_global, ani2x_eth_O, ani2x_eth_C, ani2x_eth_H]
    x2 = np.arange(len(labels))

    b3 = ax2.bar(x2 - w/2, mace_vals,  w, label="MACE-OFF-small", color=BLUE,   alpha=0.88, zorder=3)
    b4 = ax2.bar(x2 + w/2, ani2x_vals, w, label="ANI-2x",         color=ORANGE, alpha=0.88, zorder=3)

    for bar, val in zip(list(b3) + list(b4), mace_vals + ani2x_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax2.axvspan(0.5, 1.5, alpha=0.12, color="red", zorder=0)
    ax2.text(1, 0.555, "reactive\nsite", ha="center", fontsize=8.5,
             color="darkred", fontweight="bold")

    ratio = mace_eth_O / mace_eth_global
    ax2.text(1, mace_eth_O + 0.022, f"{ratio:.2f}× global",
             ha="center", fontsize=8, color=BLUE, fontweight="bold")

    ax2.set_xticks(x2)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("Force RMSE (eV/Å)")
    ax2.set_title("② MENDEL Decomposition\nrMD17 Ethanol", fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.set_ylim(0, 0.64)
    ax2.yaxis.grid(True, color="white", linewidth=1.2, zorder=0)
    ax2.set_axisbelow(True)

    # ── Panel 3: Route B ─────────────────────────────────────────────────────
    ax3 = axes[2]
    ax3.set_facecolor(LIGHT)
    cats = ["Reactive\n(O, C–O)", "Spectator\n(C, H)", "Global"]
    mendel_vals  = [mendel_reactive,  mendel_spectator,  mendel_global]
    uniform_vals = [uniform_reactive, uniform_spectator, uniform_global]
    x3 = np.arange(len(cats))

    b5 = ax3.bar(x3 - w/2, mendel_vals,  w, label="MENDEL ×3",  color=GREEN, alpha=0.88, zorder=3)
    b6 = ax3.bar(x3 + w/2, uniform_vals, w, label="Uniform ×1", color=GRAY,  alpha=0.88, zorder=3)

    for bar, val in zip(list(b5) + list(b6), mendel_vals + uniform_vals):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    deltas = [(uniform_reactive - mendel_reactive) / uniform_reactive,
              (mendel_spectator - uniform_spectator) / uniform_spectator,
              (uniform_global - mendel_global) / uniform_global]
    labels_delta = ["−4.9%", "+1.5%", "−3.7%"]
    colors_delta = [GREEN, ORANGE, GREEN]
    for xi, (lbl, col) in enumerate(zip(labels_delta, colors_delta)):
        ax3.text(xi, max(mendel_vals[xi], uniform_vals[xi]) + 0.012,
                 lbl, ha="center", fontsize=10, color=col, fontweight="bold")

    ax3.set_xticks(x3)
    ax3.set_xticklabels(cats)
    ax3.set_ylabel("Force RMSE (eV/Å)")
    ax3.set_title("③ Route B: Reactive-site Weighting\nSalicylic Acid (held-out)", fontweight="bold")
    ax3.legend(fontsize=9)
    ax3.set_ylim(0, 0.33)
    ax3.yaxis.grid(True, color="white", linewidth=1.2, zorder=0)
    ax3.set_axisbelow(True)

    fig.suptitle(
        "Pure MLIP vs MENDEL+MLIP: Reactive Site Force Error Analysis",
        fontsize=13.5, fontweight="bold", y=1.02,
    )
    fig.text(
        0.5, -0.04,
        "MENDEL reveals reactive sites are high-error atoms (Panel ②)  ·  "
        "Reactive-site weighting reduces their error by 4.9% (Panel ③)",
        ha="center", fontsize=10, style="italic", color="#555",
    )

    plt.tight_layout(pad=1.8)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160, bbox_inches="tight", facecolor="white")
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
