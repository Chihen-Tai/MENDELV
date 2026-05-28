"""One-shot comparison plot: MACE-OFF-small vs ANI-2x on rMD17 ethanol."""

from __future__ import annotations

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _load(name: str) -> dict:
    with open(ROOT / "reports" / name) as f:
        return json.load(f)


def _mean_shift_pairs(pairs: list[tuple[float, float]]) -> list[tuple[float, float]]:
    offset = sum(p - r for r, p in pairs) / len(pairs)
    return [(r, p - offset) for r, p in pairs]


def _rel_pairs(pairs: list[tuple[float, float]]) -> list[tuple[float, float]]:
    mn = min(r for r, _ in pairs)
    return [(r - mn, p - mn) for r, p in pairs]


def main() -> None:
    mb  = _load("bench_mace_small_ethanol.json")
    ab  = _load("bench_ani2x_ethanol.json")

    C_MACE = "#4C72B0"
    C_ANI  = "#DD8452"
    W = 0.35

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle(
        "MACE-OFF-small vs ANI-2x  |  rMD17 ethanol, 100 conformers, revPBE-D3 reference",
        fontsize=11, fontweight="bold", y=0.99,
    )

    # ── A: Force MAE / RMSE bar ───────────────────────────────────────────
    ax = axes[0, 0]
    labels    = ["Force MAE\n(eV/Å)", "Force RMSE\n(eV/Å)"]
    mace_vals = [mb["force_mae"], mb["force_rmse"]]
    ani_vals  = [ab["force_mae"], ab["force_rmse"]]
    x = np.arange(len(labels))
    bm = ax.bar(x - W/2, mace_vals, W, color=C_MACE, alpha=0.85, label="MACE-OFF-small")
    ba = ax.bar(x + W/2, ani_vals,  W, color=C_ANI,  alpha=0.85, label="ANI-2x")
    for bar in list(bm) + list(ba):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.006,
            f"{bar.get_height():.3f}",
            ha="center", va="bottom", fontsize=8.5,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("eV / Å")
    ax.set_ylim(0, max(mace_vals + ani_vals) * 1.3)
    ax.set_title("A  Force accuracy", loc="left", fontweight="bold")
    ax.legend(fontsize=8.5)
    ax.grid(axis="y", alpha=0.3)

    # ── B: Per-element force RMSE ─────────────────────────────────────────
    ax = axes[0, 1]
    elements = ["C", "H", "O"]
    mace_el  = [mb["per_element_force_rmse"].get(e, 0.0) for e in elements]
    ani_el   = [ab["per_element_force_rmse"].get(e, 0.0) for e in elements]
    x = np.arange(len(elements))
    ax.bar(x - W/2, mace_el, W, color=C_MACE, alpha=0.85, label="MACE-OFF-small")
    ax.bar(x + W/2, ani_el,  W, color=C_ANI,  alpha=0.85, label="ANI-2x")
    for xi, (mv, av) in enumerate(zip(mace_el, ani_el)):
        ax.text(xi - W/2, mv + 0.005, f"{mv:.3f}", ha="center", va="bottom", fontsize=8.5)
        ax.text(xi + W/2, av + 0.005, f"{av:.3f}", ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels(elements, fontsize=11)
    ax.set_ylabel("Force RMSE (eV/Å)")
    ax.set_ylim(0, max(mace_el + ani_el) * 1.3)
    ax.set_title("B  Per-element force RMSE", loc="left", fontweight="bold")
    ax.legend(fontsize=8.5)
    ax.grid(axis="y", alpha=0.3)

    # ── C: Energy parity (mean-shifted, relative) ─────────────────────────
    ax = axes[1, 0]
    mace_pairs = [
        (r["reference_energy"], r["predicted_energy"])
        for r in mb["records"] if r.get("predicted_energy") is not None
    ]
    ani_pairs = [
        (r["reference_energy"], r["predicted_energy"])
        for r in ab["records"] if r.get("predicted_energy") is not None
    ]
    mace_rp = _rel_pairs(_mean_shift_pairs(mace_pairs))
    ani_rp  = _rel_pairs(_mean_shift_pairs(ani_pairs))
    mr = [r for r, _ in mace_rp]; ms = [p for _, p in mace_rp]
    ar = [r for r, _ in ani_rp];  ap = [p for _, p in ani_rp]
    lo, hi = min(mr + ar), max(mr + ar)
    ax.scatter(mr, ms, s=22, color=C_MACE, alpha=0.75, label="MACE-OFF-small", zorder=3)
    ax.scatter(ar, ap, s=22, color=C_ANI,  alpha=0.75, label="ANI-2x", marker="^", zorder=3)
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.9, alpha=0.5, label="ideal (y = x)")
    ax.set_xlabel("Ref. energy (mean-shifted, eV)")
    ax.set_ylabel("Predicted (mean-shifted, eV)")
    ax.set_title("C  Energy parity", loc="left", fontweight="bold")
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.2)

    # ── D: Per-conformer force RMSE (sorted) ─────────────────────────────
    ax = axes[1, 1]
    mace_per = sorted(r["force_rmse"] for r in mb["records"] if r.get("force_rmse") is not None)
    ani_per  = sorted(r["force_rmse"] for r in ab["records"] if r.get("force_rmse") is not None)
    ax.plot(range(len(mace_per)), mace_per, color=C_MACE, lw=1.6, label="MACE-OFF-small")
    ax.plot(range(len(ani_per)),  ani_per,  color=C_ANI,  lw=1.6, linestyle="--", label="ANI-2x")
    ax.axhline(float(np.median(mace_per)), color=C_MACE, lw=0.8, linestyle=":", alpha=0.7)
    ax.axhline(float(np.median(ani_per)),  color=C_ANI,  lw=0.8, linestyle=":", alpha=0.7)
    ax.set_xlabel("Conformer rank (sorted by RMSE)")
    ax.set_ylabel("Force RMSE (eV/Å)")
    ax.set_title("D  Per-conformer force RMSE", loc="left", fontweight="bold")
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.2)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = ROOT / "reports" / "figures" / "mace_vs_ani2x_ethanol.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print("saved:", out)


if __name__ == "__main__":
    main()
