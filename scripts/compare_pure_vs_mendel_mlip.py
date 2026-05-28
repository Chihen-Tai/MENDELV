"""Compare pure-MLIP force error vs MENDEL-decomposed force error on rMD17 ethanol.

Pure MLIP  : global force RMSE on all atoms — one number.
MENDEL+MLIP: MENDEL identifies functional groups; force RMSE is decomposed
             per group, revealing WHERE each model struggles chemically.
"""
from __future__ import annotations

import json
import math
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from mendel.identifier import identify_functional_groups_in_mol

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _rmse(vectors: list[list[float]]) -> float:
    sq = [v[0]**2 + v[1]**2 + v[2]**2 for v in vectors]
    return math.sqrt(sum(sq) / len(sq)) if sq else 0.0


def _load(name: str) -> dict:
    with open(ROOT / "reports" / name) as f:
        return json.load(f)


def _build_ethanol_groups(ref_records: list[dict]) -> dict[str, list[int]]:
    """Atom-index sets for MENDEL-identified groups in rMD17 ethanol."""
    xyz = ref_records[0]["xyz"]
    positions = {i: (float(xyz[i][1]), float(xyz[i][2]), float(xyz[i][3]))
                 for i in range(len(xyz))}
    symbols = [xyz[i][0] for i in range(len(xyz))]
    heavy = [i for i, s in enumerate(symbols) if s != "H"]
    h_idx = [i for i, s in enumerate(symbols) if s == "H"]

    def dist(a: int, b: int) -> float:
        return math.sqrt(sum((positions[a][k] - positions[b][k])**2 for k in range(3)))

    h_parent = {h: min(heavy, key=lambda hv: dist(h, hv)) for h in h_idx}
    # identify which heavy C is alpha (bonded to O)
    c_idx = [i for i, s in enumerate(symbols) if s == "C"]
    o_idx = next(i for i, s in enumerate(symbols) if s == "O")
    alpha_c  = min(c_idx, key=lambda c: dist(c, o_idx))
    methyl_c = next(c for c in c_idx if c != alpha_c)

    alpha_h  = [h for h, p in h_parent.items() if p == alpha_c]
    methyl_h = [h for h, p in h_parent.items() if p == methyl_c]
    o_h      = [h for h, p in h_parent.items() if p == o_idx]

    # MENDEL identifies alcohol = alpha-C + O (the C–O bond anchor)
    return {
        "alcohol\n(MENDEL:\nC–O bond)":      sorted([alpha_c, o_idx]),
        "hydroxyl H\n(O–H)":                 sorted(o_h),
        "alpha C–H\n(reactive side)":         sorted(alpha_h),
        "methyl C–H\n(spectator)":            sorted(methyl_h),
        "methyl C\n(spectator)":              [methyl_c],
    }


def _per_group_rmse(bench_records: list[dict], groups: dict[str, list[int]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for label, atom_indices in groups.items():
        errors: list[list[float]] = []
        for rec in bench_records:
            fe = rec.get("force_errors")
            if not fe:
                continue
            for idx in atom_indices:
                if idx < len(fe):
                    errors.append(fe[idx])
        out[label] = _rmse(errors)
    return out


def main() -> None:
    mb = _load("bench_mace_small_ethanol.json")
    ab = _load("bench_ani2x_ethanol.json")
    with open(ROOT / "data/reference/rmd17_ethanol_sample_converted.reference.json") as f:
        ref_data = json.load(f)

    groups = _build_ethanol_groups(ref_data["records"])
    mace_group = _per_group_rmse(mb["records"], groups)
    ani_group  = _per_group_rmse(ab["records"], groups)

    C_MACE = "#4C72B0"
    C_ANI  = "#DD8452"
    W = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [1, 2.4]})
    fig.suptitle(
        "Pure MLIP  vs  MENDEL + MLIP\n"
        "rMD17 ethanol · 100 conformers · revPBE-D3 reference",
        fontsize=11, fontweight="bold",
    )

    # ── A: Pure MLIP — global force RMSE (one number per model) ─────────────
    ax = axes[0]
    pure_vals = [mb["force_rmse"], ab["force_rmse"]]
    pure_labels = ["MACE-OFF\nsmall", "ANI-2x"]
    bars = ax.bar([0, 1], pure_vals, color=[C_MACE, C_ANI], alpha=0.85, width=0.5)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.006,
                f"{bar.get_height():.3f}", ha="center", va="bottom",
                fontsize=11, fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(pure_labels, fontsize=10)
    ax.set_ylabel("Force RMSE (eV/Å)", fontsize=10)
    ax.set_ylim(0, max(pure_vals) * 1.4)
    ax.set_title("A  Pure MLIP\n(global · all atoms)", loc="left", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.text(0.5, 0.88, "one number —\nno chemical insight",
            transform=ax.transAxes, ha="center", fontsize=8.5,
            color="#888888", style="italic")

    # ── B: MENDEL + MLIP — force RMSE per functional group ──────────────────
    ax = axes[1]
    group_labels = list(groups.keys())
    x = np.arange(len(group_labels))
    mace_vals = [mace_group[k] for k in group_labels]
    ani_vals  = [ani_group[k]  for k in group_labels]

    bm = ax.bar(x - W/2, mace_vals, W, color=C_MACE, alpha=0.85, label="MACE-OFF-small")
    ba = ax.bar(x + W/2, ani_vals,  W, color=C_ANI,  alpha=0.85, label="ANI-2x")
    for bar in list(bm) + list(ba):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

    # dashed reference lines = pure MLIP global RMSE
    ax.axhline(mb["force_rmse"], color=C_MACE, lw=1.3, linestyle="--", alpha=0.55,
               label=f"MACE global ({mb['force_rmse']:.3f})")
    ax.axhline(ab["force_rmse"], color=C_ANI,  lw=1.3, linestyle="--", alpha=0.55,
               label=f"ANI-2x global ({ab['force_rmse']:.3f})")

    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, fontsize=9)
    ax.set_ylabel("Force RMSE (eV/Å)", fontsize=10)
    ax.set_ylim(0, max(mace_vals + ani_vals) * 1.35)
    ax.set_title("B  MENDEL + MLIP\n(force error decomposed by functional group agent)",
                 loc="left", fontweight="bold")
    ax.legend(fontsize=8.5, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    # bracket showing MENDEL-identified reactive region
    y_bracket = max(mace_vals[:2] + ani_vals[:2]) * 1.2
    ax.annotate("", xy=(0.5, y_bracket), xytext=(-0.5, y_bracket),
                arrowprops=dict(arrowstyle="<->", color="#333", lw=1.2))
    ax.text(0.0, y_bracket + 0.008, "MENDEL: reactive\nfunctional group",
            ha="center", va="bottom", fontsize=7.5, color="#333")

    plt.tight_layout()
    out = ROOT / "reports" / "figures" / "pure_vs_mendel_mlip.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print("saved:", out)


if __name__ == "__main__":
    main()
