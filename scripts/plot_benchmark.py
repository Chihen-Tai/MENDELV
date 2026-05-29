"""Generate benchmark comparison figures from reports/benchmark_comparison/."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPORT_DIR = Path("reports/benchmark_comparison")
OUT_FIG = REPORT_DIR / "comparison_figure.png"

d = json.loads((REPORT_DIR / "comparison.json").read_text())

MODELS = d["predictor_names"]
LABELS = {
    "rule_based_local":      "Rule-based",
    "rule_based_negotiated": "Rule + Negotiate",
    "mlp_local":             "MLP",
    "mlp_negotiated":        "MLP + Negotiate",
}
COLORS = {
    "rule_based_local":      "#6c8ebf",
    "rule_based_negotiated": "#4a7abf",
    "mlp_local":             "#d6a042",
    "mlp_negotiated":        "#c05c1f",
}

mechs = sorted(
    d["per_mechanism_accuracy"].keys(),
    key=lambda m: (-d["per_mechanism_accuracy"][m]["mlp_negotiated"], m),
)

fig = plt.figure(figsize=(16, 14))
fig.patch.set_facecolor("#1a1a2e")

def style_ax(ax, title):
    ax.set_facecolor("#16213e")
    ax.set_title(title, color="#ffffff", fontsize=13, pad=10)
    ax.tick_params(colors="#a0a0b0")
    for spine in ax.spines.values():
        spine.set_edgecolor("#3a3a5c")
    ax.grid(axis="y", color="#3a3a5c", linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)

# Panel 1: Overall accuracy
ax1 = fig.add_subplot(3, 1, 1)
style_ax(ax1, "Overall Role Accuracy")
x = np.arange(len(MODELS))
bars = ax1.bar(
    x,
    [d["overall_role_accuracy"][m] * 100 for m in MODELS],
    color=[COLORS[m] for m in MODELS],
    width=0.55,
    edgecolor="#2a2a4e",
)
for bar, m in zip(bars, MODELS):
    v = d["overall_role_accuracy"][m] * 100
    ax1.text(bar.get_x() + bar.get_width() / 2, v + 0.8, f"{v:.2f}%",
             ha="center", va="bottom", color="#e0e0e0", fontsize=11, fontweight="bold")
ax1.set_xticks(x)
ax1.set_xticklabels([LABELS[m] for m in MODELS], color="#e0e0e0", fontsize=11)
ax1.set_ylabel("Accuracy (%)", color="#a0a0b0")
ax1.set_ylim(0, 115)

# Panel 2: Per-mechanism accuracy
ax2 = fig.add_subplot(3, 1, 2)
style_ax(ax2, "Per-Mechanism Role Accuracy")
n_mechs = len(mechs)
n_models = len(MODELS)
group_w = 0.75
bar_w = group_w / n_models
x2 = np.arange(n_mechs)
for i, m in enumerate(MODELS):
    vals = [d["per_mechanism_accuracy"][mech][m] * 100 for mech in mechs]
    xpos = x2 - group_w / 2 + bar_w * i + bar_w / 2
    ax2.bar(xpos, vals, width=bar_w * 0.92, color=COLORS[m], label=LABELS[m],
            edgecolor="#1a1a2e", linewidth=0.4)

mech_labels = [m.replace("_", "\n") for m in mechs]
ax2.set_xticks(x2)
ax2.set_xticklabels(mech_labels, color="#e0e0e0", fontsize=7.5)
ax2.set_ylabel("Accuracy (%)", color="#a0a0b0")
ax2.set_ylim(0, 115)
ax2.legend(framealpha=0.2, facecolor="#1a1a2e", edgecolor="#3a3a5c",
           labelcolor="#e0e0e0", fontsize=9, ncol=4, loc="upper right")

# Panel 3: Reaction center P/R/F1
ax3 = fig.add_subplot(3, 1, 3)
style_ax(ax3, "Reaction Center Detection (Negotiated Models)")
rc_models = ["rule_based_negotiated", "mlp_negotiated"]
metrics = ["precision", "recall", "f1"]
metric_colors = ["#6ec6a0", "#e07b54", "#a78bfa"]
x3 = np.arange(len(rc_models))
bar_w3 = 0.22
for i, metric in enumerate(metrics):
    vals = [d["reaction_center"][m][metric] * 100 for m in rc_models]
    xpos = x3 - 0.3 + bar_w3 * i + bar_w3 / 2
    bars3 = ax3.bar(xpos, vals, width=bar_w3 * 0.9, color=metric_colors[i],
                    label=metric.upper(), edgecolor="#1a1a2e", linewidth=0.4)
    for bar, v in zip(bars3, vals):
        ax3.text(bar.get_x() + bar.get_width() / 2, v + 0.8, f"{v:.1f}",
                 ha="center", va="bottom", color="#e0e0e0", fontsize=9)
ax3.set_xticks(x3)
ax3.set_xticklabels([LABELS[m] for m in rc_models], color="#e0e0e0", fontsize=11)
ax3.set_ylabel("Score (%)", color="#a0a0b0")
ax3.set_ylim(0, 115)
ax3.legend(framealpha=0.2, facecolor="#1a1a2e", edgecolor="#3a3a5c",
           labelcolor="#e0e0e0", fontsize=9)

fig.suptitle(
    "MENDEL Benchmark — Role Prediction & Reaction Center Detection\n"
    "(148 reactions, 299 examples, data/reactions.center_balanced.cleaned.json)",
    color="#ffffff", fontsize=13, y=0.998,
)
plt.tight_layout(rect=[0, 0, 1, 0.985])
fig.savefig(OUT_FIG, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved: {OUT_FIG}")

# Data table
print("\n=== Overall Role Accuracy ===")
for m in MODELS:
    print(f"  {LABELS[m]:<22s}: {d['overall_role_accuracy'][m]*100:.2f}%")

print(f"\n=== Per-Mechanism  ({' | '.join(LABELS[m] for m in MODELS)}) ===")
for mech in mechs:
    row = d["per_mechanism_accuracy"][mech]
    vals = " | ".join(f"{row[m]*100:5.1f}%" for m in MODELS)
    print(f"  {mech:<35s}: {vals}")

print("\n=== Reaction Center (Negotiated models) ===")
for m in ["rule_based_negotiated", "mlp_negotiated"]:
    rc = d["reaction_center"][m]
    print(f"  {LABELS[m]:<22s}: P={rc['precision']*100:.1f}%  R={rc['recall']*100:.1f}%  F1={rc['f1']*100:.1f}%")
