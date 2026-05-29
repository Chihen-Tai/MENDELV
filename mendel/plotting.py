"""Plotting utilities for energy/force benchmark reports.

Matplotlib is imported inside plotting functions so normal ``import mendel``
does not require plotting dependencies.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

from mendel.reference_data import (
    load_mlip_predictions_json,
    load_reference_records_json,
)


def ensure_output_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def compute_energy_offset(
    reference_energies: list[float],
    predicted_energies: list[float],
) -> float:
    if len(reference_energies) != len(predicted_energies):
        raise ValueError("reference_energies and predicted_energies must have equal length")
    if not reference_energies:
        return 0.0
    errors = [
        float(predicted) - float(reference)
        for reference, predicted in zip(reference_energies, predicted_energies, strict=True)
    ]
    return sum(errors) / len(errors)


def _load_json_if_present(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    candidate = Path(path)
    if not candidate.exists():
        return {}
    return json.loads(candidate.read_text(encoding="utf-8"))


def _force_error_norms(
    reference_forces: list[list[float]],
    predicted_forces: list[list[float]],
) -> list[float]:
    norms: list[float] = []
    for ref_force, pred_force in zip(reference_forces, predicted_forces, strict=True):
        error = [
            float(predicted) - float(reference)
            for reference, predicted in zip(ref_force, pred_force, strict=True)
        ]
        norms.append(math.sqrt(sum(component * component for component in error)))
    return norms


def load_energy_force_plot_inputs(
    reference_path: str | Path,
    predictions_path: str | Path,
    benchmark_path: str | Path,
    local_analysis_path: str | Path | None = None,
) -> dict[str, object]:
    reference_records = load_reference_records_json(reference_path)
    predictions = load_mlip_predictions_json(predictions_path)
    benchmark = _load_json_if_present(benchmark_path)
    local_analysis = _load_json_if_present(local_analysis_path)
    predictions_by_id = {prediction.structure_id: prediction for prediction in predictions}

    structure_ids: list[str] = []
    reference_energies: list[float] = []
    predicted_energies: list[float] = []
    reference_forces: list[list[list[float]]] = []
    predicted_forces: list[list[list[float]]] = []
    force_error_norms: list[float] = []

    for reference in reference_records:
        prediction = predictions_by_id.get(reference.structure_id)
        if prediction is None or not prediction.success:
            continue
        if reference.reference_energy is not None and prediction.energy is not None:
            structure_ids.append(reference.structure_id)
            reference_energies.append(float(reference.reference_energy))
            predicted_energies.append(float(prediction.energy))
        if reference.reference_forces is not None and prediction.forces is not None:
            if len(reference.reference_forces) != len(prediction.forces):
                continue
            reference_forces.append(reference.reference_forces)
            predicted_forces.append(prediction.forces)
            force_error_norms.extend(
                _force_error_norms(reference.reference_forces, prediction.forces)
            )

    energy_offset = compute_energy_offset(reference_energies, predicted_energies)
    shifted_predicted = [energy - energy_offset for energy in predicted_energies]
    raw_errors = [
        predicted - reference
        for reference, predicted in zip(reference_energies, predicted_energies, strict=True)
    ]
    shifted_errors = [
        predicted - reference
        for reference, predicted in zip(reference_energies, shifted_predicted, strict=True)
    ]

    return {
        "structure_ids": structure_ids,
        "reference_energies": reference_energies,
        "predicted_energies": predicted_energies,
        "shifted_predicted_energies": shifted_predicted,
        "energy_offset": energy_offset,
        "energy_errors_raw": raw_errors,
        "energy_errors_mean_shifted": shifted_errors,
        "reference_forces": reference_forces,
        "predicted_forces": predicted_forces,
        "force_error_norms": force_error_norms,
        "raw_energy_rmse": benchmark.get("energy_rmse_raw", benchmark.get("energy_rmse")),
        "mean_shifted_energy_rmse": benchmark.get("energy_rmse_mean_shifted"),
        "global_force_rmse": benchmark.get("force_rmse"),
        "per_element_force_rmse": dict(benchmark.get("per_element_force_rmse", {})),
        "per_group_type_force_rmse": dict(
            local_analysis.get("per_group_type_force_rmse", {})
        ),
        "top_group_type_errors": list(local_analysis.get("top_group_type_errors", [])),
        "benchmark": benchmark,
        "local_analysis": local_analysis,
    }


def _pyplot():
    cache_dir = Path(os.environ.get("MPLCONFIGDIR", Path.cwd() / ".matplotlib-cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _save_current_figure(output_path: str | Path) -> None:
    plt = _pyplot()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()


def plot_energy_parity(
    reference_energies: list[float],
    predicted_energies: list[float],
    output_path: str | Path,
    title: str,
    shifted: bool = False,
    energy_unit: str = "eV",
) -> None:
    plt = _pyplot()
    plt.figure(figsize=(5.2, 4.4))
    plt.scatter(reference_energies, predicted_energies, s=24, alpha=0.75, edgecolors="none")
    if reference_energies and predicted_energies:
        values = reference_energies + predicted_energies
        min_value = min(values)
        max_value = max(values)
        padding = (max_value - min_value) * 0.05 if max_value > min_value else 1.0
        line_min = min_value - padding
        line_max = max_value + padding
        plt.plot([line_min, line_max], [line_min, line_max], color="black", linewidth=1)
    ylabel = "Mean-shifted MACE-OFF energy" if shifted else "MACE-OFF energy"
    plt.xlabel(f"DFT reference energy ({energy_unit})")
    plt.ylabel(f"{ylabel} ({energy_unit})")
    plt.title(title)
    plt.grid(True, alpha=0.25)
    _save_current_figure(output_path)


def plot_energy_rmse_bar(
    raw_rmse: float,
    mean_shifted_rmse: float,
    output_path: str | Path,
    energy_unit: str = "eV",
) -> None:
    plt = _pyplot()
    labels = ["Raw", "Mean-shifted"]
    values = [float(raw_rmse), float(mean_shifted_rmse)]
    plt.figure(figsize=(5.0, 4.0))
    bars = plt.bar(labels, values, color=["#4c78a8", "#f58518"])
    plt.ylabel(f"Energy RMSE ({energy_unit})")
    plt.title("Energy RMSE")
    plt.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values, strict=True):
        plt.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3g}", ha="center", va="bottom")
    _save_current_figure(output_path)


def plot_force_rmse_by_element(
    global_force_rmse: float,
    per_element_force_rmse: dict[str, float],
    output_path: str | Path,
    force_unit: str = "eV/Angstrom",
) -> None:
    plt = _pyplot()
    items = [("global", float(global_force_rmse))]
    items.extend(
        (element, float(value))
        for element, value in sorted(per_element_force_rmse.items())
    )
    labels = [item[0] for item in items]
    values = [item[1] for item in items]
    plt.figure(figsize=(6.0, 4.0))
    plt.bar(labels, values, color="#54a24b")
    plt.ylabel(f"Force RMSE ({force_unit})")
    plt.title("Force RMSE by element")
    plt.grid(axis="y", alpha=0.25)
    _save_current_figure(output_path)


def plot_local_force_rmse_by_group(
    per_group_type_force_rmse: dict[str, float],
    output_path: str | Path,
    force_unit: str = "eV/Angstrom",
    top_n: int = 10,
) -> None:
    plt = _pyplot()
    sorted_items = sorted(
        per_group_type_force_rmse.items(),
        key=lambda item: float(item[1]),
        reverse=True,
    )[:top_n]
    labels = [item[0] for item in sorted_items] or ["no groups"]
    values = [float(item[1]) for item in sorted_items] or [0.0]
    plt.figure(figsize=(7.0, 4.2))
    plt.bar(labels, values, color="#e45756")
    plt.ylabel(f"Local force RMSE ({force_unit})")
    plt.title("Local force RMSE by group")
    plt.xticks(rotation=35, ha="right")
    plt.grid(axis="y", alpha=0.25)
    _save_current_figure(output_path)


def plot_force_error_distribution(
    force_error_norms: list[float],
    output_path: str | Path,
    force_unit: str = "eV/Angstrom",
) -> None:
    plt = _pyplot()
    plt.figure(figsize=(6.0, 4.0))
    bins = min(30, max(5, len(force_error_norms) // 5 or 5))
    plt.hist(force_error_norms, bins=bins, color="#72b7b2")
    plt.xlabel(f"Atom force error norm ({force_unit})")
    plt.ylabel("Atom count")
    plt.title("Atom force error distribution")
    plt.grid(axis="y", alpha=0.25)
    _save_current_figure(output_path)


def save_plot_report(report: dict[str, object], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
