"""Tests for Phase 10.5 energy/force plotting utilities."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

from mendel.plotting import (
    compute_energy_offset,
    load_energy_force_plot_inputs,
    plot_energy_parity,
    plot_energy_rmse_bar,
    plot_force_error_distribution,
    plot_force_rmse_by_element,
    plot_local_force_rmse_by_group,
)
from mendel.reference_data import (
    MLIPStructurePrediction,
    ReferenceStructureRecord,
    save_mlip_predictions_json,
    save_reference_records_json,
)

_ROOT = Path(__file__).parent.parent
_SCRIPT = _ROOT / "scripts" / "plot_energy_force_comparison.py"


def _references() -> list[ReferenceStructureRecord]:
    return [
        ReferenceStructureRecord(
            structure_id="s1",
            molecule_id="m1",
            dataset_name="tiny",
            smiles=None,
            xyz=[("C", 0.0, 0.0, 0.0), ("H", 0.0, 0.0, 1.0)],
            charge=0,
            multiplicity=1,
            reference_energy=1.0,
            reference_energy_unit="eV",
            reference_forces=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            reference_force_unit="eV/Angstrom",
            reference_method="test",
            split=None,
            metadata={},
        ),
        ReferenceStructureRecord(
            structure_id="s2",
            molecule_id="m1",
            dataset_name="tiny",
            smiles=None,
            xyz=[("O", 0.0, 0.0, 0.0), ("H", 0.0, 0.0, 1.0)],
            charge=0,
            multiplicity=1,
            reference_energy=2.0,
            reference_energy_unit="eV",
            reference_forces=[[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            reference_force_unit="eV/Angstrom",
            reference_method="test",
            split=None,
            metadata={},
        ),
    ]


def _predictions() -> list[MLIPStructurePrediction]:
    return [
        MLIPStructurePrediction(
            structure_id="s1",
            backend_name="fake",
            model_name="fake",
            energy=3.0,
            energy_unit="eV",
            forces=[[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
            force_unit="eV/Angstrom",
            success=True,
            warnings=[],
            metadata={},
        ),
        MLIPStructurePrediction(
            structure_id="s2",
            backend_name="fake",
            model_name="fake",
            energy=4.0,
            energy_unit="eV",
            forces=[[0.0, 2.0, 0.0], [0.0, 0.0, 2.0]],
            force_unit="eV/Angstrom",
            success=True,
            warnings=[],
            metadata={},
        ),
    ]


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    reference = tmp_path / "reference.json"
    predictions = tmp_path / "predictions.json"
    benchmark = tmp_path / "benchmark.json"
    local = tmp_path / "local.json"
    save_reference_records_json(_references(), reference)
    save_mlip_predictions_json(_predictions(), predictions)
    benchmark.write_text(
        json.dumps(
            {
                "n_structures": 2,
                "energy_rmse_raw": 2.0,
                "energy_rmse_mean_shifted": 0.0,
                "force_rmse": 1.0,
                "per_element_force_rmse": {"C": 1.0, "H": 0.5, "O": 1.5},
                "metadata": {"energy_unit": "eV", "force_unit": "eV/Angstrom"},
            }
        ),
        encoding="utf-8",
    )
    local.write_text(
        json.dumps(
            {
                "per_group_type_force_rmse": {
                    "whole_molecule": 1.0,
                    "element_O": 1.5,
                    "element_H": 0.5,
                },
                "top_group_type_errors": [
                    {"group_type": "element_O", "force_rmse": 1.5, "n_groups": 1}
                ],
                "metadata": {"pseudo_groups_used": True},
            }
        ),
        encoding="utf-8",
    )
    return reference, predictions, benchmark, local


def test_compute_energy_offset() -> None:
    assert compute_energy_offset([1.0, 2.0], [3.0, 5.0]) == 2.5


def test_load_energy_force_plot_inputs_and_shifted_values(tmp_path: Path) -> None:
    reference, predictions, benchmark, local = _write_inputs(tmp_path)

    inputs = load_energy_force_plot_inputs(reference, predictions, benchmark, local)

    assert inputs["structure_ids"] == ["s1", "s2"]
    assert inputs["reference_energies"] == [1.0, 2.0]
    assert inputs["predicted_energies"] == [3.0, 4.0]
    assert inputs["energy_offset"] == 2.0
    assert inputs["shifted_predicted_energies"] == [1.0, 2.0]
    assert inputs["energy_errors_mean_shifted"] == [0.0, 0.0]
    assert inputs["per_element_force_rmse"]["O"] == 1.5
    assert len(inputs["force_error_norms"]) == 4


def test_load_energy_force_plot_inputs_without_local_analysis(tmp_path: Path) -> None:
    reference, predictions, benchmark, _ = _write_inputs(tmp_path)

    inputs = load_energy_force_plot_inputs(reference, predictions, benchmark, None)

    assert inputs["per_group_type_force_rmse"] == {}


def test_plot_energy_parity_writes_png(tmp_path: Path) -> None:
    output = tmp_path / "energy.png"

    plot_energy_parity([1.0, 2.0], [1.1, 2.2], output, "Energy parity")

    assert output.exists()
    assert output.stat().st_size > 0


def test_plot_energy_rmse_bar_writes_png(tmp_path: Path) -> None:
    output = tmp_path / "energy_bar.png"

    plot_energy_rmse_bar(2.0, 0.1, output)

    assert output.exists()
    assert output.stat().st_size > 0


def test_plot_force_rmse_by_element_writes_png(tmp_path: Path) -> None:
    output = tmp_path / "force_element.png"

    plot_force_rmse_by_element(1.0, {"O": 1.5, "C": 0.5}, output)

    assert output.exists()
    assert output.stat().st_size > 0


def test_plot_local_force_rmse_by_group_writes_png(tmp_path: Path) -> None:
    output = tmp_path / "local_group.png"

    plot_local_force_rmse_by_group({"whole_molecule": 1.0, "element_O": 1.5}, output)

    assert output.exists()
    assert output.stat().st_size > 0


def test_plot_force_error_distribution_writes_png(tmp_path: Path) -> None:
    output = tmp_path / "dist.png"

    plot_force_error_distribution([0.1, 0.2, 0.3], output)

    assert output.exists()
    assert output.stat().st_size > 0


def test_cli_smoke_writes_figures_and_report(tmp_path: Path) -> None:
    reference, predictions, benchmark, local = _write_inputs(tmp_path)
    output_dir = tmp_path / "figures"
    report = tmp_path / "plot_report.json"

    result = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--reference",
            str(reference),
            "--predictions",
            str(predictions),
            "--benchmark",
            str(benchmark),
            "--local-analysis",
            str(local),
            "--output-dir",
            str(output_dir),
            "--report",
            str(report),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    payload = json.loads(report.read_text(encoding="utf-8"))
    assert result.returncode == 0, result.stderr
    assert (output_dir / "energy_parity_raw.png").exists()
    assert (output_dir / "energy_parity_mean_shifted.png").exists()
    assert (output_dir / "energy_rmse_bar.png").exists()
    assert (output_dir / "force_rmse_by_element.png").exists()
    assert (output_dir / "local_force_rmse_by_group.png").exists()
    assert (output_dir / "force_error_distribution.png").exists()
    assert payload["n_structures"] == 2
    assert math.isclose(payload["energy_offset"], 2.0)


def test_no_mlip_training_or_dft_invoked() -> None:
    text = (_ROOT / "mendel" / "plotting.py").read_text(encoding="utf-8").lower()
    script = _SCRIPT.read_text(encoding="utf-8").lower()
    for token in ("train_mlp", "fine_tune", "orca", "vasp", "gaussian", "neb", "irc", "barrier"):
        assert token not in text
        assert token not in script
