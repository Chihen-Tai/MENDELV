"""Tests for Phase 10.3 real MD17/rMD17 benchmark helpers."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from mendel.reference_data import (
    MLIPStructurePrediction,
    ReferenceStructureRecord,
    compute_energy_force_benchmark,
    compute_mean_shifted_energy_errors,
)

_ROOT = Path(__file__).parent.parent
_FETCH_SCRIPT = _ROOT / "scripts" / "fetch_md17_sample.py"
_PREPARE_SCRIPT = _ROOT / "scripts" / "prepare_md17_sample.py"
_REAL_BENCH_SCRIPT = _ROOT / "scripts" / "run_md17_mace_benchmark.py"


def _record(structure_id: str, energy: float) -> ReferenceStructureRecord:
    return ReferenceStructureRecord(
        structure_id=structure_id,
        molecule_id="mol",
        dataset_name="MD17/rMD17",
        smiles=None,
        xyz=[("C", 0.0, 0.0, 0.0), ("H", 0.0, 0.0, 1.0)],
        charge=0,
        multiplicity=1,
        reference_energy=energy,
        reference_energy_unit="eV",
        reference_forces=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        reference_force_unit="eV/Angstrom",
        reference_method="dataset-provided DFT",
        split=None,
        metadata={},
    )


def _prediction(structure_id: str, energy: float) -> MLIPStructurePrediction:
    return MLIPStructurePrediction(
        structure_id=structure_id,
        backend_name="mace",
        model_name="mace-off-small",
        energy=energy,
        energy_unit="eV",
        forces=[[0.1, 0.0, 0.0], [0.2, 0.0, 0.0]],
        force_unit="eV/Angstrom",
        success=True,
        warnings=[],
        metadata={},
    )


def test_compute_mean_shifted_energy_errors_removes_constant_offset() -> None:
    result = compute_mean_shifted_energy_errors(
        reference_energies=[1.0, 2.0, 3.0],
        predicted_energies=[6.0, 7.0, 8.0],
    )

    assert result["offset"] == 5.0
    assert result["shifted_errors"] == [0.0, 0.0, 0.0]


def test_benchmark_report_includes_raw_and_mean_shifted_energy_metrics() -> None:
    records = [_record("s1", 1.0), _record("s2", 2.0), _record("s3", 3.0)]
    predictions = [_prediction("s1", 6.0), _prediction("s2", 7.0), _prediction("s3", 8.0)]

    report = compute_energy_force_benchmark(records, predictions)
    payload = report.to_dict()

    assert payload["energy_mae_raw"] == 5.0
    assert payload["energy_rmse_raw"] == 5.0
    assert payload["energy_mae_mean_shifted"] == 0.0
    assert payload["energy_rmse_mean_shifted"] == 0.0
    assert payload["energy_offset_applied"] == 5.0
    assert payload["energy_rmse_mean_shifted"] < payload["energy_rmse_raw"]
    assert payload["n_force_components_compared"] == 18
    assert payload["n_atoms_compared"] == 6
    assert payload["max_force_rmse_structure_id"] == "s1"


def test_run_md17_mace_benchmark_refuses_missing_input(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(_REAL_BENCH_SCRIPT),
            "--input",
            str(tmp_path / "missing.npz"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "input file does not exist" in result.stderr


def test_prepare_md17_sample_synthetic_mode_marks_not_scientific_reference(tmp_path: Path) -> None:
    report = tmp_path / "md17_report.json"
    output = tmp_path / "md17_reference.json"

    result = subprocess.run(
        [
            sys.executable,
            str(_PREPARE_SCRIPT),
            "--output",
            str(output),
            "--report",
            str(report),
            "--max-records",
            "2",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    payload = json.loads(report.read_text(encoding="utf-8"))
    assert result.returncode == 0
    assert "No input provided; generating synthetic test data only" in result.stdout
    assert payload["metadata"]["synthetic_test_data"] is True
    assert payload["metadata"]["not_scientific_reference"] is True


def test_prepare_md17_sample_writes_converted_units_by_default(tmp_path: Path) -> None:
    import numpy as np

    sample = tmp_path / "md17.npz"
    output = tmp_path / "md17_reference.json"
    report = tmp_path / "md17_report.json"
    np.savez(
        sample,
        z=np.array([6], dtype=int),
        R=np.array([[[0.0, 0.0, 0.0]]], dtype=float),
        E=np.array([1.0], dtype=float),
        F=np.array([[[1.0, 0.0, 0.0]]], dtype=float),
    )

    result = subprocess.run(
        [
            sys.executable,
            str(_PREPARE_SCRIPT),
            "--input",
            str(sample),
            "--output",
            str(output),
            "--report",
            str(report),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    record = json.loads(output.read_text(encoding="utf-8"))["records"][0]
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert result.returncode == 0
    assert record["reference_energy_unit"] == "eV"
    assert record["reference_force_unit"] == "eV/Angstrom"
    assert record["metadata"]["original_energy_unit"] == "kcal/mol"
    assert payload["metadata"]["unit_conversion_applied"] is True


def test_prepare_md17_sample_no_convert_units_preserves_original_units(tmp_path: Path) -> None:
    import numpy as np

    sample = tmp_path / "md17.npz"
    output = tmp_path / "md17_reference.json"
    report = tmp_path / "md17_report.json"
    np.savez(
        sample,
        z=np.array([6], dtype=int),
        R=np.array([[[0.0, 0.0, 0.0]]], dtype=float),
        E=np.array([1.0], dtype=float),
        F=np.array([[[1.0, 0.0, 0.0]]], dtype=float),
    )

    result = subprocess.run(
        [
            sys.executable,
            str(_PREPARE_SCRIPT),
            "--input",
            str(sample),
            "--output",
            str(output),
            "--report",
            str(report),
            "--no-convert-units",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    record = json.loads(output.read_text(encoding="utf-8"))["records"][0]
    assert result.returncode == 0
    assert record["reference_energy_unit"] == "kcal/mol"
    assert record["reference_force_unit"] == "kcal/mol/Angstrom"
    assert record["metadata"]["unit_conversion_applied"] is False


def test_run_md17_mace_benchmark_accepts_unit_options() -> None:
    result = subprocess.run(
        [sys.executable, str(_REAL_BENCH_SCRIPT), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--energy-unit" in result.stdout
    assert "--no-convert-units" in result.stdout


def test_fetch_md17_sample_help_runs() -> None:
    result = subprocess.run(
        [sys.executable, str(_FETCH_SCRIPT), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--allow-download" in result.stdout


def test_fetch_md17_sample_without_url_does_not_download(tmp_path: Path) -> None:
    report = tmp_path / "fetch_report.json"

    result = subprocess.run(
        [sys.executable, str(_FETCH_SCRIPT), "--report", str(report)],
        check=False,
        capture_output=True,
        text=True,
    )

    payload = json.loads(report.read_text(encoding="utf-8"))
    assert result.returncode == 0
    assert payload["download_performed"] is False
    assert "manual" in payload["message"].lower()


def test_fetch_md17_sample_refuses_large_mocked_url(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location("fetch_md17_sample", _FETCH_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class FakeResponse:
        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, *_args: Any) -> None:
            return None

        def getheader(self, name: str, default: str | None = None) -> str | None:
            return str(2 * 1024 * 1024) if name.lower() == "content-length" else default

    monkeypatch.setattr(module.urllib.request, "urlopen", lambda *_args, **_kwargs: FakeResponse())
    report = tmp_path / "fetch_report.json"

    code = module.main([
        "--url",
        "https://example.org/md17.npz",
        "--allow-download",
        "--max-size-mb",
        "1",
        "--report",
        str(report),
    ])

    payload = json.loads(report.read_text(encoding="utf-8"))
    assert code == 1
    assert payload["download_performed"] is False
    assert "exceeds" in payload["message"]


def test_no_mlip_training_invoked() -> None:
    text = _REAL_BENCH_SCRIPT.read_text(encoding="utf-8").lower()
    fetch_text = _FETCH_SCRIPT.read_text(encoding="utf-8").lower()
    for token in ("train_mlp", "fine_tune", "neb", "irc", "transition state", "barrier"):
        assert token not in text
        assert token not in fetch_text
