"""Tests for Phase 10.2 MD17/rMD17 NPZ reference adapter."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from mendel.md17 import (
    KCAL_MOL_A_TO_EV_A,
    KCAL_MOL_TO_EV,
    convert_energy_to_ev,
    convert_forces_to_ev_per_angstrom,
    convert_md17_npz_to_reference_json,
    create_tiny_synthetic_md17_npz,
    inspect_md17_npz,
    load_md17_npz_sample,
)
from mendel.reference_data import load_reference_records_json

_ROOT = Path(__file__).parent.parent
_SCRIPT = _ROOT / "scripts" / "prepare_md17_sample.py"


def _write_tiny_md17(path: Path) -> None:
    z = np.array([6, 1, 1], dtype=np.int64)
    r = np.array(
        [
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
            [[0.1, 0.0, 0.0], [0.0, 0.1, 1.0], [1.0, 0.0, 0.1]],
        ],
        dtype=float,
    )
    e = np.array([-40.0, -39.9], dtype=float)
    f = np.zeros((2, 3, 3), dtype=float)
    np.savez(path, z=z, R=r, E=e, F=f)


def test_inspect_md17_npz_detects_arrays(tmp_path: Path) -> None:
    sample = tmp_path / "md17.npz"
    _write_tiny_md17(sample)

    summary = inspect_md17_npz(sample)

    assert summary["supported"] is True
    assert summary["n_conformers"] == 2
    assert summary["n_atoms"] == 3


def test_kcal_mol_to_ev_conversion() -> None:
    assert convert_energy_to_ev(1.0, "kcal/mol") == KCAL_MOL_TO_EV
    assert convert_energy_to_ev(2.0, "eV") == 2.0


def test_kcal_mol_per_angstrom_to_ev_per_angstrom_conversion() -> None:
    converted = convert_forces_to_ev_per_angstrom([[1.0, -2.0, 0.0]], "kcal/mol/Angstrom")

    assert converted == [[KCAL_MOL_A_TO_EV_A, -2.0 * KCAL_MOL_A_TO_EV_A, 0.0]]


def test_load_md17_npz_sample_converts_records(tmp_path: Path) -> None:
    sample = tmp_path / "md17.npz"
    _write_tiny_md17(sample)

    records = load_md17_npz_sample(sample, max_records=1, molecule_id="tiny")

    assert len(records) == 1
    assert records[0].dataset_name == "MD17/rMD17"
    assert records[0].xyz[0][0] == "C"
    assert records[0].reference_energy == -40.0 * KCAL_MOL_TO_EV
    assert records[0].reference_energy_unit == "eV"
    assert records[0].reference_force_unit == "eV/Angstrom"
    assert records[0].reference_forces == [[0.0, 0.0, 0.0]] * 3
    assert records[0].metadata["original_energy_unit"] == "kcal/mol"
    assert records[0].metadata["unit_conversion_applied"] is True


def test_load_md17_npz_sample_no_convert_units_preserves_original_units(tmp_path: Path) -> None:
    sample = tmp_path / "md17.npz"
    _write_tiny_md17(sample)

    records = load_md17_npz_sample(
        sample,
        max_records=1,
        energy_unit="kcal/mol",
        force_unit="kcal/mol/Angstrom",
        convert_units=False,
    )

    assert records[0].reference_energy == -40.0
    assert records[0].reference_energy_unit == "kcal/mol"
    assert records[0].reference_force_unit == "kcal/mol/Angstrom"
    assert records[0].metadata["unit_conversion_applied"] is False


def test_convert_md17_npz_to_reference_json_writes_report(tmp_path: Path) -> None:
    sample = tmp_path / "md17.npz"
    output = tmp_path / "md17.reference.json"
    _write_tiny_md17(sample)

    report = convert_md17_npz_to_reference_json(sample, output, max_records=2)

    assert output.exists()
    assert report.n_records_written == 2
    assert len(load_reference_records_json(output)) == 2


def test_create_tiny_synthetic_md17_npz(tmp_path: Path) -> None:
    sample = tmp_path / "synthetic_md17.npz"

    create_tiny_synthetic_md17_npz(sample)

    assert inspect_md17_npz(sample)["supported"] is True


def test_prepare_md17_sample_cli_converts_local_npz(tmp_path: Path) -> None:
    sample = tmp_path / "md17.npz"
    output = tmp_path / "md17.reference.json"
    report = tmp_path / "md17_report.json"
    _write_tiny_md17(sample)

    result = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--input",
            str(sample),
            "--output",
            str(output),
            "--report",
            str(report),
            "--max-records",
            "1",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert len(load_reference_records_json(output)) == 1
    assert json.loads(report.read_text(encoding="utf-8"))["n_records_written"] == 1


def test_prepare_md17_sample_cli_creates_synthetic_when_no_input(tmp_path: Path) -> None:
    output = tmp_path / "md17.reference.json"
    report = tmp_path / "md17_report.json"

    result = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
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
    assert result.returncode == 0, result.stderr
    assert payload["metadata"]["synthetic_test_data"] is True
    assert output.exists()


def test_prepare_md17_sample_help_runs() -> None:
    result = subprocess.run(
        [sys.executable, str(_SCRIPT), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--max-records" in result.stdout


def test_no_training_invoked() -> None:
    text = _SCRIPT.read_text(encoding="utf-8").lower()
    for token in ("train_mlp", "fine_tune", "neb", "irc", "transition state", "barrier"):
        assert token not in text
