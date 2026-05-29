"""Tests for Phase 10 reference energy/force records."""

from __future__ import annotations

import json
import math
from pathlib import Path

from mendel.reference_data import (
    EnergyForceBenchmarkRecord,
    MLIPStructurePrediction,
    ReferenceStructureRecord,
    compute_energy_force_benchmark,
    compute_energy_mae,
    compute_energy_rmse,
    compute_force_mae,
    compute_force_rmse,
    load_reference_records_json,
    save_mlip_predictions_json,
    save_reference_records_json,
)


def _record(structure_id: str = "s1", n_atoms: int = 2) -> ReferenceStructureRecord:
    xyz = [("H", 0.0, 0.0, 0.0), ("H", 0.0, 0.0, 1.0)][:n_atoms]
    return ReferenceStructureRecord(
        structure_id=structure_id,
        molecule_id="m1",
        dataset_name="QO2Mol",
        smiles="[H][H]",
        xyz=xyz,
        charge=0,
        multiplicity=1,
        reference_energy=-1.0,
        reference_energy_unit="eV",
        reference_forces=[[0.0, 0.0, 0.0] for _ in xyz],
        reference_force_unit="eV/Angstrom",
        reference_method="B3LYP/def2-SVP",
        split="test",
        metadata={"synthetic_test_data": True},
    )


def _prediction(structure_id: str = "s1", n_atoms: int = 2) -> MLIPStructurePrediction:
    return MLIPStructurePrediction(
        structure_id=structure_id,
        backend_name="mace",
        model_name="mace-off-small",
        energy=-0.5,
        energy_unit="eV",
        forces=[[0.1, 0.0, 0.0] for _ in range(n_atoms)],
        force_unit="eV/Angstrom",
        success=True,
        warnings=[],
        metadata={},
    )


def test_reference_structure_record_serializes() -> None:
    payload = _record().to_dict()

    assert payload["dataset_name"] == "QO2Mol"
    assert payload["xyz"][0] == ["H", 0.0, 0.0, 0.0]


def test_mlip_structure_prediction_serializes() -> None:
    payload = _prediction().to_dict()

    assert payload["success"] is True
    assert payload["forces"] == [[0.1, 0.0, 0.0], [0.1, 0.0, 0.0]]


def test_energy_force_benchmark_record_serializes() -> None:
    record = EnergyForceBenchmarkRecord(
        structure_id="s1",
        molecule_id="m1",
        dataset_name="QO2Mol",
        reference_energy=-1.0,
        predicted_energy=-0.5,
        energy_error=0.5,
        reference_forces=[[0.0, 0.0, 0.0]],
        predicted_forces=[[0.1, 0.0, 0.0]],
        force_errors=[[0.1, 0.0, 0.0]],
        force_rmse=0.1,
        force_mae=0.03333333333333333,
        n_atoms=1,
        metadata={},
    )

    assert record.to_dict()["energy_error"] == 0.5


def test_force_metrics() -> None:
    ref = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
    pred = [[1.0, 0.0, 0.0], [2.0, 1.0, 1.0]]

    assert math.isclose(compute_force_rmse(ref, pred), math.sqrt(2 / 6))
    assert math.isclose(compute_force_mae(ref, pred), 2 / 6)


def test_energy_metrics() -> None:
    errors = [1.0, -2.0, 3.0]

    assert compute_energy_mae(errors) == 2.0
    assert math.isclose(compute_energy_rmse(errors), math.sqrt(14 / 3))


def test_compute_energy_force_benchmark_matches_by_structure_id() -> None:
    report = compute_energy_force_benchmark([_record()], [_prediction()])

    assert report.n_structures == 1
    assert report.n_success == 1
    assert report.energy_mae == 0.5
    assert report.force_mae is not None
    assert report.per_element_force_rmse["H"] > 0


def test_compute_energy_force_benchmark_handles_missing_prediction() -> None:
    report = compute_energy_force_benchmark([_record()], [])

    assert report.n_failed == 1
    assert report.failures[0]["reason"] == "missing_prediction"


def test_compute_energy_force_benchmark_handles_mismatched_atom_counts() -> None:
    report = compute_energy_force_benchmark([_record(n_atoms=2)], [_prediction(n_atoms=1)])

    assert report.n_failed == 1
    assert report.failures[0]["reason"] == "atom_count_mismatch"


def test_save_load_reference_json_and_predictions(tmp_path: Path) -> None:
    refs_path = tmp_path / "refs.json"
    preds_path = tmp_path / "preds.json"

    save_reference_records_json([_record()], refs_path)
    save_mlip_predictions_json([_prediction()], preds_path)

    assert load_reference_records_json(refs_path)[0].structure_id == "s1"
    payload = json.loads(preds_path.read_text(encoding="utf-8"))
    assert payload["predictions"][0]["structure_id"] == "s1"
