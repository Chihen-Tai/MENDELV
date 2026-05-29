"""Tests for Phase 10.4 functional-group-local force error analysis."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

from mendel.local_force_analysis import (
    AtomForceErrorRecord,
    FunctionalGroupForceAnalysisReport,
    FunctionalGroupForceErrorRecord,
    build_group_assignments,
    build_pseudo_group_assignments,
    compute_atom_force_error_records,
    compute_functional_group_force_errors,
    summarize_group_type_errors,
)
from mendel.reference_data import (
    MLIPStructurePrediction,
    ReferenceStructureRecord,
    save_mlip_predictions_json,
    save_reference_records_json,
)

_ROOT = Path(__file__).parent.parent
_SCRIPT = _ROOT / "scripts" / "analyze_functional_group_force_errors.py"


def _reference(smiles: str | None = None) -> ReferenceStructureRecord:
    return ReferenceStructureRecord(
        structure_id="s1",
        molecule_id="m1",
        dataset_name="tiny",
        smiles=smiles,
        xyz=[("C", 0.0, 0.0, 0.0), ("H", 0.0, 0.0, 1.0), ("O", 1.0, 0.0, 0.0)],
        charge=0,
        multiplicity=1,
        reference_energy=0.0,
        reference_energy_unit="eV",
        reference_forces=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
        reference_force_unit="eV/Angstrom",
        reference_method="test",
        split=None,
        metadata={},
    )


def _prediction() -> MLIPStructurePrediction:
    return MLIPStructurePrediction(
        structure_id="s1",
        backend_name="fake",
        model_name="fake",
        energy=0.0,
        energy_unit="eV",
        forces=[[1.0, 0.0, 0.0], [1.0, 2.0, 0.0], [0.0, 2.0, 3.0]],
        force_unit="eV/Angstrom",
        success=True,
        warnings=[],
        metadata={},
    )


def test_atom_force_error_record_serializes() -> None:
    record = AtomForceErrorRecord(
        structure_id="s1",
        atom_index=0,
        element="C",
        reference_force=[0.0, 0.0, 0.0],
        predicted_force=[1.0, 0.0, 0.0],
        force_error=[1.0, 0.0, 0.0],
        force_error_norm=1.0,
        reference_force_norm=0.0,
        predicted_force_norm=1.0,
        group_ids=["g1"],
        group_types=["whole_molecule"],
        metadata={"x": True},
    )

    assert record.to_dict()["force_error_norm"] == 1.0


def test_functional_group_force_error_record_serializes() -> None:
    record = FunctionalGroupForceErrorRecord(
        structure_id="s1",
        group_id="g1",
        group_type="whole_molecule",
        atom_indices=[0],
        elements=["C"],
        n_atoms=1,
        force_mae=1.0,
        force_rmse=1.0,
        mean_force_error_norm=1.0,
        max_force_error_norm=1.0,
        mean_reference_force_norm=0.0,
        mean_predicted_force_norm=1.0,
        metadata={},
    )

    assert record.to_dict()["group_type"] == "whole_molecule"


def test_functional_group_force_analysis_report_serializes() -> None:
    report = FunctionalGroupForceAnalysisReport(
        dataset_name="tiny",
        n_structures=1,
        n_atoms=1,
        n_groups=0,
        global_force_mae=1.0,
        global_force_rmse=1.0,
        per_element_force_rmse={"C": 1.0},
        per_group_type_force_rmse={},
        per_group_type_mean_error_norm={},
        top_group_type_errors=[],
        atom_records=[],
        group_records=[],
        failures=[],
        metadata={},
    )

    assert report.to_dict()["n_structures"] == 1


def test_compute_atom_force_error_records_computes_norm() -> None:
    records = compute_atom_force_error_records([_reference()], [_prediction()])

    assert len(records) == 3
    assert records[0].force_error == [1.0, 0.0, 0.0]
    assert records[0].force_error_norm == 1.0
    assert math.isclose(records[2].force_error_norm, 3.0)


def test_compute_functional_group_force_errors_aggregates_atoms() -> None:
    atom_records = compute_atom_force_error_records([_reference()], [_prediction()])
    assignments = {"s1": [{"group_id": "g1", "group_type": "test_group", "atom_indices": [0, 1]}]}

    group_records = compute_functional_group_force_errors(atom_records, assignments)

    assert len(group_records) == 1
    assert group_records[0].n_atoms == 2
    assert group_records[0].force_rmse is not None


def test_summarize_group_type_errors_ranks_group_types() -> None:
    groups = [
        FunctionalGroupForceErrorRecord(
            "s1", "g1", "low", [0], ["C"], 1, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, {}
        ),
        FunctionalGroupForceErrorRecord(
            "s1", "g2", "high", [1], ["O"], 1, 2.0, 3.0, 3.0, 3.0, 0.0, 3.0, {}
        ),
    ]

    summary = summarize_group_type_errors(groups)

    assert summary["top_group_types_by_rmse"][0]["group_type"] == "high"


def test_build_group_assignments_handles_missing_smiles_gracefully() -> None:
    assignments = build_group_assignments([_reference(smiles=None)])

    assert assignments["s1"] == []


def test_pseudo_groups_include_expected_groups() -> None:
    assignments = build_pseudo_group_assignments([_reference(smiles=None)])
    group_types = {group["group_type"] for group in assignments["s1"]}

    assert {
        "whole_molecule",
        "heavy_atoms",
        "hydrogens",
        "element_C",
        "element_H",
        "element_O",
    } <= group_types
    assert all(group["metadata"]["pseudo_group"] is True for group in assignments["s1"])


def test_cli_smoke_with_pseudo_groups(tmp_path: Path) -> None:
    reference = tmp_path / "reference.json"
    predictions = tmp_path / "predictions.json"
    output = tmp_path / "analysis.json"
    save_reference_records_json([_reference()], reference)
    save_mlip_predictions_json([_prediction()], predictions)

    result = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--reference",
            str(reference),
            "--predictions",
            str(predictions),
            "--output",
            str(output),
            "--use-pseudo-groups",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert result.returncode == 0, result.stderr
    assert payload["n_groups"] > 0
    assert "whole_molecule" in payload["per_group_type_force_rmse"]


def test_cli_require_groups_fails_without_groups(tmp_path: Path) -> None:
    reference = tmp_path / "reference.json"
    predictions = tmp_path / "predictions.json"
    output = tmp_path / "analysis.json"
    save_reference_records_json([_reference(smiles=None)], reference)
    save_mlip_predictions_json([_prediction()], predictions)

    result = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--reference",
            str(reference),
            "--predictions",
            str(predictions),
            "--output",
            str(output),
            "--require-groups",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "No functional groups" in result.stderr


def test_no_mlip_training_invoked() -> None:
    text = _SCRIPT.read_text(encoding="utf-8").lower()
    for token in ("train_mlp", "fine_tune", "neb", "irc", "transition state", "barrier"):
        assert token not in text
