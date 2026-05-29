"""Reference energy/force dataset records and benchmark math.

Phase 10 compares pretrained MLIP single-point predictions against open
molecular conformer reference data. This module has no MLIP dependency and does
not train models, run DFT, or evaluate reaction paths.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mendel.mlip import optional_import_ase

Scalar = str | int | float | bool


@dataclass
class ReferenceStructureRecord:
    structure_id: str
    molecule_id: str | None
    dataset_name: str
    smiles: str | None
    xyz: list[tuple[str, float, float, float]]
    charge: int | None
    multiplicity: int | None
    reference_energy: float | None
    reference_energy_unit: str
    reference_forces: list[list[float]] | None
    reference_force_unit: str
    reference_method: str | None
    split: str | None
    metadata: dict[str, Scalar] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "structure_id": self.structure_id,
            "molecule_id": self.molecule_id,
            "dataset_name": self.dataset_name,
            "smiles": self.smiles,
            "xyz": [[symbol, x, y, z] for symbol, x, y, z in self.xyz],
            "charge": self.charge,
            "multiplicity": self.multiplicity,
            "reference_energy": self.reference_energy,
            "reference_energy_unit": self.reference_energy_unit,
            "reference_forces": [list(force) for force in self.reference_forces]
            if self.reference_forces is not None
            else None,
            "reference_force_unit": self.reference_force_unit,
            "reference_method": self.reference_method,
            "split": self.split,
            "metadata": dict(self.metadata),
        }


@dataclass
class MLIPStructurePrediction:
    structure_id: str
    backend_name: str
    model_name: str
    energy: float | None
    energy_unit: str
    forces: list[list[float]] | None
    force_unit: str
    success: bool
    warnings: list[str]
    metadata: dict[str, Scalar] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "structure_id": self.structure_id,
            "backend_name": self.backend_name,
            "model_name": self.model_name,
            "energy": self.energy,
            "energy_unit": self.energy_unit,
            "forces": [list(force) for force in self.forces] if self.forces is not None else None,
            "force_unit": self.force_unit,
            "success": self.success,
            "warnings": list(self.warnings),
            "metadata": dict(self.metadata),
        }


@dataclass
class EnergyForceBenchmarkRecord:
    structure_id: str
    molecule_id: str | None
    dataset_name: str
    reference_energy: float | None
    predicted_energy: float | None
    energy_error: float | None
    reference_forces: list[list[float]] | None
    predicted_forces: list[list[float]] | None
    force_errors: list[list[float]] | None
    force_rmse: float | None
    force_mae: float | None
    n_atoms: int
    metadata: dict[str, Scalar] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "structure_id": self.structure_id,
            "molecule_id": self.molecule_id,
            "dataset_name": self.dataset_name,
            "reference_energy": self.reference_energy,
            "predicted_energy": self.predicted_energy,
            "energy_error": self.energy_error,
            "reference_forces": self.reference_forces,
            "predicted_forces": self.predicted_forces,
            "force_errors": self.force_errors,
            "force_rmse": self.force_rmse,
            "force_mae": self.force_mae,
            "n_atoms": self.n_atoms,
            "metadata": dict(self.metadata),
        }


@dataclass
class EnergyForceBenchmarkReport:
    dataset_name: str
    n_structures: int
    n_success: int
    n_failed: int
    energy_mae: float | None
    energy_rmse: float | None
    energy_mae_raw: float | None
    energy_rmse_raw: float | None
    energy_mae_mean_shifted: float | None
    energy_rmse_mean_shifted: float | None
    energy_offset_applied: float | None
    energy_offset_definition: str | None
    force_mae: float | None
    force_rmse: float | None
    per_element_force_rmse: dict[str, float]
    per_structure_force_rmse: list[dict[str, object]]
    max_force_rmse_structure_id: str | None
    n_force_components_compared: int
    n_atoms_compared: int
    records: list[EnergyForceBenchmarkRecord]
    failures: list[dict[str, object]]
    metadata: dict[str, Scalar] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset_name": self.dataset_name,
            "n_structures": self.n_structures,
            "n_success": self.n_success,
            "n_failed": self.n_failed,
            "energy_mae": self.energy_mae,
            "energy_rmse": self.energy_rmse,
            "energy_mae_raw": self.energy_mae_raw,
            "energy_rmse_raw": self.energy_rmse_raw,
            "energy_mae_mean_shifted": self.energy_mae_mean_shifted,
            "energy_rmse_mean_shifted": self.energy_rmse_mean_shifted,
            "energy_offset_applied": self.energy_offset_applied,
            "energy_offset_definition": self.energy_offset_definition,
            "force_mae": self.force_mae,
            "force_rmse": self.force_rmse,
            "per_element_force_rmse": dict(self.per_element_force_rmse),
            "per_structure_force_rmse": [dict(item) for item in self.per_structure_force_rmse],
            "max_force_rmse_structure_id": self.max_force_rmse_structure_id,
            "n_force_components_compared": self.n_force_components_compared,
            "n_atoms_compared": self.n_atoms_compared,
            "records": [record.to_dict() for record in self.records],
            "failures": [dict(failure) for failure in self.failures],
            "metadata": dict(self.metadata),
        }


def _record_from_dict(payload: dict[str, Any]) -> ReferenceStructureRecord:
    required = ("structure_id", "dataset_name", "xyz")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"Reference record missing required fields: {', '.join(missing)}")
    return ReferenceStructureRecord(
        structure_id=str(payload["structure_id"]),
        molecule_id=payload.get("molecule_id"),
        dataset_name=str(payload["dataset_name"]),
        smiles=payload.get("smiles"),
        xyz=[
            (str(row[0]), float(row[1]), float(row[2]), float(row[3]))
            for row in payload["xyz"]
        ],
        charge=int(payload["charge"]) if payload.get("charge") is not None else None,
        multiplicity=int(payload["multiplicity"])
        if payload.get("multiplicity") is not None
        else None,
        reference_energy=float(payload["reference_energy"])
        if payload.get("reference_energy") is not None
        else None,
        reference_energy_unit=str(payload.get("reference_energy_unit", "eV")),
        reference_forces=[
            [float(value) for value in force] for force in payload["reference_forces"]
        ]
        if payload.get("reference_forces") is not None
        else None,
        reference_force_unit=str(payload.get("reference_force_unit", "eV/Angstrom")),
        reference_method=payload.get("reference_method"),
        split=payload.get("split"),
        metadata=dict(payload.get("metadata", {})),
    )


def _prediction_from_dict(payload: dict[str, Any]) -> MLIPStructurePrediction:
    return MLIPStructurePrediction(
        structure_id=str(payload["structure_id"]),
        backend_name=str(payload["backend_name"]),
        model_name=str(payload["model_name"]),
        energy=float(payload["energy"]) if payload.get("energy") is not None else None,
        energy_unit=str(payload.get("energy_unit", "eV")),
        forces=[[float(value) for value in force] for force in payload["forces"]]
        if payload.get("forces") is not None
        else None,
        force_unit=str(payload.get("force_unit", "eV/Angstrom")),
        success=bool(payload.get("success", False)),
        warnings=[str(warning) for warning in payload.get("warnings", [])],
        metadata=dict(payload.get("metadata", {})),
    )


def xyz_to_ase_atoms(record: ReferenceStructureRecord) -> Any:
    imports = optional_import_ase()
    atoms_cls = imports["Atoms"]
    return atoms_cls(
        symbols=[symbol for symbol, _, _, _ in record.xyz],
        positions=[[x, y, z] for _, x, y, z in record.xyz],
    )


def load_reference_records_json(path: str | Path) -> list[ReferenceStructureRecord]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if "records" not in payload or not isinstance(payload["records"], list):
        raise ValueError("Reference JSON must contain a top-level 'records' list.")
    return [_record_from_dict(record) for record in payload["records"]]


def save_reference_records_json(
    records: list[ReferenceStructureRecord],
    path: str | Path,
) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"records": [record.to_dict() for record in records]}, indent=2),
        encoding="utf-8",
    )


def save_mlip_predictions_json(
    predictions: list[MLIPStructurePrediction],
    path: str | Path,
) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"predictions": [prediction.to_dict() for prediction in predictions]}, indent=2),
        encoding="utf-8",
    )


def load_mlip_predictions_json(path: str | Path) -> list[MLIPStructurePrediction]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [_prediction_from_dict(prediction) for prediction in payload.get("predictions", [])]


def compute_force_rmse(
    reference_forces: list[list[float]],
    predicted_forces: list[list[float]],
) -> float:
    errors = [
        (float(predicted) - float(reference)) ** 2
        for ref_force, pred_force in zip(reference_forces, predicted_forces, strict=True)
        for reference, predicted in zip(ref_force, pred_force, strict=True)
    ]
    return math.sqrt(sum(errors) / len(errors)) if errors else 0.0


def compute_force_mae(
    reference_forces: list[list[float]],
    predicted_forces: list[list[float]],
) -> float:
    errors = [
        abs(float(predicted) - float(reference))
        for ref_force, pred_force in zip(reference_forces, predicted_forces, strict=True)
        for reference, predicted in zip(ref_force, pred_force, strict=True)
    ]
    return sum(errors) / len(errors) if errors else 0.0


def compute_energy_rmse(errors: list[float]) -> float:
    return math.sqrt(sum(error * error for error in errors) / len(errors)) if errors else 0.0


def compute_energy_mae(errors: list[float]) -> float:
    return sum(abs(error) for error in errors) / len(errors) if errors else 0.0


def compute_mean_shifted_energy_errors(
    reference_energies: list[float],
    predicted_energies: list[float],
) -> dict[str, object]:
    if len(reference_energies) != len(predicted_energies):
        raise ValueError("reference_energies and predicted_energies must have equal length")
    if not reference_energies:
        return {"offset": None, "shifted_errors": []}
    raw_errors = [
        float(predicted) - float(reference)
        for reference, predicted in zip(reference_energies, predicted_energies, strict=True)
    ]
    offset = sum(raw_errors) / len(raw_errors)
    shifted_errors = [error - offset for error in raw_errors]
    return {"offset": offset, "shifted_errors": shifted_errors}


def _force_errors(
    reference_forces: list[list[float]],
    predicted_forces: list[list[float]],
) -> list[list[float]]:
    return [
        [
            float(predicted) - float(reference)
            for reference, predicted in zip(ref, pred, strict=True)
        ]
        for ref, pred in zip(reference_forces, predicted_forces, strict=True)
    ]


def _per_element_rmse(
    records: list[ReferenceStructureRecord],
    benchmark_records: list[EnergyForceBenchmarkRecord],
) -> dict[str, float]:
    refs_by_id = {record.structure_id: record for record in records}
    squared_by_element: dict[str, list[float]] = {}
    for record in benchmark_records:
        ref = refs_by_id.get(record.structure_id)
        if ref is None or record.force_errors is None:
            continue
        for atom, error in zip(ref.xyz, record.force_errors, strict=True):
            element = atom[0]
            squared_by_element.setdefault(element, []).extend(value * value for value in error)
    return {
        element: math.sqrt(sum(values) / len(values))
        for element, values in sorted(squared_by_element.items())
        if values
    }


def compute_energy_force_benchmark(
    reference_records: list[ReferenceStructureRecord],
    predictions: list[MLIPStructurePrediction],
    energy_unit: str = "eV",
    force_unit: str = "eV/Angstrom",
) -> EnergyForceBenchmarkReport:
    predictions_by_id = {prediction.structure_id: prediction for prediction in predictions}
    benchmark_records: list[EnergyForceBenchmarkRecord] = []
    failures: list[dict[str, object]] = []
    energy_errors: list[float] = []
    reference_energies: list[float] = []
    predicted_energies: list[float] = []
    force_maes: list[float] = []
    force_rmses: list[float] = []
    per_structure_force_rmse: list[dict[str, object]] = []
    n_force_components_compared = 0
    n_atoms_compared = 0
    dataset_name = reference_records[0].dataset_name if reference_records else "unknown"

    for ref in reference_records:
        pred = predictions_by_id.get(ref.structure_id)
        if pred is None:
            failures.append({"structure_id": ref.structure_id, "reason": "missing_prediction"})
            continue
        if not pred.success:
            failures.append({"structure_id": ref.structure_id, "reason": "prediction_failed"})
            continue
        if ref.reference_energy_unit != energy_unit or pred.energy_unit != energy_unit:
            failures.append({"structure_id": ref.structure_id, "reason": "energy_unit_mismatch"})
            continue
        if ref.reference_force_unit != force_unit or pred.force_unit != force_unit:
            failures.append({"structure_id": ref.structure_id, "reason": "force_unit_mismatch"})
            continue
        if pred.forces is not None and len(pred.forces) != len(ref.xyz):
            failures.append({"structure_id": ref.structure_id, "reason": "atom_count_mismatch"})
            continue
        energy_error = (
            pred.energy - ref.reference_energy
            if pred.energy is not None and ref.reference_energy is not None
            else None
        )
        if energy_error is not None:
            energy_errors.append(energy_error)
            reference_energies.append(float(ref.reference_energy))
            predicted_energies.append(float(pred.energy))
        force_errors = None
        force_mae = None
        force_rmse = None
        if ref.reference_forces is not None and pred.forces is not None:
            force_errors = _force_errors(ref.reference_forces, pred.forces)
            force_mae = compute_force_mae(ref.reference_forces, pred.forces)
            force_rmse = compute_force_rmse(ref.reference_forces, pred.forces)
            force_maes.append(force_mae)
            force_rmses.append(force_rmse)
            n_atoms_compared += len(ref.reference_forces)
            n_force_components_compared += sum(len(force) for force in ref.reference_forces)
            per_structure_force_rmse.append({
                "structure_id": ref.structure_id,
                "force_rmse": force_rmse,
                "n_atoms": len(ref.reference_forces),
            })
        benchmark_records.append(
            EnergyForceBenchmarkRecord(
                structure_id=ref.structure_id,
                molecule_id=ref.molecule_id,
                dataset_name=ref.dataset_name,
                reference_energy=ref.reference_energy,
                predicted_energy=pred.energy,
                energy_error=energy_error,
                reference_forces=ref.reference_forces,
                predicted_forces=pred.forces,
                force_errors=force_errors,
                force_rmse=force_rmse,
                force_mae=force_mae,
                n_atoms=len(ref.xyz),
                metadata={"backend_name": pred.backend_name, "model_name": pred.model_name},
            )
        )

    raw_energy_mae = compute_energy_mae(energy_errors) if energy_errors else None
    raw_energy_rmse = compute_energy_rmse(energy_errors) if energy_errors else None
    shifted = compute_mean_shifted_energy_errors(reference_energies, predicted_energies)
    shifted_errors = shifted["shifted_errors"]
    shifted_error_list = (
        [float(error) for error in shifted_errors] if isinstance(shifted_errors, list) else []
    )
    shifted_mae = compute_energy_mae(shifted_error_list) if shifted_error_list else None
    shifted_rmse = compute_energy_rmse(shifted_error_list) if shifted_error_list else None
    max_force_record = (
        max(per_structure_force_rmse, key=lambda item: float(item["force_rmse"]))
        if per_structure_force_rmse
        else None
    )
    return EnergyForceBenchmarkReport(
        dataset_name=dataset_name,
        n_structures=len(reference_records),
        n_success=len(benchmark_records),
        n_failed=len(failures),
        energy_mae=raw_energy_mae,
        energy_rmse=raw_energy_rmse,
        energy_mae_raw=raw_energy_mae,
        energy_rmse_raw=raw_energy_rmse,
        energy_mae_mean_shifted=shifted_mae,
        energy_rmse_mean_shifted=shifted_rmse,
        energy_offset_applied=(
            float(shifted["offset"]) if isinstance(shifted["offset"], float) else None
        ),
        energy_offset_definition="mean(predicted_energy - reference_energy)"
        if energy_errors
        else None,
        force_mae=sum(force_maes) / len(force_maes) if force_maes else None,
        force_rmse=sum(force_rmses) / len(force_rmses) if force_rmses else None,
        per_element_force_rmse=_per_element_rmse(reference_records, benchmark_records),
        per_structure_force_rmse=per_structure_force_rmse,
        max_force_rmse_structure_id=(
            str(max_force_record["structure_id"]) if max_force_record is not None else None
        ),
        n_force_components_compared=n_force_components_compared,
        n_atoms_compared=n_atoms_compared,
        records=benchmark_records,
        failures=failures,
        metadata={
            "energy_unit": energy_unit,
            "force_unit": force_unit,
            "scope_note": "Molecular conformer energy/force benchmark only; no MLIP training.",
        },
    )


def save_energy_force_benchmark_report(
    report: EnergyForceBenchmarkReport,
    path: str | Path,
) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
