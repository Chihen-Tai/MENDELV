"""Functional-group-local force error analysis for reference MLIP benchmarks."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from rdkit import Chem

from mendel.identifier import identify_functional_groups_in_mol
from mendel.reference_data import (
    MLIPStructurePrediction,
    ReferenceStructureRecord,
    compute_energy_force_benchmark,
    compute_force_mae,
    compute_force_rmse,
    load_mlip_predictions_json,
    load_reference_records_json,
)

Scalar = str | int | float | bool


@dataclass
class AtomForceErrorRecord:
    structure_id: str
    atom_index: int
    element: str
    reference_force: list[float]
    predicted_force: list[float]
    force_error: list[float]
    force_error_norm: float
    reference_force_norm: float
    predicted_force_norm: float
    group_ids: list[str]
    group_types: list[str]
    metadata: dict[str, Scalar] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "structure_id": self.structure_id,
            "atom_index": self.atom_index,
            "element": self.element,
            "reference_force": list(self.reference_force),
            "predicted_force": list(self.predicted_force),
            "force_error": list(self.force_error),
            "force_error_norm": self.force_error_norm,
            "reference_force_norm": self.reference_force_norm,
            "predicted_force_norm": self.predicted_force_norm,
            "group_ids": list(self.group_ids),
            "group_types": list(self.group_types),
            "metadata": dict(self.metadata),
        }


@dataclass
class FunctionalGroupForceErrorRecord:
    structure_id: str
    group_id: str
    group_type: str
    atom_indices: list[int]
    elements: list[str]
    n_atoms: int
    force_mae: float | None
    force_rmse: float | None
    mean_force_error_norm: float | None
    max_force_error_norm: float | None
    mean_reference_force_norm: float | None
    mean_predicted_force_norm: float | None
    metadata: dict[str, Scalar] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "structure_id": self.structure_id,
            "group_id": self.group_id,
            "group_type": self.group_type,
            "atom_indices": list(self.atom_indices),
            "elements": list(self.elements),
            "n_atoms": self.n_atoms,
            "force_mae": self.force_mae,
            "force_rmse": self.force_rmse,
            "mean_force_error_norm": self.mean_force_error_norm,
            "max_force_error_norm": self.max_force_error_norm,
            "mean_reference_force_norm": self.mean_reference_force_norm,
            "mean_predicted_force_norm": self.mean_predicted_force_norm,
            "metadata": dict(self.metadata),
        }


@dataclass
class FunctionalGroupForceAnalysisReport:
    dataset_name: str
    n_structures: int
    n_atoms: int
    n_groups: int
    global_force_mae: float | None
    global_force_rmse: float | None
    per_element_force_rmse: dict[str, float]
    per_group_type_force_rmse: dict[str, float]
    per_group_type_mean_error_norm: dict[str, float]
    top_group_type_errors: list[dict[str, object]]
    atom_records: list[AtomForceErrorRecord]
    group_records: list[FunctionalGroupForceErrorRecord]
    failures: list[dict[str, object]]
    metadata: dict[str, Scalar] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset_name": self.dataset_name,
            "n_structures": self.n_structures,
            "n_atoms": self.n_atoms,
            "n_groups": self.n_groups,
            "global_force_mae": self.global_force_mae,
            "global_force_rmse": self.global_force_rmse,
            "per_element_force_rmse": dict(self.per_element_force_rmse),
            "per_group_type_force_rmse": dict(self.per_group_type_force_rmse),
            "per_group_type_mean_error_norm": dict(self.per_group_type_mean_error_norm),
            "top_group_type_errors": [dict(item) for item in self.top_group_type_errors],
            "atom_records": [record.to_dict() for record in self.atom_records],
            "group_records": [record.to_dict() for record in self.group_records],
            "failures": [dict(failure) for failure in self.failures],
            "metadata": dict(self.metadata),
        }


def _norm(vector: list[float]) -> float:
    return math.sqrt(sum(float(value) ** 2 for value in vector))


def _group_index(
    group_assignments: dict[str, list[dict[str, object]]] | None,
) -> dict[str, dict[int, tuple[list[str], list[str]]]]:
    index: dict[str, dict[int, tuple[list[str], list[str]]]] = {}
    for structure_id, groups in (group_assignments or {}).items():
        atom_map: dict[int, tuple[list[str], list[str]]] = {}
        for group in groups:
            group_id = str(group["group_id"])
            group_type = str(group["group_type"])
            for atom_idx in group.get("atom_indices", []):
                ids, types = atom_map.setdefault(int(atom_idx), ([], []))
                ids.append(group_id)
                types.append(group_type)
        index[structure_id] = atom_map
    return index


def compute_atom_force_error_records(
    reference_records: list[ReferenceStructureRecord],
    predictions: list[MLIPStructurePrediction],
    group_assignments: dict[str, list[dict[str, object]]] | None = None,
) -> list[AtomForceErrorRecord]:
    predictions_by_id = {prediction.structure_id: prediction for prediction in predictions}
    groups_by_atom = _group_index(group_assignments)
    atom_records: list[AtomForceErrorRecord] = []
    for reference in reference_records:
        prediction = predictions_by_id.get(reference.structure_id)
        if (
            prediction is None
            or not prediction.success
            or reference.reference_forces is None
            or prediction.forces is None
            or len(reference.reference_forces) != len(prediction.forces)
        ):
            continue
        atom_group_index = groups_by_atom.get(reference.structure_id, {})
        for atom_idx, (atom, reference_force, predicted_force) in enumerate(
            zip(reference.xyz, reference.reference_forces, prediction.forces, strict=True)
        ):
            force_error = [
                float(predicted) - float(reference_component)
                for reference_component, predicted in zip(
                    reference_force,
                    predicted_force,
                    strict=True,
                )
            ]
            group_ids, group_types = atom_group_index.get(atom_idx, ([], []))
            atom_records.append(
                AtomForceErrorRecord(
                    structure_id=reference.structure_id,
                    atom_index=atom_idx,
                    element=atom[0],
                    reference_force=[float(value) for value in reference_force],
                    predicted_force=[float(value) for value in predicted_force],
                    force_error=force_error,
                    force_error_norm=_norm(force_error),
                    reference_force_norm=_norm([float(value) for value in reference_force]),
                    predicted_force_norm=_norm([float(value) for value in predicted_force]),
                    group_ids=list(group_ids),
                    group_types=list(group_types),
                    metadata={},
                )
            )
    return atom_records


def identify_groups_for_reference_record(
    record: ReferenceStructureRecord,
) -> list[dict[str, object]]:
    if not record.smiles:
        return []
    mol = Chem.MolFromSmiles(record.smiles)
    if mol is None:
        return [
            {
                "group_id": f"{record.structure_id}_invalid_smiles",
                "group_type": "unknown",
                "atom_indices": [],
                "metadata": {"warning": "RDKit could not parse record SMILES."},
            }
        ]
    groups = identify_functional_groups_in_mol(mol, molecule_index=0)
    return [
        {
            "group_id": f"{record.structure_id}_{group.group_id}",
            "group_type": group.group_type.value,
            "atom_indices": [atom_ref.atom_index for atom_ref in group.atom_refs],
            "metadata": {
                **dict(group.metadata),
                "source": "smiles_functional_group_detection",
            },
        }
        for group in groups
    ]


def build_group_assignments(
    reference_records: list[ReferenceStructureRecord],
) -> dict[str, list[dict[str, object]]]:
    return {
        record.structure_id: identify_groups_for_reference_record(record)
        for record in reference_records
    }


def build_pseudo_group_assignments(
    reference_records: list[ReferenceStructureRecord],
) -> dict[str, list[dict[str, object]]]:
    assignments: dict[str, list[dict[str, object]]] = {}
    for record in reference_records:
        all_atoms = list(range(len(record.xyz)))
        heavy_atoms = [idx for idx, atom in enumerate(record.xyz) if atom[0] != "H"]
        hydrogens = [idx for idx, atom in enumerate(record.xyz) if atom[0] == "H"]
        groups: list[dict[str, object]] = []

        def add_group(
            group_type: str,
            atom_indices: list[int],
            *,
            structure_id: str = record.structure_id,
            output_groups: list[dict[str, object]] = groups,
        ) -> None:
            if not atom_indices:
                return
            output_groups.append({
                "group_id": f"{structure_id}_{group_type}",
                "group_type": group_type,
                "atom_indices": list(atom_indices),
                "metadata": {
                    "pseudo_group": True,
                    "not_chemical_functional_group": True,
                },
            })

        add_group("whole_molecule", all_atoms)
        add_group("heavy_atoms", heavy_atoms)
        add_group("hydrogens", hydrogens)
        for element in sorted({atom[0] for atom in record.xyz}):
            add_group(
                f"element_{element}",
                [idx for idx, atom in enumerate(record.xyz) if atom[0] == element],
            )
        assignments[record.structure_id] = groups
    return assignments


def compute_functional_group_force_errors(
    atom_records: list[AtomForceErrorRecord],
    group_assignments: dict[str, list[dict[str, object]]],
) -> list[FunctionalGroupForceErrorRecord]:
    atoms_by_structure = defaultdict(dict)
    for atom in atom_records:
        atoms_by_structure[atom.structure_id][atom.atom_index] = atom
    group_records: list[FunctionalGroupForceErrorRecord] = []
    for structure_id, groups in group_assignments.items():
        for group in groups:
            atom_indices = [int(idx) for idx in group.get("atom_indices", [])]
            atoms = [
                atoms_by_structure[structure_id][idx]
                for idx in atom_indices
                if idx in atoms_by_structure.get(structure_id, {})
            ]
            if not atoms:
                continue
            reference_forces = [atom.reference_force for atom in atoms]
            predicted_forces = [atom.predicted_force for atom in atoms]
            error_norms = [atom.force_error_norm for atom in atoms]
            group_records.append(
                FunctionalGroupForceErrorRecord(
                    structure_id=structure_id,
                    group_id=str(group["group_id"]),
                    group_type=str(group["group_type"]),
                    atom_indices=[atom.atom_index for atom in atoms],
                    elements=[atom.element for atom in atoms],
                    n_atoms=len(atoms),
                    force_mae=compute_force_mae(reference_forces, predicted_forces),
                    force_rmse=compute_force_rmse(reference_forces, predicted_forces),
                    mean_force_error_norm=sum(error_norms) / len(error_norms),
                    max_force_error_norm=max(error_norms),
                    mean_reference_force_norm=sum(atom.reference_force_norm for atom in atoms)
                    / len(atoms),
                    mean_predicted_force_norm=sum(atom.predicted_force_norm for atom in atoms)
                    / len(atoms),
                    metadata=dict(group.get("metadata", {})),
                )
            )
    return group_records


def summarize_group_type_errors(
    group_records: list[FunctionalGroupForceErrorRecord],
) -> dict[str, object]:
    by_type: dict[str, list[FunctionalGroupForceErrorRecord]] = defaultdict(list)
    for record in group_records:
        by_type[record.group_type].append(record)
    per_rmse: dict[str, float] = {}
    per_mean_norm: dict[str, float] = {}
    counts: dict[str, int] = {}
    for group_type, records in by_type.items():
        rmses = [record.force_rmse for record in records if record.force_rmse is not None]
        means = [
            record.mean_force_error_norm
            for record in records
            if record.mean_force_error_norm is not None
        ]
        counts[group_type] = len(records)
        if rmses:
            per_rmse[group_type] = sum(rmses) / len(rmses)
        if means:
            per_mean_norm[group_type] = sum(means) / len(means)
    top_by_rmse = [
        {"group_type": group_type, "force_rmse": value, "n_groups": counts[group_type]}
        for group_type, value in sorted(per_rmse.items(), key=lambda item: item[1], reverse=True)
    ]
    top_by_mean = [
        {
            "group_type": group_type,
            "mean_force_error_norm": value,
            "n_groups": counts[group_type],
        }
        for group_type, value in sorted(
            per_mean_norm.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    ]
    return {
        "per_group_type_force_rmse": per_rmse,
        "per_group_type_mean_error_norm": per_mean_norm,
        "counts_per_group_type": counts,
        "top_group_types_by_rmse": top_by_rmse,
        "top_group_types_by_mean_error_norm": top_by_mean,
    }


def build_functional_group_force_analysis_report(
    reference_path: str | Path,
    predictions_path: str | Path,
    benchmark_path: str | Path | None = None,
    use_pseudo_groups: bool = False,
) -> FunctionalGroupForceAnalysisReport:
    reference_records = load_reference_records_json(reference_path)
    predictions = load_mlip_predictions_json(predictions_path)
    group_assignments = build_group_assignments(reference_records)
    true_group_count = sum(len(groups) for groups in group_assignments.values())
    missing_smiles = sum(1 for record in reference_records if not record.smiles)
    if use_pseudo_groups:
        pseudo = build_pseudo_group_assignments(reference_records)
        for structure_id, groups in pseudo.items():
            group_assignments.setdefault(structure_id, []).extend(groups)
    atom_records = compute_atom_force_error_records(
        reference_records,
        predictions,
        group_assignments,
    )
    group_records = compute_functional_group_force_errors(atom_records, group_assignments)
    benchmark = compute_energy_force_benchmark(reference_records, predictions)
    if benchmark_path is not None and Path(benchmark_path).exists():
        benchmark_payload = json.loads(Path(benchmark_path).read_text(encoding="utf-8"))
        global_force_mae = benchmark_payload.get("force_mae", benchmark.force_mae)
        global_force_rmse = benchmark_payload.get("force_rmse", benchmark.force_rmse)
        per_element = benchmark_payload.get(
            "per_element_force_rmse",
            benchmark.per_element_force_rmse,
        )
    else:
        global_force_mae = benchmark.force_mae
        global_force_rmse = benchmark.force_rmse
        per_element = benchmark.per_element_force_rmse
    summary = summarize_group_type_errors(group_records)
    failures: list[dict[str, object]] = []
    if missing_smiles:
        failures.append({
            "reason": "missing_smiles",
            "n_structures": missing_smiles,
            "message": "No SMILES available; functional group assignment skipped or limited.",
        })
    return FunctionalGroupForceAnalysisReport(
        dataset_name=reference_records[0].dataset_name if reference_records else "unknown",
        n_structures=len(reference_records),
        n_atoms=len(atom_records),
        n_groups=len(group_records),
        global_force_mae=float(global_force_mae) if global_force_mae is not None else None,
        global_force_rmse=float(global_force_rmse) if global_force_rmse is not None else None,
        per_element_force_rmse={str(k): float(v) for k, v in dict(per_element).items()},
        per_group_type_force_rmse=summary["per_group_type_force_rmse"],  # type: ignore[assignment]
        per_group_type_mean_error_norm=summary["per_group_type_mean_error_norm"],  # type: ignore[assignment]
        top_group_type_errors=summary["top_group_types_by_rmse"],  # type: ignore[assignment]
        atom_records=atom_records,
        group_records=group_records,
        failures=failures,
        metadata={
            "true_functional_groups_found": true_group_count > 0,
            "true_functional_group_count": true_group_count,
            "pseudo_groups_used": use_pseudo_groups,
            "missing_smiles_count": missing_smiles,
            "scope_note": "Molecule-conformer local force analysis only; no MLIP training.",
        },
    )


def save_functional_group_force_analysis_report(
    report: FunctionalGroupForceAnalysisReport,
    path: str | Path,
) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
