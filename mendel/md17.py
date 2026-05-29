"""MD17/rMD17-style NPZ adapter for Phase 10.2 reference benchmarks.

This module converts local molecule conformer energy/force arrays into
MENDELV ``ReferenceStructureRecord`` JSON. It does not download data, train
MLIP, run DFT, or evaluate reaction paths.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rdkit.Chem import GetPeriodicTable

from mendel.reference_data import ReferenceStructureRecord, save_reference_records_json

Scalar = str | int | float | bool
KCAL_MOL_TO_EV = 0.0433641153087705
KCAL_MOL_A_TO_EV_A = 0.0433641153087705

_ENERGY_FACTORS = {
    "ev": 1.0,
    "eV": 1.0,
    "kcal/mol": KCAL_MOL_TO_EV,
}
_FORCE_FACTORS = {
    "ev/angstrom": 1.0,
    "eV/Angstrom": 1.0,
    "eV/angstrom": 1.0,
    "kcal/mol/angstrom": KCAL_MOL_A_TO_EV_A,
    "kcal/mol/Angstrom": KCAL_MOL_A_TO_EV_A,
}


@dataclass
class MD17IngestionReport:
    dataset_name: str
    input_path: str
    output_path: str
    n_records_read: int
    n_records_written: int
    n_skipped: int
    skipped_reasons: dict[str, int]
    metadata: dict[str, Scalar] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset_name": self.dataset_name,
            "input_path": self.input_path,
            "output_path": self.output_path,
            "n_records_read": self.n_records_read,
            "n_records_written": self.n_records_written,
            "n_skipped": self.n_skipped,
            "skipped_reasons": dict(self.skipped_reasons),
            "metadata": dict(self.metadata),
        }


def md17_metadata() -> dict[str, object]:
    return {
        "dataset_name": "MD17/rMD17",
        "source_url": "user-provided local path or explicit URL",
        "reference_method": "dataset-provided DFT energies/forces",
        "data_type": "molecular conformer energy/force",
        "license_note": "verify source dataset license before redistribution",
        "access_date": datetime.now(UTC).date().isoformat(),
    }


def _normalize_unit(unit: str) -> str:
    return unit.strip().replace("Å", "Angstrom")


def _energy_factor(source_unit: str, assume_units: bool = False) -> float:
    normalized = _normalize_unit(source_unit)
    if normalized in _ENERGY_FACTORS:
        return _ENERGY_FACTORS[normalized]
    lowered = normalized.lower()
    if lowered in _ENERGY_FACTORS:
        return _ENERGY_FACTORS[lowered]
    if assume_units:
        return 1.0
    raise ValueError(f"Unknown MD17/rMD17 energy unit: {source_unit!r}")


def _force_factor(source_unit: str, assume_units: bool = False) -> float:
    normalized = _normalize_unit(source_unit)
    if normalized in _FORCE_FACTORS:
        return _FORCE_FACTORS[normalized]
    lowered = normalized.lower()
    if lowered in _FORCE_FACTORS:
        return _FORCE_FACTORS[lowered]
    if assume_units:
        return 1.0
    raise ValueError(f"Unknown MD17/rMD17 force unit: {source_unit!r}")


def convert_energy_to_ev(
    value: float,
    source_unit: str,
    assume_units: bool = False,
) -> float:
    return float(value) * _energy_factor(source_unit, assume_units=assume_units)


def convert_forces_to_ev_per_angstrom(
    forces: list[list[float]],
    source_unit: str,
    assume_units: bool = False,
) -> list[list[float]]:
    factor = _force_factor(source_unit, assume_units=assume_units)
    return [[float(value) * factor for value in force] for force in forces]


def _require_numpy() -> Any:
    try:
        import numpy as np  # type: ignore
    except ImportError as exc:  # pragma: no cover - NumPy is present in tests
        raise ImportError("NumPy is required for MD17/rMD17 NPZ conversion.") from exc
    return np


def _array_by_key(data: Any, keys: tuple[str, ...]) -> Any | None:
    for key in keys:
        if key in data.files:
            return data[key]
    return None


def _symbol_from_atomic_number(atomic_number: int) -> str:
    return str(GetPeriodicTable().GetElementSymbol(int(atomic_number)))


def inspect_md17_npz(path: str | Path) -> dict[str, object]:
    np = _require_numpy()
    target = Path(path)
    with np.load(target, allow_pickle=False) as data:
        atomic_numbers = _array_by_key(data, ("z", "Z", "atomic_numbers", "nuclear_charges"))
        coordinates = _array_by_key(data, ("R", "r", "coords", "coordinates", "positions"))
        energies = _array_by_key(data, ("E", "e", "energies", "energy"))
        forces = _array_by_key(data, ("F", "f", "forces"))
        supported = atomic_numbers is not None and coordinates is not None and energies is not None
        n_conformers = int(coordinates.shape[0]) if coordinates is not None else 0
        n_atoms = (
            int(coordinates.shape[1])
            if coordinates is not None and coordinates.ndim >= 2
            else 0
        )
        return {
            "path": str(target),
            "exists": target.exists(),
            "size_bytes": target.stat().st_size if target.exists() else None,
            "arrays": list(data.files),
            "supported": supported,
            "has_forces": forces is not None,
            "n_conformers": n_conformers,
            "n_atoms": n_atoms,
            "energy_shape": tuple(energies.shape) if energies is not None else None,
            "force_shape": tuple(forces.shape) if forces is not None else None,
        }


def _atomic_numbers_for_frame(atomic_numbers: Any, frame_idx: int) -> list[int]:
    if getattr(atomic_numbers, "ndim", 1) == 1:
        return [int(value) for value in atomic_numbers.tolist()]
    return [int(value) for value in atomic_numbers[frame_idx].tolist()]


def load_md17_npz_sample(
    path: str | Path,
    max_records: int = 100,
    molecule_id: str | None = None,
    split: str | None = None,
    energy_unit: str = "kcal/mol",
    force_unit: str = "kcal/mol/Angstrom",
    convert_units: bool = True,
    assume_units: bool = False,
) -> list[ReferenceStructureRecord]:
    np = _require_numpy()
    target = Path(path)
    with np.load(target, allow_pickle=False) as data:
        atomic_numbers = _array_by_key(data, ("z", "Z", "atomic_numbers", "nuclear_charges"))
        coordinates = _array_by_key(data, ("R", "r", "coords", "coordinates", "positions"))
        energies = _array_by_key(data, ("E", "e", "energies", "energy"))
        forces = _array_by_key(data, ("F", "f", "forces"))
        if atomic_numbers is None or coordinates is None or energies is None:
            raise ValueError(
                "MD17/rMD17 NPZ must contain atomic numbers, coordinates, and energies."
            )
        n = min(int(coordinates.shape[0]), int(max_records))
        metadata = {
            k: v for k, v in md17_metadata().items() if isinstance(v, str | int | float | bool)
        }
        metadata.update({
            "source_path": str(target),
            "local_adapter": "md17_npz",
            "molecule_conformer_benchmark_only": True,
            "original_energy_unit": energy_unit,
            "original_force_unit": force_unit,
            "converted_energy_unit": "eV" if convert_units else energy_unit,
            "converted_force_unit": "eV/Angstrom" if convert_units else force_unit,
            "unit_conversion_applied": convert_units,
            "energy_conversion_factor": _energy_factor(energy_unit, assume_units=assume_units)
            if convert_units
            else 1.0,
            "force_conversion_factor": _force_factor(force_unit, assume_units=assume_units)
            if convert_units
            else 1.0,
            "units_assumed": assume_units,
        })
        records: list[ReferenceStructureRecord] = []
        mol_id = molecule_id or target.stem
        for idx in range(n):
            frame_atomic_numbers = _atomic_numbers_for_frame(atomic_numbers, idx)
            symbols = [_symbol_from_atomic_number(value) for value in frame_atomic_numbers]
            xyz = [
                (symbol, float(position[0]), float(position[1]), float(position[2]))
                for symbol, position in zip(symbols, coordinates[idx], strict=True)
            ]
            raw_forces = (
                [[float(value) for value in force] for force in forces[idx].tolist()]
                if forces is not None
                else None
            )
            frame_forces = (
                convert_forces_to_ev_per_angstrom(
                    raw_forces,
                    force_unit,
                    assume_units=assume_units,
                )
                if raw_forces is not None and convert_units
                else raw_forces
            )
            raw_energy = float(energies[idx])
            reference_energy = (
                convert_energy_to_ev(raw_energy, energy_unit, assume_units=assume_units)
                if convert_units
                else raw_energy
            )
            records.append(
                ReferenceStructureRecord(
                    structure_id=f"md17_{mol_id}_{idx}",
                    molecule_id=mol_id,
                    dataset_name="MD17/rMD17",
                    smiles=None,
                    xyz=xyz,
                    charge=0,
                    multiplicity=1,
                    reference_energy=reference_energy,
                    reference_energy_unit="eV" if convert_units else energy_unit,
                    reference_forces=frame_forces,
                    reference_force_unit="eV/Angstrom" if convert_units else force_unit,
                    reference_method="dataset-provided DFT",
                    split=split,
                    metadata=dict(metadata),
                )
            )
        return records


def convert_md17_npz_to_reference_json(
    input_path: str | Path,
    output_path: str | Path,
    max_records: int = 100,
    molecule_id: str | None = None,
    energy_unit: str = "kcal/mol",
    force_unit: str = "kcal/mol/Angstrom",
    convert_units: bool = True,
    assume_units: bool = False,
) -> MD17IngestionReport:
    records = load_md17_npz_sample(
        input_path,
        max_records=max_records,
        molecule_id=molecule_id,
        energy_unit=energy_unit,
        force_unit=force_unit,
        convert_units=convert_units,
        assume_units=assume_units,
    )
    save_reference_records_json(records, output_path)
    report = MD17IngestionReport(
        dataset_name="MD17/rMD17",
        input_path=str(input_path),
        output_path=str(output_path),
        n_records_read=len(records),
        n_records_written=len(records),
        n_skipped=0,
        skipped_reasons={},
        metadata={
            **{
                k: v
                for k, v in md17_metadata().items()
                if isinstance(v, str | int | float | bool)
            },
            "scope_note": "Local MD17/rMD17 NPZ conversion only; no MLIP training.",
            "original_energy_unit": energy_unit,
            "original_force_unit": force_unit,
            "converted_energy_unit": "eV" if convert_units else energy_unit,
            "converted_force_unit": "eV/Angstrom" if convert_units else force_unit,
            "unit_conversion_applied": convert_units,
            "energy_conversion_factor": _energy_factor(energy_unit, assume_units=assume_units)
            if convert_units
            else 1.0,
            "force_conversion_factor": _force_factor(force_unit, assume_units=assume_units)
            if convert_units
            else 1.0,
            "units_assumed": assume_units,
        },
    )
    return report


def create_tiny_synthetic_md17_npz(path: str | Path) -> None:
    np = _require_numpy()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_numbers = np.array([6, 1, 1], dtype=np.int64)
    coordinates = np.array(
        [
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.09], [1.03, 0.0, -0.36]],
            [[0.02, 0.0, 0.0], [0.0, 0.03, 1.08], [1.02, 0.0, -0.34]],
            [[-0.01, 0.0, 0.0], [0.0, -0.02, 1.10], [1.04, 0.0, -0.35]],
        ],
        dtype=float,
    )
    energies = np.array([-40.0, -39.98, -39.99], dtype=float)
    forces = np.zeros((3, 3, 3), dtype=float)
    np.savez(
        out,
        z=atomic_numbers,
        R=coordinates,
        E=energies,
        F=forces,
        synthetic_test_data=True,
    )


def save_md17_ingestion_report(report: MD17IngestionReport, path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
