"""QO2Mol local sample ingestion for Phase 10.

This module does not download QO2Mol. It only inspects and converts local sample
files into MENDELV reference energy/force records.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_ATOMIC_SYMBOL: dict[int, str] = {
    1: "H", 6: "C", 7: "N", 8: "O", 9: "F",
    15: "P", 16: "S", 17: "Cl", 35: "Br", 53: "I",
}

from mendel.reference_data import ReferenceStructureRecord, save_reference_records_json

Scalar = str | int | float | bool


@dataclass
class QO2MolIngestionReport:
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


def qo2mol_metadata() -> dict[str, object]:
    return {
        "dataset_name": "QO2Mol",
        "citation": (
            "An Open Quantum Chemistry Property Database of 120 Kilo Molecules "
            "with 20 Million Conformers"
        ),
        "source_url": "https://github.com/saiscn/QO2Mol/",
        "reference_method": "B3LYP/def2-SVP",
        "data_type": "molecular conformer energy/force",
        "license_note": "license requires manual verification from source repository",
        "access_date": datetime.now(UTC).date().isoformat(),
    }


_BOND_TYPE_MAP: dict[str, object] = {}


def _rdkit_bond_map() -> dict[str, object]:
    if not _BOND_TYPE_MAP:
        from rdkit.Chem import BondType  # type: ignore
        _BOND_TYPE_MAP.update({
            "1": BondType.SINGLE, "2": BondType.DOUBLE,
            "3": BondType.TRIPLE, "ar": BondType.AROMATIC, "1.5": BondType.AROMATIC,
        })
    return _BOND_TYPE_MAP


def qo2mol_record_to_rdkit_mol(record: dict[str, Any]) -> "Any | None":
    """Build RDKit RWMol from QO2Mol pkl record. Returns None on sanitization failure."""
    try:
        from rdkit import Chem  # type: ignore
        from rdkit.Chem import RWMol, Atom  # type: ignore
    except ImportError:
        raise ImportError("rdkit required")
    bond_map = _rdkit_bond_map()
    mol = RWMol()
    for z, fc in zip(record["elements"], record["formal_charge"]):
        atom = Atom(int(z))
        atom.SetFormalCharge(int(fc))
        mol.AddAtom(atom)
    seen: set[tuple[int, int]] = set()
    for (i, j), bt in zip(record["edge_list"], record["edge_attr"]):
        if i < j and (i, j) not in seen:
            seen.add((i, j))
            mol.AddBond(i, j, bond_map.get(str(bt), bond_map["1"]))
    try:
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None


def qo2mol_record_to_smiles(record: dict[str, Any]) -> str | None:
    """Return canonical SMILES for a QO2Mol pkl record, or None on failure."""
    try:
        from rdkit import Chem  # type: ignore
    except ImportError:
        return None
    mol = qo2mol_record_to_rdkit_mol(record)
    return Chem.MolToSmiles(mol) if mol is not None else None


def inspect_qo2mol_path(path: str | Path) -> dict[str, object]:
    target = Path(path)
    suffix = target.suffix.lower()
    detected = "directory" if target.is_dir() else suffix.lstrip(".") if suffix else "unknown"
    if detected in {"jsonl", "ndjson"}:
        detected = "jsonl"
    if detected in {"h5", "hdf5"}:
        detected = "hdf5"
    if detected in {"xyz", "extxyz"}:
        detected = "extxyz" if detected == "extxyz" else "xyz"
    return {
        "path": str(target),
        "exists": target.exists(),
        "is_dir": target.is_dir(),
        "size_bytes": target.stat().st_size if target.exists() and target.is_file() else None,
        "detected_format": detected,
        "supported_for_loading": detected in {"json", "jsonl", "npz", "pkl"},
    }


def _iter_pkl_records(path: Path, max_records: int, seed: int) -> list[dict[str, Any]]:
    import pickle

    with path.open("rb") as fh:
        all_data: list[dict[str, Any]] = pickle.load(fh)  # noqa: S301

    rng = random.Random(seed)
    sample = rng.sample(all_data, min(max_records, len(all_data)))

    records: list[dict[str, Any]] = []
    for item in sample:
        elements: list[int] = item["elements"]
        coords: list[list[float]] = item["coordinates"]
        forces: list[list[float]] = item["forces"]
        symbols = [_ATOMIC_SYMBOL.get(z, f"X{z}") for z in elements]
        xyz = [[sym, *pos] for sym, pos in zip(symbols, coords, strict=True)]
        records.append({
            "structure_id": str(item.get("confid", f"qo2mol_pkl_{len(records)}")),
            "molecule_id": str(item.get("inchikey", "")),
            "xyz": xyz,
            "energy": float(item["energy"]),
            "reference_energy_unit": "eV",
            "forces": [[float(v) for v in f] for f in forces],
            "reference_force_unit": "eV/Angstrom",
            "charge": int(item.get("net_charge", 0)),
            "smiles": None,
            "split": "ood",
        })
    return records


def _iter_json_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("records"), list):
        return [dict(record) for record in payload["records"]]
    if isinstance(payload, list):
        return [dict(record) for record in payload]
    raise ValueError("JSON QO2Mol sample must be a list or contain a top-level records list.")


def _iter_jsonl_records(path: Path, max_records: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        records.append(dict(json.loads(line)))
        if len(records) >= max_records:
            break
    return records


def _iter_npz_records(path: Path, max_records: int) -> list[dict[str, Any]]:
    try:
        import numpy as np  # type: ignore
    except ImportError as exc:
        raise ImportError("NumPy is required to load NPZ QO2Mol samples.") from exc
    data = np.load(path, allow_pickle=True)
    if "records" in data:
        return [dict(record) for record in data["records"].tolist()[:max_records]]
    required = {"symbols", "positions", "energies"}
    if not required.issubset(set(data.files)):
        raise ValueError("NPZ sample must contain records or symbols/positions/energies arrays.")
    records: list[dict[str, Any]] = []
    n = min(len(data["energies"]), max_records)
    for idx in range(n):
        records.append(
            {
                "structure_id": f"qo2mol_npz_{idx}",
                "xyz": [
                    [str(symbol), *[float(value) for value in position]]
                    for symbol, position in zip(
                        data["symbols"][idx],
                        data["positions"][idx],
                        strict=True,
                    )
                ],
                "energy": float(data["energies"][idx]),
                "forces": data["forces"][idx].tolist() if "forces" in data else None,
            }
        )
    return records


def _record_from_qo2mol(raw: dict[str, Any], idx: int) -> ReferenceStructureRecord:
    metadata = qo2mol_metadata()
    xyz_raw = raw.get("xyz") or raw.get("atoms")
    if xyz_raw is None:
        raise ValueError("missing_xyz")
    xyz = [(str(row[0]), float(row[1]), float(row[2]), float(row[3])) for row in xyz_raw]
    record_metadata = {k: v for k, v in metadata.items() if isinstance(v, str | int | float | bool)}
    record_metadata.update({
        "source_dataset": "QO2Mol",
        "local_adapter": "generic_json_jsonl_npz",
    })
    raw_energy = raw["reference_energy"] if "reference_energy" in raw else raw.get("energy")
    raw_forces = (
        raw.get("reference_forces")
        if raw.get("reference_forces") is not None
        else raw.get("forces") or []
    )
    return ReferenceStructureRecord(
        structure_id=str(raw.get("structure_id") or raw.get("id") or f"qo2mol_sample_{idx}"),
        molecule_id=raw.get("molecule_id"),
        dataset_name="QO2Mol",
        smiles=raw.get("smiles"),
        xyz=xyz,
        charge=int(raw["charge"]) if raw.get("charge") is not None else None,
        multiplicity=int(raw["multiplicity"]) if raw.get("multiplicity") is not None else None,
        reference_energy=float(raw_energy) if raw_energy is not None else None,
        reference_energy_unit=str(
            raw.get("reference_energy_unit") or raw.get("energy_unit") or "eV"
        ),
        reference_forces=[
            [float(value) for value in force]
            for force in raw_forces
        ]
        or None,
        reference_force_unit=str(
            raw.get("reference_force_unit") or raw.get("force_unit") or "eV/Angstrom"
        ),
        reference_method="B3LYP/def2-SVP",
        split=raw.get("split"),
        metadata=record_metadata,
    )


def load_qo2mol_sample(
    path: str | Path,
    max_records: int = 100,
    seed: int = 42,
) -> list[ReferenceStructureRecord]:
    target = Path(path)
    summary = inspect_qo2mol_path(target)
    fmt = summary["detected_format"]
    if fmt == "json":
        raw_records = _iter_json_records(target)[:max_records]
    elif fmt == "jsonl":
        raw_records = _iter_jsonl_records(target, max_records)
    elif fmt == "npz":
        raw_records = _iter_npz_records(target, max_records)
    elif fmt == "pkl":
        raw_records = _iter_pkl_records(target, max_records, seed)
    else:
        raise ValueError(
            "QO2Mol format adapter not implemented for this file type. "
            "Provide JSON/NPZ/PKL sample or extend adapter."
        )

    records: list[ReferenceStructureRecord] = []
    for idx, raw in enumerate(raw_records[:max_records]):
        records.append(_record_from_qo2mol(raw, idx))
    return records


def convert_qo2mol_sample_to_reference_json(
    input_path: str | Path,
    output_path: str | Path,
    max_records: int = 100,
    seed: int = 42,
) -> QO2MolIngestionReport:
    records = load_qo2mol_sample(input_path, max_records=max_records, seed=seed)
    save_reference_records_json(records, output_path)
    report = QO2MolIngestionReport(
        dataset_name="QO2Mol",
        input_path=str(input_path),
        output_path=str(output_path),
        n_records_read=len(records),
        n_records_written=len(records),
        n_skipped=0,
        skipped_reasons={},
        metadata=qo2mol_metadata()
        | {"scope_note": "Local sample ingestion only; no QO2Mol download performed."},
    )
    return report


def save_qo2mol_ingestion_report(report: QO2MolIngestionReport, path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
