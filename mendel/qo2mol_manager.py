"""Safe local QO2Mol sample management for Phase 10.1.

This module inspects, registers, samples, and summarizes local QO2Mol-like
sample files. It never downloads the full dataset and does not require MLIP
dependencies.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from mendel.qo2mol import inspect_qo2mol_path, load_qo2mol_sample, qo2mol_metadata
from mendel.reference_data import ReferenceStructureRecord, save_reference_records_json

Scalar = str | int | float | bool
_VALID_STRATEGIES = {"first_n", "random", "element_filtered", "small_molecule_first"}


@dataclass
class QO2MolLocalSource:
    source_id: str
    root_path: str
    detected_format: str
    n_files: int
    total_size_bytes: int | None
    has_energy: bool | None
    has_forces: bool | None
    has_coordinates: bool | None
    has_smiles: bool | None
    metadata: dict[str, Scalar] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "source_id": self.source_id,
            "root_path": self.root_path,
            "detected_format": self.detected_format,
            "n_files": self.n_files,
            "total_size_bytes": self.total_size_bytes,
            "has_energy": self.has_energy,
            "has_forces": self.has_forces,
            "has_coordinates": self.has_coordinates,
            "has_smiles": self.has_smiles,
            "metadata": dict(self.metadata),
        }


@dataclass
class QO2MolSamplePlan:
    source_id: str
    max_records: int
    seed: int
    element_filter: list[str] | None
    require_forces: bool
    require_energy: bool
    strategy: str
    output_path: str
    metadata: dict[str, Scalar] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "source_id": self.source_id,
            "max_records": self.max_records,
            "seed": self.seed,
            "element_filter": list(self.element_filter) if self.element_filter else None,
            "require_forces": self.require_forces,
            "require_energy": self.require_energy,
            "strategy": self.strategy,
            "output_path": self.output_path,
            "metadata": dict(self.metadata),
        }


@dataclass
class QO2MolSampleReport:
    source_id: str
    input_path: str
    output_path: str
    n_records_seen: int
    n_records_selected: int
    n_records_written: int
    n_skipped: int
    skipped_reasons: dict[str, int]
    element_distribution: dict[str, int]
    molecule_size_distribution: dict[str, int]
    energy_unit: str | None
    force_unit: str | None
    reference_method: str | None
    warnings: list[str]
    metadata: dict[str, Scalar] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "source_id": self.source_id,
            "input_path": self.input_path,
            "output_path": self.output_path,
            "n_records_seen": self.n_records_seen,
            "n_records_selected": self.n_records_selected,
            "n_records_written": self.n_records_written,
            "n_skipped": self.n_skipped,
            "skipped_reasons": dict(self.skipped_reasons),
            "element_distribution": dict(self.element_distribution),
            "molecule_size_distribution": dict(self.molecule_size_distribution),
            "energy_unit": self.energy_unit,
            "force_unit": self.force_unit,
            "reference_method": self.reference_method,
            "warnings": list(self.warnings),
            "metadata": dict(self.metadata),
        }


def _source_from_dict(payload: dict[str, object]) -> QO2MolLocalSource:
    has_energy = payload.get("has_energy")
    has_forces = payload.get("has_forces")
    has_coordinates = payload.get("has_coordinates")
    has_smiles = payload.get("has_smiles")
    return QO2MolLocalSource(
        source_id=str(payload["source_id"]),
        root_path=str(payload["root_path"]),
        detected_format=str(payload["detected_format"]),
        n_files=int(payload["n_files"]),
        total_size_bytes=int(payload["total_size_bytes"])
        if payload.get("total_size_bytes") is not None
        else None,
        has_energy=has_energy if isinstance(has_energy, bool) else None,
        has_forces=has_forces if isinstance(has_forces, bool) else None,
        has_coordinates=has_coordinates if isinstance(has_coordinates, bool) else None,
        has_smiles=has_smiles if isinstance(has_smiles, bool) else None,
        metadata=dict(payload.get("metadata", {})),
    )


def _detect_fields(
    path: Path,
    fmt: str,
) -> tuple[bool | None, bool | None, bool | None, bool | None]:
    if fmt not in {"json", "jsonl", "npz"}:
        return None, None, None, None
    try:
        records = load_qo2mol_sample(path, max_records=3)
    except Exception:
        return None, None, None, None
    has_energy = any(record.reference_energy is not None for record in records)
    has_forces = any(record.reference_forces is not None for record in records)
    has_coordinates = any(bool(record.xyz) for record in records)
    has_smiles = any(record.smiles is not None for record in records)
    return has_energy, has_forces, has_coordinates, has_smiles


def inspect_qo2mol_source(path: str | Path) -> QO2MolLocalSource:
    target = Path(path)
    summary = inspect_qo2mol_path(target)
    if target.is_dir():
        files = [p for p in target.rglob("*") if p.is_file()]
        n_files = len(files)
        total_size = sum(path.stat().st_size for path in files)
    elif target.exists():
        n_files = 1
        total_size = target.stat().st_size
    else:
        n_files = 0
        total_size = None
    has_energy, has_forces, has_coordinates, has_smiles = _detect_fields(
        target, str(summary["detected_format"])
    )
    source_id = f"qo2mol::{target.resolve() if target.exists() else target}"
    return QO2MolLocalSource(
        source_id=source_id,
        root_path=str(target),
        detected_format=str(summary["detected_format"]),
        n_files=n_files,
        total_size_bytes=total_size,
        has_energy=has_energy,
        has_forces=has_forces,
        has_coordinates=has_coordinates,
        has_smiles=has_smiles,
        metadata={
            **{
                k: v
                for k, v in qo2mol_metadata().items()
                if isinstance(v, str | int | float | bool)
            },
            "supported_for_loading": bool(summary["supported_for_loading"]),
            "warning": "Do not commit raw QO2Mol data.",
        },
    )


def create_sample_plan(
    source: QO2MolLocalSource,
    output_path: str | Path,
    max_records: int = 100,
    seed: int = 42,
    strategy: str = "random",
    element_filter: list[str] | None = None,
    require_forces: bool = True,
    require_energy: bool = True,
) -> QO2MolSamplePlan:
    if strategy not in _VALID_STRATEGIES:
        raise ValueError(f"Unsupported QO2Mol sampling strategy: {strategy}")
    metadata = {
        k: v for k, v in qo2mol_metadata().items() if isinstance(v, str | int | float | bool)
    }
    metadata["input_path"] = source.root_path
    metadata["local_sample_only"] = True
    return QO2MolSamplePlan(
        source_id=source.source_id,
        max_records=max_records,
        seed=seed,
        element_filter=list(element_filter) if element_filter else None,
        require_forces=require_forces,
        require_energy=require_energy,
        strategy=strategy,
        output_path=str(output_path),
        metadata=metadata,
    )


def _record_elements(record: ReferenceStructureRecord) -> set[str]:
    return {symbol for symbol, _, _, _ in record.xyz}


def _filter_records(
    records: list[ReferenceStructureRecord],
    plan: QO2MolSamplePlan,
) -> tuple[list[ReferenceStructureRecord], Counter[str]]:
    skipped: Counter[str] = Counter()
    selected: list[ReferenceStructureRecord] = []
    allowed = set(plan.element_filter or [])
    for record in records:
        if not record.xyz:
            skipped["missing_coordinates"] += 1
            continue
        if plan.require_energy and record.reference_energy is None:
            skipped["missing_energy"] += 1
            continue
        if plan.require_forces and record.reference_forces is None:
            skipped["missing_forces"] += 1
            continue
        if allowed and not (_record_elements(record) & allowed):
            skipped["element_filter"] += 1
            continue
        selected.append(record)
    return selected, skipped


def _sample_records(
    records: list[ReferenceStructureRecord],
    plan: QO2MolSamplePlan,
) -> list[ReferenceStructureRecord]:
    if plan.strategy == "first_n" or plan.strategy == "element_filtered":
        return records[: plan.max_records]
    if plan.strategy == "small_molecule_first":
        return sorted(records, key=lambda record: (len(record.xyz), record.structure_id))[
            : plan.max_records
        ]
    rng = random.Random(plan.seed)
    shuffled = list(records)
    rng.shuffle(shuffled)
    return shuffled[: plan.max_records]


def execute_sample_plan(
    plan: QO2MolSamplePlan,
) -> tuple[list[ReferenceStructureRecord], QO2MolSampleReport]:
    input_path = Path(str(plan.metadata.get("input_path", "")))
    source = inspect_qo2mol_source(input_path)
    if source.detected_format == "directory":
        raise ValueError("unsupported directory source; provide a JSON/JSONL/NPZ sample file")
    if source.detected_format not in {"json", "jsonl", "npz"}:
        raise ValueError("unsupported source format; provide JSON/JSONL/NPZ sample")
    raw_records = load_qo2mol_sample(
        input_path,
        max_records=max(plan.max_records * 20, plan.max_records),
    )
    filtered, skipped = _filter_records(raw_records, plan)
    selected = _sample_records(filtered, plan)
    save_reference_records_json(selected, plan.output_path)
    summary = summarize_reference_sample(selected)
    warnings = ["Do not commit raw QO2Mol data.", "Verify dataset license before redistribution."]
    report = QO2MolSampleReport(
        source_id=plan.source_id,
        input_path=str(input_path),
        output_path=plan.output_path,
        n_records_seen=len(raw_records),
        n_records_selected=len(selected),
        n_records_written=len(selected),
        n_skipped=sum(skipped.values()),
        skipped_reasons=dict(sorted(skipped.items())),
        element_distribution=summary["element_distribution"],  # type: ignore[assignment]
        molecule_size_distribution=summary["atom_count_distribution"],  # type: ignore[assignment]
        energy_unit=selected[0].reference_energy_unit if selected else None,
        force_unit=selected[0].reference_force_unit if selected else None,
        reference_method=selected[0].reference_method if selected else None,
        warnings=warnings,
        metadata={**plan.metadata, "strategy": plan.strategy},
    )
    return selected, report


def save_sample_report(report: QO2MolSampleReport, path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")


def save_source_registry(sources: list[QO2MolLocalSource], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"sources": [source.to_dict() for source in sources]}, indent=2),
        encoding="utf-8",
    )


def load_source_registry(path: str | Path) -> list[QO2MolLocalSource]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [_source_from_dict(source) for source in payload.get("sources", [])]


def summarize_reference_sample(records: list[ReferenceStructureRecord]) -> dict[str, object]:
    element_counts: Counter[str] = Counter()
    atom_count_distribution: Counter[str] = Counter()
    energies: list[float] = []
    force_norms: list[float] = []
    n_missing_energy = 0
    n_missing_forces = 0
    n_missing_smiles = 0
    dataset_names: Counter[str] = Counter()
    for record in records:
        dataset_names[record.dataset_name] += 1
        atom_count_distribution[str(len(record.xyz))] += 1
        element_counts.update(symbol for symbol, _, _, _ in record.xyz)
        if record.reference_energy is None:
            n_missing_energy += 1
        else:
            energies.append(record.reference_energy)
        if record.reference_forces is None:
            n_missing_forces += 1
        else:
            for force in record.reference_forces:
                force_norms.append(sum(component * component for component in force) ** 0.5)
        if record.smiles is None:
            n_missing_smiles += 1

    def stats(values: list[float]) -> tuple[float | None, float | None, float | None]:
        if not values:
            return None, None, None
        return min(values), max(values), sum(values) / len(values)

    energy_min, energy_max, energy_mean = stats(energies)
    force_min, force_max, force_mean = stats(force_norms)
    return {
        "n_records": len(records),
        "element_distribution": dict(sorted(element_counts.items())),
        "atom_count_distribution": dict(sorted(atom_count_distribution.items())),
        "energy_min": energy_min,
        "energy_max": energy_max,
        "energy_mean": energy_mean,
        "force_norm_min": force_min,
        "force_norm_max": force_max,
        "force_norm_mean": force_mean,
        "n_missing_energy": n_missing_energy,
        "n_missing_forces": n_missing_forces,
        "n_missing_smiles": n_missing_smiles,
        "metadata_summary": {"dataset_distribution": dict(sorted(dataset_names.items()))},
    }


def save_reference_summary(summary: dict[str, object], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
