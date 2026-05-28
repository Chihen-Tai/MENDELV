"""Optional Phase 9 pretrained MLIP single-point backend.

This module intentionally does not import ASE, MACE, or torch at module import
time. Optional dependencies are loaded only inside functions that need them.
Phase 9 computes single-point energy and forces only; it does not train MLIP,
run DFT, NEB, IRC, MD, transition-state search, or barrier prediction.
"""

from __future__ import annotations

import inspect
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rdkit import Chem
from rdkit.Chem import AllChem

from mendel.negotiator import run_full_rule_pipeline

Scalar = str | int | float | bool


@dataclass
class MLIPConfig:
    backend_name: str = "mace"
    model_family: str = "mace-off"
    model_name: str = "mace-off-small"
    device: str = "auto"
    dtype: str = "float32"
    optimize_geometry: bool = False
    metadata: dict[str, Scalar] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "backend_name": self.backend_name,
            "model_family": self.model_family,
            "model_name": self.model_name,
            "device": self.device,
            "dtype": self.dtype,
            "optimize_geometry": self.optimize_geometry,
            "metadata": dict(self.metadata),
        }


@dataclass
class MLIPResult:
    energy: float | None
    energy_unit: str
    forces: list[list[float]] | None
    force_unit: str
    n_atoms: int
    backend_name: str
    model_name: str
    device: str
    success: bool
    warnings: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "energy": self.energy,
            "energy_unit": self.energy_unit,
            "forces": [list(force) for force in self.forces] if self.forces is not None else None,
            "force_unit": self.force_unit,
            "n_atoms": self.n_atoms,
            "backend_name": self.backend_name,
            "model_name": self.model_name,
            "device": self.device,
            "success": self.success,
            "warnings": list(self.warnings),
            "metadata": dict(self.metadata),
        }


@dataclass
class ReactionCenterForceSummary:
    reaction_center_atoms: list[int]
    n_center_atoms: int
    mean_center_force_norm: float | None
    max_center_force_norm: float | None
    mean_all_atom_force_norm: float | None
    max_all_atom_force_norm: float | None
    center_to_all_mean_force_ratio: float | None
    metadata: dict[str, Scalar] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "reaction_center_atoms": list(self.reaction_center_atoms),
            "n_center_atoms": self.n_center_atoms,
            "mean_center_force_norm": self.mean_center_force_norm,
            "max_center_force_norm": self.max_center_force_norm,
            "mean_all_atom_force_norm": self.mean_all_atom_force_norm,
            "max_all_atom_force_norm": self.max_all_atom_force_norm,
            "center_to_all_mean_force_ratio": self.center_to_all_mean_force_ratio,
            "metadata": dict(self.metadata),
        }


@dataclass
class MENDELVGuidedMLIPResult:
    reaction_smiles: str
    context: str
    mechanism_hint: str | None
    role_assignments: list[dict[str, object]]
    reaction_center_atoms: list[int]
    center_source: str
    mlip_result: MLIPResult
    force_summary: ReactionCenterForceSummary
    warnings: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "reaction_smiles": self.reaction_smiles,
            "context": self.context,
            "mechanism_hint": self.mechanism_hint,
            "role_assignments": [dict(assignment) for assignment in self.role_assignments],
            "reaction_center_atoms": list(self.reaction_center_atoms),
            "center_source": self.center_source,
            "mlip_result": self.mlip_result.to_dict(),
            "force_summary": self.force_summary.to_dict(),
            "warnings": list(self.warnings),
            "metadata": dict(self.metadata),
        }


@dataclass
class GeometrySanityReport:
    n_atoms: int
    min_interatomic_distance: float | None
    min_distance_atom_pair: list[int] | None
    max_interatomic_distance: float | None
    n_fragments: int | None
    total_formal_charge: int | None
    has_disconnected_fragments: bool
    has_charged_fragments: bool
    has_disconnected_charged_fragments: bool
    mean_force_norm: float | None
    max_force_norm: float | None
    mean_force_threshold: float
    max_force_threshold: float
    min_distance_threshold: float
    status: str
    warnings: list[str]
    metadata: dict[str, Scalar] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "n_atoms": self.n_atoms,
            "min_interatomic_distance": self.min_interatomic_distance,
            "min_distance_atom_pair": (
                list(self.min_distance_atom_pair)
                if self.min_distance_atom_pair is not None
                else None
            ),
            "max_interatomic_distance": self.max_interatomic_distance,
            "n_fragments": self.n_fragments,
            "total_formal_charge": self.total_formal_charge,
            "has_disconnected_fragments": self.has_disconnected_fragments,
            "has_charged_fragments": self.has_charged_fragments,
            "has_disconnected_charged_fragments": self.has_disconnected_charged_fragments,
            "mean_force_norm": self.mean_force_norm,
            "max_force_norm": self.max_force_norm,
            "mean_force_threshold": self.mean_force_threshold,
            "max_force_threshold": self.max_force_threshold,
            "min_distance_threshold": self.min_distance_threshold,
            "status": self.status,
            "warnings": list(self.warnings),
            "metadata": dict(self.metadata),
        }


def optional_import_ase() -> dict[str, Any]:
    try:
        from ase import Atoms  # type: ignore
        from ase.io import read  # type: ignore
        from ase.optimize import BFGS  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "ASE is required for MLIP support. Install with: pip install -e '.[mlip]'"
        ) from exc
    return {"Atoms": Atoms, "read": read, "BFGS": BFGS}


ANI2X_SUPPORTED_ELEMENTS: frozenset[str] = frozenset(
    {"H", "C", "N", "O", "F", "S", "Cl"}
)


def optional_import_torchani() -> dict[str, Any]:
    try:
        import torchani  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "torchani is required for ANI-2x support. Install with: pip install -e '.[ani2x]'"
        ) from exc
    return {"torchani": torchani, "models": torchani.models}


def diagnose_ani2x() -> dict[str, object]:
    try:
        imp = optional_import_torchani()
    except ImportError as exc:
        return {"installed": False, "error": str(exc), "supported_elements": []}
    torchani = imp["torchani"]
    return {
        "installed": True,
        "version": getattr(torchani, "__version__", "unknown"),
        "supported_elements": sorted(ANI2X_SUPPORTED_ELEMENTS),
        "error": None,
    }


def validate_ani2x_elements(atoms: Any) -> list[str]:
    """Return element symbols present in atoms that ANI-2x does not support."""
    symbols: list[str] = (
        atoms.get_chemical_symbols() if hasattr(atoms, "get_chemical_symbols") else []
    )
    return [s for s in symbols if s not in ANI2X_SUPPORTED_ELEMENTS]


def create_ani2x_calculator(config: MLIPConfig) -> Any:
    imports = optional_import_torchani()
    torchani = imports["torchani"]
    device = resolve_device(config.device)
    model = torchani.models.ANI2x(periodic_table_index=True).to(device)
    return model.ase()


def optional_import_mace() -> dict[str, Any]:
    try:
        import mace.calculators as calculators  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "mace-torch is required for MACE MLIP support. Install with: "
            "pip install -e '.[mlip]'"
        ) from exc
    return {
        "module": calculators,
        "mace_mp": getattr(calculators, "mace_mp", None),
        "mace_off": getattr(calculators, "mace_off", None),
        "MACECalculator": getattr(calculators, "MACECalculator", None),
    }


def diagnose_mace_calculators() -> dict[str, object]:
    """Return installed MACE calculator functions without creating a calculator."""
    try:
        mace = optional_import_mace()
    except ImportError as exc:
        return {"installed": False, "error": str(exc), "available_names": [], "calculators": {}}
    module = mace.get("module")
    available_names = (
        sorted(name for name in dir(module) if not name.startswith("_")) if module else []
    )
    calculators: dict[str, object] = {}
    for name in ("mace_off", "mace_mp", "MACECalculator"):
        obj = mace.get(name)
        if obj is None:
            calculators[name] = {"available": False, "signature": None}
            continue
        try:
            signature = str(inspect.signature(obj))
        except Exception:
            signature = None
        calculators[name] = {"available": True, "signature": signature}
    return {
        "installed": True,
        "error": None,
        "available_names": available_names,
        "calculators": calculators,
    }


def resolve_device(device: str = "auto") -> str:
    if device != "auto":
        return device
    try:
        import torch  # type: ignore
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def smiles_to_rdkit_mol_3d(
    smiles: str,
    add_hydrogens: bool = True,
    embed_3d: bool = True,
    optimize_rdkit: bool = True,
    seed: int = 42,
) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")
    if add_hydrogens:
        mol = Chem.AddHs(mol)
    if embed_3d:
        status = AllChem.EmbedMolecule(mol, randomSeed=int(seed))
        if status != 0:
            raise ValueError(f"RDKit 3D conformer embedding failed for SMILES: {smiles!r}")
        if optimize_rdkit:
            try:
                AllChem.UFFOptimizeMolecule(mol, maxIters=200)
            except Exception as exc:  # pragma: no cover - RDKit force-field edge case
                raise ValueError(f"RDKit UFF optimization failed: {exc}") from exc
    return mol


def rdkit_mol_to_ase_atoms(mol: Chem.Mol) -> Any:
    imports = optional_import_ase()
    atoms_cls = imports["Atoms"]
    if mol.GetNumConformers() == 0:
        raise ValueError("RDKit molecule must have a 3D conformer before ASE conversion.")
    conformer = mol.GetConformer()
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    positions = []
    for idx in range(mol.GetNumAtoms()):
        pos = conformer.GetAtomPosition(idx)
        positions.append([float(pos.x), float(pos.y), float(pos.z)])
    return atoms_cls(symbols=symbols, positions=positions)


def smiles_to_ase_atoms(
    smiles: str,
    add_hydrogens: bool = True,
    embed_3d: bool = True,
    optimize_rdkit: bool = True,
    seed: int = 42,
) -> Any:
    mol = smiles_to_rdkit_mol_3d(
        smiles,
        add_hydrogens=add_hydrogens,
        embed_3d=embed_3d,
        optimize_rdkit=optimize_rdkit,
        seed=seed,
    )
    return rdkit_mol_to_ase_atoms(mol)


def atoms_from_xyz(path: str | Path) -> Any:
    imports = optional_import_ase()
    return imports["read"](str(path))


def normalize_mace_model_config(config: MLIPConfig) -> dict[str, str | bool]:
    """Normalize MACE-OFF aliases and local model path configuration."""
    raw_name = str(config.model_name).strip()
    raw_family = str(config.model_family or "").strip() or "mace-off"
    path = Path(raw_name).expanduser()
    looks_like_path = any(sep in raw_name for sep in ("/", "\\")) or path.suffix in {
        ".model",
        ".pt",
        ".pth",
    }
    if looks_like_path:
        return {
            "model_family": "local",
            "model_name": raw_name,
            "model_path": str(path),
            "is_local_path": True,
        }
    lowered = raw_name.lower().replace("_", "-")
    aliases = {
        "mace-off-small": ("mace-off", "small"),
        "mace-off-medium": ("mace-off", "medium"),
        "mace-off-large": ("mace-off", "large"),
        "small": ("mace-off", "small"),
        "medium": ("mace-off", "medium"),
        "large": ("mace-off", "large"),
    }
    family, name = aliases.get(lowered, (raw_family.lower(), lowered))
    if family in {"mace_off", "maceoff"}:
        family = "mace-off"
    return {
        "model_family": family,
        "model_name": name,
        "mace_off_model": name if family == "mace-off" else "",
        "model_path": "",
        "is_local_path": False,
    }


def _call_with_supported_kwargs(callable_obj: Any, kwargs: dict[str, Any]) -> Any:
    try:
        signature = inspect.signature(callable_obj)
    except Exception:
        return callable_obj(**kwargs)
    if any(param.kind is inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return callable_obj(**kwargs)
    supported = {key: value for key, value in kwargs.items() if key in signature.parameters}
    return callable_obj(**supported)


def create_mlip_calculator(config: MLIPConfig) -> Any:
    backend = config.backend_name.strip().lower()
    if backend in {"ani2x", "ani-2x", "ani"}:
        return create_ani2x_calculator(config)
    if backend != "mace":
        raise ValueError(
            f"Unsupported MLIP backend {config.backend_name!r}; supported: mace, ani2x"
        )
    mace = optional_import_mace()
    device = resolve_device(config.device)
    normalized = normalize_mace_model_config(config)
    try:
        if normalized["is_local_path"]:
            calculator_cls = mace.get("MACECalculator")
            if calculator_cls is None:
                raise RuntimeError("Installed mace-torch does not expose MACECalculator.")
            return _call_with_supported_kwargs(
                calculator_cls,
                {
                    "model_paths": str(normalized["model_path"]),
                    "device": device,
                    "default_dtype": config.dtype,
                },
            )
        if normalized["model_family"] == "mace-off" and mace.get("mace_off") is not None:
            return _call_with_supported_kwargs(
                mace["mace_off"],
                {
                    "model": normalized["mace_off_model"],
                    "device": device,
                    "default_dtype": config.dtype,
                },
            )
        if mace.get("mace_mp") is not None:
            model_arg = (
                f"{normalized['model_family']}-{normalized['model_name']}"
                if normalized["model_family"] != "mace"
                else normalized["model_name"]
            )
            return _call_with_supported_kwargs(
                mace["mace_mp"],
                {"model": model_arg, "device": device, "default_dtype": config.dtype},
            )
        raise RuntimeError("Installed mace-torch exposes no supported calculator factory.")
    except Exception as exc:
        raise RuntimeError(
            "Could not create MACE calculator. For MACE-OFF use --model-name small "
            "or --model-name medium, or pass a local model file path. Model availability "
            "depends on installed mace-torch version and local model cache. "
            f"Underlying error: {exc}"
        ) from exc


def _atoms_positions(atoms: Any) -> list[list[float]]:
    if hasattr(atoms, "get_positions"):
        positions = atoms.get_positions()
    else:
        positions = getattr(atoms, "positions", [])
    if hasattr(positions, "tolist"):
        positions = positions.tolist()
    return [[float(value) for value in position] for position in positions]


def compute_interatomic_distances(atoms: Any) -> dict[str, object]:
    positions = _atoms_positions(atoms)
    n_atoms = len(positions)
    if n_atoms < 2:
        return {
            "n_atoms": n_atoms,
            "min_interatomic_distance": None,
            "min_distance_atom_pair": None,
            "max_interatomic_distance": None,
        }
    min_distance: float | None = None
    max_distance: float | None = None
    min_pair: list[int] | None = None
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            distance = math.sqrt(
                sum((positions[i][axis] - positions[j][axis]) ** 2 for axis in range(3))
            )
            if min_distance is None or distance < min_distance:
                min_distance = distance
                min_pair = [i, j]
            if max_distance is None or distance > max_distance:
                max_distance = distance
    return {
        "n_atoms": n_atoms,
        "min_interatomic_distance": min_distance,
        "min_distance_atom_pair": min_pair,
        "max_interatomic_distance": max_distance,
    }


def detect_rdkit_fragment_charge_info(mol: Chem.Mol | None) -> dict[str, object]:
    if mol is None:
        return {
            "n_fragments": None,
            "fragment_charges": [],
            "total_formal_charge": None,
            "has_disconnected_fragments": False,
            "has_charged_fragments": False,
            "has_disconnected_charged_fragments": False,
            "warnings": ["RDKit molecule unavailable; fragment charge check skipped."],
        }
    fragments = Chem.GetMolFrags(mol, asMols=False, sanitizeFrags=False)
    fragment_charges: list[int] = []
    for fragment in fragments:
        charge = sum(int(mol.GetAtomWithIdx(atom_idx).GetFormalCharge()) for atom_idx in fragment)
        fragment_charges.append(charge)
    total_charge = sum(fragment_charges)
    has_disconnected = len(fragments) > 1
    has_charged = any(charge != 0 for charge in fragment_charges)
    return {
        "n_fragments": len(fragments),
        "fragment_charges": fragment_charges,
        "total_formal_charge": total_charge,
        "has_disconnected_fragments": has_disconnected,
        "has_charged_fragments": has_charged,
        "has_disconnected_charged_fragments": has_disconnected and has_charged,
        "warnings": [],
    }


_BRACKET_CHARGE_RE = re.compile(r"\[[^\]]*[+-][^\]]*\]")


def detect_reaction_smiles_charge_risk(reaction_smiles: str) -> dict[str, object]:
    reactants = reaction_smiles.partition(">>")[0]
    fragments = [fragment for fragment in reactants.split(".") if fragment]
    charged_fragments = [
        fragment for fragment in fragments if _BRACKET_CHARGE_RE.search(fragment) is not None
    ]
    has_dot_separated = len(fragments) > 1
    has_charged = bool(charged_fragments)
    has_disconnected_charged = has_dot_separated and has_charged
    warnings: list[str] = []
    if has_disconnected_charged:
        warnings.append(
            "Disconnected charged reaction SMILES may generate arbitrary relative geometry."
        )
        warnings.append(
            "Provide a physically meaningful 3D complex / XYZ for reliable MLIP forces."
        )
    return {
        "reactant_fragments": fragments,
        "charged_reactant_fragments": charged_fragments,
        "has_dot_separated_reactants": has_dot_separated,
        "has_charged_reactants": has_charged,
        "has_disconnected_charged_reactants": has_disconnected_charged,
        "warnings": warnings,
    }


def check_geometry_sanity(
    atoms: Any,
    forces: list[list[float]] | None = None,
    rdkit_mol: Chem.Mol | None = None,
    reaction_smiles: str | None = None,
    mean_force_threshold: float = 100.0,
    max_force_threshold: float = 1000.0,
    min_distance_threshold: float = 0.6,
) -> GeometrySanityReport:
    distances = compute_interatomic_distances(atoms)
    n_atoms = int(distances["n_atoms"])
    min_distance = distances["min_interatomic_distance"]
    max_distance = distances["max_interatomic_distance"]
    min_pair = distances["min_distance_atom_pair"]
    fragment_info = detect_rdkit_fragment_charge_info(rdkit_mol)
    reaction_risk = (
        detect_reaction_smiles_charge_risk(reaction_smiles)
        if reaction_smiles is not None
        else {
            "has_disconnected_charged_reactants": False,
            "warnings": [],
        }
    )
    force_norms = compute_force_norms(forces) if forces is not None else []
    mean_force_norm = sum(force_norms) / len(force_norms) if force_norms else None
    max_force_norm = max(force_norms) if force_norms else None
    warnings: list[str] = []
    fail = False
    warning = False
    if isinstance(min_distance, float) and min_distance < min_distance_threshold:
        fail = True
        warnings.append(
            f"Minimum interatomic distance {min_distance:.3f} Angstrom is below "
            f"threshold {min_distance_threshold:.3f}; geometry may be unphysical."
        )
    if mean_force_norm is not None and mean_force_norm > mean_force_threshold:
        fail = True
        warnings.append(
            f"MLIP mean force norm {mean_force_norm:.3f} eV/Angstrom exceeds "
            f"threshold {mean_force_threshold:.3f}; geometry may be unphysical."
        )
    if max_force_norm is not None and max_force_norm > max_force_threshold:
        fail = True
        warnings.append(
            f"MLIP max force norm {max_force_norm:.3f} eV/Angstrom exceeds "
            f"threshold {max_force_threshold:.3f}; geometry may be unphysical."
        )
    fragment_disconnected_charged = bool(fragment_info["has_disconnected_charged_fragments"])
    reaction_disconnected_charged = bool(reaction_risk["has_disconnected_charged_reactants"])
    if fragment_disconnected_charged:
        warning = True
        warnings.append("RDKit molecule contains disconnected charged fragments.")
    if reaction_disconnected_charged:
        warning = True
        warnings.extend(str(item) for item in reaction_risk.get("warnings", []))
    if fail:
        status = "fail"
    elif warning:
        status = "warning"
    elif n_atoms == 0:
        status = "unknown"
        warnings.append("No atoms available for geometry sanity check.")
    else:
        status = "pass"
    return GeometrySanityReport(
        n_atoms=n_atoms,
        min_interatomic_distance=float(min_distance) if isinstance(min_distance, float) else None,
        min_distance_atom_pair=list(min_pair) if isinstance(min_pair, list) else None,
        max_interatomic_distance=float(max_distance) if isinstance(max_distance, float) else None,
        n_fragments=(
            int(fragment_info["n_fragments"])
            if isinstance(fragment_info["n_fragments"], int)
            else None
        ),
        total_formal_charge=(
            int(fragment_info["total_formal_charge"])
            if isinstance(fragment_info["total_formal_charge"], int)
            else None
        ),
        has_disconnected_fragments=bool(fragment_info["has_disconnected_fragments"])
        or bool(reaction_risk.get("has_dot_separated_reactants", False)),
        has_charged_fragments=bool(fragment_info["has_charged_fragments"])
        or bool(reaction_risk.get("has_charged_reactants", False)),
        has_disconnected_charged_fragments=fragment_disconnected_charged
        or reaction_disconnected_charged,
        mean_force_norm=mean_force_norm,
        max_force_norm=max_force_norm,
        mean_force_threshold=mean_force_threshold,
        max_force_threshold=max_force_threshold,
        min_distance_threshold=min_distance_threshold,
        status=status,
        warnings=warnings,
        metadata={
            "reaction_smiles_checked": reaction_smiles is not None,
            "rdkit_fragment_check_available": rdkit_mol is not None,
        },
    )


def attach_geometry_sanity_to_mlip_result(
    result: MLIPResult,
    sanity_report: GeometrySanityReport,
) -> MLIPResult:
    metadata = dict(result.metadata)
    metadata["geometry_sanity"] = sanity_report.to_dict()
    metadata["geometry_sanity_status"] = sanity_report.status
    metadata["geometry_sanity_failed"] = sanity_report.status == "fail"
    warnings = list(result.warnings)
    if sanity_report.status in {"warning", "fail"}:
        for warning in sanity_report.warnings:
            if warning not in warnings:
                warnings.append(warning)
    return MLIPResult(
        energy=result.energy,
        energy_unit=result.energy_unit,
        forces=[list(force) for force in result.forces] if result.forces is not None else None,
        force_unit=result.force_unit,
        n_atoms=result.n_atoms,
        backend_name=result.backend_name,
        model_name=result.model_name,
        device=result.device,
        success=result.success,
        warnings=warnings,
        metadata=metadata,
    )


def compute_mlip_singlepoint(
    atoms: Any,
    config: MLIPConfig | None = None,
    calculator: Any | None = None,
    rdkit_mol: Chem.Mol | None = None,
    reaction_smiles: str | None = None,
    mean_force_threshold: float = 100.0,
    max_force_threshold: float = 1000.0,
    min_distance_threshold: float = 0.6,
) -> MLIPResult:
    cfg = config or MLIPConfig()
    warnings = [
        "Phase 9 computes single-point energy/forces only.",
        "MLIP result is not a DFT reference.",
    ]
    if cfg.backend_name.strip().lower() in {"ani2x", "ani-2x", "ani"}:
        unsupported = validate_ani2x_elements(atoms)
        if unsupported:
            warnings.append(
                f"ANI-2x does not support elements: {sorted(set(unsupported))}. "
                "Results for these atoms will be incorrect."
            )
    device = resolve_device(cfg.device)
    calc = calculator if calculator is not None else create_mlip_calculator(cfg)
    atoms.calc = calc
    if cfg.optimize_geometry:
        imports = optional_import_ase()
        warnings.append("Geometry optimization is local only; this is not a reaction path.")
        optimizer = imports["BFGS"](atoms, logfile=None)
        optimizer.run(fmax=0.05, steps=20)
    energy = float(atoms.get_potential_energy())
    raw_forces = atoms.get_forces()
    forces = [[float(value) for value in force] for force in raw_forces.tolist()]
    result = MLIPResult(
        energy=energy,
        energy_unit="eV",
        forces=forces,
        force_unit="eV/Angstrom",
        n_atoms=len(atoms),
        backend_name=cfg.backend_name,
        model_name=cfg.model_name,
        device=device,
        success=True,
        warnings=warnings,
        metadata={
            "optimize_geometry": cfg.optimize_geometry,
            "calculator_reused": calculator is not None,
            "calculator_initialized_once": calculator is not None,
        },
    )
    sanity_report = check_geometry_sanity(
        atoms,
        forces=forces,
        rdkit_mol=rdkit_mol,
        reaction_smiles=reaction_smiles,
        mean_force_threshold=mean_force_threshold,
        max_force_threshold=max_force_threshold,
        min_distance_threshold=min_distance_threshold,
    )
    return attach_geometry_sanity_to_mlip_result(result, sanity_report)


def compute_force_norms(forces: list[list[float]]) -> list[float]:
    return [math.sqrt(sum(float(component) ** 2 for component in force)) for force in forces]


def summarize_reaction_center_forces(
    result: MLIPResult,
    reaction_center_atoms: list[int],
) -> ReactionCenterForceSummary:
    metadata: dict[str, Scalar] = {}
    if result.forces is None:
        metadata["missing_forces"] = True
        return ReactionCenterForceSummary(
            reaction_center_atoms=list(reaction_center_atoms),
            n_center_atoms=0,
            mean_center_force_norm=None,
            max_center_force_norm=None,
            mean_all_atom_force_norm=None,
            max_all_atom_force_norm=None,
            center_to_all_mean_force_ratio=None,
            metadata=metadata,
        )
    norms = compute_force_norms(result.forces)
    all_mean = sum(norms) / len(norms) if norms else None
    all_max = max(norms) if norms else None
    valid_center_atoms = [idx for idx in reaction_center_atoms if 0 <= idx < len(norms)]
    out_of_range = [idx for idx in reaction_center_atoms if idx not in valid_center_atoms]
    if not reaction_center_atoms:
        metadata["empty_reaction_center"] = True
    if out_of_range:
        metadata["out_of_range_center_atoms"] = ",".join(str(idx) for idx in out_of_range)
    center_norms = [norms[idx] for idx in valid_center_atoms]
    center_mean = sum(center_norms) / len(center_norms) if center_norms else None
    center_max = max(center_norms) if center_norms else None
    ratio = center_mean / all_mean if center_mean is not None and all_mean else None
    return ReactionCenterForceSummary(
        reaction_center_atoms=list(valid_center_atoms),
        n_center_atoms=len(valid_center_atoms),
        mean_center_force_norm=center_mean,
        max_center_force_norm=center_max,
        mean_all_atom_force_norm=all_mean,
        max_all_atom_force_norm=all_max,
        center_to_all_mean_force_ratio=ratio,
        metadata=metadata,
    )


def _reactant_smiles(reaction_smiles: str) -> str:
    reactants, sep, _ = reaction_smiles.partition(">>")
    if not sep:
        raise ValueError("reaction_smiles must contain '>>'")
    return reactants


def _center_atom_indices_from_negotiation(result: Any) -> list[int]:
    left = _reactant_smiles(result.reaction_smiles)
    offsets: list[int] = []
    running = 0
    for smiles in [part for part in left.split(".") if part]:
        offsets.append(running)
        mol = Chem.MolFromSmiles(smiles)
        running += mol.GetNumAtoms() if mol is not None else 0
    indices: list[int] = []
    for ref in result.reaction_center_atoms:
        offset = offsets[ref.molecule_index] if ref.molecule_index < len(offsets) else 0
        value = offset + ref.atom_index
        if value not in indices:
            indices.append(value)
    return indices


def run_mendel_guided_mlip_singlepoint(
    reaction_smiles: str,
    context: str = "unknown",
    config: MLIPConfig | None = None,
    center_source: str = "auto",
    mean_force_threshold: float = 100.0,
    max_force_threshold: float = 1000.0,
    min_distance_threshold: float = 0.6,
) -> MENDELVGuidedMLIPResult:
    warnings = [
        "Phase 9 computes single-point energy/forces only.",
        "This is not NEB/IRC/MD/TS/barrier prediction.",
        "Disconnected reactant complexes may have arbitrary conformer geometry.",
        "MLIP result is not a DFT reference.",
    ]
    negotiation = run_full_rule_pipeline(reaction_smiles, context=context)
    center_atoms = _center_atom_indices_from_negotiation(negotiation)
    resolved_center_source = "negotiated"
    if center_source == "auto":
        warnings.append(
            "Atom-center-head inference for raw reactions is not yet wired into Phase 9; "
            "using negotiated reaction-center atoms."
        )
    charge_risk = detect_reaction_smiles_charge_risk(reaction_smiles)
    for warning in charge_risk["warnings"]:
        if warning not in warnings:
            warnings.append(str(warning))
    reactant_smiles = _reactant_smiles(reaction_smiles)
    rdkit_reactants = Chem.MolFromSmiles(reactant_smiles)
    atoms = smiles_to_ase_atoms(reactant_smiles)
    mlip_result = compute_mlip_singlepoint(
        atoms,
        config,
        rdkit_mol=rdkit_reactants,
        reaction_smiles=reaction_smiles,
        mean_force_threshold=mean_force_threshold,
        max_force_threshold=max_force_threshold,
        min_distance_threshold=min_distance_threshold,
    )
    force_summary = summarize_reaction_center_forces(mlip_result, center_atoms)
    return MENDELVGuidedMLIPResult(
        reaction_smiles=reaction_smiles,
        context=context,
        mechanism_hint=negotiation.mechanism_hint,
        role_assignments=[assignment.to_dict() for assignment in negotiation.assignments],
        reaction_center_atoms=center_atoms,
        center_source=resolved_center_source,
        mlip_result=mlip_result,
        force_summary=force_summary,
        warnings=[*warnings, *mlip_result.warnings],
        metadata={
            "scope_note": "Optional pretrained MLIP single-point prototype.",
            "reaction_charge_risk": charge_risk,
            "geometry_sanity": mlip_result.metadata.get("geometry_sanity"),
        },
    )


def save_json(payload: dict[str, object], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
