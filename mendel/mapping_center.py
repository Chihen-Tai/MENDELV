"""Phase 8.14 atom-mapping-aware reaction-center utilities.

These helpers use atom-map numbers and bond changes to audit or suggest
reaction_center_atoms labels. They do not train MLIP, MACE, energies, forces,
transition states, IRC, NEB, MD, or barriers.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field, replace
from pathlib import Path

from rdkit import Chem

from mendel.labels import LabeledReaction

Scalar = str | int | float | bool

_CONTROL_MECHANISMS = frozenset({"control", "ester_control", "nitrile_control", "no_reaction"})
_CONFIDENCE_RANK = {"low": 1, "medium": 2, "high": 3}


@dataclass
class MappedAtomPair:
    reactant_atom_map: int
    reactant_molecule_index: int | None
    reactant_atom_index: int
    product_molecule_index: int | None
    product_atom_index: int | None
    atom_symbol: str | None
    metadata: dict[str, Scalar] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "reactant_atom_map": self.reactant_atom_map,
            "reactant_molecule_index": self.reactant_molecule_index,
            "reactant_atom_index": self.reactant_atom_index,
            "product_molecule_index": self.product_molecule_index,
            "product_atom_index": self.product_atom_index,
            "atom_symbol": self.atom_symbol,
            "metadata": dict(self.metadata),
        }


@dataclass
class BondChangeRecord:
    atom_map_1: int
    atom_map_2: int
    reactant_bond_order: float | None
    product_bond_order: float | None
    change_type: str
    reactant_atom_indices: list[int]
    product_atom_indices: list[int]
    metadata: dict[str, Scalar] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "atom_map_1": self.atom_map_1,
            "atom_map_2": self.atom_map_2,
            "reactant_bond_order": self.reactant_bond_order,
            "product_bond_order": self.product_bond_order,
            "change_type": self.change_type,
            "reactant_atom_indices": list(self.reactant_atom_indices),
            "product_atom_indices": list(self.product_atom_indices),
            "metadata": dict(self.metadata),
        }


@dataclass
class MappingCenterSuggestion:
    reaction_id: str
    mechanism_type: str
    mapped: bool
    suggested_center_atoms: list[int]
    bond_changes: list[BondChangeRecord]
    confidence: str
    warnings: list[str]
    metadata: dict[str, Scalar] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "reaction_id": self.reaction_id,
            "mechanism_type": self.mechanism_type,
            "mapped": self.mapped,
            "suggested_center_atoms": list(self.suggested_center_atoms),
            "bond_changes": [change.to_dict() for change in self.bond_changes],
            "confidence": self.confidence,
            "warnings": list(self.warnings),
            "metadata": dict(self.metadata),
        }


@dataclass
class MappingCenterAuditRecord:
    reaction_id: str
    mechanism_type: str
    labeled_center_atoms: list[int]
    suggested_center_atoms: list[int]
    missing_from_label: list[int]
    extra_in_label: list[int]
    exact_match: bool
    overlap_f1: float | None
    issue_code: str
    severity: str
    message: str
    metadata: dict[str, Scalar] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "reaction_id": self.reaction_id,
            "mechanism_type": self.mechanism_type,
            "labeled_center_atoms": list(self.labeled_center_atoms),
            "suggested_center_atoms": list(self.suggested_center_atoms),
            "missing_from_label": list(self.missing_from_label),
            "extra_in_label": list(self.extra_in_label),
            "exact_match": self.exact_match,
            "overlap_f1": self.overlap_f1,
            "issue_code": self.issue_code,
            "severity": self.severity,
            "message": self.message,
            "metadata": dict(self.metadata),
        }


@dataclass
class MappingCenterAuditReport:
    dataset_path: str
    n_reactions: int
    n_mapped_reactions: int
    n_unmapped_reactions: int
    n_exact_matches: int
    mean_overlap_f1: float | None
    issue_counts_by_severity: dict[str, int]
    issue_counts_by_type: dict[str, int]
    records: list[MappingCenterAuditRecord]
    suggestions: list[MappingCenterSuggestion]
    recommendations: list[str]
    metadata: dict[str, Scalar] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset_path": self.dataset_path,
            "n_reactions": self.n_reactions,
            "n_mapped_reactions": self.n_mapped_reactions,
            "n_unmapped_reactions": self.n_unmapped_reactions,
            "n_exact_matches": self.n_exact_matches,
            "mean_overlap_f1": self.mean_overlap_f1,
            "issue_counts_by_severity": dict(self.issue_counts_by_severity),
            "issue_counts_by_type": dict(self.issue_counts_by_type),
            "records": [record.to_dict() for record in self.records],
            "suggestions": [suggestion.to_dict() for suggestion in self.suggestions],
            "recommendations": list(self.recommendations),
            "metadata": dict(self.metadata),
        }


def _split_reaction_smiles(reaction_smiles: str) -> tuple[list[Chem.Mol], list[Chem.Mol]]:
    reactants, sep, products = reaction_smiles.partition(">>")
    if not sep:
        raise ValueError("reaction_smiles must contain '>>'")

    def parse_side(side: str) -> list[Chem.Mol]:
        mols: list[Chem.Mol] = []
        for smiles in [part for part in side.split(".") if part]:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f"RDKit could not parse mapped SMILES component: {smiles}")
            mols.append(mol)
        return mols

    return parse_side(reactants), parse_side(products)


def has_atom_mapping(reaction_smiles: str) -> bool:
    try:
        reactants, products = _split_reaction_smiles(reaction_smiles)
    except ValueError:
        return False
    return any(
        atom.GetAtomMapNum() > 0 for mol in [*reactants, *products] for atom in mol.GetAtoms()
    )


def extract_mapped_atom_pairs(reaction_smiles: str) -> list[MappedAtomPair]:
    reactants, products = _split_reaction_smiles(reaction_smiles)
    product_by_map: dict[int, tuple[int, int, str]] = {}
    for mol_idx, mol in enumerate(products):
        for atom in mol.GetAtoms():
            atom_map = atom.GetAtomMapNum()
            if atom_map:
                product_by_map.setdefault(atom_map, (mol_idx, atom.GetIdx(), atom.GetSymbol()))

    pairs: list[MappedAtomPair] = []
    for mol_idx, mol in enumerate(reactants):
        for atom in mol.GetAtoms():
            atom_map = atom.GetAtomMapNum()
            if not atom_map:
                continue
            product = product_by_map.get(atom_map)
            pairs.append(
                MappedAtomPair(
                    reactant_atom_map=atom_map,
                    reactant_molecule_index=mol_idx,
                    reactant_atom_index=atom.GetIdx(),
                    product_molecule_index=product[0] if product else None,
                    product_atom_index=product[1] if product else None,
                    atom_symbol=atom.GetSymbol(),
                    metadata={"product_atom_missing": product is None},
                )
            )
    return sorted(pairs, key=lambda pair: pair.reactant_atom_map)


def _bond_index(
    mols: list[Chem.Mol],
) -> tuple[dict[tuple[int, int], float], dict[int, int]]:
    bonds: dict[tuple[int, int], float] = {}
    atom_idx_by_map: dict[int, int] = {}
    for mol in mols:
        for atom in mol.GetAtoms():
            atom_map = atom.GetAtomMapNum()
            if atom_map:
                atom_idx_by_map.setdefault(atom_map, atom.GetIdx())
        for bond in mol.GetBonds():
            begin_map = bond.GetBeginAtom().GetAtomMapNum()
            end_map = bond.GetEndAtom().GetAtomMapNum()
            if not begin_map or not end_map:
                continue
            key = tuple(sorted((begin_map, end_map)))
            bonds[key] = float(bond.GetBondTypeAsDouble())
    return bonds, atom_idx_by_map


def extract_bond_changes(reaction_smiles: str) -> list[BondChangeRecord]:
    reactants, products = _split_reaction_smiles(reaction_smiles)
    reactant_bonds, reactant_atom_idx = _bond_index(reactants)
    product_bonds, product_atom_idx = _bond_index(products)
    changes: list[BondChangeRecord] = []
    for atom_map_1, atom_map_2 in sorted(set(reactant_bonds) | set(product_bonds)):
        reactant_order = reactant_bonds.get((atom_map_1, atom_map_2))
        product_order = product_bonds.get((atom_map_1, atom_map_2))
        if reactant_order is None and product_order is not None:
            change_type = "bond_formed"
        elif reactant_order is not None and product_order is None:
            change_type = "bond_broken"
        elif reactant_order != product_order:
            change_type = "bond_order_changed"
        else:
            continue
        changes.append(
            BondChangeRecord(
                atom_map_1=atom_map_1,
                atom_map_2=atom_map_2,
                reactant_bond_order=reactant_order,
                product_bond_order=product_order,
                change_type=change_type,
                reactant_atom_indices=[
                    idx
                    for idx in (
                        reactant_atom_idx.get(atom_map_1),
                        reactant_atom_idx.get(atom_map_2),
                    )
                    if idx is not None
                ],
                product_atom_indices=[
                    idx
                    for idx in (product_atom_idx.get(atom_map_1), product_atom_idx.get(atom_map_2))
                    if idx is not None
                ],
                metadata={},
            )
        )
    return changes


def infer_center_atoms_from_mapping(reaction: LabeledReaction) -> MappingCenterSuggestion:
    warnings: list[str] = []
    if not has_atom_mapping(reaction.reaction_smiles):
        return MappingCenterSuggestion(
            reaction_id=reaction.reaction_id,
            mechanism_type=reaction.mechanism_type,
            mapped=False,
            suggested_center_atoms=[],
            bond_changes=[],
            confidence="low",
            warnings=["reaction SMILES does not contain atom-map numbers"],
            metadata={},
        )
    try:
        changes = extract_bond_changes(reaction.reaction_smiles)
    except ValueError as exc:
        return MappingCenterSuggestion(
            reaction_id=reaction.reaction_id,
            mechanism_type=reaction.mechanism_type,
            mapped=False,
            suggested_center_atoms=[],
            bond_changes=[],
            confidence="low",
            warnings=[str(exc)],
            metadata={"parse_failed": True},
        )
    mechanism = reaction.mechanism_type.strip().lower()
    center = sorted({atom for change in changes for atom in (change.atom_map_1, change.atom_map_2)})
    if mechanism in _CONTROL_MECHANISMS and center:
        warnings.append("control/no-reaction mechanism has mapped bond changes")
    if not changes:
        warnings.append("atom mapping contains no bond changes")
    mapped_pairs = extract_mapped_atom_pairs(reaction.reaction_smiles)
    if any(pair.product_atom_index is None for pair in mapped_pairs):
        warnings.append("mapping is incomplete for at least one reactant atom")

    confidence = "high" if changes and not warnings else "medium" if changes else "low"
    return MappingCenterSuggestion(
        reaction_id=reaction.reaction_id,
        mechanism_type=reaction.mechanism_type,
        mapped=True,
        suggested_center_atoms=center,
        bond_changes=changes,
        confidence=confidence,
        warnings=warnings,
        metadata={"n_bond_changes": len(changes)},
    )


def _overlap_f1(label: set[int], suggestion: set[int]) -> float | None:
    if not label and not suggestion:
        return 1.0
    if not label or not suggestion:
        return 0.0
    overlap = len(label & suggestion)
    precision = overlap / len(suggestion)
    recall = overlap / len(label)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _audit_record(
    reaction: LabeledReaction,
    suggestion: MappingCenterSuggestion,
) -> MappingCenterAuditRecord:
    labeled = set(reaction.reaction_center_atoms)
    suggested = set(suggestion.suggested_center_atoms)
    missing = sorted(suggested - labeled)
    extra = sorted(labeled - suggested)
    exact = labeled == suggested
    f1 = _overlap_f1(labeled, suggested)
    mechanism = reaction.mechanism_type.strip().lower()
    if not suggestion.mapped:
        issue_code = "unmapped_reaction"
        severity = "info"
        message = "reaction SMILES is not atom-mapped; mapping audit skipped"
    elif mechanism in _CONTROL_MECHANISMS and suggested:
        issue_code = "mapping_control_bond_change"
        severity = "warning"
        message = "mapped control/no-reaction has bond changes"
    elif exact:
        issue_code = "exact_match"
        severity = "info"
        message = "labeled center exactly matches mapping-derived center"
    elif (missing or extra) and (labeled & suggested):
        issue_code = "partial_overlap"
        severity = "warning"
        message = "labeled center partially overlaps mapping-derived center"
    elif missing:
        issue_code = "label_missing_mapping_center"
        severity = "warning"
        message = "labeled center is missing mapping-derived atoms"
    elif extra:
        issue_code = "label_has_extra_atoms"
        severity = "warning"
        message = "labeled center contains atoms outside mapping-derived center"
    else:
        issue_code = "mapping_incomplete"
        severity = "info"
        message = "mapping audit did not produce a usable issue category"
    return MappingCenterAuditRecord(
        reaction_id=reaction.reaction_id,
        mechanism_type=reaction.mechanism_type,
        labeled_center_atoms=sorted(labeled),
        suggested_center_atoms=sorted(suggested),
        missing_from_label=missing,
        extra_in_label=extra,
        exact_match=exact,
        overlap_f1=f1,
        issue_code=issue_code,
        severity=severity,
        message=message,
        metadata={"mapping_confidence": suggestion.confidence},
    )


def audit_labeled_centers_against_mapping(
    reactions: list[LabeledReaction],
    dataset_path: str = "",
) -> MappingCenterAuditReport:
    suggestions = [infer_center_atoms_from_mapping(rxn) for rxn in reactions]
    records = [
        _audit_record(rxn, suggestion)
        for rxn, suggestion in zip(reactions, suggestions, strict=True)
    ]
    mapped = [suggestion for suggestion in suggestions if suggestion.mapped]
    f1_values = [
        record.overlap_f1
        for record in records
        if record.overlap_f1 is not None and record.issue_code != "unmapped_reaction"
    ]
    by_severity: dict[str, int] = {}
    by_type: dict[str, int] = {}
    for record in records:
        by_severity[record.severity] = by_severity.get(record.severity, 0) + 1
        by_type[record.issue_code] = by_type.get(record.issue_code, 0) + 1

    recommendations: list[str] = []
    if not mapped:
        recommendations.append(
            "No mapped reactions found; continue balanced validation and add "
            "atom-mapped data later."
        )
    if by_type.get("label_missing_mapping_center") or by_type.get("partial_overlap"):
        recommendations.append(
            "Spot-check high-confidence mapping suggestions before using them as labels."
        )
    if by_type.get("mapping_control_bond_change"):
        recommendations.append(
            "Review mapped control reactions with bond changes; they may not be true controls."
        )
    if not recommendations:
        recommendations.append("Mapping audit found no blocking center-label issues.")

    return MappingCenterAuditReport(
        dataset_path=dataset_path,
        n_reactions=len(reactions),
        n_mapped_reactions=len(mapped),
        n_unmapped_reactions=len(reactions) - len(mapped),
        n_exact_matches=sum(
            1
            for record in records
            if record.exact_match and record.issue_code != "unmapped_reaction"
        ),
        mean_overlap_f1=sum(f1_values) / len(f1_values) if f1_values else None,
        issue_counts_by_severity=dict(sorted(by_severity.items())),
        issue_counts_by_type=dict(sorted(by_type.items())),
        records=records,
        suggestions=suggestions,
        recommendations=recommendations,
        metadata={"scope_note": "Mapping center audit only; no MLIP or energy/force modeling."},
    )


def apply_mapping_center_suggestions(
    reactions: list[LabeledReaction],
    min_confidence: str = "high",
    conservative: bool = True,
) -> tuple[list[LabeledReaction], list[MappingCenterAuditRecord]]:
    min_rank = _CONFIDENCE_RANK.get(min_confidence, _CONFIDENCE_RANK["high"])
    updated: list[LabeledReaction] = []
    applied: list[MappingCenterAuditRecord] = []
    for reaction in reactions:
        suggestion = infer_center_atoms_from_mapping(reaction)
        record = _audit_record(reaction, suggestion)
        should_apply = (
            suggestion.mapped
            and _CONFIDENCE_RANK.get(suggestion.confidence, 0) >= min_rank
            and suggestion.suggested_center_atoms != reaction.reaction_center_atoms
        )
        if conservative and reaction.mechanism_type.strip().lower() in _CONTROL_MECHANISMS:
            should_apply = should_apply and not suggestion.suggested_center_atoms
        if should_apply:
            metadata = copy.deepcopy(reaction.metadata)
            metadata["original_reaction_center_atoms"] = list(reaction.reaction_center_atoms)
            metadata["mapping_center_applied"] = True
            metadata["mapping_center_confidence"] = suggestion.confidence
            metadata["mapping_center_phase"] = "8.14"
            updated.append(
                replace(
                    reaction,
                    reaction_center_atoms=list(suggestion.suggested_center_atoms),
                    metadata=metadata,
                )
            )
            applied.append(record)
        else:
            updated.append(replace(reaction, metadata=copy.deepcopy(reaction.metadata)))
    return updated, applied


def save_mapping_center_audit_report(
    report: MappingCenterAuditReport,
    path: str | Path,
) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")


def save_labeled_reactions_json(reactions: list[LabeledReaction], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"reactions": [rxn.to_dict() for rxn in reactions]}, indent=2),
        encoding="utf-8",
    )
