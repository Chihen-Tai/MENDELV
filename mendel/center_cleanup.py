"""Phase 8.12 conservative reaction-center label cleanup."""

from __future__ import annotations

import copy
import json
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path

from rdkit import Chem

from mendel.center_validation import audit_center_labels, summarize_center_label_issues
from mendel.labels import LabeledReaction, load_labeled_reactions
from mendel.parser import parse_reaction_smiles
from mendel.types import FunctionalGroupType, Role

Scalar = str | int | float | bool
_CONTROL_MECHANISMS = {"control", "ester_control", "nitrile_control", "no_reaction"}


@dataclass
class CenterLabelCorrection:
    reaction_id: str
    mechanism_type: str
    correction_type: str
    severity: str
    old_center_atoms: list[int]
    new_center_atoms: list[int]
    reason: str
    confidence: str
    metadata: dict[str, Scalar] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "reaction_id": self.reaction_id,
            "mechanism_type": self.mechanism_type,
            "correction_type": self.correction_type,
            "severity": self.severity,
            "old_center_atoms": list(self.old_center_atoms),
            "new_center_atoms": list(self.new_center_atoms),
            "reason": self.reason,
            "confidence": self.confidence,
            "metadata": dict(self.metadata),
        }


@dataclass
class CenterCleanupReport:
    input_path: str
    output_path: str
    n_reactions: int
    n_corrected_reactions: int
    n_skipped_reactions: int
    n_corrections: int
    corrections_by_type: dict[str, int]
    remaining_issues_by_severity: dict[str, int]
    remaining_issues_by_type: dict[str, int]
    corrections: list[CenterLabelCorrection]
    skipped_reactions: list[dict[str, object]]
    recommendations: list[str]
    metadata: dict[str, Scalar] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "input_path": self.input_path,
            "output_path": self.output_path,
            "n_reactions": self.n_reactions,
            "n_corrected_reactions": self.n_corrected_reactions,
            "n_skipped_reactions": self.n_skipped_reactions,
            "n_corrections": self.n_corrections,
            "corrections_by_type": dict(self.corrections_by_type),
            "remaining_issues_by_severity": dict(self.remaining_issues_by_severity),
            "remaining_issues_by_type": dict(self.remaining_issues_by_type),
            "corrections": [correction.to_dict() for correction in self.corrections],
            "skipped_reactions": list(self.skipped_reactions),
            "recommendations": list(self.recommendations),
            "metadata": dict(self.metadata),
        }


def _mechanism(reaction: LabeledReaction) -> str:
    return reaction.mechanism_type.lower().replace("-", "_").replace(" ", "_")


def _center_values_for_label(
    reaction: LabeledReaction, group_type: FunctionalGroupType
) -> set[int]:
    parsed = parse_reaction_smiles(reaction.reaction_smiles, context=reaction.context)
    values: set[int] = set()
    labels = [role for role in reaction.group_roles if role.group_type is group_type]
    for label in labels:
        mol = next(
            (mol for mol in parsed.reactants if mol.molecule_index == label.molecule_index), None
        )
        rd_mol = Chem.MolFromSmiles(mol.smiles) if mol is not None else None
        for atom_index in label.atom_indices:
            value = atom_index
            if rd_mol is not None and atom_index < rd_mol.GetNumAtoms():
                atom_map = rd_mol.GetAtomWithIdx(atom_index).GetAtomMapNum()
                value = atom_map if atom_map else atom_index
            values.add(value)
    return values


def _correction(
    reaction: LabeledReaction,
    correction_type: str,
    old_center: list[int],
    new_center: list[int],
    reason: str,
    severity: str = "warning",
    confidence: str = "conservative_rule",
    metadata: dict[str, Scalar] | None = None,
) -> CenterLabelCorrection | None:
    if list(old_center) == list(new_center):
        return None
    return CenterLabelCorrection(
        reaction_id=reaction.reaction_id,
        mechanism_type=reaction.mechanism_type,
        correction_type=correction_type,
        severity=severity,
        old_center_atoms=list(old_center),
        new_center_atoms=list(new_center),
        reason=reason,
        confidence=confidence,
        metadata=metadata or {},
    )


def cleanup_control_centers(reaction: LabeledReaction) -> CenterLabelCorrection | None:
    if _mechanism(reaction) in _CONTROL_MECHANISMS and reaction.reaction_center_atoms:
        return _correction(
            reaction,
            "control_empty_center",
            reaction.reaction_center_atoms,
            [],
            "control/no-reaction mechanism should not have reaction-center atoms",
            severity="error",
        )
    return None


def cleanup_spectator_only_centers(reaction: LabeledReaction) -> CenterLabelCorrection | None:
    if (
        reaction.reaction_center_atoms
        and reaction.group_roles
        and all(role.role is Role.spectator for role in reaction.group_roles)
        and not reaction.metadata.get("explicit_reactive_center")
    ):
        return _correction(
            reaction,
            "spectator_only_empty_center",
            reaction.reaction_center_atoms,
            [],
            "spectator-only reaction should not define non-empty center",
            severity="warning",
        )
    return None


def cleanup_diels_alder_substituent_centers(
    reaction: LabeledReaction,
) -> CenterLabelCorrection | None:
    if _mechanism(reaction) != "diels_alder":
        return None
    alkene = _center_values_for_label(reaction, FunctionalGroupType.alkene)
    substituents = set()
    for group_type in (
        FunctionalGroupType.nitrile,
        FunctionalGroupType.ester,
        FunctionalGroupType.carbonyl,
        FunctionalGroupType.alpha_carbon,
        FunctionalGroupType.aromatic,
    ):
        substituents.update(_center_values_for_label(reaction, group_type))
    old = list(reaction.reaction_center_atoms)
    new = [atom for atom in old if atom not in substituents or atom in alkene]
    if old and not any(atom in alkene for atom in new):
        return None
    return _correction(
        reaction,
        "diels_alder_substituent_center_cleanup",
        old,
        new,
        "Diels-Alder center should keep reacting alkene atoms and exclude substituent atoms",
    )


def cleanup_carbonyl_addition_centers(reaction: LabeledReaction) -> CenterLabelCorrection | None:
    if _mechanism(reaction) != "carbonyl_addition":
        return None
    alpha = _center_values_for_label(reaction, FunctionalGroupType.alpha_carbon)
    carbonyl = _center_values_for_label(reaction, FunctionalGroupType.carbonyl)
    old = list(reaction.reaction_center_atoms)
    new = [atom for atom in old if atom not in alpha or atom in carbonyl]
    return _correction(
        reaction,
        "carbonyl_addition_alpha_center_cleanup",
        old,
        new,
        "carbonyl addition center should not include spectator alpha carbon",
    )


def cleanup_sn2_e2_centers(reaction: LabeledReaction) -> CenterLabelCorrection | None:
    if _mechanism(reaction) not in {"sn2", "e2"}:
        return None
    halide = _center_values_for_label(reaction, FunctionalGroupType.halide)
    if not halide:
        return None
    old = list(reaction.reaction_center_atoms)
    new = sorted(set(old) | halide)
    return _correction(
        reaction,
        "sn2_e2_halide_center_completion",
        old,
        new,
        "SN2/E2 center should include detected halide and attached carbon when represented",
    )


def cleanup_radical_bromination_centers(reaction: LabeledReaction) -> CenterLabelCorrection | None:
    if _mechanism(reaction) not in {"radical_bromination", "benzylic_radical_bromination"}:
        return None
    benzylic = _center_values_for_label(reaction, FunctionalGroupType.benzylic_site)
    aromatic = _center_values_for_label(reaction, FunctionalGroupType.aromatic)
    if not benzylic:
        return None
    old = list(reaction.reaction_center_atoms)
    new = [atom for atom in old if atom in benzylic or atom not in aromatic]
    return _correction(
        reaction,
        "radical_bromination_aromatic_center_cleanup",
        old,
        new,
        "radical bromination center should keep benzylic atoms and exclude aromatic spectator ring",
    )


def remove_center_atoms_outside_detected_groups(
    reaction: LabeledReaction,
    allow_if_metadata: bool = True,
) -> CenterLabelCorrection | None:
    if allow_if_metadata:
        return None
    return None


def _apply_metadata(
    reaction: LabeledReaction,
    correction: CenterLabelCorrection,
) -> LabeledReaction:
    metadata = copy.deepcopy(reaction.metadata)
    metadata["center_cleanup_applied"] = True
    metadata["center_cleanup_phase"] = "8.12"
    metadata["original_reaction_center_atoms"] = ",".join(
        str(atom) for atom in correction.old_center_atoms
    )
    metadata["center_cleanup_reason"] = correction.reason
    return replace(
        reaction, reaction_center_atoms=list(correction.new_center_atoms), metadata=metadata
    )


def cleanup_center_labels(
    reactions: list[LabeledReaction],
    conservative: bool = True,
) -> tuple[list[LabeledReaction], list[CenterLabelCorrection]]:
    cleaned: list[LabeledReaction] = []
    corrections: list[CenterLabelCorrection] = []
    rules = [
        cleanup_control_centers,
        cleanup_spectator_only_centers,
        cleanup_diels_alder_substituent_centers,
        cleanup_carbonyl_addition_centers,
        cleanup_sn2_e2_centers,
        cleanup_radical_bromination_centers,
        remove_center_atoms_outside_detected_groups,
    ]
    for reaction in reactions:
        current = replace(reaction, metadata=copy.deepcopy(reaction.metadata))
        for rule in rules:
            correction = rule(current)
            if correction is None:
                continue
            corrections.append(correction)
            current = _apply_metadata(current, correction)
        cleaned.append(current)
    return cleaned, corrections


def build_center_cleanup_report(
    input_path: str | Path,
    output_path: str | Path,
    original: list[LabeledReaction],
    cleaned: list[LabeledReaction],
    corrections: list[CenterLabelCorrection],
) -> CenterCleanupReport:
    remaining_summary = summarize_center_label_issues(audit_center_labels(cleaned))
    corrected_ids = {correction.reaction_id for correction in corrections}
    by_type = Counter(correction.correction_type for correction in corrections)
    recommendations: list[str] = []
    if remaining_summary["by_severity"].get("error", 0):  # type: ignore[index,union-attr]
        recommendations.append("Manually review remaining error-severity center-label issues.")
    if not recommendations:
        recommendations.append("Rerun leakage-resistant strict validation on cleaned labels.")
    return CenterCleanupReport(
        input_path=str(input_path),
        output_path=str(output_path),
        n_reactions=len(original),
        n_corrected_reactions=len(corrected_ids),
        n_skipped_reactions=0,
        n_corrections=len(corrections),
        corrections_by_type=dict(sorted(by_type.items())),
        remaining_issues_by_severity=dict(remaining_summary["by_severity"]),  # type: ignore[arg-type]
        remaining_issues_by_type=dict(remaining_summary["by_issue_code"]),  # type: ignore[arg-type]
        corrections=corrections,
        skipped_reactions=[],
        recommendations=recommendations,
        metadata={"conservative": True},
    )


def save_labeled_reactions_json(reactions: list[LabeledReaction], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"reactions": [rxn.to_dict() for rxn in reactions]}, indent=2), encoding="utf-8"
    )


def save_center_cleanup_report(report: CenterCleanupReport, path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")


def cleanup_dataset(
    input_path: str | Path,
    output_path: str | Path,
    report_path: str | Path,
    dry_run: bool = False,
) -> CenterCleanupReport:
    reactions = load_labeled_reactions(input_path)
    cleaned, corrections = cleanup_center_labels(reactions)
    report = build_center_cleanup_report(input_path, output_path, reactions, cleaned, corrections)
    if not dry_run:
        save_labeled_reactions_json(cleaned, output_path)
        save_center_cleanup_report(report, report_path)
    return report
