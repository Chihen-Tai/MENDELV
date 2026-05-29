"""Phase 8.13 conservative review for center-expansion reactions."""

from __future__ import annotations

import copy
import json
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path

from rdkit import Chem

from mendel.labels import LabeledReaction
from mendel.parser import parse_reaction_smiles
from mendel.types import FunctionalGroupType, Role

Scalar = str | int | float | bool
_CONTROL_MECHANISMS = {"control", "ester_control", "nitrile_control", "no_reaction"}


@dataclass
class CenterExpansionPromotionRecord:
    reaction_id: str
    mechanism_type: str
    promoted: bool
    skip_reason: str | None
    old_group_roles: list[dict[str, object]]
    new_group_roles: list[dict[str, object]]
    old_reaction_center_atoms: list[int]
    new_reaction_center_atoms: list[int]
    center_policy: str
    corrections: list[str]
    metadata: dict[str, Scalar] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "reaction_id": self.reaction_id,
            "mechanism_type": self.mechanism_type,
            "promoted": self.promoted,
            "skip_reason": self.skip_reason,
            "old_group_roles": list(self.old_group_roles),
            "new_group_roles": list(self.new_group_roles),
            "old_reaction_center_atoms": list(self.old_reaction_center_atoms),
            "new_reaction_center_atoms": list(self.new_reaction_center_atoms),
            "center_policy": self.center_policy,
            "corrections": list(self.corrections),
            "metadata": dict(self.metadata),
        }


@dataclass
class CenterExpansionPromotionReport:
    input_path: str
    output_path: str
    merged_output_path: str
    n_input_reactions: int
    n_promoted_reactions: int
    n_skipped_reactions: int
    n_promoted_group_labels: int
    promoted_mechanism_distribution: dict[str, int]
    promoted_role_distribution: dict[str, int]
    promoted_center_atom_count_distribution: dict[str, int]
    promotion_records: list[CenterExpansionPromotionRecord]
    skipped_reactions: list[dict[str, object]]
    warnings: list[str]
    metadata: dict[str, Scalar] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "input_path": self.input_path,
            "output_path": self.output_path,
            "merged_output_path": self.merged_output_path,
            "n_input_reactions": self.n_input_reactions,
            "n_promoted_reactions": self.n_promoted_reactions,
            "n_skipped_reactions": self.n_skipped_reactions,
            "n_promoted_group_labels": self.n_promoted_group_labels,
            "promoted_mechanism_distribution": dict(self.promoted_mechanism_distribution),
            "promoted_role_distribution": dict(self.promoted_role_distribution),
            "promoted_center_atom_count_distribution": dict(
                self.promoted_center_atom_count_distribution
            ),
            "promotion_records": [record.to_dict() for record in self.promotion_records],
            "skipped_reactions": list(self.skipped_reactions),
            "warnings": list(self.warnings),
            "metadata": dict(self.metadata),
        }


def _mechanism(reaction: LabeledReaction) -> str:
    return reaction.mechanism_type.lower().replace("-", "_").replace(" ", "_")


def _mapped_values(reaction: LabeledReaction, role) -> list[int]:
    parsed = parse_reaction_smiles(reaction.reaction_smiles, context=reaction.context)
    values: list[int] = []
    molecule_index = role.molecule_index
    if role.group_id.startswith("mol"):
        prefix = role.group_id.split("_", 1)[0]
        try:
            molecule_index = int(prefix.removeprefix("mol"))
        except ValueError:
            molecule_index = role.molecule_index
    mol = next((m for m in parsed.reactants if m.molecule_index == molecule_index), None)
    rd_mol = Chem.MolFromSmiles(mol.smiles) if mol is not None else None
    for atom_index in role.atom_indices:
        value = atom_index
        if rd_mol is not None and atom_index < rd_mol.GetNumAtoms():
            atom_map = rd_mol.GetAtomWithIdx(atom_index).GetAtomMapNum()
            value = atom_map if atom_map else atom_index
        if value not in values:
            values.append(value)
    return values


def _set_roles(reaction: LabeledReaction, updates: dict[FunctionalGroupType, Role]) -> list:
    roles = []
    for role in reaction.group_roles:
        new_role = updates.get(role.group_type, role.role)
        roles.append(replace(role, role=new_role, confidence="manual"))
    return roles


def _with_review_metadata(
    reaction: LabeledReaction,
    roles: list,
    center_atoms: list[int],
    policy: str,
) -> LabeledReaction:
    metadata = copy.deepcopy(reaction.metadata)
    metadata["original_source"] = str(metadata.get("source", ""))
    metadata["source"] = "center_expansion_human_review_promoted"
    metadata["center_expansion_promoted"] = True
    metadata["needs_manual_review"] = False
    metadata["reviewed_by"] = "MENDELV Phase 8.13 conservative center expansion"
    metadata["center_policy"] = policy
    return replace(
        reaction,
        split="train",
        group_roles=roles,
        reaction_center_atoms=list(dict.fromkeys(center_atoms)),
        metadata=metadata,
    )


def apply_mechanism_center_policy(
    reaction: LabeledReaction,
) -> tuple[LabeledReaction | None, list[str], str | None]:
    mechanism = _mechanism(reaction)
    corrections: list[str] = []

    if mechanism in _CONTROL_MECHANISMS:
        roles = [
            replace(role, role=Role.spectator, confidence="manual") for role in reaction.group_roles
        ]
        policy = "Control/no-reaction center policy: all groups spectator; center=[]."
        corrections.append("control_empty_center")
        return _with_review_metadata(reaction, roles, [], policy), corrections, policy

    if mechanism in {"sn2", "e2"}:
        halides = [
            role for role in reaction.group_roles if role.group_type is FunctionalGroupType.halide
        ]
        if not halides:
            return None, corrections, "skip: no halide group for SN2/E2 center policy"
        roles = _set_roles(reaction, {FunctionalGroupType.halide: Role.leaving_group})
        center = []
        for role in halides:
            center.extend(_mapped_values(reaction, role))
        policy = "SN2/E2 center policy: halide and attached carbon are center atoms."
        corrections.append("halide_leaving_group_center")
        return (
            _with_review_metadata(reaction, roles, sorted(set(center)), policy),
            corrections,
            policy,
        )

    if mechanism == "diels_alder":
        alkenes = [
            role for role in reaction.group_roles if role.group_type is FunctionalGroupType.alkene
        ]
        if not alkenes:
            return None, corrections, "skip: no alkene groups for Diels-Alder center policy"
        roles = []
        center: list[int] = []
        for idx, role in enumerate(reaction.group_roles):
            if role.group_type is FunctionalGroupType.alkene:
                role_value = (
                    Role.reactive_electrophile
                    if idx == len(reaction.group_roles) - 1
                    else Role.reactive_nucleophile
                )
                roles.append(replace(role, role=role_value, confidence="manual"))
                center.extend(_mapped_values(reaction, role))
            else:
                roles.append(replace(role, role=Role.spectator, confidence="manual"))
        policy = "Diels-Alder center policy: reacting alkene atoms only."
        corrections.append("diels_alder_alkene_center_only")
        return _with_review_metadata(reaction, roles, center, policy), corrections, policy

    if mechanism == "carbonyl_addition":
        carbonyls = [
            role for role in reaction.group_roles if role.group_type is FunctionalGroupType.carbonyl
        ]
        if not carbonyls:
            return None, corrections, "skip: no carbonyl group for carbonyl addition policy"
        roles = _set_roles(
            reaction,
            {
                FunctionalGroupType.carbonyl: Role.reactive_electrophile,
                FunctionalGroupType.alpha_carbon: Role.spectator,
            },
        )
        center: list[int] = []
        for role in carbonyls:
            center.extend(_mapped_values(reaction, role))
        policy = "Carbonyl addition center policy: carbonyl carbon and oxygen only."
        corrections.append("carbonyl_center_only")
        return _with_review_metadata(reaction, roles, center, policy), corrections, policy

    if mechanism in {"aldol", "cross_aldol"}:
        alphas = [
            role
            for role in reaction.group_roles
            if role.group_type is FunctionalGroupType.alpha_carbon
        ]
        carbonyls = [
            role for role in reaction.group_roles if role.group_type is FunctionalGroupType.carbonyl
        ]
        if len(alphas) != 1 or len(carbonyls) != 1:
            return None, corrections, "skip: ambiguous aldol donor/acceptor assignment"
        roles = _set_roles(
            reaction,
            {
                FunctionalGroupType.alpha_carbon: Role.reactive_nucleophile,
                FunctionalGroupType.carbonyl: Role.reactive_electrophile,
                FunctionalGroupType.aromatic: Role.spectator,
            },
        )
        center = _mapped_values(reaction, alphas[0]) + _mapped_values(reaction, carbonyls[0])
        policy = "Aldol center policy: donor alpha carbon and acceptor carbonyl."
        corrections.append("aldol_donor_acceptor_center")
        return _with_review_metadata(reaction, roles, center, policy), corrections, policy

    if mechanism in {"benzylic_radical_bromination", "radical_bromination"}:
        benzylic = [
            role
            for role in reaction.group_roles
            if role.group_type is FunctionalGroupType.benzylic_site
        ]
        if not benzylic:
            return None, corrections, "skip: no benzylic_site for radical bromination policy"
        roles = _set_roles(
            reaction,
            {
                FunctionalGroupType.benzylic_site: Role.reactive_radical,
                FunctionalGroupType.aromatic: Role.spectator,
            },
        )
        center = []
        for role in benzylic:
            center.extend(_mapped_values(reaction, role))
        policy = "Radical bromination center policy: benzylic atom only."
        corrections.append("benzylic_radical_center")
        return _with_review_metadata(reaction, roles, center, policy), corrections, policy

    if mechanism == "nitroalkane_deprotonation":
        alphas = [
            role
            for role in reaction.group_roles
            if role.group_type is FunctionalGroupType.alpha_carbon
        ]
        if not alphas:
            return None, corrections, "skip: no alpha_carbon for nitroalkane center policy"
        roles = _set_roles(
            reaction,
            {
                FunctionalGroupType.alpha_carbon: Role.reactive_nucleophile,
                FunctionalGroupType.nitro: Role.spectator,
            },
        )
        center = []
        for role in alphas:
            center.extend(_mapped_values(reaction, role))
        policy = "Nitroalkane policy: alpha carbon represents nitronate center."
        corrections.append("nitroalkane_alpha_center")
        return _with_review_metadata(reaction, roles, center, policy), corrections, policy

    return None, corrections, f"skip: unsupported mechanism {reaction.mechanism_type}"


def _empty_report() -> CenterExpansionPromotionReport:
    return CenterExpansionPromotionReport("", "", "", 0, 0, 0, 0, {}, {}, {}, [], [], [], {})


def promote_center_expansion_reactions(
    reactions: list[LabeledReaction],
    conservative: bool = True,
    include_aldol: bool = True,
    include_controls: bool = True,
) -> tuple[list[LabeledReaction], CenterExpansionPromotionReport]:
    promoted: list[LabeledReaction] = []
    records: list[CenterExpansionPromotionRecord] = []
    skipped: list[dict[str, object]] = []
    for reaction in reactions:
        mechanism = _mechanism(reaction)
        if mechanism in {"aldol", "cross_aldol"} and not include_aldol:
            reviewed = None
            corrections: list[str] = []
            policy = "skip: aldol disabled"
        elif mechanism in _CONTROL_MECHANISMS and not include_controls:
            reviewed = None
            corrections = []
            policy = "skip: controls disabled"
        else:
            reviewed, corrections, policy = apply_mechanism_center_policy(reaction)
        if reviewed is not None:
            promoted.append(reviewed)
            promoted_flag = True
            skip_reason = None
        else:
            promoted_flag = False
            skip_reason = policy
            skipped.append({"reaction_id": reaction.reaction_id, "reason": str(policy)})
        records.append(
            CenterExpansionPromotionRecord(
                reaction_id=reaction.reaction_id,
                mechanism_type=reaction.mechanism_type,
                promoted=promoted_flag,
                skip_reason=skip_reason,
                old_group_roles=[role.to_dict() for role in reaction.group_roles],
                new_group_roles=[role.to_dict() for role in reviewed.group_roles]
                if reviewed
                else [],
                old_reaction_center_atoms=list(reaction.reaction_center_atoms),
                new_reaction_center_atoms=list(reviewed.reaction_center_atoms) if reviewed else [],
                center_policy=str(policy),
                corrections=corrections,
                metadata={},
            )
        )
    mechanism_counts = Counter(rxn.mechanism_type for rxn in promoted)
    role_counts = Counter(role.role.value for rxn in promoted for role in rxn.group_roles)
    center_counts = Counter(str(len(rxn.reaction_center_atoms)) for rxn in promoted)
    report = CenterExpansionPromotionReport(
        input_path="",
        output_path="",
        merged_output_path="",
        n_input_reactions=len(reactions),
        n_promoted_reactions=len(promoted),
        n_skipped_reactions=len(skipped),
        n_promoted_group_labels=sum(len(rxn.group_roles) for rxn in promoted),
        promoted_mechanism_distribution=dict(sorted(mechanism_counts.items())),
        promoted_role_distribution=dict(sorted(role_counts.items())),
        promoted_center_atom_count_distribution=dict(sorted(center_counts.items())),
        promotion_records=records,
        skipped_reactions=skipped,
        warnings=[],
        metadata={"conservative": conservative},
    )
    return promoted, report


def merge_center_expansion_with_cleaned_base(
    base_reactions: list[LabeledReaction],
    promoted_reactions: list[LabeledReaction],
) -> list[LabeledReaction]:
    merged = list(base_reactions)
    seen = {rxn.reaction_id for rxn in merged}
    for reaction in promoted_reactions:
        if reaction.reaction_id in seen:
            candidate_id = f"{reaction.reaction_id}_center_expansion"
            reaction = replace(reaction, reaction_id=candidate_id)
        seen.add(reaction.reaction_id)
        merged.append(reaction)
    return merged


def save_labeled_reactions_json(reactions: list[LabeledReaction], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"reactions": [rxn.to_dict() for rxn in reactions]}, indent=2),
        encoding="utf-8",
    )


def save_center_expansion_promotion_report(
    report: CenterExpansionPromotionReport,
    path: str | Path,
) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
