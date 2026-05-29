"""Phase 8.6: Conservative promotion of reviewed auto-candidate labels.

This module promotes draft labels from a review queue into a curated
supplemental dataset using deterministic, conservative policy rules.
It does not train MLP models and does not modify base datasets in place.
"""

from __future__ import annotations

import copy
import hashlib
from collections import Counter
from typing import Any

from mendel.labels import LabeledReaction
from mendel.types import FunctionalGroupType, Role

PROMOTION_POLICY = "conservative_phase_8_6"
PROMOTED_SOURCE = "auto_candidate_human_review_promoted"
REVIEWED_BY = "MENDELV Phase 8.6 conservative promotion"
REPORT_NOTE = (
    "Promoted labels are conservative manually reviewed-style labels, "
    "but should still be spot-checked before final publication."
)


def stable_split_from_reaction_id(reaction_id: str) -> str:
    """Assign a deterministic train/val/test split from reaction_id."""
    digest = hashlib.sha256(reaction_id.encode("utf-8")).hexdigest()
    bucket = int(digest, 16) % 10
    if bucket == 0:
        return "test"
    if bucket in {1, 2}:
        return "val"
    return "train"


def _append_note(existing: str | None, extra: str) -> str:
    if existing:
        return f"{existing}; {extra}"
    return extra


def _mechanism_key(mechanism_type: str) -> str:
    return mechanism_type.strip().lower()


def _set_promoted_group_role(
    reaction_id: str,
    group_role: Any,
    new_role: Role,
    reason: str,
    corrected_labels: list[dict[str, str]],
) -> None:
    old_role = group_role.role
    old_role_value = old_role.value
    new_role_value = new_role.value
    group_role.role = new_role
    group_role.confidence = "manual"

    note = (
        f"original_draft_role={old_role_value}; "
        f"promotion_reason={reason}"
    )
    if old_role != new_role:
        note = f"{note}; role_correction={old_role_value}->{new_role_value}"
        corrected_labels.append({
            "reaction_id": reaction_id,
            "group_id": group_role.group_id,
            "group_type": group_role.group_type.value,
            "old_role": old_role_value,
            "new_role": new_role_value,
            "reason": reason,
        })
    if group_role.notes:
        note = _append_note(note, f"draft_notes={group_role.notes}")
    group_role.notes = note


def _finalize_promoted_reaction(
    reaction: LabeledReaction,
    promotion_note: str,
) -> None:
    metadata = dict(reaction.metadata)
    old_source = metadata.get("source")
    if old_source is not None:
        metadata["original_source"] = old_source
    metadata["review_status"] = "promoted_manual_review"
    metadata["source"] = PROMOTED_SOURCE
    metadata["reviewed_by"] = REVIEWED_BY
    metadata["needs_manual_review"] = False
    metadata["exclude_from_ground_truth_until_review"] = False
    metadata["promotion_policy"] = PROMOTION_POLICY
    metadata["promotion_note"] = promotion_note
    reaction.metadata = metadata
    reaction.split = stable_split_from_reaction_id(reaction.reaction_id)


def _promote_sn2(
    reaction: LabeledReaction,
    corrected_labels: list[dict[str, str]],
) -> tuple[str | None, str]:
    has_halide = any(
        role.group_type == FunctionalGroupType.halide for role in reaction.group_roles
    )
    if not has_halide:
        return "sn2_missing_halide_group", ""

    for role in reaction.group_roles:
        if role.group_type == FunctionalGroupType.halide:
            _set_promoted_group_role(
                reaction.reaction_id,
                role,
                Role.leaving_group,
                "SN2 substrate halide promoted as leaving_group.",
                corrected_labels,
            )
            continue
        if role.group_type in {FunctionalGroupType.aromatic, FunctionalGroupType.benzylic_site}:
            _set_promoted_group_role(
                reaction.reaction_id,
                role,
                Role.spectator,
                "SN2 context: aromatic/benzylic-site labels are spectators in v0.1.",
                corrected_labels,
            )
            continue
        _set_promoted_group_role(
            reaction.reaction_id,
            role,
            Role.spectator,
            "SN2 context: non-halide substrate groups are treated as spectators.",
            corrected_labels,
        )

    return (
        None,
        "external nucleophile is not represented by current functional-group schema.",
    )


def _promote_e2(
    reaction: LabeledReaction,
    corrected_labels: list[dict[str, str]],
) -> tuple[str | None, str]:
    has_halide = any(
        role.group_type == FunctionalGroupType.halide for role in reaction.group_roles
    )
    if not has_halide:
        return "e2_missing_halide_group", ""

    for role in reaction.group_roles:
        if role.group_type == FunctionalGroupType.halide:
            _set_promoted_group_role(
                reaction.reaction_id,
                role,
                Role.leaving_group,
                "E2 substrate halide promoted as leaving_group.",
                corrected_labels,
            )
            continue
        _set_promoted_group_role(
            reaction.reaction_id,
            role,
            Role.spectator,
            "E2 context: non-halide substrate groups are treated as spectators.",
            corrected_labels,
        )

    return (
        None,
        "base and beta C-H abstraction are not represented as functional-group agents in v0.1.",
    )


def _promote_benzylic_radical_bromination(
    reaction: LabeledReaction,
    corrected_labels: list[dict[str, str]],
) -> tuple[str | None, str]:
    has_benzylic = any(
        role.group_type == FunctionalGroupType.benzylic_site for role in reaction.group_roles
    )
    if not has_benzylic:
        return "benzylic_radical_missing_benzylic_site", ""

    for role in reaction.group_roles:
        if role.group_type == FunctionalGroupType.benzylic_site:
            _set_promoted_group_role(
                reaction.reaction_id,
                role,
                Role.reactive_radical,
                "Benzylic radical bromination: benzylic_site is the radical center.",
                corrected_labels,
            )
            continue
        if role.group_type == FunctionalGroupType.aromatic:
            _set_promoted_group_role(
                reaction.reaction_id,
                role,
                Role.spectator,
                "Benzylic radical bromination: aromatic ring remains spectator.",
                corrected_labels,
            )
            continue
        _set_promoted_group_role(
            reaction.reaction_id,
            role,
            Role.spectator,
            "Benzylic radical bromination: non-benzylic groups are treated as spectators.",
            corrected_labels,
        )

    return (
        None,
        "Br2/radical chain carrier is not represented by current functional-group schema.",
    )


def _promote_diels_alder(
    reaction: LabeledReaction,
    corrected_labels: list[dict[str, str]],
) -> tuple[str | None, str]:
    alkene_roles = [
        role for role in reaction.group_roles if role.group_type == FunctionalGroupType.alkene
    ]
    if len(alkene_roles) < 3:
        return "diels_alder_insufficient_alkene_groups", ""

    per_molecule = Counter(role.molecule_index for role in alkene_roles)
    diene_molecule, diene_count = max(
        per_molecule.items(),
        key=lambda item: (item[1], -item[0]),
    )
    if diene_count < 2:
        return "diels_alder_ambiguous_diene_selection", ""

    dienophile_candidates = [
        role for role in alkene_roles if role.molecule_index != diene_molecule
    ]
    if len(dienophile_candidates) != 1:
        return "diels_alder_ambiguous_dienophile_selection", ""

    diene_ids = {
        role.group_id for role in alkene_roles if role.molecule_index == diene_molecule
    }
    dienophile_id = dienophile_candidates[0].group_id
    substituent_types = {
        FunctionalGroupType.nitrile,
        FunctionalGroupType.ester,
        FunctionalGroupType.carbonyl,
        FunctionalGroupType.alpha_carbon,
        FunctionalGroupType.aromatic,
    }

    for role in reaction.group_roles:
        if role.group_id in diene_ids:
            _set_promoted_group_role(
                reaction.reaction_id,
                role,
                Role.reactive_nucleophile,
                "Diels-Alder: diene alkene promoted to reactive_nucleophile in flat v0.1 taxonomy.",
                corrected_labels,
            )
            continue
        if role.group_id == dienophile_id:
            _set_promoted_group_role(
                reaction.reaction_id,
                role,
                Role.reactive_electrophile,
                "Diels-Alder: dienophile alkene promoted to reactive_electrophile.",
                corrected_labels,
            )
            continue
        if role.group_type in substituent_types:
            _set_promoted_group_role(
                reaction.reaction_id,
                role,
                Role.spectator,
                "Diels-Alder: substituent groups modulate reactivity but are "
                "spectators in flat v0.1 roles.",
                corrected_labels,
            )
            continue
        _set_promoted_group_role(
            reaction.reaction_id,
            role,
            Role.spectator,
            "Diels-Alder: non-partner groups are treated as spectators conservatively.",
            corrected_labels,
        )

    return (
        None,
        "Reacting pi partners are the alkene groups; EWG substituents are "
        "spectators in flat v0.1 roles.",
    )


def _promote_carbonyl_addition(
    reaction: LabeledReaction,
    corrected_labels: list[dict[str, str]],
) -> tuple[str | None, str]:
    has_carbonyl = any(
        role.group_type == FunctionalGroupType.carbonyl for role in reaction.group_roles
    )
    if not has_carbonyl:
        return "carbonyl_addition_missing_carbonyl_group", ""

    for role in reaction.group_roles:
        if role.group_type == FunctionalGroupType.carbonyl:
            _set_promoted_group_role(
                reaction.reaction_id,
                role,
                Role.reactive_electrophile,
                "Carbonyl addition: carbonyl is the substrate electrophile.",
                corrected_labels,
            )
            continue
        if role.group_type == FunctionalGroupType.alpha_carbon:
            _set_promoted_group_role(
                reaction.reaction_id,
                role,
                Role.spectator,
                "Carbonyl addition: alpha_carbon is spectator when external "
                "hydride/cyanide reacts.",
                corrected_labels,
            )
            continue
        _set_promoted_group_role(
            reaction.reaction_id,
            role,
            Role.spectator,
            "Carbonyl addition: non-carbonyl substrate groups are treated as "
            "spectators conservatively.",
            corrected_labels,
        )

    return (
        None,
        "External hydride/cyanide nucleophile is not represented by current "
        "functional-group schema.",
    )


def _promote_control_like(
    reaction: LabeledReaction,
    corrected_labels: list[dict[str, str]],
) -> tuple[str | None, str]:
    for role in reaction.group_roles:
        _set_promoted_group_role(
            reaction.reaction_id,
            role,
            Role.spectator,
            "Control/no-reaction example: groups set to spectator conservatively.",
            corrected_labels,
        )

    return (
        None,
        "No-reaction control in v0.1: detected substrate groups are curated as spectators.",
    )


def _promote_nitroalkane_deprotonation(
    reaction: LabeledReaction,
    corrected_labels: list[dict[str, str]],
) -> tuple[str | None, str]:
    has_alpha = any(
        role.group_type == FunctionalGroupType.alpha_carbon for role in reaction.group_roles
    )
    has_nitro = any(
        role.group_type == FunctionalGroupType.nitro for role in reaction.group_roles
    )
    if not has_alpha or not has_nitro:
        return "nitroalkane_deprotonation_missing_alpha_or_nitro", ""

    for role in reaction.group_roles:
        if role.group_type == FunctionalGroupType.alpha_carbon:
            _set_promoted_group_role(
                reaction.reaction_id,
                role,
                Role.reactive_nucleophile,
                "Nitroalkane deprotonation: alpha carbon represents the "
                "nitronate-like nucleophilic center.",
                corrected_labels,
            )
            continue
        if role.group_type == FunctionalGroupType.nitro:
            _set_promoted_group_role(
                reaction.reaction_id,
                role,
                Role.spectator,
                "Nitroalkane deprotonation: nitro group is curated as "
                "spectator in flat v0.1 roles.",
                corrected_labels,
            )
            continue
        _set_promoted_group_role(
            reaction.reaction_id,
            role,
            Role.spectator,
            "Nitroalkane deprotonation: non-alpha/non-nitro groups are spectators conservatively.",
            corrected_labels,
        )

    return (
        None,
        "alpha carbon represents nitronate-like nucleophilic center after "
        "deprotonation in the flat v0.1 role taxonomy.",
    )


def _promote_aldol_if_unambiguous(
    reaction: LabeledReaction,
    corrected_labels: list[dict[str, str]],
) -> tuple[str | None, str]:
    group_ids = [role.group_id for role in reaction.group_roles]
    if len(group_ids) != len(set(group_ids)):
        return "aldol_inconsistent_group_ids", ""

    donor_alpha = [
        role
        for role in reaction.group_roles
        if role.group_type == FunctionalGroupType.alpha_carbon
        and role.role == Role.reactive_nucleophile
    ]
    acceptor_carbonyl = [
        role
        for role in reaction.group_roles
        if role.group_type == FunctionalGroupType.carbonyl
        and role.role == Role.reactive_electrophile
    ]
    if len(donor_alpha) != 1 or len(acceptor_carbonyl) != 1:
        return "aldol_ambiguous_donor_acceptor", ""

    donor_id = donor_alpha[0].group_id
    acceptor_id = acceptor_carbonyl[0].group_id
    for role in reaction.group_roles:
        if (
            role.group_type == FunctionalGroupType.alpha_carbon
            and role.group_id != donor_id
            and role.role != Role.spectator
        ):
            return "aldol_secondary_alpha_not_spectator", ""
        if role.group_type == FunctionalGroupType.aromatic and role.role != Role.spectator:
            return "aldol_aromatic_not_spectator", ""

    for role in reaction.group_roles:
        if role.group_id == donor_id:
            _set_promoted_group_role(
                reaction.reaction_id,
                role,
                Role.reactive_nucleophile,
                "Aldol (unambiguous): selected donor alpha_carbon promoted as "
                "reactive_nucleophile.",
                corrected_labels,
            )
            continue
        if role.group_id == acceptor_id:
            _set_promoted_group_role(
                reaction.reaction_id,
                role,
                Role.reactive_electrophile,
                "Aldol (unambiguous): selected acceptor carbonyl promoted as "
                "reactive_electrophile.",
                corrected_labels,
            )
            continue
        _set_promoted_group_role(
            reaction.reaction_id,
            role,
            Role.spectator,
            "Aldol (unambiguous): non-primary groups set to spectator conservatively.",
            corrected_labels,
        )

    return (
        None,
        "Aldol promoted only under unambiguous donor/acceptor selection in flat v0.1 taxonomy.",
    )


def promote_review_queue_reaction(
    reaction: LabeledReaction,
    include_aldol: bool = False,
    include_controls: bool = True,
) -> tuple[LabeledReaction | None, str | None, list[dict[str, str]], list[str]]:
    """Promote one review-queue reaction conservatively or return a skip reason."""
    if not reaction.group_roles:
        return None, "no_group_roles", [], []

    promoted = copy.deepcopy(reaction)
    corrected_labels: list[dict[str, str]] = []
    warnings: list[str] = []
    mechanism = _mechanism_key(promoted.mechanism_type)
    skip_reason: str | None = None
    promotion_note = ""

    if mechanism == "sn2":
        skip_reason, promotion_note = _promote_sn2(promoted, corrected_labels)
    elif mechanism == "e2":
        skip_reason, promotion_note = _promote_e2(promoted, corrected_labels)
    elif mechanism == "benzylic_radical_bromination":
        skip_reason, promotion_note = _promote_benzylic_radical_bromination(
            promoted,
            corrected_labels,
        )
    elif mechanism == "diels_alder":
        skip_reason, promotion_note = _promote_diels_alder(promoted, corrected_labels)
    elif mechanism == "carbonyl_addition":
        skip_reason, promotion_note = _promote_carbonyl_addition(promoted, corrected_labels)
    elif mechanism in {"control", "ester_control", "nitrile_control"}:
        if not include_controls:
            return None, "control_promotion_disabled", [], []
        skip_reason, promotion_note = _promote_control_like(promoted, corrected_labels)
    elif mechanism == "nitroalkane_deprotonation":
        skip_reason, promotion_note = _promote_nitroalkane_deprotonation(promoted, corrected_labels)
    elif mechanism in {"aldol", "cross_aldol"}:
        if not include_aldol:
            return None, "aldol_skipped_by_default", [], []
        skip_reason, promotion_note = _promote_aldol_if_unambiguous(promoted, corrected_labels)
    else:
        if promoted.metadata.get("product_simplified") is True:
            return None, "product_simplified_assignment_ambiguous", [], []
        return None, "unsupported_mechanism_for_conservative_promotion", [], []

    if skip_reason is not None:
        return None, skip_reason, [], warnings

    _finalize_promoted_reaction(promoted, promotion_note)
    return promoted, None, corrected_labels, warnings


def promote_review_queue(
    reactions: list[LabeledReaction],
    include_aldol: bool = False,
    include_controls: bool = True,
    max_promote: int | None = None,
) -> tuple[list[LabeledReaction], list[dict[str, str]], list[dict[str, str]], list[str]]:
    """Promote a review queue according to conservative Phase 8.6 policy."""
    promoted: list[LabeledReaction] = []
    skipped: list[dict[str, str]] = []
    corrected_labels: list[dict[str, str]] = []
    warnings: list[str] = []

    for reaction in reactions:
        if max_promote is not None and len(promoted) >= max_promote:
            skipped.append({
                "reaction_id": reaction.reaction_id,
                "mechanism_type": reaction.mechanism_type,
                "reason": "max_promote_reached",
            })
            continue

        promoted_reaction, skip_reason, corrections, promotion_warnings = (
            promote_review_queue_reaction(
                reaction,
                include_aldol=include_aldol,
                include_controls=include_controls,
            )
        )
        if promoted_reaction is None:
            skipped.append({
                "reaction_id": reaction.reaction_id,
                "mechanism_type": reaction.mechanism_type,
                "reason": skip_reason or "promotion_skipped",
            })
            continue

        promoted.append(promoted_reaction)
        corrected_labels.extend(corrections)
        warnings.extend(promotion_warnings)

    return promoted, skipped, corrected_labels, warnings


def merge_promoted_with_base(
    base_reactions: list[LabeledReaction],
    promoted_reactions: list[LabeledReaction],
) -> tuple[list[LabeledReaction], list[str]]:
    """Merge promoted reactions with the base curated dataset by reaction_id."""
    merged = list(base_reactions)
    seen_ids = {reaction.reaction_id for reaction in base_reactions}
    warnings: list[str] = []

    for reaction in promoted_reactions:
        if reaction.reaction_id in seen_ids:
            warnings.append(
                f"Reaction {reaction.reaction_id} already exists in base dataset; "
                "keeping base record."
            )
            continue
        merged.append(reaction)
        seen_ids.add(reaction.reaction_id)

    return merged, warnings


def build_promotion_report(
    input_reactions: list[LabeledReaction],
    promoted_reactions: list[LabeledReaction],
    skipped_reactions: list[dict[str, str]],
    corrected_labels: list[dict[str, str]],
    warnings: list[str],
    output_paths: dict[str, str],
) -> dict[str, Any]:
    """Build the Phase 8.6 promotion report payload."""
    promoted_mechanisms = Counter(
        reaction.mechanism_type for reaction in promoted_reactions
    )
    promoted_roles = Counter(
        role.role.value
        for reaction in promoted_reactions
        for role in reaction.group_roles
    )
    promoted_group_types = Counter(
        role.group_type.value
        for reaction in promoted_reactions
        for role in reaction.group_roles
    )

    return {
        "n_input_reactions": len(input_reactions),
        "n_promoted_reactions": len(promoted_reactions),
        "n_skipped_reactions": len(skipped_reactions),
        "n_promoted_group_labels": sum(
            len(reaction.group_roles) for reaction in promoted_reactions
        ),
        "n_corrected_labels": len(corrected_labels),
        "promoted_mechanism_distribution": dict(sorted(promoted_mechanisms.items())),
        "promoted_role_distribution": dict(sorted(promoted_roles.items())),
        "promoted_group_type_distribution": dict(sorted(promoted_group_types.items())),
        "skipped_reactions": skipped_reactions,
        "corrected_labels": corrected_labels,
        "warnings": warnings,
        "output_paths": output_paths,
        "note": REPORT_NOTE,
    }
