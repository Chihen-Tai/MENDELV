"""Reaction-center atom collection and marking."""

from __future__ import annotations

from mendel.negotiation.types import (
    NegotiatedRoleAssignment,
    _deduplicate_atom_refs,
    _group_by_id,
)
from mendel.parser import ParsedReaction
from mendel.types import (
    AtomRef,
    FunctionalGroup,
)


def infer_reaction_center_atoms(
    parsed_reaction: ParsedReaction,
    groups: list[FunctionalGroup],
    assignments: list[NegotiatedRoleAssignment],
) -> list[AtomRef]:
    """Collect atom refs from groups marked as reaction center.

    Deduplicates by (molecule_index, atom_index) in stable input order.
    """
    group_lookup = _group_by_id(groups)
    all_refs: list[AtomRef] = []
    for assignment in assignments:
        if not assignment.is_reaction_center:
            continue
        group = group_lookup.get(assignment.group_id)
        if group is None:
            continue
        all_refs.extend(group.atom_refs)
    return _deduplicate_atom_refs(all_refs)


def clear_centers(assign_by_id: dict[str, NegotiatedRoleAssignment]) -> None:
    for assignment in assign_by_id.values():
        assignment.is_reaction_center = False
        assignment.metadata["center_selection_reason"] = ""


def mark_center(
    assign_by_id: dict[str, NegotiatedRoleAssignment],
    group_id: str,
    reason: str,
) -> None:
    assignment = assign_by_id.get(group_id)
    if assignment is None:
        return
    assignment.is_reaction_center = True
    assignment.metadata["center_selection_reason"] = reason
    assignment.metadata["final_role"] = assignment.final_role.value
    assignment.metadata["role_changed_by_negotiation"] = (
        assignment.raw_role != assignment.final_role
    )
