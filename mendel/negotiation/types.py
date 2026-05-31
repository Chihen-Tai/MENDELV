"""Negotiation data contracts and shared helpers (extracted from negotiator.py)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mendel.predictor import RolePrediction
from mendel.types import AtomRef, FunctionalGroup, FunctionalGroupType, ReactionContext, Role

_PI_GROUP_TYPES: frozenset[FunctionalGroupType] = frozenset({
    FunctionalGroupType.alkene,
    FunctionalGroupType.alkyne,
    FunctionalGroupType.aromatic,
})

@dataclass
class NegotiationWarning:
    """A warning produced during negotiation.

    severity: "info" | "warning" | "error"
    """

    code: str
    message: str
    severity: str
    metadata: dict[str, str | int | float | bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict."""
        return {
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
            "metadata": dict(self.metadata),
        }


# ---------------------------------------------------------------------------
# NegotiatedRoleAssignment
# ---------------------------------------------------------------------------


@dataclass
class NegotiatedRoleAssignment:
    """Final role assignment for a single functional-group agent after negotiation.

    raw_role / raw_confidence: values from Phase 5 predictor (never mutated).
    final_role / final_confidence: values after global consistency rules.
    subrole: optional fine-grained label; does not introduce a new Role enum value.
    is_reaction_center: True when this group's atoms belong to the reaction center.
    """

    group_id: str
    group_type: FunctionalGroupType
    raw_role: Role
    final_role: Role
    raw_confidence: float
    final_confidence: float
    reason: str
    subrole: str | None = None
    is_reaction_center: bool = False
    metadata: dict[str, str | int | float | bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict."""
        return {
            "group_id": self.group_id,
            "group_type": self.group_type.value,
            "raw_role": self.raw_role.value,
            "final_role": self.final_role.value,
            "raw_confidence": self.raw_confidence,
            "final_confidence": self.final_confidence,
            "reason": self.reason,
            "subrole": self.subrole,
            "is_reaction_center": self.is_reaction_center,
            "metadata": dict(self.metadata),
        }


# ---------------------------------------------------------------------------
# NegotiationResult
# ---------------------------------------------------------------------------


@dataclass
class NegotiationResult:
    """Full negotiation output for a single reaction."""

    reaction_smiles: str
    context: ReactionContext
    mechanism_hint: str
    assignments: list[NegotiatedRoleAssignment]
    reaction_center_atoms: list[AtomRef]
    warnings: list[NegotiationWarning]
    metadata: dict[str, str | int | float | bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict."""
        return {
            "reaction_smiles": self.reaction_smiles,
            "context": self.context.value,
            "mechanism_hint": self.mechanism_hint,
            "assignments": [a.to_dict() for a in self.assignments],
            "reaction_center_atoms": [a.to_dict() for a in self.reaction_center_atoms],
            "warnings": [w.to_dict() for w in self.warnings],
            "metadata": dict(self.metadata),
        }


@dataclass
class NegotiatorConfig:
    """Configuration for the rule-based negotiator."""

    mode: str = "rule_based"
    use_confidence: bool = True
    spectator_confidence_threshold: float = 0.70
    reactive_confidence_threshold: float = 0.50
    suppress_control_centers: bool = True
    prefer_mechanism_specific_centers: bool = True
    metadata: dict[str, str | int | float | bool] = field(default_factory=dict)
    min_center_confidence: float = 0.45
    prefer_high_confidence_candidates: bool = True
    allow_role_downgrade_to_spectator: bool = True
    require_ionic_pair: bool = True
    require_pericyclic_partners: bool = True
    require_radical_center: bool = True


def _warn(
    code: str,
    message: str,
    severity: str,
    meta: dict[str, str | int | float | bool] | None = None,
) -> NegotiationWarning:
    return NegotiationWarning(
        code=code,
        message=message,
        severity=severity,
        metadata=meta or {},
    )


def _has_group_type(groups: list[FunctionalGroup], gt: FunctionalGroupType) -> bool:
    return any(g.group_type == gt for g in groups)


def _groups_of_type(
    groups: list[FunctionalGroup], gt: FunctionalGroupType
) -> list[FunctionalGroup]:
    return [g for g in groups if g.group_type == gt]


def _group_by_id(groups: list[FunctionalGroup]) -> dict[str, FunctionalGroup]:
    return {g.group_id: g for g in groups}


def _pred_by_id(predictions: list[RolePrediction]) -> dict[str, RolePrediction]:
    return {p.group_id: p for p in predictions}


def _deduplicate_atom_refs(atom_refs: list[AtomRef]) -> list[AtomRef]:
    """Deduplicate AtomRefs by (molecule_index, atom_index), stable order."""
    seen: set[tuple[int, int]] = set()
    result: list[AtomRef] = []
    for ref in atom_refs:
        key = (ref.molecule_index, ref.atom_index)
        if key not in seen:
            seen.add(key)
            result.append(ref)
    return result


def _is_reactive_role(role: Role) -> bool:
    return role in {
        Role.reactive_nucleophile,
        Role.reactive_electrophile,
        Role.reactive_radical,
        Role.leaving_group,
    }
