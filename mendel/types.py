"""Core data contracts for MENDEL.

These are lightweight schema definitions shared across all phases.
They do NOT contain chemistry parsing, SMARTS matching, or ML logic.

Central abstraction: functional group = agent.
Each FunctionalGroup is the unit that predicts and negotiates a reaction Role.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ReactionContext(str, Enum):
    """Broad mechanistic category for a reaction."""

    ionic = "ionic"
    radical = "radical"
    pericyclic = "pericyclic"
    unknown = "unknown"


class Role(str, Enum):
    """Role a functional group plays in a single reaction step.

    Roles are mutually exclusive per group per step.
    """

    reactive_nucleophile = "reactive_nucleophile"
    reactive_electrophile = "reactive_electrophile"
    reactive_radical = "reactive_radical"
    leaving_group = "leaving_group"
    spectator = "spectator"


class FunctionalGroupType(str, Enum):
    """Recognised functional group categories.

    Ordering follows the DESIGN.md priority list (specific before general).
    """

    alkene = "alkene"
    alkyne = "alkyne"
    aromatic = "aromatic"
    alcohol = "alcohol"
    phenol = "phenol"
    ether = "ether"
    carbonyl = "carbonyl"
    carboxylic_acid = "carboxylic_acid"
    ester = "ester"
    amine = "amine"
    amide = "amide"
    halide = "halide"
    nitrile = "nitrile"
    nitro = "nitro"
    alpha_carbon = "alpha_carbon"
    benzylic_site = "benzylic_site"
    unknown = "unknown"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class AtomRef:
    """Pointer to a specific atom within the reaction SMILES.

    molecule_index: 0-based index into the reactants list.
    atom_index: 0-based heavy-atom index within that molecule.
    atom_map_num: optional atom-map number from the SMILES string.
    """

    molecule_index: int
    atom_index: int
    atom_map_num: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "molecule_index": self.molecule_index,
            "atom_index": self.atom_index,
            "atom_map_num": self.atom_map_num,
        }


@dataclass
class FunctionalGroup:
    """A detected functional group acting as a local agent.

    group_id: unique identifier within a reaction record (e.g. "mol0_halide_0").
    group_type: enum from FunctionalGroupType.
    atom_refs: atoms that belong to this group.
    smarts: SMARTS pattern that matched this group (filled by Module 1).
    metadata: arbitrary key/value pairs for extensibility.
    """

    group_id: str
    group_type: FunctionalGroupType
    atom_refs: list[AtomRef]
    smarts: str | None = None
    metadata: dict[str, str | int | float | bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "group_type": self.group_type.value,
            "atom_refs": [a.to_dict() for a in self.atom_refs],
            "smarts": self.smarts,
            "metadata": dict(self.metadata),
        }


@dataclass
class RoleAssignment:
    """Predicted (or ground-truth) role for a single functional group.

    group_id: matches FunctionalGroup.group_id.
    role: assigned role.
    confidence: model probability in [0, 1]; None for ground-truth labels.
    reason: human-readable explanation (rule or model output).
    """

    group_id: str
    role: Role
    confidence: float | None = None
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "role": self.role.value,
            "confidence": self.confidence,
            "reason": self.reason,
        }


@dataclass
class ReactionRecord:
    """Top-level container for a single reaction example.

    Used for both training data and inference results.

    reaction_id: stable identifier (e.g. "sn2_methyl_bromide_oh").
    reaction_smiles: full reaction SMILES (reactants>>products).
    context: broad mechanistic category.
    expected_roles: ground-truth RoleAssignments (empty for inference input).
    expected_reaction_center: ground-truth reactive atoms.
    metadata: arbitrary key/value pairs for extensibility.
    """

    reaction_id: str
    reaction_smiles: str
    context: ReactionContext
    expected_roles: list[RoleAssignment] = field(default_factory=list)
    expected_reaction_center: list[AtomRef] = field(default_factory=list)
    metadata: dict[str, str | int | float | bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "reaction_id": self.reaction_id,
            "reaction_smiles": self.reaction_smiles,
            "context": self.context.value,
            "expected_roles": [r.to_dict() for r in self.expected_roles],
            "expected_reaction_center": [a.to_dict() for a in self.expected_reaction_center],
            "metadata": dict(self.metadata),
        }
