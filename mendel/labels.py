"""Phase 4 — Labeled reaction dataset schema and utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mendel.types import FunctionalGroupType, ReactionContext, Role


class LabelValidationError(Exception):
    """Raised when a labeled reaction or dataset fails validation."""


@dataclass
class LabeledGroupRole:
    """Ground-truth role label for one functional group in a reaction.

    group_id: matches the identifier assigned by identifier.py.
    molecule_index: 0-based reactant index.
    group_type: functional group category.
    atom_indices: 0-based atom indices within the molecule.
    role: ground-truth role.
    confidence: 'manual' for human-curated labels, else a float string.
    notes: optional free-text annotation.
    """

    group_id: str
    molecule_index: int
    group_type: FunctionalGroupType
    atom_indices: list[int]
    role: Role
    confidence: str = "manual"
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "molecule_index": self.molecule_index,
            "group_type": self.group_type.value,
            "atom_indices": list(self.atom_indices),
            "role": self.role.value,
            "confidence": self.confidence,
            "notes": self.notes,
        }


@dataclass
class LabeledReaction:
    """A fully labeled reaction record used for training and evaluation.

    reaction_id: stable human-readable identifier.
    reaction_smiles: atom-mapped reaction SMILES (reactants>>products).
    context: broad mechanistic category.
    mechanism_type: fine-grained label (e.g. 'SN2', 'E2', 'Diels-Alder').
    split: dataset partition — 'train', 'val', or 'test'.
    group_roles: ground-truth role labels for all relevant groups.
    reaction_center_atoms: atom-map numbers of atoms that change bonds.
    metadata: arbitrary key/value pairs.
    """

    reaction_id: str
    reaction_smiles: str
    context: ReactionContext
    mechanism_type: str
    split: str
    group_roles: list[LabeledGroupRole] = field(default_factory=list)
    reaction_center_atoms: list[int] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "reaction_id": self.reaction_id,
            "reaction_smiles": self.reaction_smiles,
            "context": self.context.value,
            "mechanism_type": self.mechanism_type,
            "split": self.split,
            "group_roles": [r.to_dict() for r in self.group_roles],
            "reaction_center_atoms": list(self.reaction_center_atoms),
            "metadata": dict(self.metadata),
        }


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def _lgr_from_dict(d: dict[str, Any]) -> LabeledGroupRole:
    return LabeledGroupRole(
        group_id=d["group_id"],
        molecule_index=int(d["molecule_index"]),
        group_type=FunctionalGroupType(d["group_type"]),
        atom_indices=list(d["atom_indices"]),
        role=Role(d["role"]),
        confidence=str(d.get("confidence", "manual")),
        notes=d.get("notes"),
    )


def _lr_from_dict(d: dict[str, Any]) -> LabeledReaction:
    return LabeledReaction(
        reaction_id=d["reaction_id"],
        reaction_smiles=d["reaction_smiles"],
        context=ReactionContext(d["context"]),
        mechanism_type=d["mechanism_type"],
        split=d["split"],
        group_roles=[_lgr_from_dict(r) for r in d.get("group_roles", [])],
        reaction_center_atoms=list(d.get("reaction_center_atoms", [])),
        metadata=dict(d.get("metadata", {})),
    )


def load_labeled_reactions(path: str | Path) -> list[LabeledReaction]:
    """Load a labeled dataset from a JSON file.

    The file must contain a top-level 'reactions' list.
    Raises FileNotFoundError or json.JSONDecodeError on bad input.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [_lr_from_dict(r) for r in data["reactions"]]


def save_labeled_reactions(reactions: list[LabeledReaction], path: str | Path) -> None:
    """Serialise labeled reactions to a JSON file with a 'reactions' wrapper."""
    payload = {"reactions": [r.to_dict() for r in reactions]}
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

_VALID_SPLITS = {"train", "val", "test"}


def validate_labeled_reaction(rxn: LabeledReaction) -> bool:
    """Validate a single LabeledReaction.

    Raises LabelValidationError describing the first problem found.
    Returns True on success.
    """
    if not rxn.reaction_id:
        raise LabelValidationError(f"Empty reaction_id: {rxn!r}")
    if ">>" not in rxn.reaction_smiles:
        raise LabelValidationError(f"{rxn.reaction_id}: reaction_smiles must contain '>>'")
    if rxn.split not in _VALID_SPLITS:
        raise LabelValidationError(
            f"{rxn.reaction_id}: invalid split '{rxn.split}', expected one of {_VALID_SPLITS}"
        )
    seen_ids: set[str] = set()
    for lgr in rxn.group_roles:
        if lgr.group_id in seen_ids:
            raise LabelValidationError(
                f"{rxn.reaction_id}: duplicate group_id '{lgr.group_id}'"
            )
        seen_ids.add(lgr.group_id)
        if not lgr.atom_indices:
            raise LabelValidationError(
                f"{rxn.reaction_id}/{lgr.group_id}: atom_indices must not be empty"
            )
    return True


def validate_labeled_dataset(reactions: list[LabeledReaction]) -> bool:
    """Validate every reaction in the dataset.

    Raises LabelValidationError on the first failure.
    Returns True if all pass.
    """
    seen_ids: set[str] = set()
    for rxn in reactions:
        if rxn.reaction_id in seen_ids:
            raise LabelValidationError(f"Duplicate reaction_id across dataset: '{rxn.reaction_id}'")
        seen_ids.add(rxn.reaction_id)
        validate_labeled_reaction(rxn)
    return True


# ---------------------------------------------------------------------------
# Summarisation and training utilities
# ---------------------------------------------------------------------------


def summarize_labeled_dataset(reactions: list[LabeledReaction]) -> dict[str, Any]:
    """Return counts and distributions for a labeled dataset."""
    role_counts: dict[str, int] = {}
    mechanism_counts: dict[str, int] = {}
    split_counts: dict[str, int] = {}
    total_labels = 0

    for rxn in reactions:
        mechanism_counts[rxn.mechanism_type] = mechanism_counts.get(rxn.mechanism_type, 0) + 1
        split_counts[rxn.split] = split_counts.get(rxn.split, 0) + 1
        for lgr in rxn.group_roles:
            role_counts[lgr.role.value] = role_counts.get(lgr.role.value, 0) + 1
            total_labels += 1

    return {
        "n_reactions": len(reactions),
        "n_labels": total_labels,
        "role_distribution": role_counts,
        "mechanism_distribution": mechanism_counts,
        "split_distribution": split_counts,
    }


def labels_to_training_rows(
    reactions: list[LabeledReaction],
) -> list[dict[str, Any]]:
    """Flatten labeled reactions into one row per group role.

    Each row contains reaction metadata + the group role fields, suitable
    for passing alongside descriptor vectors to a training loop.
    """
    rows: list[dict[str, Any]] = []
    for rxn in reactions:
        base = {
            "reaction_id": rxn.reaction_id,
            "reaction_smiles": rxn.reaction_smiles,
            "context": rxn.context.value,
            "mechanism_type": rxn.mechanism_type,
            "split": rxn.split,
        }
        for lgr in rxn.group_roles:
            rows.append({**base, **lgr.to_dict()})
    return rows
