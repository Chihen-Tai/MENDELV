"""Tests for mechanism-balanced template splitting."""

from __future__ import annotations

from mendel.center_validation import assign_leakage_resistant_splits
from mendel.labels import LabeledReaction
from mendel.types import ReactionContext


def _rxn(idx: int, mechanism: str, template: str) -> LabeledReaction:
    return LabeledReaction(
        reaction_id=f"{mechanism}_{idx}",
        reaction_smiles="[CH3:1][Br:2]>>[CH3:1][OH:3]",
        context=ReactionContext.ionic,
        mechanism_type=mechanism,
        split="train",
        group_roles=[],
        reaction_center_atoms=[1, 2],
        metadata={"template_name": template, "generation_method": "unit_test"},
    )


def test_mechanism_balanced_template_keeps_template_together() -> None:
    reactions = [
        _rxn(0, "sn2", "same"),
        _rxn(1, "sn2", "same"),
        _rxn(2, "e2", "other"),
        _rxn(3, "diels_alder", "third"),
    ]

    split_reactions, _ = assign_leakage_resistant_splits(
        reactions,
        strategy="mechanism_balanced_template",
    )

    same_splits = {rxn.split for rxn in split_reactions if rxn.metadata["leakage_group"] == "same"}
    assert len(same_splits) == 1


def test_mechanism_balanced_template_test_has_multiple_mechanisms_when_possible() -> None:
    reactions = [
        _rxn(i, mech, f"{mech}_{i}")
        for i, mech in enumerate(
            ["sn2", "e2", "diels_alder", "carbonyl_addition", "control", "aldol"] * 3
        )
    ]

    split_reactions, _ = assign_leakage_resistant_splits(
        reactions,
        strategy="mechanism_balanced_template",
        train_fraction=0.5,
        val_fraction=0.2,
        test_fraction=0.3,
    )
    test_mechanisms = {rxn.mechanism_type for rxn in split_reactions if rxn.split == "test"}

    assert len(test_mechanisms) >= 2


def test_mechanism_balanced_template_deterministic() -> None:
    reactions = [
        _rxn(i, mech, f"{mech}_{i}")
        for i, mech in enumerate(["sn2", "e2", "diels_alder", "carbonyl_addition"] * 2)
    ]

    first, _ = assign_leakage_resistant_splits(reactions, strategy="mechanism_balanced_template")
    second, _ = assign_leakage_resistant_splits(reactions, strategy="mechanism_balanced_template")

    assert [(rxn.reaction_id, rxn.split) for rxn in first] == [
        (rxn.reaction_id, rxn.split) for rxn in second
    ]


def test_mechanism_balanced_template_warns_when_impossible() -> None:
    reactions = [_rxn(0, "sn2", "only")]

    split_reactions, _ = assign_leakage_resistant_splits(
        reactions,
        strategy="mechanism_balanced_template",
    )

    assert "split_warning" in split_reactions[0].metadata
