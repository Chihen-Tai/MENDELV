"""Tests for Phase 8.14 val/test balanced center validation."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from mendel.center_validation import (
    analyze_split_performance_gap,
    assign_leakage_resistant_splits,
    build_leakage_validation_report,
)
from mendel.labels import LabeledGroupRole, LabeledReaction
from mendel.types import FunctionalGroupType, ReactionContext, Role

_ROOT = Path(__file__).parent.parent
_RUN_SCRIPT = _ROOT / "scripts" / "run_balanced_center_validation.py"


def _rxn(mechanism: str, template: str, idx: int) -> LabeledReaction:
    smiles = "[CH3:1][Br:2].[OH-:3]>>[CH3:1][OH:3].[Br-:2]"
    role = Role.leaving_group
    group_type = FunctionalGroupType.halide
    center = [1, 2]
    context = ReactionContext.ionic
    if mechanism == "control":
        smiles = "[CH3:1][CH3:2]>>[CH3:1][CH3:2]"
        role = Role.spectator
        group_type = FunctionalGroupType.alkene
        center = []
        context = ReactionContext.unknown
    return LabeledReaction(
        reaction_id=f"{mechanism}_{template}_{idx}",
        reaction_smiles=smiles,
        context=context,
        mechanism_type=mechanism,
        split="train",
        group_roles=[
            LabeledGroupRole(
                group_id="mol0_group_0",
                molecule_index=0,
                group_type=group_type,
                atom_indices=[0, 1],
                role=role,
            )
        ],
        reaction_center_atoms=center,
        metadata={"template_name": template, "generation_method": "unit_test"},
    )


def _broad_dataset() -> list[LabeledReaction]:
    mechanisms = [
        "sn2",
        "e2",
        "diels_alder",
        "carbonyl_addition",
        "benzylic_radical_bromination",
        "control",
    ]
    reactions: list[LabeledReaction] = []
    for mechanism in mechanisms:
        for template_idx in range(3):
            template = f"{mechanism}_template_{template_idx}"
            for idx in range(3):
                reactions.append(_rxn(mechanism, template, idx))
    return reactions


def _write_dataset(path: Path, reactions: list[LabeledReaction]) -> None:
    path.write_text(
        json.dumps({"reactions": [rxn.to_dict() for rxn in reactions]}),
        encoding="utf-8",
    )


def test_val_test_balanced_template_keeps_leakage_group_in_one_split() -> None:
    split_reactions, _ = assign_leakage_resistant_splits(
        _broad_dataset(),
        strategy="val_test_balanced_template",
    )

    by_group: dict[str, set[str]] = {}
    for rxn in split_reactions:
        by_group.setdefault(str(rxn.metadata["leakage_group"]), set()).add(rxn.split)

    assert all(len(splits) == 1 for splits in by_group.values())


def test_val_and_test_both_get_multiple_mechanisms_when_possible() -> None:
    split_reactions, _ = assign_leakage_resistant_splits(
        _broad_dataset(),
        strategy="val_test_balanced_template",
    )
    val_mechanisms = {rxn.mechanism_type for rxn in split_reactions if rxn.split == "val"}
    test_mechanisms = {rxn.mechanism_type for rxn in split_reactions if rxn.split == "test"}

    assert len(val_mechanisms) >= 5
    assert len(test_mechanisms) >= 5


def test_split_diagnostics_include_mechanisms_and_positive_atom_counts(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.json"
    _write_dataset(dataset, _broad_dataset())

    report = build_leakage_validation_report(dataset, "val_test_balanced_template")
    diagnostics = report.metrics["split_diagnostics"]

    assert diagnostics["positive_atom_counts_by_split"]["val"] > 0
    assert diagnostics["positive_atom_counts_by_split"]["test"] > 0
    assert diagnostics["target_coverage"]["val"]["mechanism_target_met"] is True
    assert diagnostics["target_coverage"]["test"]["mechanism_target_met"] is True


def test_analyze_split_performance_gap_reports_likely_causes() -> None:
    analysis = analyze_split_performance_gap(
        {
            "val": {"reaction_center_f1": 0.6, "per_mechanism_f1": {"sn2": 0.5}},
            "test": {"reaction_center_f1": 0.9, "per_mechanism_f1": {"sn2": 0.9}},
        },
        {
            "mechanism_distribution_by_split": {
                "val": {"sn2": 3},
                "test": {"sn2": 3, "e2": 3, "control": 3},
            },
            "metrics": {
                "split_diagnostics": {
                    "positive_atom_counts_by_split": {"val": 4, "test": 20},
                }
            },
        },
    )

    assert analysis["f1_gap"] == 0.3
    assert analysis["recommendations"]


def test_run_balanced_center_validation_dry_run_smoke(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.json"
    _write_dataset(dataset, _broad_dataset())

    result = subprocess.run(
        [
            sys.executable,
            str(_RUN_SCRIPT),
            "--data",
            str(dataset),
            "--epochs",
            "1",
            "--output-prefix",
            "tiny_balanced",
            "--dry-run",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "models/atom_center_head_tiny_balanced.pt" in result.stdout
    assert "models/atom_center_head.pt" not in result.stdout


def test_no_base_checkpoint_overwrite_in_balanced_script() -> None:
    text = _RUN_SCRIPT.read_text(encoding="utf-8")

    assert "models/atom_center_head_center_balanced.pt" in text
    assert "models/atom_center_head.pt" not in text
