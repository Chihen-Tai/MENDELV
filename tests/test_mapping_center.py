"""Tests for Phase 8.14 mapping-aware center labels."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from mendel.labels import LabeledGroupRole, LabeledReaction, load_labeled_reactions
from mendel.mapping_center import (
    MappingCenterAuditReport,
    apply_mapping_center_suggestions,
    audit_labeled_centers_against_mapping,
    extract_bond_changes,
    has_atom_mapping,
    infer_center_atoms_from_mapping,
    save_mapping_center_audit_report,
)
from mendel.types import FunctionalGroupType, ReactionContext, Role

_ROOT = Path(__file__).parent.parent
_SCRIPT = _ROOT / "scripts" / "audit_mapping_centers.py"


def _rxn(center: list[int] | None = None, smiles: str | None = None) -> LabeledReaction:
    return LabeledReaction(
        reaction_id="mapped_sn2",
        reaction_smiles=smiles
        or "[CH3:1][Br:2].[OH-:3]>>[CH3:1][OH:3].[Br-:2]",
        context=ReactionContext.ionic,
        mechanism_type="sn2",
        split="train",
        group_roles=[
            LabeledGroupRole(
                group_id="mol0_halide_0",
                molecule_index=0,
                group_type=FunctionalGroupType.halide,
                atom_indices=[1],
                role=Role.leaving_group,
            )
        ],
        reaction_center_atoms=[1, 2, 3] if center is None else center,
        metadata={"template_name": "mapped_sn2_template"},
    )


def _write_dataset(path: Path, reactions: list[LabeledReaction]) -> None:
    path.write_text(
        json.dumps({"reactions": [rxn.to_dict() for rxn in reactions]}),
        encoding="utf-8",
    )


def test_has_atom_mapping_detects_mapped_reaction_smiles() -> None:
    assert has_atom_mapping("[CH3:1][Br:2]>>[CH3:1].[Br-:2]")
    assert not has_atom_mapping("CCBr>>CCO")


def test_extract_bond_changes_detects_bond_formation_and_breaking() -> None:
    changes = extract_bond_changes("[CH3:1][Br:2].[OH-:3]>>[CH3:1][OH:3].[Br-:2]")
    by_type = {change.change_type for change in changes}
    changed_pairs = {tuple(sorted((change.atom_map_1, change.atom_map_2))) for change in changes}

    assert "bond_broken" in by_type
    assert "bond_formed" in by_type
    assert (1, 2) in changed_pairs
    assert (1, 3) in changed_pairs


def test_infer_center_atoms_from_mapping_uses_changed_bond_atoms() -> None:
    suggestion = infer_center_atoms_from_mapping(_rxn())

    assert suggestion.mapped is True
    assert {1, 2, 3}.issubset(set(suggestion.suggested_center_atoms))
    assert suggestion.confidence == "high"


def test_audit_labeled_centers_against_mapping_detects_exact_match() -> None:
    report = audit_labeled_centers_against_mapping([_rxn(center=[1, 2, 3])])

    assert report.n_mapped_reactions == 1
    assert report.n_exact_matches == 1
    assert report.records[0].issue_code == "exact_match"


def test_audit_labeled_centers_against_mapping_detects_partial_overlap() -> None:
    report = audit_labeled_centers_against_mapping([_rxn(center=[1, 2])])

    assert report.records[0].issue_code == "partial_overlap"
    assert report.records[0].missing_from_label == [3]
    assert report.records[0].overlap_f1 is not None


def test_apply_mapping_center_suggestions_does_not_mutate_input() -> None:
    rxn = _rxn(center=[1, 2])

    updated, records = apply_mapping_center_suggestions([rxn])

    assert rxn.reaction_center_atoms == [1, 2]
    assert updated[0].reaction_center_atoms == [1, 2, 3]
    assert updated[0] is not rxn
    assert records


def test_unmapped_reactions_are_handled_gracefully() -> None:
    rxn = _rxn(center=[], smiles="CCBr>>CCO")
    suggestion = infer_center_atoms_from_mapping(rxn)
    report = audit_labeled_centers_against_mapping([rxn])

    assert suggestion.mapped is False
    assert suggestion.suggested_center_atoms == []
    assert report.n_unmapped_reactions == 1
    assert report.records[0].issue_code == "unmapped_reaction"


def test_mapping_center_report_serialization(tmp_path: Path) -> None:
    report = audit_labeled_centers_against_mapping([_rxn()])
    out = tmp_path / "mapping_report.json"

    save_mapping_center_audit_report(report, out)

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["n_mapped_reactions"] == 1
    assert payload["suggestions"]
    assert MappingCenterAuditReport(
        **{
            **report.to_dict(),
            "records": report.records,
            "suggestions": report.suggestions,
        }
    )


def test_cli_audit_mapping_centers_smoke_writes_report(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.json"
    report = tmp_path / "report.json"
    applied = tmp_path / "applied.json"
    _write_dataset(dataset, [_rxn(center=[1, 2])])

    result = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--data",
            str(dataset),
            "--output",
            str(report),
            "--apply-high-confidence",
            "--apply-suggestions-output",
            str(applied),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert report.exists()
    assert load_labeled_reactions(applied)[0].reaction_center_atoms == [1, 2, 3]


def test_no_mlip_training_invoked() -> None:
    text = _SCRIPT.read_text(encoding="utf-8").lower()
    for token in ("mace", "transition1x", "dft", "forces", "barrier"):
        assert token not in text
