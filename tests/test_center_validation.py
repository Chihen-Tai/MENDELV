"""Tests for Phase 8.11 leakage-resistant center validation."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from mendel.center_validation import (
    CenterLabelIssue,
    assign_leakage_resistant_splits,
    audit_center_labels,
    build_leakage_validation_report,
    infer_template_key,
    save_labeled_reactions_json,
    save_leakage_validation_report,
    summarize_center_label_issues,
)
from mendel.labels import LabeledGroupRole, LabeledReaction, load_labeled_reactions
from mendel.types import FunctionalGroupType, ReactionContext, Role

_ROOT = Path(__file__).parent.parent
_SCRIPT = _ROOT / "scripts" / "validate_center_labels.py"
_STRICT_TRAIN_SCRIPT = _ROOT / "scripts" / "retrain_center_head_strict_split.py"
_STRICT_BENCH_SCRIPT = _ROOT / "scripts" / "benchmark_center_head_strict_split.py"


def _rxn(
    reaction_id: str,
    mechanism: str = "sn2",
    center: list[int] | None = None,
    split: str = "train",
    template: str = "alkyl_halide_sn2",
) -> LabeledReaction:
    return LabeledReaction(
        reaction_id=reaction_id,
        reaction_smiles="[CH3:1][Br:2].[OH-:3]>>[CH3:1][OH:3].[Br-:2]",
        context=ReactionContext.ionic,
        mechanism_type=mechanism,
        split=split,
        group_roles=[
            LabeledGroupRole(
                group_id="mol0_halide_0",
                molecule_index=0,
                group_type=FunctionalGroupType.halide,
                atom_indices=[0, 1],
                role=Role.leaving_group,
            )
        ],
        reaction_center_atoms=[1, 2] if center is None else center,
        metadata={"template_name": template, "generation_method": "unit_test"},
    )


def _control(center: list[int] | None = None) -> LabeledReaction:
    return LabeledReaction(
        reaction_id="control_tiny",
        reaction_smiles="[CH3:1][CH3:2]>>[CH3:1][CH3:2]",
        context=ReactionContext.unknown,
        mechanism_type="control",
        split="val",
        group_roles=[],
        reaction_center_atoms=[] if center is None else center,
        metadata={"template_name": "control_template"},
    )


def _dataset(path: Path, reactions: list[LabeledReaction]) -> None:
    path.write_text(
        json.dumps({"reactions": [rxn.to_dict() for rxn in reactions]}),
        encoding="utf-8",
    )


def test_center_label_issue_serializes() -> None:
    issue = CenterLabelIssue(
        reaction_id="r1",
        mechanism_type="sn2",
        issue_code="invalid_center_atom",
        severity="error",
        message="bad atom",
        reaction_center_atoms=[99],
        metadata={"max_atom_value": 2},
    )

    assert issue.to_dict()["issue_code"] == "invalid_center_atom"
    assert issue.to_dict()["metadata"]["max_atom_value"] == 2


def test_infer_template_key_is_deterministic() -> None:
    rxn = _rxn("sn2_template_001")

    assert infer_template_key(rxn) == infer_template_key(rxn)
    assert "alkyl_halide_sn2" in infer_template_key(rxn)


def test_assign_leakage_resistant_splits_keeps_template_together() -> None:
    reactions = [
        _rxn("a_1", template="same"),
        _rxn("a_2", template="same", split="test"),
        _rxn("b_1", template="other"),
    ]

    split_reactions, records = assign_leakage_resistant_splits(reactions, strategy="template")

    same_splits = {rxn.split for rxn in split_reactions if rxn.metadata["leakage_group"] == "same"}
    assert len(same_splits) == 1
    assert {record.leakage_group for record in records} == {"same", "other"}


def test_assign_leakage_resistant_splits_does_not_mutate_input() -> None:
    reactions = [_rxn("a_1", split="test")]

    split_reactions, _ = assign_leakage_resistant_splits(reactions, strategy="template")

    assert reactions[0].split == "test"
    assert reactions[0].metadata.get("leakage_group") is None
    assert split_reactions[0] is not reactions[0]


def test_audit_center_labels_catches_invalid_atom_indices() -> None:
    issues = audit_center_labels([_rxn("bad", center=[999])])

    assert any(issue.issue_code == "invalid_center_atom" for issue in issues)


def test_audit_center_labels_catches_control_with_non_empty_center() -> None:
    issues = audit_center_labels([_control(center=[1])])

    assert any(issue.issue_code == "control_has_center_atoms" for issue in issues)


def test_audit_center_labels_catches_reactive_with_empty_center() -> None:
    issues = audit_center_labels([_rxn("missing", center=[])])

    assert any(issue.issue_code == "reactive_empty_center" for issue in issues)


def test_summarize_center_label_issues_counts() -> None:
    issues = audit_center_labels([
        _rxn("bad", center=[999]),
        _control(center=[1]),
    ])

    summary = summarize_center_label_issues(issues)

    assert summary["by_severity"]["error"] >= 1
    assert summary["by_issue_code"]["invalid_center_atom"] == 1


def test_save_labeled_reactions_json_writes_valid_dataset(tmp_path: Path) -> None:
    out = tmp_path / "dataset.json"

    save_labeled_reactions_json([_rxn("a")], out)

    assert load_labeled_reactions(out)[0].reaction_id == "a"


def test_build_and_save_leakage_validation_report(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.json"
    out_dataset = tmp_path / "template_split.json"
    report_path = tmp_path / "report.json"
    _dataset(dataset, [_rxn("a"), _rxn("b", template="other"), _control()])

    report = build_leakage_validation_report(dataset, "template", out_dataset)
    save_leakage_validation_report(report, report_path)

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["split_strategy"] == "template"
    assert payload["n_reactions"] == 3
    assert out_dataset.exists()


def test_cli_validate_center_labels_smoke_writes_report(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.json"
    out_dataset = tmp_path / "template_split.json"
    report = tmp_path / "report.json"
    _dataset(dataset, [_rxn("a"), _rxn("b", template="other"), _control()])

    result = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--data",
            str(dataset),
            "--output-data",
            str(out_dataset),
            "--report",
            str(report),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert report.exists()
    assert "split distribution" in result.stdout.lower()


def test_strict_cli_help_runs() -> None:
    for script in (_STRICT_TRAIN_SCRIPT, _STRICT_BENCH_SCRIPT):
        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "MLIP" in result.stdout or "strict" in result.stdout.lower()


def test_no_mlip_training_invoked() -> None:
    for script in (_SCRIPT, _STRICT_TRAIN_SCRIPT, _STRICT_BENCH_SCRIPT):
        text = script.read_text(encoding="utf-8").lower()
        for token in ("mace", "transition1x", "dft", "forces", "barrier"):
            assert token not in text
