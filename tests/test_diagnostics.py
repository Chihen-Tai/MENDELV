"""Tests for Phase 8.8 MLP diagnostics."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from mendel.diagnostics import (
    MLPCalibrationBin,
    build_diagnostics_report,
    collect_reaction_center_failures,
    compute_mlp_calibration,
    save_diagnostics_report,
    summarize_failure_patterns,
)

_ROOT = Path(__file__).parent.parent
_SCRIPT = _ROOT / "scripts" / "diagnose_mlp.py"


def _group(
    reaction_id: str,
    true_role: str,
    predicted_role: str,
    confidence: float | None,
    correct: bool,
) -> dict[str, object]:
    return {
        "reaction_id": reaction_id,
        "group_id": f"{reaction_id}_g",
        "group_type": "halide",
        "true_role": true_role,
        "predicted_role": predicted_role,
        "predicted_confidence": confidence,
        "predictor_name": "fake",
        "correct": correct,
        "split": "train",
        "mechanism_type": "sn2",
        "metadata": {},
    }


def _reaction_record(
    reaction_id: str,
    f1: float | None,
    accuracy: float,
    n_correct: int,
) -> dict[str, object]:
    return {
        "reaction_id": reaction_id,
        "reaction_smiles": "CBr.[OH-]>>CO.[Br-]",
        "split": "train",
        "mechanism_type": "sn2",
        "predictor_name": "fake",
        "n_labeled_groups": 1,
        "n_correct_roles": n_correct,
        "role_accuracy": accuracy,
        "mechanism_hint": "sn2_or_e2_like",
        "reaction_center_precision": None if f1 is None else f1,
        "reaction_center_recall": None if f1 is None else f1,
        "reaction_center_f1": f1,
        "warnings": [],
        "metadata": {
            "true_reaction_center_atoms": [1, 2],
            "predicted_reaction_center_atoms": [] if f1 == 0 else [1],
        },
    }


def _report(path: Path, predictor_name: str, f1: float, correct: bool) -> None:
    payload = {
        "predictor_name": predictor_name,
        "n_reactions": 1,
        "n_group_labels": 1,
        "overall_role_accuracy": 1.0 if correct else 0.0,
        "role_accuracy_by_role": {"leaving_group": 1.0 if correct else 0.0},
        "role_accuracy_by_group_type": {"halide": 1.0 if correct else 0.0},
        "role_accuracy_by_mechanism": {"sn2": 1.0 if correct else 0.0},
        "split_accuracy": {"train": 1.0 if correct else 0.0},
        "confusion_matrix": {},
        "reaction_center_precision": f1,
        "reaction_center_recall": f1,
        "reaction_center_f1": f1,
        "group_records": [
            _group(
                "rxn1",
                "leaving_group",
                "leaving_group" if correct else "spectator",
                0.9,
                correct,
            )
        ],
        "reaction_records": [_reaction_record("rxn1", f1, 1.0 if correct else 0.0, int(correct))],
        "metadata": {},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _dataset(path: Path) -> None:
    path.write_text(
        json.dumps({
            "reactions": [{
                "reaction_id": "rxn1",
                "reaction_smiles": "CBr.[OH-]>>CO.[Br-]",
                "context": "ionic",
                "mechanism_type": "sn2",
                "split": "train",
                "group_roles": [{
                    "group_id": "rxn1_g",
                    "molecule_index": 0,
                    "group_type": "halide",
                    "atom_indices": [1],
                    "role": "leaving_group",
                    "confidence": "manual",
                    "notes": "test",
                }],
                "reaction_center_atoms": [1, 2],
                "metadata": {},
            }]
        }),
        encoding="utf-8",
    )


def test_mlp_calibration_bin_serializes() -> None:
    payload = MLPCalibrationBin(0.0, 0.1, 1, 1.0, 0.8, 0.02).to_dict()

    assert payload["bin_start"] == 0.0
    assert payload["accuracy"] == 1.0


def test_compute_mlp_calibration_with_fake_records() -> None:
    bins = compute_mlp_calibration([
        _group("a", "spectator", "spectator", 0.8, True),
        _group("b", "spectator", "leaving_group", 0.8, False),
    ], n_bins=2)

    populated = [bin_ for bin_ in bins if bin_.n]
    assert populated
    assert populated[0].accuracy == 0.5
    assert populated[0].mean_confidence == 0.8


def test_collect_reaction_center_failures_classifies_cases() -> None:
    report = {
        "reaction_records": [
            _reaction_record("empty", 0.0, 1.0, 1),
            _reaction_record("partial", 0.5, 0.0, 0),
            _reaction_record("role_correct_center_wrong", 0.5, 1.0, 1),
        ]
    }

    failures = collect_reaction_center_failures(report, "fake", f1_threshold=0.8)
    types = {failure.failure_type for failure in failures}

    assert "empty_prediction" in types
    assert "partial_overlap" in types
    assert "role_correct_center_wrong" in types


def test_summarize_failure_patterns_non_empty() -> None:
    failures = collect_reaction_center_failures(
        {"reaction_records": [_reaction_record("partial", 0.5, 0.0, 0)]},
        "fake",
    )

    patterns = summarize_failure_patterns(failures, [])

    assert patterns


def test_build_and_save_diagnostics_report_from_fake_json(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.json"
    rule = tmp_path / "rule.json"
    new = tmp_path / "new.json"
    old = tmp_path / "old.json"
    out = tmp_path / "diagnostics.json"
    _dataset(dataset)
    _report(rule, "rule_based_negotiated", 1.0, True)
    _report(new, "new_mlp_negotiated", 0.0, False)
    _report(old, "old_mlp_negotiated", 0.0, False)

    report = build_diagnostics_report(dataset, rule, new, old)
    save_diagnostics_report(report, out)

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["n_reactions"] == 1
    assert payload["reaction_center_failures"]
    assert payload["recommendations"]


def test_cli_smoke_creates_output_from_fake_reports(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.json"
    rule = tmp_path / "rule.json"
    new = tmp_path / "new.json"
    old = tmp_path / "old.json"
    out = tmp_path / "diagnostics.json"
    _dataset(dataset)
    _report(rule, "rule_based_negotiated", 1.0, True)
    _report(new, "new_mlp_negotiated", 0.0, False)
    _report(old, "old_mlp_negotiated", 0.0, False)

    result = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--data",
            str(dataset),
            "--rule-report",
            str(rule),
            "--new-mlp-report",
            str(new),
            "--old-mlp-report",
            str(old),
            "--output",
            str(out),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert out.exists()


def test_no_mlp_training_invoked() -> None:
    text = (_ROOT / "mendel" / "diagnostics.py").read_text(encoding="utf-8")
    if _SCRIPT.exists():
        text += _SCRIPT.read_text(encoding="utf-8")
    assert "train_mlp_role_predictor" not in text
    assert "train_promoted_mlp" not in text
