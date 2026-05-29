"""Tests for Phase 8.10 atom-level reaction-center head."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from mendel.center_head import (
    AtomCenterExample,
    AtomCenterPrediction,
    aggregate_atom_predictions_to_reaction_centers,
    benchmark_atom_center_head,
    benchmark_atom_center_head_by_split,
    build_atom_center_examples,
    predict_atom_centers,
    save_atom_center_benchmark_report,
    summarize_atom_center_examples,
    train_atom_center_head,
)
from mendel.labels import LabeledGroupRole, LabeledReaction
from mendel.types import FunctionalGroupType, ReactionContext, Role

_ROOT = Path(__file__).parent.parent
_TRAIN_SCRIPT = _ROOT / "scripts" / "train_center_head.py"
_BENCH_SCRIPT = _ROOT / "scripts" / "benchmark_center_head.py"


def _sn2_reaction() -> LabeledReaction:
    return LabeledReaction(
        reaction_id="sn2_tiny",
        reaction_smiles="[CH3:1][Br:2].[OH-:3]>>[CH3:1][OH:3].[Br-:2]",
        context=ReactionContext.ionic,
        mechanism_type="sn2",
        split="train",
        group_roles=[
            LabeledGroupRole(
                group_id="mol0_halide_0",
                molecule_index=0,
                group_type=FunctionalGroupType.halide,
                atom_indices=[0, 1],
                role=Role.leaving_group,
            )
        ],
        reaction_center_atoms=[1, 2],
        metadata={},
    )


def _control_reaction() -> LabeledReaction:
    return LabeledReaction(
        reaction_id="control_tiny",
        reaction_smiles="CC>>CC",
        context=ReactionContext.unknown,
        mechanism_type="control",
        split="val",
        group_roles=[],
        reaction_center_atoms=[],
        metadata={},
    )


def _dataset(path: Path) -> None:
    payload = {"reactions": [_sn2_reaction().to_dict(), _control_reaction().to_dict()]}
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_atom_center_example_serializes() -> None:
    example = AtomCenterExample(
        reaction_id="r1",
        mechanism_type="sn2",
        split="train",
        atom_index=1,
        molecule_index=0,
        atomic_number=35,
        atom_symbol="Br",
        group_ids=["mol0_halide_0"],
        group_types=["halide"],
        group_roles=["leaving_group"],
        role_confidences=[0.9],
        is_labeled_center=True,
        features=[35.0, 1.0],
        metadata={"atom_map_num": 2},
    )

    payload = example.to_dict()

    assert payload["reaction_id"] == "r1"
    assert payload["group_roles"] == ["leaving_group"]
    assert payload["is_labeled_center"] is True


def test_atom_center_prediction_serializes() -> None:
    prediction = AtomCenterPrediction(
        reaction_id="r1",
        atom_index=1,
        molecule_index=0,
        probability=0.8,
        predicted_center=True,
        threshold=0.5,
        contributing_group_ids=["mol0_halide_0"],
        contributing_roles=["leaving_group"],
        metadata={"atom_map_num": 2},
    )

    payload = prediction.to_dict()

    assert payload["probability"] == 0.8
    assert payload["predicted_center"] is True


def test_build_atom_center_examples_from_labeled_reaction() -> None:
    examples = build_atom_center_examples([_sn2_reaction()])

    assert examples
    assert {ex.atom_index for ex in examples if ex.is_labeled_center} == {0, 1}
    bromine = next(ex for ex in examples if ex.atom_symbol == "Br")
    assert bromine.group_ids == ["mol0_halide_0"]
    assert bromine.group_roles == ["leaving_group"]
    assert bromine.features


def test_control_empty_reaction_center_is_all_negative() -> None:
    examples = build_atom_center_examples([_control_reaction()])

    assert examples
    assert not any(ex.is_labeled_center for ex in examples)


def test_summarize_atom_center_examples_counts_classes() -> None:
    examples = build_atom_center_examples([_sn2_reaction(), _control_reaction()])
    summary = summarize_atom_center_examples(examples)

    assert summary["n_examples"] == len(examples)
    assert summary["n_positive"] == 2
    assert summary["n_negative"] == len(examples) - 2
    assert summary["split_distribution"]["train"] > 0


def test_aggregate_atom_predictions_to_reaction_centers_uses_atom_map_when_present() -> None:
    predictions = [
        AtomCenterPrediction("r1", 0, 0, 0.9, True, 0.5, [], [], {"atom_map_num": 1}),
        AtomCenterPrediction("r1", 1, 0, 0.1, False, 0.5, [], [], {"atom_map_num": 2}),
        AtomCenterPrediction("r1", 2, 1, 0.8, True, 0.5, [], [], {}),
    ]

    centers = aggregate_atom_predictions_to_reaction_centers(predictions)

    assert centers == {"r1": [1, 2]}


def test_benchmark_atom_center_head_metrics() -> None:
    predictions = [
        AtomCenterPrediction("sn2_tiny", 0, 0, 0.9, True, 0.5, [], [], {"atom_map_num": 1}),
        AtomCenterPrediction("sn2_tiny", 1, 0, 0.8, True, 0.5, [], [], {"atom_map_num": 2}),
        AtomCenterPrediction("sn2_tiny", 0, 1, 0.1, False, 0.5, [], [], {"atom_map_num": 3}),
    ]

    report = benchmark_atom_center_head([_sn2_reaction()], predictions)

    assert report.atom_precision == 1.0
    assert report.atom_recall == 1.0
    assert report.atom_f1 == 1.0
    assert report.reaction_center_f1 == 1.0
    assert report.per_mechanism_f1["sn2"] == 1.0


def test_benchmark_atom_center_head_by_split_returns_split_metrics() -> None:
    train_rxn = _sn2_reaction()
    val_rxn = _control_reaction()
    predictions = [
        AtomCenterPrediction("sn2_tiny", 0, 0, 0.9, True, 0.5, [], [], {"atom_map_num": 1}),
        AtomCenterPrediction("sn2_tiny", 1, 0, 0.8, True, 0.5, [], [], {"atom_map_num": 2}),
        AtomCenterPrediction("control_tiny", 0, 0, 0.1, False, 0.5, [], [], {"atom_map_num": 0}),
    ]

    reports = benchmark_atom_center_head_by_split([train_rxn, val_rxn], predictions)

    assert set(reports) == {"overall", "train", "val", "test"}
    assert reports["overall"]["reaction_center_f1"] == 1.0
    assert reports["train"]["reaction_center_f1"] == 1.0
    assert reports["val"]["n_reactions"] == 1


def test_save_atom_center_benchmark_report_writes_json(tmp_path: Path) -> None:
    report = benchmark_atom_center_head([_sn2_reaction()], [])
    out = tmp_path / "benchmark.json"

    save_atom_center_benchmark_report(report, out)

    assert json.loads(out.read_text(encoding="utf-8"))["predictor_name"] == "atom_center_head"


def test_train_smoke_if_torch_installed(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    examples = build_atom_center_examples([_sn2_reaction(), _control_reaction()])
    checkpoint = tmp_path / "atom_center_head.pt"
    report_path = tmp_path / "training.json"

    report = train_atom_center_head(
        examples,
        checkpoint,
        report_path,
        hidden_dim=8,
        epochs=2,
        batch_size=4,
        learning_rate=1e-2,
        seed=0,
        device="cpu",
    )

    assert checkpoint.exists()
    assert report_path.exists()
    assert report.n_examples == len(examples)


def test_predict_smoke_if_torch_installed(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    examples = build_atom_center_examples([_sn2_reaction(), _control_reaction()])
    checkpoint = tmp_path / "atom_center_head.pt"
    report_path = tmp_path / "training.json"
    train_atom_center_head(
        examples,
        checkpoint,
        report_path,
        hidden_dim=8,
        epochs=2,
        batch_size=4,
        seed=0,
        device="cpu",
    )

    predictions = predict_atom_centers(examples, checkpoint, threshold=0.5, device="cpu")

    assert len(predictions) == len(examples)
    assert all(0.0 <= pred.probability <= 1.0 for pred in predictions)


def test_cli_help_runs() -> None:
    for script in (_TRAIN_SCRIPT, _BENCH_SCRIPT):
        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "MLIP" in result.stdout or "reaction-center" in result.stdout


def test_train_cli_refuses_dataset_without_positive_centers(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.json"
    payload = {"reactions": [_control_reaction().to_dict()]}
    dataset.write_text(json.dumps(payload), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(_TRAIN_SCRIPT),
            "--data",
            str(dataset),
            "--output",
            str(tmp_path / "head.pt"),
            "--report",
            str(tmp_path / "report.json"),
            "--epochs",
            "1",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "no positive" in result.stderr.lower()


def test_no_mlip_training_invoked() -> None:
    train_text = _TRAIN_SCRIPT.read_text(encoding="utf-8")
    bench_text = _BENCH_SCRIPT.read_text(encoding="utf-8")

    forbidden = ("mace", "transition1x", "dft", "forces", "barrier")
    for token in forbidden:
        assert token not in train_text.lower()
        assert token not in bench_text.lower()
