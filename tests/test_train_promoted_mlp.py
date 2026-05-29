"""Tests for Phase 8.7 promoted-dataset MLP training script."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from mendel.labels import LabeledGroupRole, LabeledReaction
from mendel.types import FunctionalGroupType, ReactionContext, Role

torch = pytest.importorskip("torch")

_ROOT = Path(__file__).parent.parent
_SCRIPT = _ROOT / "scripts" / "train_promoted_mlp.py"
_MINIMAL = _ROOT / "data" / "reactions.minimal.json"


def _draft_dataset(path: Path) -> None:
    reaction = LabeledReaction(
        reaction_id="draft_sn2",
        reaction_smiles="CBr.[OH-]>>CO.[Br-]",
        context=ReactionContext.ionic,
        mechanism_type="sn2",
        split="draft",
        group_roles=[
            LabeledGroupRole(
                group_id="mol0_halide_0",
                molecule_index=0,
                group_type=FunctionalGroupType.halide,
                atom_indices=[1],
                role=Role.leaving_group,
                confidence="draft",
                notes="draft label",
            )
        ],
        reaction_center_atoms=[0, 1],
        metadata={"needs_manual_review": True},
    )
    path.write_text(
        json.dumps({"reactions": [reaction.to_dict()]}, indent=2),
        encoding="utf-8",
    )


def test_script_refuses_missing_dataset(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--data",
            str(tmp_path / "missing.json"),
            "--output",
            str(tmp_path / "model.pt"),
            "--report",
            str(tmp_path / "report.json"),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "does not exist" in result.stderr


def test_script_refuses_draft_labels_without_flag(tmp_path: Path) -> None:
    dataset = tmp_path / "draft.json"
    _draft_dataset(dataset)

    result = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--data",
            str(dataset),
            "--output",
            str(tmp_path / "model.pt"),
            "--report",
            str(tmp_path / "report.json"),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "draft labels" in result.stderr


def test_tiny_smoke_training_creates_checkpoint_and_report(tmp_path: Path) -> None:
    checkpoint = tmp_path / "role_mlp_promoted.pt"
    report = tmp_path / "training_report.json"
    protected = _ROOT / "models" / "role_mlp.pt"
    before = protected.read_bytes() if protected.exists() else None

    result = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--data",
            str(_MINIMAL),
            "--output",
            str(checkpoint),
            "--report",
            str(report),
            "--epochs",
            "2",
            "--hidden-dim",
            "8",
            "--batch-size",
            "4",
            "--device",
            "cpu",
        ],
        check=False,
        text=True,
        capture_output=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr
    assert checkpoint.exists()
    assert report.exists()
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["checkpoint_path"] == str(checkpoint)
    assert payload["training_history"]["metadata"]["epochs_run"] >= 1
    if before is not None:
        assert protected.read_bytes() == before
