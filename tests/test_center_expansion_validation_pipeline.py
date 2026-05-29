"""Tests for Phase 8.13 center expansion validation pipeline."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from scripts.run_center_expansion_validation import build_pipeline_commands

_ROOT = Path(__file__).parent.parent
_SCRIPT = _ROOT / "scripts" / "run_center_expansion_validation.py"


def test_pipeline_commands_are_deterministic() -> None:
    first = build_pipeline_commands(max_promote=2, device="cpu", epochs=3, threshold=0.5)
    second = build_pipeline_commands(max_promote=2, device="cpu", epochs=3, threshold=0.5)

    assert first == second


def test_pipeline_does_not_overwrite_default_atom_checkpoint() -> None:
    commands = build_pipeline_commands(max_promote=2, device="cpu", epochs=3, threshold=0.5)
    flat = " ".join(" ".join(command) for command in commands)

    assert "models/atom_center_head.pt" not in flat
    assert "models/atom_center_head_center_expanded_strict.pt" in flat


def test_end_to_end_script_tiny_dry_run() -> None:
    result = subprocess.run(
        [sys.executable, str(_SCRIPT), "--max-promote", "2", "--dry-run"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "promote_center_expansion.py" in result.stdout


def test_no_mlip_training_invoked() -> None:
    text = _SCRIPT.read_text(encoding="utf-8").lower()
    for token in ("mace", "transition1x", "dft", "forces", "barrier"):
        assert token not in text
