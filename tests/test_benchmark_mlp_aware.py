"""Tests for Phase 8.9 MLP-aware benchmark mode."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).parent.parent
_SCRIPT = _ROOT / "scripts" / "benchmark_promoted_mlp.py"
_MINIMAL = _ROOT / "data" / "reactions.minimal.json"
_MLP_MINIMAL = _ROOT / "models" / "role_mlp_minimal.pt"


def test_mlp_aware_benchmark_includes_mode_when_checkpoint_exists(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    if not _MLP_MINIMAL.exists():
        pytest.skip("models/role_mlp_minimal.pt is not available")

    result = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--data",
            str(_MINIMAL),
            "--new-mlp-checkpoint",
            str(_MLP_MINIMAL),
            "--old-mlp-checkpoint",
            str(tmp_path / "missing_old.pt"),
            "--output-dir",
            str(tmp_path),
            "--device",
            "cpu",
            "--include-mlp-aware-negotiation",
        ],
        check=False,
        text=True,
        capture_output=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr
    comparison = json.loads((tmp_path / "comparison.json").read_text(encoding="utf-8"))
    assert "new_mlp_aware_negotiated" in comparison["predictor_names"]
    assert "new_mlp_aware_negotiated" in comparison["reaction_center"]
    assert (tmp_path / "new_mlp_aware_negotiated.json").exists()


def test_mlp_aware_benchmark_gracefully_skips_missing_checkpoint(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--data",
            str(_MINIMAL),
            "--new-mlp-checkpoint",
            str(tmp_path / "missing_new.pt"),
            "--skip-old-mlp",
            "--output-dir",
            str(tmp_path),
            "--device",
            "cpu",
            "--include-mlp-aware-negotiation",
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    comparison = json.loads((tmp_path / "comparison.json").read_text(encoding="utf-8"))
    assert "new_mlp_aware_negotiated" not in comparison["predictor_names"]
    assert "rule_based_negotiated" in comparison["predictor_names"]


def test_mlp_aware_benchmark_does_not_invoke_training() -> None:
    text = _SCRIPT.read_text(encoding="utf-8")
    assert "train_mlp_role_predictor" not in text
    assert "train_promoted_mlp" not in text
