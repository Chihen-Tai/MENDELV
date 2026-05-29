"""Tests for Phase 8.7 promoted-dataset benchmark script."""

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


def test_benchmark_handles_missing_old_checkpoint_gracefully(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--data",
            str(_MINIMAL),
            "--new-mlp-checkpoint",
            str(tmp_path / "missing_new.pt"),
            "--old-mlp-checkpoint",
            str(tmp_path / "missing_old.pt"),
            "--output-dir",
            str(tmp_path),
            "--device",
            "cpu",
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    comparison = json.loads((tmp_path / "comparison.json").read_text(encoding="utf-8"))
    assert "rule_based_local" in comparison["predictor_names"]
    assert "rule_based_negotiated" in comparison["predictor_names"]
    assert "new_mlp_local" not in comparison["predictor_names"]
    assert "old_mlp_local" not in comparison["predictor_names"]


def test_rule_reports_and_comparison_are_written(tmp_path: Path) -> None:
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
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "rule_based_local.json").exists()
    assert (tmp_path / "rule_based_negotiated.json").exists()
    assert (tmp_path / "comparison.json").exists()


def test_benchmark_runs_with_new_checkpoint_and_preserves_custom_names(tmp_path: Path) -> None:
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
        ],
        check=False,
        text=True,
        capture_output=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr
    comparison = json.loads((tmp_path / "comparison.json").read_text(encoding="utf-8"))
    assert "new_mlp_local" in comparison["predictor_names"]
    assert "new_mlp_negotiated" in comparison["predictor_names"]
    assert (tmp_path / "new_mlp_local.json").exists()
    assert (tmp_path / "new_mlp_negotiated.json").exists()
