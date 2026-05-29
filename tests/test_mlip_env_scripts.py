"""Tests for safe optional MLIP environment helpers."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
_CHECK = _ROOT / "scripts" / "check_mlip_env.py"
_SETUP = _ROOT / "scripts" / "setup_mlip_env.py"


def test_check_mlip_env_runs_without_ase_mace(tmp_path: Path) -> None:
    report = tmp_path / "mlip_env_report.json"

    result = subprocess.run(
        [sys.executable, str(_CHECK), "--output", str(report)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "Install MLIP extras with: pip install -e '.[mlip]'" in result.stdout
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["mendel_import"]["ok"] is True
    assert "ase" in payload
    assert "mace_torch" in payload


def test_setup_mlip_env_without_install_does_not_install() -> None:
    result = subprocess.run(
        [sys.executable, str(_SETUP)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "No installation performed" in result.stdout
    assert "pip install -e \".[mlip]\"" in result.stdout
    assert "uv pip install -e \".[mlip]\"" in result.stdout


def test_setup_mlip_env_help_works() -> None:
    result = subprocess.run(
        [sys.executable, str(_SETUP), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--install" in result.stdout


def test_no_network_or_training_invoked() -> None:
    combined = (
        _CHECK.read_text(encoding="utf-8").lower()
        + _SETUP.read_text(encoding="utf-8").lower()
    )
    for token in ("requests.get", "urlretrieve", "train_mlp", "fit(", "neb", "irc", "md"):
        assert token not in combined
