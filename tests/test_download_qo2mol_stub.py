"""Tests for safe QO2Mol downloader stub."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
_SCRIPT = _ROOT / "scripts" / "download_qo2mol_stub.py"


def test_download_stub_help_runs() -> None:
    result = subprocess.run(
        [sys.executable, str(_SCRIPT), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0


def test_download_stub_without_url_does_not_download() -> None:
    result = subprocess.run(
        [sys.executable, str(_SCRIPT)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "https://github.com/saiscn/QO2Mol/" in result.stdout
    assert "Do not commit raw QO2Mol data" in result.stdout
    assert "Verify dataset license" in result.stdout


def test_download_stub_has_no_network_dependency() -> None:
    text = _SCRIPT.read_text(encoding="utf-8")

    assert "requests.get" not in text
    assert "urlretrieve" not in text
