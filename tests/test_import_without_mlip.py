"""Import safety tests for optional Phase 9 MLIP support."""

from __future__ import annotations

import subprocess
import sys


def test_import_mendel_without_mlip_dependencies() -> None:
    result = subprocess.run(
        [sys.executable, "-c", "import mendel; print(mendel.__version__)"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0


def test_import_mendel_mlip_without_optional_dependencies() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import mendel.mlip as m; print(m.MLIPConfig().backend_name)",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "mace" in result.stdout


def test_optional_functions_raise_only_when_called() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from mendel.mlip import optional_import_ase; "
            "\ntry:\n optional_import_ase()\nexcept ImportError as e:\n print(str(e))",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "ASE is required" in result.stdout or result.stdout == ""
