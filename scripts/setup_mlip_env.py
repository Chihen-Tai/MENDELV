"""Cautious optional MLIP environment setup helper."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Print or run optional MLIP dependency installation commands.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--install", action="store_true")
    installer = parser.add_mutually_exclusive_group()
    installer.add_argument("--uv", action="store_true", help="Use uv pip install.")
    installer.add_argument("--pip", action="store_true", help="Use python -m pip install.")
    return parser


def _install_command(use_uv: bool) -> list[str]:
    if use_uv:
        return ["uv", "pip", "install", "-e", ".[mlip]"]
    return [sys.executable, "-m", "pip", "install", "-e", ".[mlip]"]


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    print("Recommended MLIP extra install commands:")
    print('  pip install -e ".[mlip]"')
    print('  uv pip install -e ".[mlip]"')
    print("No QO2Mol data will be installed.")
    print("No MLIP training will be run.")

    if not args.install:
        print("No installation performed. Pass --install with --pip or --uv to install.")
        return 0

    command = _install_command(use_uv=args.uv)
    if not args.uv and not args.pip:
        print("No installer selected; defaulting to python -m pip.")
    print("Running install command:")
    print("  " + " ".join(command))
    install_result = subprocess.run(command, cwd=_ROOT, check=False)
    if install_result.returncode != 0:
        return install_result.returncode
    check_command = [sys.executable, str(_ROOT / "scripts" / "check_mlip_env.py")]
    return subprocess.run(check_command, cwd=_ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
