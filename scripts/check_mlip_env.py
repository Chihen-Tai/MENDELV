"""Check optional MLIP environment readiness without requiring MLIP extras."""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _module_status(module_name: str) -> dict[str, object]:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        return {"installed": False, "version": None, "error": str(exc)}
    return {
        "installed": True,
        "version": getattr(module, "__version__", None),
        "error": None,
    }


def _torch_status() -> dict[str, object]:
    status = _module_status("torch")
    status["cuda_available"] = False
    status["mps_available"] = False
    if not status["installed"]:
        return status
    torch = importlib.import_module("torch")
    status["cuda_available"] = bool(torch.cuda.is_available())
    mps = getattr(torch.backends, "mps", None)
    status["mps_available"] = bool(mps is not None and mps.is_available())
    return status


def build_report() -> dict[str, Any]:
    report: dict[str, Any] = {
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "torch": _torch_status(),
        "ase": _module_status("ase"),
        "mace_torch": _module_status("mace"),
        "install_hint": "Install MLIP extras with: pip install -e '.[mlip]'",
    }
    try:
        import mendel

        report["mendel_import"] = {
            "ok": True,
            "version": getattr(mendel, "__version__", None),
            "error": None,
        }
    except Exception as exc:
        report["mendel_import"] = {"ok": False, "version": None, "error": str(exc)}
    try:
        import mendel.mlip as mlip

        report["mendel_mlip_import"] = {"ok": True, "error": None}
        report["resolve_device_auto"] = {
            "ok": True,
            "device": mlip.resolve_device("auto"),
            "error": None,
        }
    except Exception as exc:
        report["mendel_mlip_import"] = {"ok": False, "error": str(exc)}
        report["resolve_device_auto"] = {"ok": False, "device": None, "error": str(exc)}
    return report


def _print_summary(report: dict[str, Any]) -> None:
    print("MENDELV optional MLIP environment check")
    print(f"Python: {report['python']['version'].split()[0]}")
    print(f"Executable: {report['python']['executable']}")
    print(f"mendel import: {report['mendel_import']['ok']}")
    print(f"mendel.mlip import: {report['mendel_mlip_import']['ok']}")
    torch = report["torch"]
    print(f"torch installed: {torch['installed']} version: {torch['version']}")
    print(f"cuda available: {torch['cuda_available']}")
    print(f"mps available: {torch['mps_available']}")
    print(f"ase installed: {report['ase']['installed']} version: {report['ase']['version']}")
    print(
        "mace-torch installed: "
        f"{report['mace_torch']['installed']} version: {report['mace_torch']['version']}"
    )
    print(f"resolve_device('auto'): {report['resolve_device_auto']['device']}")
    print(report["install_hint"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check optional MLIP dependencies without making them mandatory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = build_report()
    _print_summary(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Report: {args.output}")
    return 0 if report["mendel_import"]["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
