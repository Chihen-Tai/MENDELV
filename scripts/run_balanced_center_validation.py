"""Run Phase 8.14 balanced center validation without MLIP training.

Default checkpoint path: models/atom_center_head_center_balanced.pt
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run mapping audit, cleanup, balanced split, center-head train/benchmark.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=_ROOT / "data" / "reactions.center_expanded.cleaned.json",
    )
    parser.add_argument("--use-mapping-suggestions", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda", "mps", "auto"), default="cpu")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output-prefix", default="center_balanced")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _paths(prefix: str) -> dict[str, Path]:
    return {
        "mapping_data": _ROOT / "data" / f"reactions.{prefix}.mapping_audited.json",
        "cleaned_data": _ROOT / "data" / f"reactions.{prefix}.cleaned.json",
        "split_data": _ROOT / "data" / f"reactions.{prefix}.split.json",
        "mapping_report": _ROOT / "reports" / "mapping_center_audit_report.json",
        "cleanup_report": _ROOT / "reports" / f"{prefix}_cleanup_report.json",
        "validation_report": _ROOT / "reports" / f"{prefix}_validation_report.json",
        "checkpoint": _ROOT / "models" / f"atom_center_head_{prefix}.pt",
        "training_report": _ROOT / "reports" / f"atom_center_training_{prefix}_report.json",
        "benchmark_report": _ROOT / "reports" / f"atom_center_benchmark_{prefix}_report.json",
        "comparison_report": _ROOT / "reports" / f"center_head_{prefix}_comparison.json",
    }


def _run(command: list[str], dry_run: bool) -> None:
    printable = " ".join(command)
    print(printable)
    if dry_run:
        return
    subprocess.run(command, cwd=_ROOT, check=True)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if not args.data.exists():
        print(f"ERROR: dataset does not exist: {args.data}", file=sys.stderr)
        return 1
    paths = _paths(args.output_prefix)
    python = sys.executable
    current_data = args.data

    if args.use_mapping_suggestions:
        _run(
            [
                python,
                str(_ROOT / "scripts" / "audit_mapping_centers.py"),
                "--data",
                str(args.data),
                "--output",
                str(paths["mapping_report"]),
                "--apply-high-confidence",
                "--apply-suggestions-output",
                str(paths["mapping_data"]),
            ],
            args.dry_run,
        )
        current_data = paths["mapping_data"]

    _run(
        [
            python,
            str(_ROOT / "scripts" / "cleanup_center_labels.py"),
            "--input",
            str(current_data),
            "--output",
            str(paths["cleaned_data"]),
            "--report",
            str(paths["cleanup_report"]),
            "--conservative",
        ],
        args.dry_run,
    )
    _run(
        [
            python,
            str(_ROOT / "scripts" / "validate_center_labels.py"),
            "--data",
            str(paths["cleaned_data"]),
            "--strategy",
            "val_test_balanced_template",
            "--output-data",
            str(paths["split_data"]),
            "--report",
            str(paths["validation_report"]),
        ],
        args.dry_run,
    )
    _run(
        [
            python,
            str(_ROOT / "scripts" / "retrain_center_head_strict_split.py"),
            "--data",
            str(paths["split_data"]),
            "--role-checkpoint",
            str(_ROOT / "models" / "role_mlp_promoted.pt"),
            "--output",
            str(paths["checkpoint"]),
            "--report",
            str(paths["training_report"]),
            "--epochs",
            str(args.epochs),
            "--threshold",
            str(args.threshold),
            "--device",
            args.device,
            "--use-class-weights",
        ],
        args.dry_run,
    )
    _run(
        [
            python,
            str(_ROOT / "scripts" / "benchmark_center_head_strict_split.py"),
            "--data",
            str(paths["split_data"]),
            "--center-checkpoint",
            str(paths["checkpoint"]),
            "--role-checkpoint",
            str(_ROOT / "models" / "role_mlp_promoted.pt"),
            "--threshold",
            str(args.threshold),
            "--device",
            args.device,
            "--output",
            str(paths["benchmark_report"]),
            "--comparison-output",
            str(paths["comparison_report"]),
        ],
        args.dry_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
