"""Run Phase 8.13 center expansion validation workflow."""

from __future__ import annotations

import argparse
import subprocess
import sys


def build_pipeline_commands(
    max_promote: int | None = None,
    device: str = "cpu",
    epochs: int = 80,
    threshold: float = 0.5,
) -> list[list[str]]:
    py = sys.executable
    promote = [
        py,
        "scripts/promote_center_expansion.py",
        "--output",
        "data/reactions.center_expansion.promoted.json",
        "--merged-output",
        "data/reactions.center_expanded.cleaned_input.json",
        "--report",
        "reports/center_expansion_promotion_report.json",
    ]
    if max_promote is not None:
        promote.extend(["--max-promote", str(max_promote)])
    return [
        promote,
        [
            py,
            "scripts/cleanup_center_labels.py",
            "--input",
            "data/reactions.center_expanded.cleaned_input.json",
            "--output",
            "data/reactions.center_expanded.cleaned.json",
            "--report",
            "reports/center_expanded_cleanup_report.json",
        ],
        [
            py,
            "scripts/validate_center_labels.py",
            "--data",
            "data/reactions.center_expanded.cleaned.json",
            "--strategy",
            "mechanism_balanced_template",
            "--output-data",
            "data/reactions.center_expanded.mechanism_balanced_split.json",
            "--report",
            "reports/center_expanded_validation_report.json",
        ],
        [
            py,
            "scripts/retrain_center_head_strict_split.py",
            "--data",
            "data/reactions.center_expanded.mechanism_balanced_split.json",
            "--output",
            "models/atom_center_head_center_expanded_strict.pt",
            "--report",
            "reports/atom_center_training_center_expanded_strict_report.json",
            "--epochs",
            str(epochs),
        ],
        [
            py,
            "scripts/benchmark_center_head_strict_split.py",
            "--data",
            "data/reactions.center_expanded.mechanism_balanced_split.json",
            "--center-checkpoint",
            "models/atom_center_head_center_expanded_strict.pt",
            "--device",
            device,
            "--threshold",
            str(threshold),
            "--output",
            "reports/atom_center_benchmark_center_expanded_strict_report.json",
            "--comparison-output",
            "reports/center_head_center_expanded_strict_comparison.json",
        ],
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Phase 8.13 center expansion validation.")
    parser.add_argument("--max-promote", type=int, default=None)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps", "auto"), default="cpu")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    commands = build_pipeline_commands(args.max_promote, args.device, args.epochs, args.threshold)
    for command in commands:
        print(" ".join(command))
        if not args.dry_run:
            completed = subprocess.run(command, check=False)
            if completed.returncode != 0:
                return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
