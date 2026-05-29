"""Run Phase 8.12 cleanup plus strict validation/training/benchmark."""

from __future__ import annotations

import argparse
import subprocess
import sys


def _run(cmd: list[str]) -> None:
    completed = subprocess.run(cmd, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run Phase 8.12 center cleanup validation workflow."
    )
    parser.parse_args(argv)
    py = sys.executable
    _run([py, "scripts/cleanup_center_labels.py"])
    _run(
        [
            py,
            "scripts/validate_center_labels.py",
            "--data",
            "data/reactions.center_cleaned.json",
            "--strategy",
            "mechanism_balanced_template",
            "--output-data",
            "data/reactions.center_cleaned.mechanism_balanced_split.json",
            "--report",
            "reports/center_validation_cleaned_report.json",
        ]
    )
    _run(
        [
            py,
            "scripts/retrain_center_head_strict_split.py",
            "--data",
            "data/reactions.center_cleaned.mechanism_balanced_split.json",
            "--output",
            "models/atom_center_head_cleaned_strict.pt",
            "--report",
            "reports/atom_center_training_cleaned_strict_report.json",
        ]
    )
    _run(
        [
            py,
            "scripts/benchmark_center_head_strict_split.py",
            "--data",
            "data/reactions.center_cleaned.mechanism_balanced_split.json",
            "--center-checkpoint",
            "models/atom_center_head_cleaned_strict.pt",
            "--output",
            "reports/atom_center_benchmark_cleaned_strict_report.json",
            "--comparison-output",
            "reports/center_head_cleaned_strict_comparison.json",
        ]
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
