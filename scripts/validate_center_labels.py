"""CLI: validate reaction-center labels and create leakage-resistant splits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mendel.center_validation import (  # noqa: E402
    build_leakage_validation_report,
    save_leakage_validation_report,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate MENDELV atom-center labels and leakage-resistant splits.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=_ROOT / "data" / "reactions.proposed_with_auto_promoted.normalized.json",
    )
    parser.add_argument(
        "--strategy",
        choices=(
            "template",
            "mechanism",
            "source",
            "reaction_id_prefix",
            "mechanism_balanced_template",
            "val_test_balanced_template",
        ),
        default="template",
    )
    parser.add_argument(
        "--output-data",
        type=Path,
        default=_ROOT / "data" / "reactions.center_validated.template_split.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=_ROOT / "reports" / "center_validation_report.json",
    )
    parser.add_argument("--no-write-data", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if not args.data.exists():
        print(f"ERROR: dataset does not exist: {args.data}", file=sys.stderr)
        return 1
    output_dataset = None if args.no_write_data else args.output_data
    report = build_leakage_validation_report(args.data, args.strategy, output_dataset)
    save_leakage_validation_report(report, args.report)
    issue_summary = report.metrics.get("center_label_issue_summary", {})
    by_severity = issue_summary.get("by_severity", {}) if isinstance(issue_summary, dict) else {}
    by_type = issue_summary.get("by_issue_code", {}) if isinstance(issue_summary, dict) else {}

    print(f"n_reactions: {report.n_reactions}")
    print(f"split distribution: {report.new_split_distribution}")
    print(f"issue counts by severity: {by_severity}")
    print(f"issue counts by type: {by_type}")
    print("recommendations:")
    for recommendation in report.recommendations:
        print(f"  - {recommendation}")
    print(f"Report: {args.report}")
    if output_dataset is not None:
        print(f"Split dataset: {output_dataset}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
