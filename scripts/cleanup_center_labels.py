"""CLI: conservatively clean MENDELV reaction-center labels."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mendel.center_cleanup import (  # noqa: E402
    build_center_cleanup_report,
    cleanup_center_labels,
    save_center_cleanup_report,
    save_labeled_reactions_json,
)
from mendel.center_validation import (  # noqa: E402
    audit_center_labels,
    summarize_center_label_issues,
)
from mendel.labels import load_labeled_reactions  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Conservative Phase 8.12 center-label cleanup.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=_ROOT / "data" / "reactions.proposed_with_auto_promoted.normalized.json",
    )
    parser.add_argument(
        "--output", type=Path, default=_ROOT / "data" / "reactions.center_cleaned.json"
    )
    parser.add_argument(
        "--report", type=Path, default=_ROOT / "reports" / "center_cleanup_report.json"
    )
    parser.add_argument("--conservative", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fail-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if not args.input.exists():
        print(f"ERROR: input dataset does not exist: {args.input}", file=sys.stderr)
        return 1
    reactions = load_labeled_reactions(args.input)
    before_summary = summarize_center_label_issues(audit_center_labels(reactions))
    cleaned, corrections = cleanup_center_labels(reactions, conservative=args.conservative)
    after_summary = summarize_center_label_issues(audit_center_labels(cleaned))
    report = build_center_cleanup_report(args.input, args.output, reactions, cleaned, corrections)
    report.metadata["before_issues_by_type"] = str(before_summary["by_issue_code"])
    if args.fail_on_error and after_summary["by_severity"].get("error", 0):  # type: ignore[index,union-attr]
        print("ERROR: cleanup left error-severity center-label issues.", file=sys.stderr)
        return 1
    if not args.dry_run:
        save_labeled_reactions_json(cleaned, args.output)
        save_center_cleanup_report(report, args.report)

    print(f"n_reactions: {report.n_reactions}")
    print(f"n_corrections: {report.n_corrections}")
    print(f"corrections by type: {report.corrections_by_type}")
    print(f"remaining issue counts: {report.remaining_issues_by_type}")
    print(f"remaining issues by severity: {report.remaining_issues_by_severity}")
    print("recommendations:")
    for recommendation in report.recommendations:
        print(f"  - {recommendation}")
    if not args.dry_run:
        print(f"Cleaned dataset: {args.output}")
        print(f"Report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
