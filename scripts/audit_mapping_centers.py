"""CLI: audit atom-mapped reaction centers without training MLIP."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mendel.labels import load_labeled_reactions  # noqa: E402
from mendel.mapping_center import (  # noqa: E402
    apply_mapping_center_suggestions,
    audit_labeled_centers_against_mapping,
    save_labeled_reactions_json,
    save_mapping_center_audit_report,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit MENDELV center labels against atom mapping / bond changes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=_ROOT / "data" / "reactions.center_expanded.cleaned.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_ROOT / "reports" / "mapping_center_audit_report.json",
    )
    parser.add_argument(
        "--apply-suggestions-output",
        type=Path,
        default=_ROOT / "data" / "reactions.center_expanded.mapping_audited.json",
    )
    parser.add_argument("--apply-high-confidence", action="store_true")
    parser.add_argument("--conservative", action="store_true", default=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if not args.data.exists():
        print(f"ERROR: dataset does not exist: {args.data}", file=sys.stderr)
        return 1
    reactions = load_labeled_reactions(args.data)
    report = audit_labeled_centers_against_mapping(reactions, dataset_path=str(args.data))
    save_mapping_center_audit_report(report, args.output)

    n_applied = 0
    if args.apply_high_confidence:
        updated, applied = apply_mapping_center_suggestions(
            reactions,
            min_confidence="high",
            conservative=args.conservative,
        )
        n_applied = len(applied)
        save_labeled_reactions_json(updated, args.apply_suggestions_output)

    print(f"n_reactions: {report.n_reactions}")
    print(f"n_mapped: {report.n_mapped_reactions}")
    print(f"n_unmapped: {report.n_unmapped_reactions}")
    print(f"n_exact_matches: {report.n_exact_matches}")
    print(f"mean_overlap_f1: {report.mean_overlap_f1}")
    print(f"issue counts: {report.issue_counts_by_type}")
    print(f"suggestions applied: {n_applied}")
    print("recommendations:")
    for recommendation in report.recommendations:
        print(f"  - {recommendation}")
    print(f"Report: {args.output}")
    if args.apply_high_confidence:
        print(f"Applied dataset: {args.apply_suggestions_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
