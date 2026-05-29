"""CLI script: promote conservative Phase 8.6 review-queue labels.

This script reviews deterministic auto-candidate draft labels and writes a
curated supplemental dataset plus a proposed merged dataset. It does not train
models and never modifies data/reactions.json in place.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mendel.dataset_quality import save_labeled_reactions_json  # noqa: E402
from mendel.labels import load_labeled_reactions  # noqa: E402
from mendel.promotion import (  # noqa: E402
    build_promotion_report,
    merge_promoted_with_base,
    promote_review_queue,
)


def _default_base_path() -> Path:
    normalized = _ROOT / "data" / "reactions.normalized.json"
    if normalized.exists():
        return normalized
    return _ROOT / "data" / "reactions.json"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Promote conservative MENDELV auto review-queue labels.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=_ROOT / "data" / "reactions.auto_review_queue.json",
    )
    parser.add_argument("--base", type=Path, default=_default_base_path())
    parser.add_argument(
        "--output",
        type=Path,
        default=_ROOT / "data" / "reactions.auto_curated_promoted.json",
    )
    parser.add_argument(
        "--merged-output",
        type=Path,
        default=_ROOT / "data" / "reactions.proposed_with_auto_promoted.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=_ROOT / "reports" / "auto_promotion_review_report.json",
    )
    parser.add_argument("--max-promote", type=int)
    parser.add_argument("--include-aldol", action="store_true", default=False)
    parser.add_argument("--include-controls", action="store_true", default=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _write_report(report: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def _print_summary(report: dict[str, object], dry_run: bool) -> None:
    prefix = "DRY RUN: " if dry_run else ""
    print(f"{prefix}Phase 8.6 promotion complete")
    print(f"  input reactions:          {report['n_input_reactions']}")
    print(f"  promoted reactions:       {report['n_promoted_reactions']}")
    print(f"  skipped reactions:        {report['n_skipped_reactions']}")
    print(f"  promoted group labels:    {report['n_promoted_group_labels']}")
    print(f"  corrected labels:         {len(report['corrected_labels'])}")
    print(f"  promoted mechanisms:      {report['promoted_mechanism_distribution']}")


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    input_reactions = load_labeled_reactions(args.input)
    base_reactions = load_labeled_reactions(args.base)

    promoted, skipped, corrected_labels, warnings = promote_review_queue(
        input_reactions,
        include_aldol=args.include_aldol,
        include_controls=args.include_controls,
        max_promote=args.max_promote,
    )
    merged, merge_warnings = merge_promoted_with_base(base_reactions, promoted)
    warnings.extend(merge_warnings)

    output_paths = {
        "input": str(args.input),
        "base": str(args.base),
        "output": str(args.output),
        "merged_output": str(args.merged_output),
        "report": str(args.report),
    }
    report = build_promotion_report(
        input_reactions=input_reactions,
        promoted_reactions=promoted,
        skipped_reactions=skipped,
        corrected_labels=corrected_labels,
        warnings=warnings,
        output_paths=output_paths,
    )

    if not args.dry_run:
        save_labeled_reactions_json(promoted, args.output)
        save_labeled_reactions_json(merged, args.merged_output)
        _write_report(report, args.report)

    _print_summary(report, dry_run=args.dry_run)
    if args.dry_run:
        print("  no files written")
    else:
        print(f"  promoted dataset:         {args.output}")
        print(f"  proposed merged dataset:  {args.merged_output}")
        print(f"  report:                   {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
