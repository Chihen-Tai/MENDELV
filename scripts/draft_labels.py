"""Phase 6.5 CLI: generate draft labeled reactions for manual curation.

Usage examples:
  python scripts/draft_labels.py --core --output data/reactions.draft.core.json --report reports/draft_core_report.json
  python scripts/draft_labels.py --core --extended --output data/reactions.draft.json --report reports/draft_report.json
  python scripts/draft_labels.py --input data/draft_inputs.json --output data/reactions.draft.json --report reports/draft_report.json
  python scripts/draft_labels.py --input data/draft_inputs.json --merge-existing data/reactions.json --output data/reactions.merged.json --report reports/draft_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mendel.curation import (
    DraftLabelConfig,
    create_core_draft_inputs,
    create_extended_draft_inputs,
    draft_labeled_reactions,
    load_draft_inputs,
    merge_labeled_reactions,
    save_draft_labeled_reactions,
    summarize_draft_labels,
)
from mendel.labels import load_labeled_reactions


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate draft labeled reactions for Phase 6.5 curation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    source = p.add_argument_group("Input sources (at least one required)")
    source.add_argument("--core", action="store_true", help="Include core benchmark draft inputs")
    source.add_argument(
        "--extended", action="store_true", help="Include extended benchmark draft inputs"
    )
    source.add_argument(
        "--input", metavar="PATH", help="Load draft inputs from a JSON file"
    )

    p.add_argument(
        "--output", metavar="PATH", required=True, help="Output draft labeled reactions JSON"
    )
    p.add_argument("--report", metavar="PATH", required=True, help="Output report JSON")

    p.add_argument(
        "--merge-existing",
        metavar="PATH",
        help="Merge drafts into an existing labeled dataset",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="When merging, overwrite existing records with the same reaction_id",
    )
    p.add_argument("--include-spectators", action="store_true", help="Include spectator groups")
    p.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        metavar="FLOAT",
        help="Minimum final_confidence; lower-confidence groups are excluded (default: 0.0)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not (args.core or args.extended or args.input):
        parser.error("Specify at least one of --core, --extended, or --input")

    inputs = []
    if args.core:
        inputs.extend(create_core_draft_inputs())
    if args.extended:
        inputs.extend(create_extended_draft_inputs())
    if args.input:
        try:
            inputs.extend(load_draft_inputs(args.input))
        except (ValueError, FileNotFoundError) as exc:
            print(f"ERROR loading --input {args.input}: {exc}", file=sys.stderr)
            return 1

    if not inputs:
        print("No draft inputs to process.", file=sys.stderr)
        return 1

    # De-duplicate by reaction_id (first occurrence wins)
    seen: set[str] = set()
    deduped = []
    for inp in inputs:
        if inp.reaction_id not in seen:
            deduped.append(inp)
            seen.add(inp.reaction_id)
    inputs = deduped

    cfg = DraftLabelConfig(
        include_spectators=args.include_spectators,
        include_low_confidence=args.min_confidence == 0.0,
        min_confidence=args.min_confidence,
    )

    print(f"Processing {len(inputs)} draft input(s)...")
    reactions, report = draft_labeled_reactions(inputs, cfg)

    if report.skipped:
        print(f"WARNING: {len(report.skipped)} input(s) failed:")
        for skip in report.skipped:
            print(f"  - {skip['reaction_id']}: {skip['error']}")

    if args.merge_existing:
        try:
            existing = load_labeled_reactions(args.merge_existing)
            reactions = merge_labeled_reactions(existing, reactions, overwrite=args.overwrite)
            print(
                f"Merged with {len(existing)} existing record(s) "
                f"(overwrite={args.overwrite})"
            )
        except (FileNotFoundError, KeyError) as exc:
            print(f"ERROR loading --merge-existing {args.merge_existing}: {exc}", file=sys.stderr)
            return 1

    out_path = Path(args.output)
    report_path = Path(args.report)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    save_draft_labeled_reactions(reactions, out_path)
    report_path.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )

    summary = summarize_draft_labels(reactions)
    print(f"\nDraft labels saved to:  {out_path}")
    print(f"Report saved to:        {report_path}")
    print(f"\nSummary:")
    print(f"  Reactions generated:    {report.n_outputs} / {report.n_inputs}")
    print(f"  Total group roles:      {report.n_group_roles}")
    print(f"  needs_manual_review:    {summary['needs_manual_review_count']}")
    print(f"  Role distribution:      {summary['role_counts']}")
    print(f"  Mechanism distribution: {summary['mechanism_counts']}")
    if report.skipped:
        print(f"  Skipped (errors):       {len(report.skipped)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
