"""CLI: conservatively promote Phase 8.13 center-expansion reactions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mendel.center_expansion_review import (  # noqa: E402
    merge_center_expansion_with_cleaned_base,
    promote_center_expansion_reactions,
    save_center_expansion_promotion_report,
    save_labeled_reactions_json,
)
from mendel.curation import (  # noqa: E402
    DraftLabelConfig,
    DraftReactionInput,
    draft_labeled_reactions,
)
from mendel.labels import load_labeled_reactions  # noqa: E402
from mendel.types import ReactionContext  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Promote conservative center-expansion labels. No training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", type=Path, default=_ROOT / "data" / "reactions.draft.center_expansion.json"
    )
    parser.add_argument(
        "--fallback-input", type=Path, default=_ROOT / "data" / "draft_inputs.center_expansion.json"
    )
    parser.add_argument(
        "--base", type=Path, default=_ROOT / "data" / "reactions.center_cleaned.json"
    )
    parser.add_argument(
        "--output", type=Path, default=_ROOT / "data" / "reactions.center_expansion.promoted.json"
    )
    parser.add_argument(
        "--merged-output",
        type=Path,
        default=_ROOT / "data" / "reactions.center_expanded.cleaned_input.json",
    )
    parser.add_argument(
        "--report", type=Path, default=_ROOT / "reports" / "center_expansion_promotion_report.json"
    )
    parser.add_argument("--conservative", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-aldol", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-controls", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-promote", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _load_fallback_inputs(path: Path) -> list[DraftReactionInput]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw = payload.get("candidates", payload) if isinstance(payload, dict) else payload
    inputs: list[DraftReactionInput] = []
    for item in raw:
        inputs.append(
            DraftReactionInput(
                reaction_id=item["reaction_id"],
                reaction_smiles=item["reaction_smiles"],
                context=ReactionContext(item["context"]),
                mechanism_type=item["mechanism_type"],
                split=item.get("split", "draft"),
                metadata=dict(item.get("metadata", {})),
            )
        )
    return inputs


def _load_or_draft(input_path: Path, fallback_path: Path):
    if input_path.exists():
        return load_labeled_reactions(input_path)
    if not fallback_path.exists():
        raise FileNotFoundError(f"No input or fallback input found: {input_path}, {fallback_path}")
    inputs = _load_fallback_inputs(fallback_path)
    reactions, _ = draft_labeled_reactions(
        inputs,
        DraftLabelConfig(include_spectators=True, source_tag="center_expansion_draft"),
    )
    input_path.parent.mkdir(parents=True, exist_ok=True)
    save_labeled_reactions_json(reactions, input_path)
    return reactions


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        reactions = _load_or_draft(args.input, args.fallback_input)
    except Exception as exc:
        print(f"ERROR: could not load center expansion input: {exc}", file=sys.stderr)
        return 1
    if args.max_promote is not None:
        reactions = reactions[: args.max_promote]
    base = load_labeled_reactions(args.base) if args.base.exists() else []
    promoted, report = promote_center_expansion_reactions(
        reactions,
        conservative=args.conservative,
        include_aldol=args.include_aldol,
        include_controls=args.include_controls,
    )
    merged = merge_center_expansion_with_cleaned_base(base, promoted)
    report.input_path = str(args.input if args.input.exists() else args.fallback_input)
    report.output_path = str(args.output)
    report.merged_output_path = str(args.merged_output)
    if not args.dry_run:
        save_labeled_reactions_json(promoted, args.output)
        save_labeled_reactions_json(merged, args.merged_output)
        save_center_expansion_promotion_report(report, args.report)
    print(f"n_input_reactions: {report.n_input_reactions}")
    print(f"n_promoted_reactions: {report.n_promoted_reactions}")
    print(f"n_skipped_reactions: {report.n_skipped_reactions}")
    print(f"promoted mechanisms: {report.promoted_mechanism_distribution}")
    if not args.dry_run:
        print(f"Promoted output: {args.output}")
        print(f"Merged output: {args.merged_output}")
        print(f"Report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
