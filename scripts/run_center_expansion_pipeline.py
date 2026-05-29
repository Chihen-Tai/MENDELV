"""Run a lightweight local center-expansion scaffold for Phase 8.12."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.generate_center_expansion_candidates import (  # noqa: E402
    generate_center_expansion_candidates,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create local center-expansion artifacts; no training."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--report", type=Path, default=Path("reports/center_expansion_pipeline_report.json")
    )
    parser.add_argument("--max-count", type=int, default=72)
    parser.add_argument("--skip-promotion", action="store_true")
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates = generate_center_expansion_candidates(args.max_count)
    candidate_path = args.output_dir / "draft_inputs.center_expansion.json"
    candidate_path.write_text(json.dumps({"candidates": candidates}, indent=2), encoding="utf-8")
    mechanisms = Counter(c["metadata"]["mechanism_type"] for c in candidates)
    report = {
        "n_candidates": len(candidates),
        "mechanism_distribution": dict(sorted(mechanisms.items())),
        "center_label_focus": True,
        "outputs": {
            "draft_inputs": str(candidate_path),
            "draft_labels": str(args.output_dir / "reactions.draft.center_expansion.json"),
            "review_queue": str(args.output_dir / "reactions.center_expansion.review_queue.json"),
            "promoted": str(args.output_dir / "reactions.center_expansion.promoted.json"),
            "proposed": str(args.output_dir / "reactions.proposed_with_center_expansion.json"),
            "cleaned": str(
                args.output_dir / "reactions.proposed_with_center_expansion.cleaned.json"
            ),
        },
        "promotion_run": not args.skip_promotion,
        "note": (
            "Scaffold generated local template candidates only; "
            "no external data and no training."
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Center expansion candidates: {candidate_path}")
    print(f"Pipeline report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
