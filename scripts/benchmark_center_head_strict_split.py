"""CLI: benchmark atom-center head on leakage-resistant splits.

This evaluates an existing atom classifier. It does not train MLIP.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mendel.center_head import (  # noqa: E402
    benchmark_atom_center_head_by_split,
    build_atom_center_examples,
    predict_atom_centers,
)
from mendel.labels import load_labeled_reactions  # noqa: E402
from scripts.benchmark_center_head import _reference_f1, _role_predictions  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Strict-split benchmark for MENDELV atom reaction-center head.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=_ROOT / "data" / "reactions.center_validated.template_split.json",
    )
    parser.add_argument(
        "--center-checkpoint",
        type=Path,
        default=_ROOT / "models" / "atom_center_head_template_split.pt",
    )
    parser.add_argument(
        "--role-checkpoint",
        type=Path,
        default=_ROOT / "models" / "role_mlp_promoted.pt",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps", "auto"), default="cpu")
    parser.add_argument(
        "--output",
        type=Path,
        default=_ROOT / "reports" / "atom_center_benchmark_template_split_report.json",
    )
    parser.add_argument(
        "--comparison-output",
        type=Path,
        default=_ROOT / "reports" / "center_head_template_split_comparison.json",
    )
    return parser


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if not args.data.exists():
        print(f"ERROR: dataset does not exist: {args.data}", file=sys.stderr)
        return 1
    if not args.center_checkpoint.exists():
        print(f"ERROR: center checkpoint does not exist: {args.center_checkpoint}", file=sys.stderr)
        return 1
    reactions = load_labeled_reactions(args.data)
    role_predictions = _role_predictions(reactions, args.role_checkpoint, args.device)
    examples = build_atom_center_examples(reactions, role_predictions)
    try:
        predictions = predict_atom_centers(
            examples,
            args.center_checkpoint,
            threshold=args.threshold,
            device=args.device if args.device != "auto" else "cpu",
        )
    except ImportError:
        print("ERROR: torch is required to benchmark the atom center head.", file=sys.stderr)
        return 1
    split_reports = benchmark_atom_center_head_by_split(
        reactions,
        predictions,
        threshold=args.threshold,
        predictor_name="atom_center_head_template_split",
    )
    _write_json(args.output, split_reports)
    refs = _reference_f1()
    comparison = {
        "strict_split_reports": split_reports,
        "references": {
            **refs,
            "non_strict_atom_center_head": 0.9054,
        },
        "primary_metric": "test.reaction_center_f1",
        "strict_test_reaction_center_f1": split_reports["test"].get("reaction_center_f1"),
        "scope_note": "Strict split atom-center benchmark only; no MLIP training.",
    }
    _write_json(args.comparison_output, comparison)

    print("Strict-split atom center head benchmark")
    for split in ("train", "val", "test", "overall"):
        report = split_reports[split]
        print(f"  {split} reaction-center F1: {report.get('reaction_center_f1')}")
    print(f"Benchmark report:  {args.output}")
    print(f"Comparison report: {args.comparison_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
