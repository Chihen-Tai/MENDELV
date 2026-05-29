"""CLI: benchmark the Phase 8.10 atom-level reaction-center head.

This evaluates an existing binary atom classifier. It does not train.
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
    benchmark_atom_center_head,
    build_atom_center_examples,
    predict_atom_centers,
    save_atom_center_benchmark_report,
)
from mendel.descriptor import build_descriptors  # noqa: E402
from mendel.identifier import identify_functional_groups  # noqa: E402
from mendel.labels import load_labeled_reactions  # noqa: E402
from mendel.parser import parse_reaction_smiles  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark MENDELV Phase 8.10 atom-level reaction-center head.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=_ROOT / "data" / "reactions.proposed_with_auto_promoted.normalized.json",
    )
    parser.add_argument(
        "--center-checkpoint",
        type=Path,
        default=_ROOT / "models" / "atom_center_head.pt",
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
        default=_ROOT / "reports" / "atom_center_benchmark_report.json",
    )
    parser.add_argument(
        "--comparison-output",
        type=Path,
        default=_ROOT / "reports" / "center_head_comparison.json",
    )
    return parser


def _role_predictions(reactions, checkpoint: Path, device: str):
    if not checkpoint.exists():
        print(f"WARNING: role checkpoint missing; using labeled roles: {checkpoint}")
        return None
    try:
        from mendel.mlp import MLPRolePredictor
    except ImportError:
        print("WARNING: torch unavailable for role checkpoint; using labeled roles.")
        return None
    predictor = MLPRolePredictor.load(checkpoint, device=device if device != "auto" else "cpu")
    predictions = {}
    for rxn in reactions:
        parsed = parse_reaction_smiles(rxn.reaction_smiles, context=rxn.context)
        groups = identify_functional_groups(parsed)
        descriptors = build_descriptors(parsed, groups)
        predictions[rxn.reaction_id] = predictor.predict_descriptors(descriptors)
    return predictions


def _reference_f1() -> dict[str, float | None]:
    path = _ROOT / "reports" / "benchmark_mlp_aware_full" / "comparison.json"
    if not path.exists():
        return {
            "rule_based_negotiated": 0.8929,
            "new_mlp_aware_negotiated": 0.7973,
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    center = payload.get("reaction_center", {})
    refs: dict[str, float | None] = {}
    for name in ("rule_based_negotiated", "new_mlp_aware_negotiated"):
        value = center.get(name, {}) if isinstance(center, dict) else {}
        refs[name] = value.get("f1") if isinstance(value, dict) else None
    return refs


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
    report = benchmark_atom_center_head(
        reactions,
        predictions,
        threshold=args.threshold,
        predictor_name="atom_center_head",
    )
    save_atom_center_benchmark_report(report, args.output)

    refs = _reference_f1()
    comparison = {
        "atom_center_head": report.to_dict(),
        "references": refs,
        "improves_over_new_mlp_aware_negotiated": (
            report.reaction_center_f1 is not None
            and refs.get("new_mlp_aware_negotiated") is not None
            and report.reaction_center_f1 > refs["new_mlp_aware_negotiated"]
        ),
        "approaches_rule_based_negotiated": (
            report.reaction_center_f1 is not None
            and refs.get("rule_based_negotiated") is not None
            and report.reaction_center_f1 >= refs["rule_based_negotiated"]
        ),
        "scope_note": "Atom reaction-center classifier only; no MLIP training.",
    }
    args.comparison_output.parent.mkdir(parents=True, exist_ok=True)
    args.comparison_output.write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("Atom center head benchmark")
    print(f"  atom precision:          {report.atom_precision}")
    print(f"  atom recall:             {report.atom_recall}")
    print(f"  atom F1:                 {report.atom_f1}")
    print(f"  reaction-center F1:      {report.reaction_center_f1}")
    print("  per-mechanism F1:")
    for mechanism, f1 in sorted(report.per_mechanism_f1.items()):
        print(f"    {mechanism}: {f1}")
    print(f"Benchmark report:  {args.output}")
    print(f"Comparison report: {args.comparison_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
