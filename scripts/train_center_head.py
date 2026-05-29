"""CLI: train the Phase 8.10 atom-level reaction-center head.

This trains only a binary atom classifier. It is not MLIP training.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_CONTROL_MECHANISMS = {"control", "ester_control", "nitrile_control", "no_reaction"}

from mendel.center_head import (  # noqa: E402
    build_atom_center_examples,
    summarize_atom_center_examples,
    train_atom_center_head,
)
from mendel.descriptor import build_descriptors  # noqa: E402
from mendel.identifier import identify_functional_groups  # noqa: E402
from mendel.labels import load_labeled_reactions  # noqa: E402
from mendel.parser import parse_reaction_smiles  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train MENDELV Phase 8.10 atom-level reaction-center head. Not MLIP.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=_ROOT / "data" / "reactions.proposed_with_auto_promoted.normalized.json",
    )
    parser.add_argument(
        "--role-checkpoint",
        type=Path,
        default=_ROOT / "models" / "role_mlp_promoted.pt",
    )
    parser.add_argument("--output", type=Path, default=_ROOT / "models" / "atom_center_head.pt")
    parser.add_argument(
        "--report",
        type=Path,
        default=_ROOT / "reports" / "atom_center_training_report.json",
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--use-class-weights",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use positive-class weighting for rare center atoms.",
    )
    parser.add_argument("--device", choices=("cpu", "cuda", "mps", "auto"), default="auto")
    parser.add_argument("--allow-missing-centers", action="store_true")
    return parser


def _role_predictions(reactions, checkpoint: Path, device: str):
    if not checkpoint.exists():
        return None, [f"role checkpoint missing; using labeled roles: {checkpoint}"]
    try:
        from mendel.mlp import MLPRolePredictor
    except ImportError:
        return None, ["torch unavailable; using labeled roles as atom-head features"]
    predictor = MLPRolePredictor.load(checkpoint, device=device if device != "auto" else "cpu")
    predictions = {}
    warnings: list[str] = []
    for rxn in reactions:
        try:
            parsed = parse_reaction_smiles(rxn.reaction_smiles, context=rxn.context)
            groups = identify_functional_groups(parsed)
            descriptors = build_descriptors(parsed, groups)
            predictions[rxn.reaction_id] = predictor.predict_descriptors(descriptors)
        except Exception as exc:  # pragma: no cover - defensive CLI path
            warnings.append(f"role prediction failed for {rxn.reaction_id}: {exc}")
    return predictions, warnings


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if not args.data.exists():
        print(f"ERROR: dataset does not exist: {args.data}", file=sys.stderr)
        return 1
    reactions = load_labeled_reactions(args.data)
    missing = [
        rxn.reaction_id
        for rxn in reactions
        if not rxn.reaction_center_atoms
        and rxn.mechanism_type.lower() not in _CONTROL_MECHANISMS
    ]
    if missing and not args.allow_missing_centers:
        print(
            "WARNING: non-control reactions without center labels are included "
            "as all-negative atom examples; pass --allow-missing-centers to "
            "acknowledge explicitly."
        )
    role_predictions, warnings = _role_predictions(reactions, args.role_checkpoint, args.device)
    examples = build_atom_center_examples(reactions, role_predictions)
    summary = summarize_atom_center_examples(examples)
    if int(summary["n_positive"]) == 0:
        print("ERROR: no positive center atoms available for training.", file=sys.stderr)
        return 1

    print(f"Training atom reaction-center head from {args.data}")
    print(f"  atom examples: {summary['n_examples']}")
    print(f"  positives:     {summary['n_positive']}")
    print(f"  negatives:     {summary['n_negative']}")
    print(f"  positive frac: {summary['positive_fraction']}")
    for warning in [*warnings, *summary["warnings"]]:
        print(f"WARNING: {warning}")

    try:
        report = train_atom_center_head(
            examples,
            args.output,
            args.report,
            hidden_dim=args.hidden_dim,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            threshold=args.threshold,
            use_class_weights=args.use_class_weights,
            device=args.device,
        )
    except ImportError:
        print("ERROR: torch is required to train the atom center head.", file=sys.stderr)
        return 1

    print(f"Checkpoint saved: {args.output}")
    print(f"Training report:  {args.report}")
    print(f"Train accuracy:   {report.train_accuracy}")
    print(f"Val F1:           {report.val_f1}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
