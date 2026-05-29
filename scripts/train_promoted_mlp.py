"""CLI script: train the Phase 8.7 promoted-data MLP role predictor.

This trains only the small descriptor-to-role classifier. It does not train
MLIP, MACE, energies, forces, transition states, or barrier models.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mendel.labels import load_labeled_reactions, summarize_labeled_dataset  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train MENDELV Phase 8.7 MLP on promoted curated labels.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=_ROOT / "data" / "reactions.proposed_with_auto_promoted.normalized.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_ROOT / "models" / "role_mlp_promoted.pt",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=_ROOT / "reports" / "mlp_promoted_training_report.json",
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--use-class-weights",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use inverse-frequency class weights.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-draft-labels", action="store_true", default=False)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps", "auto"), default="auto")
    return parser


def _has_draft_labels(reactions) -> bool:
    for reaction in reactions:
        if reaction.metadata.get("needs_manual_review") is True:
            return True
        if any(role.confidence == "draft" for role in reaction.group_roles):
            return True
    return False


def _dataset_warnings(summary: dict[str, object]) -> list[str]:
    warnings: list[str] = []
    n_labels = int(summary["n_labels"])
    role_distribution = dict(summary["role_distribution"])
    if n_labels < 100:
        warnings.append("n_labels_below_100")
    for role, count in sorted(role_distribution.items()):
        if int(count) < 10:
            warnings.append(f"role_below_10:{role}")
    return warnings


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if not args.data.exists():
        print(f"ERROR: dataset does not exist: {args.data}", file=sys.stderr)
        return 1
    if args.output.resolve() == (_ROOT / "models" / "role_mlp.pt").resolve():
        print("ERROR: refusing to overwrite models/role_mlp.pt", file=sys.stderr)
        return 1

    reactions = load_labeled_reactions(args.data)
    dataset_summary = summarize_labeled_dataset(reactions)
    warnings = _dataset_warnings(dataset_summary)

    if _has_draft_labels(reactions) and not args.allow_draft_labels:
        print(
            "ERROR: dataset contains draft labels; pass --allow-draft-labels only "
            "for smoke testing.",
            file=sys.stderr,
        )
        return 1

    from mendel.mlp import (  # noqa: PLC0415
        TrainingConfig,
        build_training_examples,
        evaluate_mlp_predictor,
        save_training_report,
        summarize_training_examples,
        train_mlp_role_predictor,
    )

    examples = build_training_examples(
        reactions,
        strict_group_matching=True,
        allow_draft_labels=args.allow_draft_labels,
    )
    if not examples:
        print("ERROR: no MLP training examples could be built.", file=sys.stderr)
        return 1

    config = TrainingConfig(
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        use_class_weights=args.use_class_weights,
        seed=args.seed,
        device=args.device,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    print(f"Training promoted MLP from {args.data}")
    print(f"  reactions: {dataset_summary['n_reactions']}")
    print(f"  labels:    {dataset_summary['n_labels']}")
    for warning in warnings:
        print(f"WARNING: {warning}")

    predictor, history = train_mlp_role_predictor(examples, config=config)
    predictor.save(args.output)

    example_summary = summarize_training_examples(examples)
    evaluation = evaluate_mlp_predictor(predictor, examples)
    report = {
        "phase": "8.7",
        "dataset_path": str(args.data),
        "checkpoint_path": str(args.output),
        "n_reactions": dataset_summary["n_reactions"],
        "n_labels": dataset_summary["n_labels"],
        "n_training_examples": len(examples),
        "role_distribution": dataset_summary["role_distribution"],
        "mechanism_distribution": dataset_summary["mechanism_distribution"],
        "split_distribution": dataset_summary["split_distribution"],
        "training_example_summary": example_summary.to_dict(),
        "hyperparameters": {
            "epochs": args.epochs,
            "hidden_dim": args.hidden_dim,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "use_class_weights": args.use_class_weights,
            "seed": args.seed,
            "device": args.device,
        },
        "training_history": history.to_dict(),
        "final_train_accuracy": history.train_accuracy[-1] if history.train_accuracy else None,
        "validation_accuracy": history.val_accuracy[-1] if history.val_accuracy else None,
        "evaluation_on_all_examples": evaluation,
        "warnings": warnings,
        "class_weights_used": args.use_class_weights,
        "random_seed": args.seed,
        "scope_note": "MLP role classifier only; no MLIP, MACE, energy, or force training.",
    }
    save_training_report(report, args.report)

    print(f"Checkpoint saved: {args.output}")
    print(f"Training report:  {args.report}")
    print(f"Final train accuracy: {report['final_train_accuracy']}")
    print(f"Validation accuracy:  {report['validation_accuracy']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
