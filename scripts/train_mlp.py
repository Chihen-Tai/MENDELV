"""CLI script: train the MENDEL Phase 7 MLP role predictor.

Usage:
    python scripts/train_mlp.py [options]

Default output paths:
    Checkpoint: models/role_mlp.pt
    Report:     reports/mlp_training_report.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Resolve project root so the script works from any working directory
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mendel.labels import load_labeled_reactions
from mendel.mlp import (
    TrainingConfig,
    build_training_examples,
    evaluate_mlp_predictor,
    save_training_report,
    summarize_training_examples,
    train_mlp_role_predictor,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train the MENDEL MLP role predictor (Phase 7).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data",
        type=Path,
        default=_ROOT / "data" / "reactions.json",
        help="Path to labeled reactions JSON file.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=_ROOT / "models" / "role_mlp.pt",
        help="Path to save the trained model checkpoint.",
    )
    p.add_argument(
        "--report",
        type=Path,
        default=_ROOT / "reports" / "mlp_training_report.json",
        help="Path to save the evaluation report JSON.",
    )
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--hidden-dim", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument(
        "--use-class-weights",
        action="store_true",
        default=False,
        help="Weight loss by inverse class frequency.",
    )
    p.add_argument(
        "--strict-group-matching",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Warn and skip labeled groups that have no identified descriptor match.",
    )
    p.add_argument(
        "--allow-draft-labels",
        action="store_true",
        default=False,
        help="Allow draft (unreviewed) labels. FOR SMOKE TESTING ONLY.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading labeled reactions from {args.data} ...")
    reactions = load_labeled_reactions(args.data)
    print(f"  {len(reactions)} reactions loaded.")

    if args.allow_draft_labels:
        print("WARNING: Training on draft labels is for smoke testing only.")

    examples = build_training_examples(
        reactions,
        strict_group_matching=args.strict_group_matching,
        allow_draft_labels=args.allow_draft_labels,
    )

    # If no examples and draft labels exist but flag not set, give a clear error
    if not examples and not args.allow_draft_labels:
        all_examples_with_drafts = build_training_examples(
            reactions,
            strict_group_matching=False,
            allow_draft_labels=True,
        )
        if all_examples_with_drafts:
            print(
                "\nERROR: No training examples found. The dataset contains only draft "
                "labels (needs_manual_review=true or confidence='draft').\n"
                "Either manually review and curate the labels, or pass "
                "--allow-draft-labels for smoke testing only.",
                file=sys.stderr,
            )
            sys.exit(1)

    summary = summarize_training_examples(examples)

    print(f"\nTraining examples: {summary.n_examples}")
    print(f"  Features per example: {summary.n_features}")
    print("  Role distribution:")
    for role, count in sorted(summary.role_counts.items()):
        print(f"    {role}: {count}")
    missing_roles = str(summary.metadata.get("missing_roles", ""))
    roles_below_10 = str(summary.metadata.get("roles_below_10", ""))
    if missing_roles:
        print(f"  Missing roles: {missing_roles}")
    if roles_below_10:
        print(f"  Roles below 10 labels: {roles_below_10}")
    if summary.n_examples < 50:
        print(
            "WARNING: Fewer than 50 training examples; MLP results are "
            "smoke/early-training only and should not replace rule-based defaults."
        )

    if summary.n_examples == 0:
        print("\nNo training examples found. Exiting.")
        sys.exit(1)

    config = TrainingConfig(
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        use_class_weights=args.use_class_weights,
    )

    print(f"\nTraining MLP (hidden={config.hidden_dim}, epochs={config.epochs}) ...")
    predictor, history = train_mlp_role_predictor(examples, config=config)

    train_loss_final = history.train_loss[-1] if history.train_loss else float("nan")
    val_loss_final = history.val_loss[-1] if history.val_loss else float("nan")
    val_acc_final = history.val_accuracy[-1] if history.val_accuracy else float("nan")
    epochs_run = int(history.metadata.get("epochs_run", len(history.train_loss)))

    val_is_holdout = bool(
        history.metadata.get(
            "val_is_holdout", not history.metadata.get("fallback_split", False)
        )
    )
    print(f"  Epochs run:         {epochs_run}")
    print(f"  Final train loss:   {train_loss_final:.4f}")
    if val_is_holdout:
        print(f"  Final val loss:     {val_loss_final:.4f}")
        print(f"  Final val accuracy: {val_acc_final:.4f}")
    else:
        print("  Validation:         NONE (fallback split — train data mirrored as val).")
        print("                      val_loss/val_accuracy are NOT held-out KPIs; suppressed.")
    dataset_warnings = str(history.metadata.get("dataset_warnings", ""))
    if dataset_warnings:
        print(f"  Dataset warnings:    {dataset_warnings}")

    predictor.save(args.output)
    print(f"\nCheckpoint saved to {args.output}")

    report = evaluate_mlp_predictor(predictor, examples)
    report["training_summary"] = summary.to_dict()
    report["training_history"] = history.to_dict()
    save_training_report(report, args.report)
    print(f"Report saved to {args.report}")

    print(f"\nFinal accuracy on all examples: {report['accuracy']:.4f}")


if __name__ == "__main__":
    main()
