"""CLI script: benchmark Phase 8.7 promoted-data MLP checkpoints.

This evaluates existing predictors only. It does not train.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mendel.benchmark import (  # noqa: E402
    BenchmarkReport,
    compare_benchmark_reports,
    evaluate_mlp_aware_negotiated_checkpoint,
    evaluate_mlp_checkpoint,
    evaluate_negotiated_mlp_checkpoint,
    evaluate_negotiated_rule_based,
    evaluate_rule_based_predictor,
    save_benchmark_comparison,
    save_benchmark_report,
)
from mendel.labels import load_labeled_reactions  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark promoted MLP checkpoint against MENDELV baselines.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=_ROOT / "data" / "reactions.proposed_with_auto_promoted.normalized.json",
    )
    parser.add_argument(
        "--new-mlp-checkpoint",
        type=Path,
        default=_ROOT / "models" / "role_mlp_promoted.pt",
    )
    parser.add_argument(
        "--old-mlp-checkpoint",
        type=Path,
        default=_ROOT / "models" / "role_mlp.pt",
    )
    parser.add_argument("--device", choices=("cpu", "cuda", "mps", "auto"), default="auto")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_ROOT / "reports" / "benchmark_promoted_full",
    )
    parser.add_argument("--skip-old-mlp", action="store_true")
    parser.add_argument("--include-mlp-aware-negotiation", action="store_true")
    return parser


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _rename_report(report: BenchmarkReport, predictor_name: str) -> BenchmarkReport:
    original = report.predictor_name
    report.predictor_name = predictor_name
    report.metadata["original_predictor_name"] = original
    for record in report.group_records:
        record.predictor_name = predictor_name
    for record in report.reaction_records:
        record.predictor_name = predictor_name
    return report


def _print_report(report: BenchmarkReport) -> None:
    print(f"\n{report.predictor_name}")
    print(f"  overall role accuracy: {report.overall_role_accuracy:.4f}")
    print(f"  reaction-center F1:    {report.reaction_center_f1}")
    print("  per-role accuracy:")
    for role, accuracy in sorted(report.role_accuracy_by_role.items()):
        print(f"    {role}: {accuracy:.4f}")
    print("  per-mechanism accuracy:")
    for mechanism, accuracy in sorted(report.role_accuracy_by_mechanism.items()):
        print(f"    {mechanism}: {accuracy:.4f}")


def _add_phase87_interpretation(comparison: dict[str, object]) -> None:
    overall = comparison.get("overall_role_accuracy", {})
    if not isinstance(overall, dict):
        return
    reaction_center = comparison.get("reaction_center", {})
    new_local = overall.get("new_mlp_local")
    new_negotiated = overall.get("new_mlp_negotiated")
    old_local = overall.get("old_mlp_local")
    old_negotiated = overall.get("old_mlp_negotiated")
    rule_local = overall.get("rule_based_local")
    rule_negotiated = overall.get("rule_based_negotiated")
    new_rc_f1 = None
    rule_rc_f1 = None
    if isinstance(reaction_center, dict):
        new_rc = reaction_center.get("new_mlp_negotiated")
        rule_rc = reaction_center.get("rule_based_negotiated")
        if isinstance(new_rc, dict):
            new_rc_f1 = new_rc.get("f1")
        if isinstance(rule_rc, dict):
            rule_rc_f1 = rule_rc.get("f1")
    role_beats_negotiated = (
        new_negotiated is not None
        and rule_negotiated is not None
        and new_negotiated > rule_negotiated
    )
    rc_not_worse = (
        new_rc_f1 is not None
        and rule_rc_f1 is not None
        and new_rc_f1 >= rule_rc_f1
    )
    comparison["phase8_7"] = {
        "new_mlp_improved_over_old_local": (
            new_local is not None and old_local is not None and new_local > old_local
        ),
        "new_mlp_improved_over_old_negotiated": (
            new_negotiated is not None
            and old_negotiated is not None
            and new_negotiated > old_negotiated
        ),
        "new_mlp_beats_rule_based_local": (
            new_local is not None and rule_local is not None and new_local > rule_local
        ),
        "new_mlp_beats_rule_based_negotiated": role_beats_negotiated,
        "new_mlp_negotiated_reaction_center_f1_not_worse": rc_not_worse,
        "rule_based_negotiated_remains_default": not (
            role_beats_negotiated and rc_not_worse
        ),
    }


def _evaluate_checkpoint_pair(
    reactions,
    checkpoint: Path,
    device: str,
    local_name: str,
    negotiated_name: str,
) -> list[BenchmarkReport]:
    if not checkpoint.exists():
        print(f"WARNING: checkpoint missing; skipping {checkpoint}")
        return []
    local = _rename_report(
        evaluate_mlp_checkpoint(reactions, checkpoint, device=device),
        local_name,
    )
    negotiated = _rename_report(
        evaluate_negotiated_mlp_checkpoint(reactions, checkpoint, device=device),
        negotiated_name,
    )
    local.metadata["checkpoint_path"] = str(checkpoint)
    negotiated.metadata["checkpoint_path"] = str(checkpoint)
    return [local, negotiated]


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if not args.data.exists():
        print(f"ERROR: dataset does not exist: {args.data}", file=sys.stderr)
        return 1

    reactions = load_labeled_reactions(args.data)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(args.device)

    reports: list[BenchmarkReport] = [
        evaluate_rule_based_predictor(reactions),
        evaluate_negotiated_rule_based(reactions),
    ]
    reports.extend(
        _evaluate_checkpoint_pair(
            reactions,
            args.new_mlp_checkpoint,
            device,
            "new_mlp_local",
            "new_mlp_negotiated",
        )
    )
    if args.include_mlp_aware_negotiation and args.new_mlp_checkpoint.exists():
        aware = evaluate_mlp_aware_negotiated_checkpoint(
            reactions,
            args.new_mlp_checkpoint,
            device=device,
            predictor_name="new_mlp_aware_negotiated",
        )
        aware.metadata["checkpoint_path"] = str(args.new_mlp_checkpoint)
        reports.append(aware)
    elif args.include_mlp_aware_negotiation:
        print(f"WARNING: checkpoint missing; skipping {args.new_mlp_checkpoint}")
    if not args.skip_old_mlp:
        reports.extend(
            _evaluate_checkpoint_pair(
                reactions,
                args.old_mlp_checkpoint,
                device,
                "old_mlp_local",
                "old_mlp_negotiated",
            )
        )

    for report in reports:
        save_benchmark_report(report, output_dir / f"{report.predictor_name}.json")
    comparison = compare_benchmark_reports(reports)
    _add_phase87_interpretation(comparison)
    save_benchmark_comparison(comparison, output_dir / "comparison.json")

    print(f"Loaded {len(reactions)} reactions from {args.data}")
    for report in reports:
        _print_report(report)
    phase = comparison.get("phase8_7", {})
    if isinstance(phase, dict):
        print("\nPhase 8.7 interpretation:")
        for key, value in phase.items():
            print(f"  {key}: {value}")
    print(f"\nComparison saved: {output_dir / 'comparison.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
