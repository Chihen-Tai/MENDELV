"""CLI script: Phase 8.8 MLP diagnostics.

Analyzes saved benchmark reports only. Does not train.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mendel.diagnostics import build_diagnostics_report, save_diagnostics_report  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Diagnose MLP role/center failures from saved benchmark reports.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=_ROOT / "data" / "reactions.proposed_with_auto_promoted.normalized.json",
    )
    parser.add_argument(
        "--rule-report",
        type=Path,
        default=_ROOT / "reports" / "benchmark_promoted_full" / "rule_based_negotiated.json",
    )
    parser.add_argument(
        "--new-mlp-report",
        type=Path,
        default=_ROOT / "reports" / "benchmark_promoted_full" / "new_mlp_negotiated.json",
    )
    parser.add_argument(
        "--old-mlp-report",
        type=Path,
        default=_ROOT / "reports" / "benchmark_promoted_full" / "old_mlp_negotiated.json",
    )
    parser.add_argument(
        "--comparison",
        type=Path,
        default=_ROOT / "reports" / "benchmark_promoted_full" / "comparison.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_ROOT / "reports" / "mlp_diagnostics_report.json",
    )
    parser.add_argument("--f1-threshold", type=float, default=0.8)
    return parser


def _print_summary(report) -> None:
    payload = report.to_dict()
    failures = report.reaction_center_failures
    high_conf_wrong = report.disagreement_counts.get("new_mlp_high_confidence_wrong", 0)
    mechanism_counts = Counter(failure.mechanism_type for failure in failures)

    print("Phase 8.8 MLP diagnostics")
    print(f"  dataset: {report.dataset_path}")
    print(f"  n_reactions: {report.n_reactions}")
    print(f"  n_group_labels: {report.n_group_labels}")
    print("  role accuracy:")
    for name, value in sorted(report.role_accuracy_summary.items()):
        print(f"    {name}: {value:.4f}")
    print("  reaction-center F1:")
    for name, metrics in sorted(report.reaction_center_summary.items()):
        print(f"    {name}: {metrics.get('f1')}")
    predictor_names = list(report.predictor_names)
    target_predictor = predictor_names[1] if len(predictor_names) > 1 else "new_mlp_negotiated"
    mlp_failures = sum(
        1 for failure in failures
        if failure.predictor_name == target_predictor
    )
    print(f"  MLP reaction-center failures: {mlp_failures}")
    print(f"  high-confidence wrong predictions: {high_conf_wrong}")
    print("  top failure mechanisms:")
    for mechanism, count in mechanism_counts.most_common(5):
        print(f"    {mechanism}: {count}")
    print("  recommendations:")
    for recommendation in payload["recommendations"]:
        print(f"    - {recommendation}")


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    del args.comparison
    try:
        report = build_diagnostics_report(
            dataset_path=args.data,
            rule_report_path=args.rule_report,
            new_mlp_report_path=args.new_mlp_report,
            old_mlp_report_path=args.old_mlp_report,
            f1_threshold=args.f1_threshold,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    save_diagnostics_report(report, args.output)
    _print_summary(report)
    print(f"Diagnostics report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
