"""Analyze atom- and functional-group-local MLIP force errors."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mendel.local_force_analysis import (  # noqa: E402
    build_functional_group_force_analysis_report,
    save_functional_group_force_analysis_report,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze functional-group-local force errors from reference predictions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=_ROOT / "data" / "reference" / "rmd17_ethanol_sample_converted.reference.json",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=_ROOT / "reports" / "mlip_rmd17_ethanol_converted_predictions.json",
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=_ROOT / "reports" / "mlip_rmd17_ethanol_converted_benchmark.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_ROOT / "reports" / "functional_group_force_analysis.json",
    )
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--require-groups", action="store_true")
    parser.add_argument("--use-pseudo-groups", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if not args.reference.exists():
        print(f"ERROR: reference file does not exist: {args.reference}", file=sys.stderr)
        return 1
    if not args.predictions.exists():
        print(f"ERROR: predictions file does not exist: {args.predictions}", file=sys.stderr)
        return 1
    benchmark = args.benchmark if args.benchmark.exists() else None
    report = build_functional_group_force_analysis_report(
        args.reference,
        args.predictions,
        benchmark_path=benchmark,
        use_pseudo_groups=args.use_pseudo_groups,
    )
    if args.require_groups and not report.metadata.get("true_functional_groups_found"):
        print(
            "ERROR: No functional groups could be identified. "
            "Provide SMILES or omit --require-groups.",
            file=sys.stderr,
        )
        return 1
    save_functional_group_force_analysis_report(report, args.output)
    print(f"n structures: {report.n_structures}")
    print(f"n groups: {report.n_groups}")
    print(f"global force RMSE: {report.global_force_rmse}")
    print(f"per-element force RMSE: {report.per_element_force_rmse}")
    print("top functional group types by force RMSE:")
    for item in report.top_group_type_errors[: args.top_n]:
        print(f"  {item['group_type']}: {item['force_rmse']}")
    for failure in report.failures:
        print(f"warning: {failure.get('message')}")
    print(f"Output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
