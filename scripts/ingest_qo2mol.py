"""CLI: ingest a local QO2Mol sample into MENDELV reference JSON."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mendel.qo2mol import (  # noqa: E402
    convert_qo2mol_sample_to_reference_json,
    inspect_qo2mol_path,
    save_qo2mol_ingestion_report,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect or ingest a local QO2Mol sample. No downloads are performed.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=_ROOT / "data" / "reference" / "qo2mol_sample.reference.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=_ROOT / "reports" / "qo2mol_ingestion_report.json",
    )
    parser.add_argument("--max-records", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--inspect-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    summary = inspect_qo2mol_path(args.input)
    print(f"path: {summary['path']}")
    print(f"exists: {summary['exists']}")
    print(f"detected_format: {summary['detected_format']}")
    print(f"supported_for_loading: {summary['supported_for_loading']}")
    if args.inspect_only:
        return 0
    if not args.input.exists():
        print(f"ERROR: input does not exist: {args.input}", file=sys.stderr)
        return 1
    try:
        report = convert_qo2mol_sample_to_reference_json(
            args.input,
            args.output,
            max_records=args.max_records,
            seed=args.seed,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    save_qo2mol_ingestion_report(report, args.report)
    print(f"n_records_written: {report.n_records_written}")
    print(f"Output: {args.output}")
    print(f"Report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
