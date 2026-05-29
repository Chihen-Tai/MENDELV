"""CLI for safe local QO2Mol source inspection, sampling, and summarization."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mendel.qo2mol_manager import (  # noqa: E402
    create_sample_plan,
    execute_sample_plan,
    inspect_qo2mol_source,
    load_source_registry,
    save_reference_summary,
    save_sample_report,
    save_source_registry,
    summarize_reference_sample,
)
from mendel.reference_data import load_reference_records_json  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Manage local QO2Mol samples without downloading the full dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    inspect = sub.add_parser("inspect", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    inspect.add_argument("--input", type=Path, required=True)
    inspect.add_argument("--registry", type=Path, default=_ROOT / "reports" / "qo2mol_sources.json")

    sample = sub.add_parser("sample", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sample.add_argument("--input", type=Path, required=True)
    sample.add_argument(
        "--output",
        type=Path,
        default=_ROOT / "data" / "reference" / "qo2mol_sample.reference.json",
    )
    sample.add_argument(
        "--report",
        type=Path,
        default=_ROOT / "reports" / "qo2mol_sample_report.json",
    )
    sample.add_argument("--max-records", type=int, default=100)
    sample.add_argument("--seed", type=int, default=42)
    sample.add_argument(
        "--strategy",
        choices=("first_n", "random", "element_filtered", "small_molecule_first"),
        default="random",
    )
    sample.add_argument("--elements")
    sample.add_argument("--allow-missing-forces", action="store_true")
    sample.add_argument("--allow-missing-energy", action="store_true")

    summarize = sub.add_parser("summarize", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    summarize.add_argument("--reference", type=Path, required=True)
    summarize.add_argument("--output", type=Path)
    return parser


def _print_source(source) -> None:  # noqa: ANN001
    print(f"source_id: {source.source_id}")
    print(f"detected_format: {source.detected_format}")
    print(f"n_files: {source.n_files}")
    print(f"total_size_bytes: {source.total_size_bytes}")
    print(f"has_energy: {source.has_energy}")
    print(f"has_forces: {source.has_forces}")
    print(f"has_coordinates: {source.has_coordinates}")
    print(f"has_smiles: {source.has_smiles}")


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "inspect":
        source = inspect_qo2mol_source(args.input)
        existing = load_source_registry(args.registry) if args.registry.exists() else []
        sources = [item for item in existing if item.source_id != source.source_id]
        sources.append(source)
        save_source_registry(sources, args.registry)
        _print_source(source)
        print(f"Registry: {args.registry}")
        return 0

    if args.command == "sample":
        source = inspect_qo2mol_source(args.input)
        elements = (
            [part.strip() for part in args.elements.split(",") if part.strip()]
            if args.elements
            else None
        )
        plan = create_sample_plan(
            source,
            args.output,
            max_records=args.max_records,
            seed=args.seed,
            strategy=args.strategy,
            element_filter=elements,
            require_forces=not args.allow_missing_forces,
            require_energy=not args.allow_missing_energy,
        )
        try:
            records, report = execute_sample_plan(plan)
        except Exception as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        save_sample_report(report, args.report)
        summary = summarize_reference_sample(records)
        print(f"n_records_written: {report.n_records_written}")
        print(f"element_distribution: {summary['element_distribution']}")
        print(f"Output: {args.output}")
        print(f"Report: {args.report}")
        return 0

    if args.command == "summarize":
        records = load_reference_records_json(args.reference)
        summary = summarize_reference_sample(records)
        print(f"n_records: {summary['n_records']}")
        print(f"element_distribution: {summary['element_distribution']}")
        print(f"atom_count_distribution: {summary['atom_count_distribution']}")
        if args.output:
            save_reference_summary(summary, args.output)
            print(f"Summary: {args.output}")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
