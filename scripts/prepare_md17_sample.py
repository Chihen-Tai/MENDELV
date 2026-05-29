"""Prepare a small MD17/rMD17-style NPZ sample as MENDELV reference JSON."""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mendel.md17 import (  # noqa: E402
    convert_md17_npz_to_reference_json,
    create_tiny_synthetic_md17_npz,
    save_md17_ingestion_report,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a local MD17/rMD17 NPZ sample to MENDELV reference JSON.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path)
    parser.add_argument("--url", help="Explicit MD17/rMD17 NPZ URL; requires --allow-download.")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--max-size-mb", type=float, default=200.0)
    parser.add_argument(
        "--download-output",
        type=Path,
        default=_ROOT / "data" / "external" / "md17_sample" / "md17_sample.npz",
    )
    parser.add_argument(
        "--synthetic-npz",
        type=Path,
        default=_ROOT / "data" / "reference" / "md17_tiny_synthetic.npz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_ROOT / "data" / "reference" / "md17_sample.reference.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=_ROOT / "reports" / "md17_sample_report.json",
    )
    parser.add_argument("--max-records", type=int, default=100)
    parser.add_argument("--molecule-id")
    parser.add_argument("--energy-unit", default="kcal/mol")
    parser.add_argument("--force-unit", default="kcal/mol/Angstrom")
    parser.add_argument("--no-convert-units", action="store_true")
    parser.add_argument("--assume-units", action="store_true")
    return parser


def _content_length(url: str) -> int | None:
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "MENDELV-MD17"})
    with urllib.request.urlopen(request, timeout=10.0) as response:
        header = response.getheader("Content-Length")
    return int(header) if header is not None and header.isdigit() else None


def _download_explicit_url(url: str, output: Path, max_size_bytes: int) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "MENDELV-MD17"})
    written = 0
    with urllib.request.urlopen(request, timeout=30.0) as response, output.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            written += len(chunk)
            if written > max_size_bytes:
                output.unlink(missing_ok=True)
                raise ValueError("explicit MD17/rMD17 URL exceeds --max-size-mb")
            handle.write(chunk)
    return output


def _resolve_input(args: argparse.Namespace) -> tuple[Path, bool]:
    if args.input is not None:
        return args.input, False
    if args.url:
        if not args.allow_download:
            raise ValueError("--url requires --allow-download; no automatic download is performed.")
        max_size_bytes = int(args.max_size_mb * 1024 * 1024)
        size = _content_length(args.url)
        if size is not None and size > max_size_bytes:
            raise ValueError("explicit MD17/rMD17 URL exceeds --max-size-mb")
        return _download_explicit_url(args.url, args.download_output, max_size_bytes), False
    print(
        "No input provided; generating synthetic test data only. "
        "This is not scientific reference data."
    )
    create_tiny_synthetic_md17_npz(args.synthetic_npz)
    return args.synthetic_npz, True


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        input_path, synthetic = _resolve_input(args)
        report = convert_md17_npz_to_reference_json(
            input_path,
            args.output,
            max_records=args.max_records,
            molecule_id=args.molecule_id,
            energy_unit=args.energy_unit,
            force_unit=args.force_unit,
            convert_units=False if synthetic else not args.no_convert_units,
            assume_units=args.assume_units,
        )
        if synthetic:
            report.metadata["synthetic_test_data"] = True
            report.metadata["not_scientific_reference"] = True
            report.metadata["unit_conversion_applied"] = False
            report.metadata["synthetic_unit_policy"] = (
                "Synthetic test data only; units are not scientific."
            )
        if args.url:
            report.metadata["explicit_url"] = args.url
            report.metadata["raw_data_policy"] = "Do not commit raw MD17/rMD17 data."
        if not synthetic:
            report.metadata["synthetic_test_data"] = False
            report.metadata["units"] = (
                f"energy={report.metadata['converted_energy_unit']}, "
                f"forces={report.metadata['converted_force_unit']}"
            )
            report.metadata["units_warning"] = (
                "Units are assumed from CLI/defaults; verify against dataset documentation."
            )
        save_md17_ingestion_report(report, args.report)
    except (OSError, urllib.error.URLError, ValueError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"records written: {report.n_records_written}")
    print(f"original energy unit: {report.metadata.get('original_energy_unit')}")
    print(f"original force unit: {report.metadata.get('original_force_unit')}")
    print(f"converted energy unit: {report.metadata.get('converted_energy_unit')}")
    print(f"converted force unit: {report.metadata.get('converted_force_unit')}")
    print(f"unit conversion applied: {report.metadata.get('unit_conversion_applied')}")
    print(f"reference JSON: {args.output}")
    print(f"report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
