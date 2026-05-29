"""Instructional QO2Mol downloader stub.

This script intentionally does not download the full QO2Mol dataset by default.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

SOURCE_URL = "https://github.com/saiscn/QO2Mol/"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Print safe QO2Mol download instructions; no default download.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--url")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--metadata-output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    print(f"QO2Mol source: {SOURCE_URL}")
    print("Do not commit raw QO2Mol data.")
    print("Verify dataset license before redistribution.")
    print("This stub does not download the full dataset by default.")
    if not args.url:
        print("Provide --url and --output-dir only for an explicitly selected small sample URL.")
        return 0
    if args.output_dir is None:
        print("ERROR: --output-dir is required when --url is provided.")
        return 1
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "url": args.url,
        "output_dir": str(args.output_dir),
        "created_at": datetime.now(UTC).isoformat(),
        "download_performed": False,
        "note": "Network download intentionally not implemented in this stub.",
    }
    if args.metadata_output:
        args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
        args.metadata_output.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Network download is not implemented in this safe stub.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
