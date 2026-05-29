"""Safe helper for acquiring a local MD17/rMD17 NPZ sample."""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

DEFAULT_OUTPUT_DIR = Path("data/external/md17")
DEFAULT_REPORT = Path("reports/md17_fetch_report.json")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Safely download a user-selected small MD17/rMD17 NPZ file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--url")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--max-size-mb", type=float, default=500.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser


def _base_report(args: argparse.Namespace) -> dict[str, object]:
    return {
        "dataset_name": "MD17/rMD17",
        "source_url": args.url,
        "access_date": datetime.now(UTC).isoformat(),
        "output_dir": str(args.output_dir),
        "max_size_mb": args.max_size_mb,
        "download_allowed": bool(args.allow_download),
        "download_performed": False,
        "downloaded_files": [],
        "file_size_bytes": None,
        "license_note": "verify source dataset license before redistribution",
        "do_not_commit_raw_data": True,
        "message": "",
        "warnings": [
            "Do not commit raw MD17/rMD17 data.",
            "Use only official project URLs or a user-provided source.",
            "This helper does not download anything unless --allow-download is provided.",
        ],
    }


def _save_report(report: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def _content_length(url: str) -> int | None:
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "MENDELV-MD17"})
    with urllib.request.urlopen(request, timeout=10.0) as response:
        header = response.getheader("Content-Length")
    return int(header) if header is not None and header.isdigit() else None


def _download(url: str, output_dir: Path, max_size_bytes: int) -> tuple[Path, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    name = Path(url.split("?", maxsplit=1)[0]).name or "md17_sample.npz"
    output = output_dir / name
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
                raise ValueError("download exceeds --max-size-mb")
            handle.write(chunk)
    return output, written


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = _base_report(args)
    if not args.url:
        report["message"] = (
            "No URL provided. Download an MD17/rMD17 NPZ manually, or rerun with "
            "--url and --allow-download after verifying source and license."
        )
        _save_report(report, args.report)
        print(report["message"])
        print(f"Report: {args.report}")
        return 0
    max_size_bytes = int(args.max_size_mb * 1024 * 1024)
    try:
        size = _content_length(args.url)
    except (OSError, urllib.error.URLError) as exc:
        report["message"] = f"Could not inspect URL size: {exc}"
        _save_report(report, args.report)
        print(report["message"])
        print(f"Report: {args.report}")
        return 1
    report["file_size_bytes"] = size
    if size is not None and size > max_size_bytes:
        report["message"] = (
            f"Refusing download: Content-Length {size} bytes exceeds "
            f"--max-size-mb limit ({max_size_bytes} bytes)."
        )
        _save_report(report, args.report)
        print(report["message"])
        print(f"Report: {args.report}")
        return 1
    if not args.allow_download:
        report["message"] = "URL inspected; no download performed without --allow-download."
        _save_report(report, args.report)
        print(report["message"])
        print(f"Report: {args.report}")
        return 0
    try:
        output, written = _download(args.url, args.output_dir, max_size_bytes)
    except (OSError, urllib.error.URLError, ValueError) as exc:
        report["message"] = f"Download failed or was refused: {exc}"
        _save_report(report, args.report)
        print(report["message"])
        print(f"Report: {args.report}")
        return 1
    report["download_performed"] = True
    report["downloaded_files"] = [str(output)]
    report["file_size_bytes"] = written
    report["message"] = "Downloaded explicit MD17/rMD17 NPZ URL."
    _save_report(report, args.report)
    print(report["message"])
    print(f"Output: {output}")
    print(f"Report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
