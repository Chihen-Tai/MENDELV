"""Safe Phase 10.2a QO2Mol small-sample fetcher.

The default mode writes a report only. Network download happens only when
``--allow-download`` is provided, and raw files are written under ignored
external-data paths.
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

OFFICIAL_REPO_URL = "https://github.com/saiscn/QO2Mol/"
LICENSE_NOTE = "verify from QO2Mol source repository before redistribution"
NO_SAMPLE_MESSAGE = (
    "No official small sample was automatically found. Please download a QO2Mol "
    "sample manually and run qo2mol_sample_manager.py inspect."
)
NO_METADATA_SAMPLE_MESSAGE = "No official small sample found in repository metadata."
SAMPLE_EXTENSIONS = (".json", ".jsonl", ".npz", ".npy", ".xyz", ".extxyz", ".csv")
README_EXTENSIONS = (".md", ".rst", ".txt")
SAMPLE_HINTS = ("sample", "demo", "example", "test", "tiny", "subset")
LIGHTWEIGHT_TEXT_MAX_BYTES = 2 * 1024 * 1024


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Safely fetch or prepare a small QO2Mol sample.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--repo-url", default=OFFICIAL_REPO_URL)
    parser.add_argument("--output-dir", type=Path, default=Path("data/external/qo2mol_sample"))
    parser.add_argument("--max-size-mb", type=float, default=200.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--url", help="Explicit user-provided official small sample URL.")
    parser.add_argument("--report", type=Path, default=Path("reports/qo2mol_fetch_report.json"))
    return parser


def _access_date() -> str:
    return datetime.now(UTC).isoformat()


def _base_report(args: argparse.Namespace) -> dict[str, object]:
    return {
        "dataset_name": "QO2Mol",
        "source_url": args.url or args.repo_url,
        "repo_url": args.repo_url,
        "access_date": _access_date(),
        "output_dir": str(args.output_dir),
        "max_size_mb": args.max_size_mb,
        "download_allowed": bool(args.allow_download),
        "download_performed": False,
        "downloaded_files": [],
        "candidates": [],
        "candidate_urls": [],
        "file_size_bytes": None,
        "license_note": LICENSE_NOTE,
        "do_not_commit_raw_data": True,
        "message": "",
        "warnings": [
            "Do not commit raw QO2Mol data.",
            "Verify dataset license before redistribution.",
            "This fetcher does not download the full QO2Mol dataset by default.",
        ],
        "metadata": {
            "phase": "10.2a",
            "data_type": "molecular conformer energy/force",
            "no_training": True,
        },
    }


def _save_report(report: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def _github_api_base(repo_url: str) -> str | None:
    parsed = urllib.parse.urlparse(repo_url.rstrip("/"))
    if parsed.netloc.lower() != "github.com":
        return None
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < 2:
        return None
    owner, repo = parts[0], parts[1]
    return f"https://api.github.com/repos/{owner}/{repo}/contents"


def _safe_filename_from_url(url: str) -> str:
    name = Path(urllib.parse.urlparse(url).path).name
    return name or "qo2mol_sample"


def _read_json_url(url: str, timeout: float = 10.0) -> object:
    request = urllib.request.Request(url, headers={"User-Agent": "MENDELV-QO2Mol-fetcher"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _read_text_url(url: str, timeout: float = 10.0) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "MENDELV-QO2Mol-fetcher"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read(LIGHTWEIGHT_TEXT_MAX_BYTES).decode("utf-8", errors="replace")


def _content_length_or_none(url: str) -> int | None:
    try:
        return _content_length(url)
    except (OSError, urllib.error.URLError):
        return None


def _candidate_score(item: dict[str, object]) -> tuple[int, str]:
    url = str(item.get("url", "")).lower()
    source_file = str(item.get("source_file", "")).lower()
    hinted = any(hint in url or hint in source_file for hint in SAMPLE_HINTS)
    supported = urllib.parse.urlparse(url).path.lower().endswith(SAMPLE_EXTENSIONS)
    size = int(item.get("estimated_size_bytes") or 0)
    score = 0
    if supported:
        score -= 10
    if hinted:
        score -= 20
    score += min(size // 1024, 10000)
    return score, source_file


def _is_sample_like(text: str) -> bool:
    lowered = text.lower()
    return any(hint in lowered for hint in SAMPLE_HINTS)


def _is_supported_sample_url(url: str) -> bool:
    path = urllib.parse.urlparse(url).path.lower()
    return path.endswith(SAMPLE_EXTENSIONS) and _is_sample_like(path)


def _candidate_from_file_item(item: dict[str, Any]) -> dict[str, object] | None:
    name = str(item.get("name", ""))
    path = str(item.get("path", name))
    lowered = f"{name} {path}".lower()
    download_url = item.get("download_url")
    if not isinstance(download_url, str):
        return None
    if not path.lower().endswith(SAMPLE_EXTENSIONS):
        return None
    if not _is_sample_like(lowered):
        return None
    return {
        "url": download_url,
        "estimated_size_bytes": int(item.get("size") or 0),
        "source_file": path,
        "found_via": "github_repository_file_listing",
    }


_URL_RE = re.compile(r"https?://[^\s)\"'>]+")


def _candidates_from_text(source_file: str, text: str) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    for match in _URL_RE.finditer(text):
        url = match.group(0).rstrip(".,;")
        if not _is_supported_sample_url(url):
            continue
        candidates.append({
            "url": url,
            "estimated_size_bytes": _content_length_or_none(url),
            "source_file": source_file,
            "found_via": "repository_text_link",
        })
    return candidates


def _dedupe_candidates(candidates: list[dict[str, object]]) -> list[dict[str, object]]:
    seen: set[str] = set()
    unique: list[dict[str, object]] = []
    for candidate in candidates:
        url = str(candidate.get("url", ""))
        if not url or url in seen:
            continue
        seen.add(url)
        unique.append(candidate)
    return sorted(unique, key=_candidate_score)


def discover_official_small_sample(
    repo_url: str,
    max_size_bytes: int,
    max_dirs: int = 25,
) -> dict[str, object] | None:
    """Inspect official GitHub metadata for the best small sample-like file."""
    candidates = discover_official_sample_candidates(repo_url, max_size_bytes, max_dirs)
    return candidates[0] if candidates else None


def discover_official_sample_candidates(
    repo_url: str,
    max_size_bytes: int,
    max_dirs: int = 25,
) -> list[dict[str, object]]:
    """Inspect lightweight official GitHub metadata and text files for sample candidates."""
    api_base = _github_api_base(repo_url)
    if api_base is None:
        return []
    queue = [api_base]
    seen_dirs = 0
    candidates: list[dict[str, object]] = []
    while queue and seen_dirs < max_dirs:
        url = queue.pop(0)
        seen_dirs += 1
        try:
            payload = _read_json_url(url)
        except (OSError, urllib.error.URLError, json.JSONDecodeError):
            continue
        if not isinstance(payload, list):
            continue
        for item in payload:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            name = str(item.get("name", ""))
            if item_type == "dir":
                lowered = name.lower()
                if any(hint in lowered for hint in ("sample", "demo", "example", "test", "data")):
                    next_url = item.get("url")
                    if isinstance(next_url, str):
                        queue.append(next_url)
                continue
            if item_type != "file":
                continue
            size = int(item.get("size") or 0)
            download_url = item.get("download_url")
            candidate = _candidate_from_file_item(item)
            if (
                candidate is not None
                and int(candidate["estimated_size_bytes"] or 0) <= max_size_bytes
            ):
                candidates.append(candidate)
                continue
            path = str(item.get("path", name))
            if (
                isinstance(download_url, str)
                and path.lower().endswith(README_EXTENSIONS)
                and size <= LIGHTWEIGHT_TEXT_MAX_BYTES
            ):
                try:
                    text = _read_text_url(download_url)
                except (OSError, urllib.error.URLError, UnicodeDecodeError):
                    continue
                for linked_candidate in _candidates_from_text(path, text):
                    linked_size = linked_candidate.get("estimated_size_bytes")
                    if linked_size is None or int(linked_size) <= max_size_bytes:
                        candidates.append(linked_candidate)
    return _dedupe_candidates(candidates)


def _content_length(url: str, timeout: float = 10.0) -> int | None:
    request = urllib.request.Request(
        url,
        method="HEAD",
        headers={"User-Agent": "MENDELV-QO2Mol-fetcher"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        header = response.getheader("Content-Length")
    return int(header) if header is not None and header.isdigit() else None


def _download_file(url: str, output_dir: Path, max_size_bytes: int) -> tuple[Path, int]:
    request = urllib.request.Request(url, headers={"User-Agent": "MENDELV-QO2Mol-fetcher"})
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / _safe_filename_from_url(url)
    bytes_written = 0
    with (
        urllib.request.urlopen(request, timeout=30.0) as response,
        output_path.open("wb") as handle,
    ):
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            bytes_written += len(chunk)
            if bytes_written > max_size_bytes:
                output_path.unlink(missing_ok=True)
                raise ValueError(
                    f"download exceeded max size of {max_size_bytes} bytes; refusing file"
                )
            handle.write(chunk)
    return output_path, bytes_written


def _handle_explicit_url(args: argparse.Namespace, report: dict[str, object]) -> int:
    max_size_bytes = int(args.max_size_mb * 1024 * 1024)
    try:
        size = _content_length(args.url)
    except (OSError, urllib.error.URLError) as exc:
        report["message"] = f"Could not inspect URL size: {exc}"
        return 1
    report["file_size_bytes"] = size
    if size is not None and size > max_size_bytes:
        report["message"] = (
            f"Refusing download: Content-Length {size} bytes exceeds "
            f"--max-size-mb limit ({max_size_bytes} bytes)."
        )
        return 1
    if not args.allow_download or args.dry_run:
        report["message"] = "Dry run complete; explicit URL was inspected but not downloaded."
        return 0
    try:
        output_path, bytes_written = _download_file(
            args.url,
            args.output_dir,
            max_size_bytes,
        )
    except (OSError, urllib.error.URLError, ValueError) as exc:
        report["message"] = f"Download failed or was refused: {exc}"
        return 1
    report["download_performed"] = True
    report["downloaded_files"] = [str(output_path)]
    report["file_size_bytes"] = bytes_written
    report["message"] = "Downloaded explicit QO2Mol sample URL."
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = _base_report(args)
    if args.url:
        exit_code = _handle_explicit_url(args, report)
        _save_report(report, args.report)
        print(report["message"])
        print(f"Report: {args.report}")
        return exit_code
    max_size_bytes = int(args.max_size_mb * 1024 * 1024)
    candidates = discover_official_sample_candidates(args.repo_url, max_size_bytes)
    report["candidates"] = candidates
    report["candidate_urls"] = [candidate["url"] for candidate in candidates]
    if not candidates:
        report["message"] = NO_METADATA_SAMPLE_MESSAGE
        _save_report(report, args.report)
        print(report["message"])
        print(f"Report: {args.report}")
        return 0
    candidate = candidates[0]
    report["source_url"] = candidate["url"]
    report["file_size_bytes"] = candidate.get("estimated_size_bytes")
    if args.dry_run or not args.allow_download:
        report["message"] = "Official small-sample candidate found; dry run did not download."
        report["metadata"]["selected_candidate"] = candidate  # type: ignore[index]
        _save_report(report, args.report)
        print(report["message"])
        print(f"Candidate: {candidate['url']}")
        print(f"Report: {args.report}")
        return 0
    try:
        output_path, bytes_written = _download_file(
            str(candidate["url"]),
            args.output_dir,
            max_size_bytes,
        )
    except (OSError, urllib.error.URLError, ValueError) as exc:
        report["message"] = f"Download failed or was refused: {exc}"
        _save_report(report, args.report)
        print(report["message"])
        print(f"Report: {args.report}")
        return 1
    report["download_performed"] = True
    report["downloaded_files"] = [str(output_path)]
    report["file_size_bytes"] = bytes_written
    report["message"] = "Downloaded official QO2Mol small-sample candidate."
    _save_report(report, args.report)
    print(report["message"])
    print(f"Output directory: {args.output_dir}")
    print(f"Report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
