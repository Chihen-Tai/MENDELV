"""Tests for the safe Phase 10.2a QO2Mol sample fetcher."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

_ROOT = Path(__file__).parent.parent
_SCRIPT = _ROOT / "scripts" / "fetch_qo2mol_sample.py"


def _load_fetcher_module() -> Any:
    import importlib.util

    spec = importlib.util.spec_from_file_location("fetch_qo2mol_sample", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fetch_qo2mol_sample_help_runs() -> None:
    result = subprocess.run(
        [sys.executable, str(_SCRIPT), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--allow-download" in result.stdout


def test_dry_run_does_not_download(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    output_dir = tmp_path / "sample"

    result = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--dry-run",
            "--output-dir",
            str(output_dir),
            "--report",
            str(report),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    payload = json.loads(report.read_text(encoding="utf-8"))
    assert result.returncode == 0
    assert payload["download_performed"] is False
    assert not output_dir.exists()


def test_missing_sample_is_handled_gracefully(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_fetcher_module()
    report = tmp_path / "report.json"
    monkeypatch.setattr(module, "discover_official_sample_candidates", lambda *_args: [])

    code = module.main([
        "--dry-run",
        "--report",
        str(report),
        "--output-dir",
        str(tmp_path / "sample"),
    ])

    payload = json.loads(report.read_text(encoding="utf-8"))
    assert code == 0
    assert payload["download_performed"] is False
    assert payload["message"] == "No official small sample found in repository metadata."


def test_dry_run_reports_mocked_github_candidates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_fetcher_module()

    def fake_read_json_url(url: str, timeout: float = 10.0) -> object:
        if url.endswith("/contents"):
            return [
                {
                    "type": "file",
                    "name": "sample.json",
                    "path": "examples/sample.json",
                    "size": 1234,
                    "download_url": "https://raw.githubusercontent.com/saiscn/QO2Mol/main/examples/sample.json",
                    "html_url": "https://github.com/saiscn/QO2Mol/blob/main/examples/sample.json",
                },
                {
                    "type": "file",
                    "name": "README.md",
                    "path": "README.md",
                    "size": 100,
                    "download_url": "https://raw.githubusercontent.com/saiscn/QO2Mol/main/README.md",
                    "html_url": "https://github.com/saiscn/QO2Mol/blob/main/README.md",
                },
            ]
        raise AssertionError(f"unexpected JSON URL: {url}")

    def fake_read_text_url(url: str, timeout: float = 10.0) -> str:
        assert url.endswith("README.md")
        return "Use https://github.com/saiscn/QO2Mol/raw/main/demo/test_subset.npz"

    monkeypatch.setattr(module, "_read_json_url", fake_read_json_url)
    monkeypatch.setattr(module, "_read_text_url", fake_read_text_url)
    monkeypatch.setattr(module, "_content_length_or_none", lambda _url: 2048)
    report = tmp_path / "report.json"

    code = module.main(["--dry-run", "--report", str(report)])

    payload = json.loads(report.read_text(encoding="utf-8"))
    assert code == 0
    assert payload["download_performed"] is False
    assert len(payload["candidate_urls"]) == 2
    assert payload["candidates"][0]["estimated_size_bytes"] == 1234
    assert payload["candidates"][0]["source_file"] == "examples/sample.json"
    assert payload["candidates"][1]["source_file"] == "README.md"


def test_explicit_url_larger_than_max_size_is_refused(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_fetcher_module()

    class FakeResponse:
        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, *_args: Any) -> None:
            return None

        def getheader(self, name: str, default: str | None = None) -> str | None:
            return str(2 * 1024 * 1024) if name.lower() == "content-length" else default

    def fake_urlopen(request: Any, timeout: float = 10.0) -> FakeResponse:
        return FakeResponse()

    monkeypatch.setattr(module.urllib.request, "urlopen", fake_urlopen)
    report = tmp_path / "report.json"
    code = module.main([
        "--url",
        "https://github.com/saiscn/QO2Mol/raw/main/sample.json",
        "--allow-download",
        "--max-size-mb",
        "1",
        "--report",
        str(report),
        "--output-dir",
        str(tmp_path / "sample"),
    ])

    payload = json.loads(report.read_text(encoding="utf-8"))
    assert code == 1
    assert payload["download_performed"] is False
    assert "exceeds" in payload["message"]


def test_report_json_is_written_for_dry_run(tmp_path: Path) -> None:
    report = tmp_path / "fetch_report.json"

    result = subprocess.run(
        [sys.executable, str(_SCRIPT), "--dry-run", "--report", str(report)],
        check=False,
        capture_output=True,
        text=True,
    )

    payload = json.loads(report.read_text(encoding="utf-8"))
    assert result.returncode == 0
    assert payload["license_note"] == "verify from QO2Mol source repository before redistribution"
    assert payload["do_not_commit_raw_data"] is True


def test_no_mlip_training_invoked() -> None:
    text = _SCRIPT.read_text(encoding="utf-8").lower()
    for token in ("train_mlp", "fit(", "neb", "irc", "transition1x", "dft"):
        assert token not in text
