"""Tests for Phase 8.12 center expansion scaffolding."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.generate_center_expansion_candidates import generate_center_expansion_candidates

_ROOT = Path(__file__).parent.parent
_GEN_SCRIPT = _ROOT / "scripts" / "generate_center_expansion_candidates.py"
_PIPELINE_SCRIPT = _ROOT / "scripts" / "run_center_expansion_pipeline.py"


def test_center_expansion_generator_creates_candidates() -> None:
    candidates = generate_center_expansion_candidates(max_count=24)

    assert candidates
    assert all(c["metadata"]["center_label_focus"] is True for c in candidates)


def test_center_expansion_mechanisms_include_core_set() -> None:
    candidates = generate_center_expansion_candidates(max_count=80)
    mechanisms = {c["metadata"]["mechanism_type"] for c in candidates}

    assert {"sn2", "e2", "diels_alder", "carbonyl_addition", "control"} <= mechanisms


def test_center_expansion_reaction_ids_are_unique() -> None:
    candidates = generate_center_expansion_candidates(max_count=80)
    ids = [c["reaction_id"] for c in candidates]

    assert len(ids) == len(set(ids))


def test_generator_cli_writes_output(tmp_path: Path) -> None:
    output = tmp_path / "draft_inputs.center_expansion.json"

    result = subprocess.run(
        [sys.executable, str(_GEN_SCRIPT), "--output", str(output), "--max-count", "12"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["candidates"]


def test_pipeline_smoke_creates_output_and_report(tmp_path: Path) -> None:
    output_dir = tmp_path / "data"
    report = tmp_path / "report.json"

    result = subprocess.run(
        [
            sys.executable,
            str(_PIPELINE_SCRIPT),
            "--output-dir",
            str(output_dir),
            "--report",
            str(report),
            "--max-count",
            "12",
            "--skip-promotion",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert (output_dir / "draft_inputs.center_expansion.json").exists()
    assert report.exists()


def test_no_external_data_required() -> None:
    text = _GEN_SCRIPT.read_text(encoding="utf-8").lower()
    for token in ("requests", "urllib", "http://", "https://"):
        assert token not in text
