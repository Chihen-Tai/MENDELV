"""Tests for Phase 10.1 QO2Mol sample manager."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from mendel.qo2mol_manager import (
    create_sample_plan,
    execute_sample_plan,
    inspect_qo2mol_source,
    load_source_registry,
    save_source_registry,
    summarize_reference_sample,
)
from mendel.reference_data import load_reference_records_json

_ROOT = Path(__file__).parent.parent
_SCRIPT = _ROOT / "scripts" / "qo2mol_sample_manager.py"


def _sample_record(structure_id: str, atoms: list[str] | None = None) -> dict[str, object]:
    atoms = atoms or ["C", "H", "H"]
    return {
        "structure_id": structure_id,
        "molecule_id": "mol",
        "smiles": "C",
        "xyz": [[atom, float(idx), 0.0, 0.0] for idx, atom in enumerate(atoms)],
        "energy": -1.0,
        "energy_unit": "eV",
        "forces": [[0.0, 0.0, 0.0] for _ in atoms],
        "force_unit": "eV/Angstrom",
    }


def _write_sample(path: Path) -> None:
    path.write_text(
        json.dumps({
            "records": [
                _sample_record("c1", ["C", "H", "H"]),
                _sample_record("o1", ["O", "H"]),
                _sample_record("cl1", ["Cl", "C"]),
            ]
        }),
        encoding="utf-8",
    )


def test_inspect_qo2mol_source_on_tiny_json_file(tmp_path: Path) -> None:
    sample = tmp_path / "sample.json"
    _write_sample(sample)

    source = inspect_qo2mol_source(sample)

    assert source.detected_format == "json"
    assert source.n_files == 1
    assert source.has_coordinates is True
    assert source.has_energy is True
    assert source.has_forces is True


def test_inspect_qo2mol_source_on_directory(tmp_path: Path) -> None:
    _write_sample(tmp_path / "a.json")
    _write_sample(tmp_path / "b.json")

    source = inspect_qo2mol_source(tmp_path)

    assert source.detected_format == "directory"
    assert source.n_files == 2
    assert source.total_size_bytes is not None


def test_create_sample_plan_serializes(tmp_path: Path) -> None:
    sample = tmp_path / "sample.json"
    _write_sample(sample)
    source = inspect_qo2mol_source(sample)

    plan = create_sample_plan(source, tmp_path / "out.json", max_records=2, strategy="first_n")

    payload = plan.to_dict()
    assert payload["max_records"] == 2
    assert payload["metadata"]["dataset_name"] == "QO2Mol"


def test_execute_sample_plan_respects_max_records(tmp_path: Path) -> None:
    sample = tmp_path / "sample.json"
    _write_sample(sample)
    source = inspect_qo2mol_source(sample)
    plan = create_sample_plan(source, tmp_path / "out.json", max_records=2, strategy="first_n")

    records, report = execute_sample_plan(plan)

    assert len(records) == 2
    assert report.n_records_written == 2


def test_execute_sample_plan_respects_element_filter(tmp_path: Path) -> None:
    sample = tmp_path / "sample.json"
    _write_sample(sample)
    source = inspect_qo2mol_source(sample)
    plan = create_sample_plan(
        source,
        tmp_path / "out.json",
        max_records=10,
        strategy="element_filtered",
        element_filter=["O"],
    )

    records, report = execute_sample_plan(plan)

    assert [record.structure_id for record in records] == ["o1"]
    assert report.element_distribution["O"] == 1


def test_summarize_reference_sample_returns_element_distribution(tmp_path: Path) -> None:
    sample = tmp_path / "sample.json"
    _write_sample(sample)
    source = inspect_qo2mol_source(sample)
    records, _ = execute_sample_plan(create_sample_plan(source, tmp_path / "out.json"))

    summary = summarize_reference_sample(records)

    assert summary["n_records"] == 3
    assert summary["element_distribution"]["C"] == 2
    assert summary["n_missing_energy"] == 0


def test_source_registry_save_load(tmp_path: Path) -> None:
    sample = tmp_path / "sample.json"
    registry = tmp_path / "registry.json"
    _write_sample(sample)
    source = inspect_qo2mol_source(sample)

    save_source_registry([source], registry)

    loaded = load_source_registry(registry)
    assert loaded[0].source_id == source.source_id


def test_cli_inspect_writes_registry(tmp_path: Path) -> None:
    sample = tmp_path / "sample.json"
    registry = tmp_path / "registry.json"
    _write_sample(sample)

    result = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "inspect",
            "--input",
            str(sample),
            "--registry",
            str(registry),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert registry.exists()


def test_cli_sample_writes_reference_output_and_report(tmp_path: Path) -> None:
    sample = tmp_path / "sample.json"
    output = tmp_path / "out.reference.json"
    report = tmp_path / "sample_report.json"
    _write_sample(sample)

    result = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "sample",
            "--input",
            str(sample),
            "--output",
            str(output),
            "--report",
            str(report),
            "--max-records",
            "2",
            "--strategy",
            "first_n",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert len(load_reference_records_json(output)) == 2
    assert report.exists()


def test_cli_summarize_writes_summary_output(tmp_path: Path) -> None:
    sample = tmp_path / "sample.json"
    output = tmp_path / "out.reference.json"
    summary = tmp_path / "summary.json"
    _write_sample(sample)
    source = inspect_qo2mol_source(sample)
    records, _ = execute_sample_plan(create_sample_plan(source, output))
    from mendel.reference_data import save_reference_records_json

    save_reference_records_json(records, output)

    result = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "summarize",
            "--reference",
            str(output),
            "--output",
            str(summary),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(summary.read_text(encoding="utf-8"))["n_records"] == 3


def test_unsupported_source_fails_clearly(tmp_path: Path) -> None:
    unsupported = tmp_path / "sample.txt"
    unsupported.write_text("unsupported", encoding="utf-8")
    source = inspect_qo2mol_source(unsupported)
    plan = create_sample_plan(source, tmp_path / "out.json")

    with pytest.raises(ValueError, match="unsupported"):
        execute_sample_plan(plan)


def test_no_web_download_or_training_invoked() -> None:
    import mendel.qo2mol_manager as manager

    text = Path(manager.__file__).read_text(encoding="utf-8").lower()
    script_text = _SCRIPT.read_text(encoding="utf-8").lower()
    assert "requests.get" not in text
    assert "urlretrieve" not in text
    assert "train_mlp" not in script_text
