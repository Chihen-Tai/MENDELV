"""Tests for Phase 10 QO2Mol local sample ingestion."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mendel.qo2mol import (
    convert_qo2mol_sample_to_reference_json,
    inspect_qo2mol_path,
    load_qo2mol_sample,
    qo2mol_metadata,
)
from mendel.reference_data import load_reference_records_json


def _sample_record(structure_id: str = "q1") -> dict[str, object]:
    return {
        "structure_id": structure_id,
        "molecule_id": "mol1",
        "smiles": "O",
        "xyz": [["O", 0.0, 0.0, 0.0], ["H", 0.0, 0.0, 1.0]],
        "charge": 0,
        "multiplicity": 1,
        "energy": -76.0,
        "energy_unit": "eV",
        "forces": [[0.0, 0.0, 0.1], [0.0, 0.0, -0.1]],
        "force_unit": "eV/Angstrom",
    }


def test_qo2mol_metadata_contains_required_source_fields() -> None:
    metadata = qo2mol_metadata()

    assert metadata["dataset_name"] == "QO2Mol"
    assert metadata["reference_method"] == "B3LYP/def2-SVP"
    assert metadata["source_url"] == "https://github.com/saiscn/QO2Mol/"


def test_inspect_qo2mol_path_on_json_file(tmp_path: Path) -> None:
    sample = tmp_path / "sample.json"
    sample.write_text(json.dumps({"records": [_sample_record()]}), encoding="utf-8")

    summary = inspect_qo2mol_path(sample)

    assert summary["exists"] is True
    assert summary["detected_format"] == "json"


def test_load_qo2mol_sample_from_json_and_jsonl(tmp_path: Path) -> None:
    json_path = tmp_path / "sample.json"
    jsonl_path = tmp_path / "sample.jsonl"
    json_path.write_text(
        json.dumps({"records": [_sample_record("a"), _sample_record("b")]}),
        encoding="utf-8",
    )
    jsonl_path.write_text(
        "\n".join(json.dumps(_sample_record(key)) for key in ("c", "d")),
        encoding="utf-8",
    )

    assert len(load_qo2mol_sample(json_path, max_records=1)) == 1
    assert len(load_qo2mol_sample(jsonl_path, max_records=2)) == 2
    assert load_qo2mol_sample(json_path, max_records=1)[0].dataset_name == "QO2Mol"


def test_convert_qo2mol_sample_to_reference_json(tmp_path: Path) -> None:
    sample = tmp_path / "sample.json"
    output = tmp_path / "reference.json"
    sample.write_text(json.dumps({"records": [_sample_record()]}), encoding="utf-8")

    report = convert_qo2mol_sample_to_reference_json(sample, output, max_records=100)

    assert report.n_records_written == 1
    assert load_reference_records_json(output)[0].metadata["dataset_name"] == "QO2Mol"


def test_unsupported_format_fails_clearly(tmp_path: Path) -> None:
    unsupported = tmp_path / "sample.txt"
    unsupported.write_text("not a supported sample", encoding="utf-8")

    with pytest.raises(ValueError, match="QO2Mol format adapter not implemented"):
        load_qo2mol_sample(unsupported)


def test_no_web_download_occurs() -> None:
    import mendel.qo2mol as qo2mol

    text = Path(qo2mol.__file__).read_text(encoding="utf-8")
    assert "requests.get" not in text
    assert "urlretrieve" not in text
