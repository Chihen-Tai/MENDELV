"""Create a tiny synthetic reference dataset for benchmark math tests."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mendel.reference_data import (  # noqa: E402
    ReferenceStructureRecord,
    save_reference_records_json,
)


def _records() -> list[ReferenceStructureRecord]:
    metadata = {
        "synthetic_test_data": True,
        "not_scientific_reference": True,
        "scope_note": "Tiny made-up data for software tests only.",
    }
    return [
        ReferenceStructureRecord(
            structure_id="tiny_water_0",
            molecule_id="tiny_water",
            dataset_name="synthetic_tiny_reference",
            smiles="O",
            xyz=[
                ("O", 0.0, 0.0, 0.0),
                ("H", 0.0, 0.8, 0.6),
                ("H", 0.0, -0.8, 0.6),
            ],
            charge=0,
            multiplicity=1,
            reference_energy=-76.0,
            reference_energy_unit="eV",
            reference_forces=[[0.0, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, -0.1, 0.0]],
            reference_force_unit="eV/Angstrom",
            reference_method="synthetic",
            split="test",
            metadata=dict(metadata),
        ),
        ReferenceStructureRecord(
            structure_id="tiny_methane_0",
            molecule_id="tiny_methane",
            dataset_name="synthetic_tiny_reference",
            smiles="C",
            xyz=[
                ("C", 0.0, 0.0, 0.0),
                ("H", 0.6, 0.6, 0.6),
                ("H", -0.6, -0.6, 0.6),
            ],
            charge=0,
            multiplicity=1,
            reference_energy=-40.0,
            reference_energy_unit="eV",
            reference_forces=[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]],
            reference_force_unit="eV/Angstrom",
            reference_method="synthetic",
            split="test",
            metadata=dict(metadata),
        ),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create tiny synthetic reference JSON; not scientific data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_ROOT / "data" / "reference" / "tiny_reference_example.json",
    )
    args = parser.parse_args(argv)
    save_reference_records_json(_records(), args.output)
    print(f"Wrote {len(_records())} synthetic records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
