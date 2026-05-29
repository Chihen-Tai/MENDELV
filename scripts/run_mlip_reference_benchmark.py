"""Run pretrained MLIP single-point benchmark on reference conformer records."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mendel.mlip import MLIPConfig, compute_mlip_singlepoint, create_mlip_calculator  # noqa: E402
from mendel.reference_data import (  # noqa: E402
    MLIPStructurePrediction,
    compute_energy_force_benchmark,
    load_reference_records_json,
    save_energy_force_benchmark_report,
    save_mlip_predictions_json,
    xyz_to_ase_atoms,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark pretrained MLIP single-point energy/forces on reference data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=_ROOT / "data" / "reference" / "qo2mol_sample.reference.json",
    )
    parser.add_argument("--backend", default="mace")
    parser.add_argument("--model-family", default="mace-off")
    parser.add_argument("--model-name", default="mace-off-small")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--predictions-output",
        type=Path,
        default=_ROOT / "reports" / "mlip_qo2mol_predictions.json",
    )
    parser.add_argument(
        "--benchmark-output",
        type=Path,
        default=_ROOT / "reports" / "mlip_qo2mol_benchmark.json",
    )
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--continue-on-error", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if not args.reference.exists():
        print(f"ERROR: reference file does not exist: {args.reference}", file=sys.stderr)
        return 1
    records = load_reference_records_json(args.reference)
    if args.max_records is not None:
        records = records[: args.max_records]
    config = MLIPConfig(
        backend_name=args.backend,
        model_family=args.model_family,
        model_name=args.model_name,
        device=args.device,
    )
    predictions: list[MLIPStructurePrediction] = []
    try:
        calculator = create_mlip_calculator(config)
    except Exception as exc:
        if not args.continue_on_error:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        calculator = None
        calculator_error = str(exc)
    else:
        calculator_error = None
    for record in records:
        try:
            if calculator is None:
                raise RuntimeError(calculator_error or "MLIP calculator unavailable")
            atoms = xyz_to_ase_atoms(record)
            result = compute_mlip_singlepoint(atoms, config, calculator=calculator)
            predictions.append(
                MLIPStructurePrediction(
                    structure_id=record.structure_id,
                    backend_name=result.backend_name,
                    model_name=result.model_name,
                    energy=result.energy,
                    energy_unit=result.energy_unit,
                    forces=result.forces,
                    force_unit=result.force_unit,
                    success=result.success,
                    warnings=result.warnings,
                    metadata=result.metadata,
                )
            )
        except Exception as exc:
            if not args.continue_on_error:
                print(f"ERROR: {exc}", file=sys.stderr)
                return 1
            predictions.append(
                MLIPStructurePrediction(
                    structure_id=record.structure_id,
                    backend_name=args.backend,
                    model_name=args.model_name,
                    energy=None,
                    energy_unit="eV",
                    forces=None,
                    force_unit="eV/Angstrom",
                    success=False,
                    warnings=[str(exc)],
                    metadata={"error": str(exc)},
                )
            )
    save_mlip_predictions_json(predictions, args.predictions_output)
    report = compute_energy_force_benchmark(records, predictions)
    report.metadata["calculator_reused"] = calculator is not None
    report.metadata["calculator_initialized_once"] = calculator is not None
    save_energy_force_benchmark_report(report, args.benchmark_output)
    print(f"n structures: {report.n_structures}")
    print(f"n success: {report.n_success}")
    print(f"n failed: {report.n_failed}")
    print(f"energy MAE/RMSE: {report.energy_mae} / {report.energy_rmse}")
    print(f"force MAE/RMSE: {report.force_mae} / {report.force_rmse}")
    print(f"per-element force RMSE: {report.per_element_force_rmse}")
    print(f"Predictions: {args.predictions_output}")
    print(f"Benchmark: {args.benchmark_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
