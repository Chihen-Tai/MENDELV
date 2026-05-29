"""Convert a real MD17/rMD17 NPZ and run a pretrained MACE-OFF benchmark."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mendel.md17 import convert_md17_npz_to_reference_json, save_md17_ingestion_report  # noqa: E402
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
        description="Run a real MD17/rMD17 conformer MACE-OFF energy/force benchmark.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output-reference",
        type=Path,
        default=_ROOT / "data" / "reference" / "md17_real_sample.reference.json",
    )
    parser.add_argument(
        "--sample-report",
        type=Path,
        default=_ROOT / "reports" / "md17_real_sample_report.json",
    )
    parser.add_argument(
        "--predictions-output",
        type=Path,
        default=_ROOT / "reports" / "mlip_md17_real_predictions.json",
    )
    parser.add_argument(
        "--benchmark-output",
        type=Path,
        default=_ROOT / "reports" / "mlip_md17_real_benchmark.json",
    )
    parser.add_argument("--max-records", type=int, default=100)
    parser.add_argument("--backend", default="mace")
    parser.add_argument("--model-family", default="mace-off")
    parser.add_argument("--model-name", default="mace-off-small")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--energy-unit", default="kcal/mol")
    parser.add_argument("--force-unit", default="kcal/mol/Angstrom")
    parser.add_argument("--no-convert-units", action="store_true")
    parser.add_argument("--assume-units", action="store_true")
    return parser


def _predict_records(
    records: list,
    config: MLIPConfig,
    continue_on_error: bool,
    calculator: object | None,
    calculator_error: str | None = None,
) -> list[MLIPStructurePrediction]:
    predictions: list[MLIPStructurePrediction] = []
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
            if not continue_on_error:
                raise
            predictions.append(
                MLIPStructurePrediction(
                    structure_id=record.structure_id,
                    backend_name=config.backend_name,
                    model_name=config.model_name,
                    energy=None,
                    energy_unit="eV",
                    forces=None,
                    force_unit="eV/Angstrom",
                    success=False,
                    warnings=[str(exc)],
                    metadata={"error": str(exc)},
                )
            )
    return predictions


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if not args.input.exists():
        print(f"ERROR: input file does not exist: {args.input}", file=sys.stderr)
        return 1
    del args.seed
    sample_report = convert_md17_npz_to_reference_json(
        args.input,
        args.output_reference,
        max_records=args.max_records,
        energy_unit=args.energy_unit,
        force_unit=args.force_unit,
        convert_units=not args.no_convert_units,
        assume_units=args.assume_units,
    )
    sample_report.metadata["synthetic_test_data"] = False
    sample_report.metadata["units"] = (
        f"energy={sample_report.metadata['converted_energy_unit']}, "
        f"forces={sample_report.metadata['converted_force_unit']}"
    )
    sample_report.metadata["units_warning"] = (
        "Units are assumed from CLI/defaults; verify against dataset documentation."
    )
    save_md17_ingestion_report(sample_report, args.sample_report)
    records = load_reference_records_json(args.output_reference)
    config = MLIPConfig(
        backend_name=args.backend,
        model_family=args.model_family,
        model_name=args.model_name,
        device=args.device,
        dtype=args.dtype,
    )
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
    try:
        predictions = _predict_records(
            records,
            config,
            args.continue_on_error,
            calculator,
            calculator_error,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    save_mlip_predictions_json(predictions, args.predictions_output)
    benchmark = compute_energy_force_benchmark(records, predictions)
    benchmark.metadata.update({
        "reference_units_validated": True,
        "original_energy_unit": sample_report.metadata.get("original_energy_unit"),
        "original_force_unit": sample_report.metadata.get("original_force_unit"),
        "benchmark_energy_unit": sample_report.metadata.get("converted_energy_unit"),
        "benchmark_force_unit": sample_report.metadata.get("converted_force_unit"),
        "unit_conversion_applied": sample_report.metadata.get("unit_conversion_applied"),
        "unit_conversion_factor_energy": sample_report.metadata.get("energy_conversion_factor"),
        "unit_conversion_factor_force": sample_report.metadata.get("force_conversion_factor"),
        "calculator_reused": calculator is not None,
        "calculator_initialized_once": calculator is not None,
        "pre_unit_validation_comparison_note": (
            "Older MD17/rMD17 benchmarks without explicit conversion should be treated as "
            "pre-unit-validation."
        ),
    })
    save_energy_force_benchmark_report(benchmark, args.benchmark_output)
    print(f"original energy unit: {sample_report.metadata.get('original_energy_unit')}")
    print(f"original force unit: {sample_report.metadata.get('original_force_unit')}")
    print(f"converted energy unit: {sample_report.metadata.get('converted_energy_unit')}")
    print(f"converted force unit: {sample_report.metadata.get('converted_force_unit')}")
    print(f"energy conversion factor: {sample_report.metadata.get('energy_conversion_factor')}")
    print(f"force conversion factor: {sample_report.metadata.get('force_conversion_factor')}")
    print(f"n structures: {benchmark.n_structures}")
    print(f"n success: {benchmark.n_success}")
    print(f"n failed: {benchmark.n_failed}")
    print(f"raw energy MAE/RMSE: {benchmark.energy_mae_raw} / {benchmark.energy_rmse_raw}")
    print(
        "mean-shifted energy MAE/RMSE: "
        f"{benchmark.energy_mae_mean_shifted} / {benchmark.energy_rmse_mean_shifted}"
    )
    print(f"force MAE/RMSE: {benchmark.force_mae} / {benchmark.force_rmse}")
    print(f"per-element force RMSE: {benchmark.per_element_force_rmse}")
    print(f"Reference: {args.output_reference}")
    print(f"Predictions: {args.predictions_output}")
    print(f"Benchmark: {args.benchmark_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
