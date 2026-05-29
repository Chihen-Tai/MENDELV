"""Generate Phase 10.5 energy/force benchmark figures."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mendel.plotting import (  # noqa: E402
    ensure_output_dir,
    load_energy_force_plot_inputs,
    plot_energy_parity,
    plot_energy_rmse_bar,
    plot_force_error_distribution,
    plot_force_rmse_by_element,
    plot_local_force_rmse_by_group,
    save_plot_report,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate energy/force comparison figures for a reference MLIP benchmark.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=_ROOT / "data" / "reference" / "rmd17_ethanol_sample_converted.reference.json",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=_ROOT / "reports" / "mlip_rmd17_ethanol_converted_predictions.json",
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=_ROOT / "reports" / "mlip_rmd17_ethanol_converted_benchmark.json",
    )
    parser.add_argument(
        "--local-analysis",
        type=Path,
        default=_ROOT / "reports" / "functional_group_force_analysis_ethanol.json",
    )
    parser.add_argument("--output-dir", type=Path, default=_ROOT / "reports" / "figures")
    parser.add_argument(
        "--report",
        type=Path,
        default=_ROOT / "reports" / "energy_force_plot_report.json",
    )
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--format", default="png", choices=["png"])
    return parser


def _require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} file does not exist: {path}")


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        _require_file(args.reference, "reference")
        _require_file(args.predictions, "predictions")
        _require_file(args.benchmark, "benchmark")
        local_analysis = args.local_analysis if args.local_analysis.exists() else None
        output_dir = ensure_output_dir(args.output_dir)
        inputs = load_energy_force_plot_inputs(
            args.reference,
            args.predictions,
            args.benchmark,
            local_analysis,
        )

        figure_paths = {
            "energy_parity_raw": output_dir / "energy_parity_raw.png",
            "energy_parity_mean_shifted": output_dir / "energy_parity_mean_shifted.png",
            "energy_rmse_bar": output_dir / "energy_rmse_bar.png",
            "force_rmse_by_element": output_dir / "force_rmse_by_element.png",
            "local_force_rmse_by_group": output_dir / "local_force_rmse_by_group.png",
            "force_error_distribution": output_dir / "force_error_distribution.png",
        }
        reference_energies = list(inputs["reference_energies"])  # type: ignore[arg-type]
        predicted_energies = list(inputs["predicted_energies"])  # type: ignore[arg-type]
        shifted_energies = list(inputs["shifted_predicted_energies"])  # type: ignore[arg-type]
        force_error_norms = list(inputs["force_error_norms"])  # type: ignore[arg-type]
        per_element = dict(inputs["per_element_force_rmse"])  # type: ignore[arg-type]
        per_group = dict(inputs["per_group_type_force_rmse"])  # type: ignore[arg-type]

        plot_energy_parity(
            reference_energies,
            predicted_energies,
            figure_paths["energy_parity_raw"],
            "Raw energy parity",
        )
        plot_energy_parity(
            reference_energies,
            shifted_energies,
            figure_paths["energy_parity_mean_shifted"],
            "Mean-shifted energy parity",
            shifted=True,
        )
        raw_rmse = float(inputs["raw_energy_rmse"] or 0.0)
        shifted_rmse = float(inputs["mean_shifted_energy_rmse"] or 0.0)
        global_force_rmse = float(inputs["global_force_rmse"] or 0.0)
        plot_energy_rmse_bar(raw_rmse, shifted_rmse, figure_paths["energy_rmse_bar"])
        plot_force_rmse_by_element(
            global_force_rmse,
            {str(k): float(v) for k, v in per_element.items()},
            figure_paths["force_rmse_by_element"],
        )
        plot_local_force_rmse_by_group(
            {str(k): float(v) for k, v in per_group.items()},
            figure_paths["local_force_rmse_by_group"],
            top_n=args.top_n,
        )
        plot_force_error_distribution(
            [float(value) for value in force_error_norms],
            figure_paths["force_error_distribution"],
        )

        top_local = sorted(
            [{"group_type": str(k), "force_rmse": float(v)} for k, v in per_group.items()],
            key=lambda item: item["force_rmse"],
            reverse=True,
        )[: args.top_n]
        report = {
            "input_paths": {
                "reference": str(args.reference),
                "predictions": str(args.predictions),
                "benchmark": str(args.benchmark),
                "local_analysis": str(local_analysis) if local_analysis is not None else None,
            },
            "output_figure_paths": {key: str(path) for key, path in figure_paths.items()},
            "n_structures": len(reference_energies),
            "raw_energy_rmse": raw_rmse,
            "mean_shifted_energy_rmse": shifted_rmse,
            "energy_offset": float(inputs["energy_offset"]),
            "force_rmse": global_force_rmse,
            "per_element_force_rmse": {str(k): float(v) for k, v in per_element.items()},
            "top_local_group_rmse": top_local,
            "notes": [
                "DFT reference vs pure MACE-OFF predictions are plotted on fixed conformers.",
                "MENDELV organizes local force errors; it does not correct MACE-OFF predictions.",
                "This is fixed-conformer energy/force analysis, not a reaction workflow.",
            ],
            "limitations": [
                "rMD17 ethanol lacks SMILES, so local groups may be pseudo-groups.",
                "No MLIP training or DFT calculation is performed.",
            ],
        }
        save_plot_report(report, args.report)
        print(f"n structures: {report['n_structures']}")
        print(f"raw energy RMSE: {raw_rmse}")
        print(f"mean-shifted energy RMSE: {shifted_rmse}")
        print(f"force RMSE: {global_force_rmse}")
        print(f"per-element force RMSE: {report['per_element_force_rmse']}")
        print(f"top local group RMSE: {top_local}")
        print(f"figures: {output_dir}")
        print(f"report: {args.report}")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
