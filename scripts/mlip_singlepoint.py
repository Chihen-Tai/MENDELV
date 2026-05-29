"""CLI for optional Phase 9 pretrained MLIP single-point calculations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mendel.mlip import (  # noqa: E402
    MLIPConfig,
    atoms_from_xyz,
    compute_force_norms,
    compute_mlip_singlepoint,
    run_mendel_guided_mlip_singlepoint,
    save_json,
    smiles_to_ase_atoms,
    summarize_reaction_center_forces,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run optional pretrained MLIP single-point energy/force calculation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--smiles")
    source.add_argument("--reaction-smiles")
    source.add_argument("--xyz", type=Path)
    parser.add_argument("--context", default="unknown")
    parser.add_argument("--backend", default="mace")
    parser.add_argument("--model-family", default="mace-off")
    parser.add_argument("--model-name", default="mace-off-small")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--output", type=Path, default=_ROOT / "reports" / "mlip_singlepoint.json")
    parser.add_argument("--reaction-center-from-mendelv", action="store_true")
    parser.add_argument("--center-source", default="auto")
    parser.add_argument("--optimize-geometry", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--fail-on-geometry-sanity",
        action="store_true",
        help="Exit nonzero if the geometry sanity status is fail after writing JSON output.",
    )
    parser.add_argument("--mean-force-threshold", type=float, default=100.0)
    parser.add_argument("--max-force-threshold", type=float, default=1000.0)
    parser.add_argument("--min-distance-threshold", type=float, default=0.6)
    return parser


def _geometry_sanity_from_payload(payload: dict[str, object]) -> dict[str, object] | None:
    mlip = payload.get("mlip_result", payload)
    if not isinstance(mlip, dict):
        return None
    metadata = mlip.get("metadata")
    if not isinstance(metadata, dict):
        return None
    sanity = metadata.get("geometry_sanity")
    return sanity if isinstance(sanity, dict) else None


def _print_result(payload: dict[str, object]) -> None:
    mlip = payload.get("mlip_result", payload)
    force_summary = payload.get("force_summary")
    if isinstance(mlip, dict):
        print(f"success: {mlip.get('success')}")
        print(f"energy: {mlip.get('energy')} {mlip.get('energy_unit')}")
        print(f"n_atoms: {mlip.get('n_atoms')}")
        forces = mlip.get("forces")
        if isinstance(forces, list):
            norms = compute_force_norms(forces)  # type: ignore[arg-type]
            mean_norm = sum(norms) / len(norms) if norms else None
            print(f"mean_force_norm: {mean_norm}")
        sanity = _geometry_sanity_from_payload(payload)
        if sanity is not None:
            print(f"geometry_sanity_status: {sanity.get('status')}")
            print(f"min_interatomic_distance: {sanity.get('min_interatomic_distance')}")
            print(f"max_force_norm: {sanity.get('max_force_norm')}")
        warnings = mlip.get("warnings", [])
        if warnings:
            print("warnings:")
            for warning in warnings:
                print(f"  - {warning}")
    if isinstance(force_summary, dict):
        print("reaction-center force summary:")
        for key in (
            "n_center_atoms",
            "mean_center_force_norm",
            "max_center_force_norm",
            "center_to_all_mean_force_ratio",
        ):
            print(f"  {key}: {force_summary.get(key)}")


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config = MLIPConfig(
        backend_name=args.backend,
        model_family=args.model_family,
        model_name=args.model_name,
        device=args.device,
        dtype=args.dtype,
        optimize_geometry=args.optimize_geometry,
    )
    try:
        if args.reaction_smiles:
            result = run_mendel_guided_mlip_singlepoint(
                args.reaction_smiles,
                context=args.context,
                config=config,
                center_source=args.center_source,
                mean_force_threshold=args.mean_force_threshold,
                max_force_threshold=args.max_force_threshold,
                min_distance_threshold=args.min_distance_threshold,
            )
            payload = result.to_dict()
        else:
            atoms = (
                atoms_from_xyz(args.xyz)
                if args.xyz
                else smiles_to_ase_atoms(args.smiles, seed=args.seed)
            )
            result = compute_mlip_singlepoint(
                atoms,
                config,
                mean_force_threshold=args.mean_force_threshold,
                max_force_threshold=args.max_force_threshold,
                min_distance_threshold=args.min_distance_threshold,
            )
            payload = result.to_dict()
            if args.reaction_center_from_mendelv:
                summary = summarize_reaction_center_forces(result, [])
                payload = {"mlip_result": payload, "force_summary": summary.to_dict()}
    except Exception as exc:
        payload = {
            "success": False,
            "error": str(exc),
            "scope_note": "Single-point MLIP prototype only; no training or reaction path search.",
        }
        save_json(payload, args.output)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    save_json(payload, args.output)
    _print_result(payload)
    print(f"Output: {args.output}")
    sanity = _geometry_sanity_from_payload(payload)
    if (
        args.fail_on_geometry_sanity
        and sanity is not None
        and sanity.get("status") == "fail"
    ):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
