"""Multi-molecule ANI-2x fine-tuning with MENDEL reactive-site weighting (Route B).

Trains on several rMD17 molecules jointly, tests on a held-out molecule.
This is the correct experimental design for validating whether MENDEL's
heteroatom-based reactive-site identification generalises across molecules.

Usage:
  python scripts/finetune_ani2x_multimol.py \\
    --train-npz ethanol:data/external/md17/rmd17_ethanol.npz \\
               malonaldehyde:data/external/md17/rmd17_malonaldehyde.npz \\
               aspirin:data/external/md17/rmd17_aspirin.npz \\
    --test-npz  salicylic:data/external/md17/rmd17_salicylic.npz \\
    --output models/ani2x_multimol_finetuned.pt \\
    --report reports/finetune_ani2x_multimol.json \\
    --epochs 30 --lr 5e-5 --device cpu

Control run (uniform weights — no MENDEL):
  python scripts/finetune_ani2x_multimol.py ... --reactive-weight 1.0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mendel.weighted_finetune import (
    ConformerRecord,
    FineTuneConfig,
    _eval_force_rmse,
    finetune_ani2x,
    load_md17_npz_records,
    save_model,
)


def _parse_mol_arg(spec: str) -> tuple[str, Path]:
    if ":" not in spec:
        raise argparse.ArgumentTypeError(f"Expected 'name:path', got {spec!r}")
    name, path_str = spec.split(":", 1)
    return name.strip(), ROOT / path_str.strip()


def _load_molecules(
    specs: list[str],
    max_per_mol: int,
    reactive_weight: float,
    seed: int,
) -> list[ConformerRecord]:
    all_records: list[ConformerRecord] = []
    for spec in specs:
        name, path = _parse_mol_arg(spec)
        if not path.exists():
            raise FileNotFoundError(f"NPZ not found: {path}")
        records = load_md17_npz_records(
            path,
            molecule_name=name,
            max_records=max_per_mol,
            reactive_weight=reactive_weight,
            seed=seed,
        )
        n_reactive = sum(1 for w in records[0].atom_weights if w > 1.0)
        print(f"  {name:<20} {len(records):>4} conformers  "
              f"{len(records[0].symbols):>2} atoms  "
              f"{n_reactive} reactive")
        all_records.extend(records)
    return all_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-molecule ANI-2x reactive-site fine-tuning")
    parser.add_argument(
        "--train-npz",
        nargs="+",
        required=True,
        metavar="NAME:PATH",
    )
    parser.add_argument(
        "--test-npz",
        nargs="+",
        required=True,
        metavar="NAME:PATH",
    )
    parser.add_argument("--output", default="models/ani2x_multimol_finetuned.pt")
    parser.add_argument("--report", default="reports/finetune_ani2x_multimol.json")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument(
        "--reactive-weight",
        type=float,
        default=3.0,
        help="Force loss multiplier for reactive atoms (1.0 = uniform control)",
    )
    parser.add_argument("--max-per-mol", type=int, default=300)
    parser.add_argument("--max-test-per-mol", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    out_path = ROOT / args.output
    report_path = ROOT / args.report

    label = "MENDEL-weighted" if args.reactive_weight > 1.0 else "uniform control"
    print(f"reactive_weight = {args.reactive_weight}  ({label})\n")

    print("Loading training molecules...")
    train_records = _load_molecules(
        args.train_npz, args.max_per_mol, args.reactive_weight, args.seed,
    )
    print(f"  total: {len(train_records)} conformers\n")

    print("Loading test (held-out) molecules...")
    test_records = _load_molecules(
        args.test_npz, args.max_test_per_mol, args.reactive_weight, args.seed,
    )
    print(f"  total: {len(test_records)} conformers\n")

    config = FineTuneConfig(
        lr=args.lr,
        epochs=args.epochs,
        reactive_loss_weight=args.reactive_weight,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
    )

    print("Fine-tuning ANI-2x...")
    model, result = finetune_ani2x(train_records, test_records, config)

    save_model(model, out_path)

    import torch
    device = torch.device(args.device)
    held_out_rmse = _eval_force_rmse(model, test_records, device)
    print(f"\nheld-out test force RMSE: {held_out_rmse:.4f} eV/Å")

    report = {
        **result.to_dict(),
        "held_out_test_force_rmse": held_out_rmse,
        "train_molecules": [s.split(":")[0] for s in args.train_npz],
        "test_molecules": [s.split(":")[0] for s in args.test_npz],
        "n_train_total": len(train_records),
        "n_test_total": len(test_records),
        "reactive_weight": args.reactive_weight,
        "experiment_type": label,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"saved report: {report_path}")


if __name__ == "__main__":
    main()
