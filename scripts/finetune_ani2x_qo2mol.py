"""Route B reactive-site weighted fine-tuning on QO2Mol OOD set.

Train on diverse QO2Mol molecules (ANI-2x compatible elements only).
Test on a held-out sample (different random seed = different molecules).

Usage:
  # MENDEL ×3
  python scripts/finetune_ani2x_qo2mol.py \
    --pkl data/external/QO2Mol26_OOD_large.pkl \
    --output models/ani2x_qo2mol_mendel.pt \
    --report reports/finetune_ani2x_qo2mol_mendel.json \
    --reactive-weight 3.0

  # Uniform control
  python scripts/finetune_ani2x_qo2mol.py \
    --pkl data/external/QO2Mol26_OOD_large.pkl \
    --output models/ani2x_qo2mol_uniform.pt \
    --report reports/finetune_ani2x_qo2mol_uniform.json \
    --reactive-weight 1.0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mendel.weighted_finetune import (
    FineTuneConfig,
    _eval_force_rmse,
    finetune_ani2x,
    load_qo2mol_pkl_records,
    save_model,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="ANI-2x reactive-site fine-tuning on QO2Mol")
    parser.add_argument("--pkl", type=Path, required=True)
    parser.add_argument("--output", default="models/ani2x_qo2mol_finetuned.pt")
    parser.add_argument("--report", default="reports/finetune_ani2x_qo2mol.json")
    parser.add_argument("--reactive-weight", type=float, default=3.0)
    parser.add_argument("--n-train", type=int, default=300)
    parser.add_argument("--n-test", type=int, default=100)
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--test-seed", type=int, default=99)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mlp-checkpoint", type=Path, default=None,
                        help="MENDEL MLP checkpoint for role-based reactive detection.")
    args = parser.parse_args()

    pkl_path = ROOT / args.pkl if not Path(args.pkl).is_absolute() else Path(args.pkl)
    out_path = ROOT / args.output
    report_path = ROOT / args.report
    mlp_ckpt = (ROOT / args.mlp_checkpoint
                if args.mlp_checkpoint and not args.mlp_checkpoint.is_absolute()
                else args.mlp_checkpoint)

    detection = "MLP-guided" if mlp_ckpt else "heteroatom-proximity"
    label = "MENDEL-weighted" if args.reactive_weight > 1.0 else "uniform control"
    print(f"reactive_weight = {args.reactive_weight}  ({label})")
    print(f"reactive detection = {detection}\n")

    print("Loading training records...")
    train = load_qo2mol_pkl_records(
        pkl_path, max_records=args.n_train,
        reactive_weight=args.reactive_weight, seed=args.train_seed,
        mlp_checkpoint=mlp_ckpt,
    )
    n_reactive = sum(1 for r in train for w in r.atom_weights if w > 1.0)
    n_total_atoms = sum(len(r.symbols) for r in train)
    print(f"  {len(train)} conformers, {n_total_atoms} total atoms, "
          f"{n_reactive} reactive atom-slots ({100*n_reactive/n_total_atoms:.1f}%)\n")

    print("Loading test (held-out) records...")
    test = load_qo2mol_pkl_records(
        pkl_path, max_records=args.n_test,
        reactive_weight=args.reactive_weight, seed=args.test_seed,
        mlp_checkpoint=mlp_ckpt,
    )
    print(f"  {len(test)} conformers\n")

    config = FineTuneConfig(
        lr=args.lr,
        epochs=args.epochs,
        reactive_loss_weight=args.reactive_weight,
        batch_size=args.batch_size,
        device=args.device,
    )

    print("Fine-tuning ANI-2x...")
    model, result = finetune_ani2x(train, test, config)
    save_model(model, out_path)

    import torch
    held_rmse = _eval_force_rmse(model, test, torch.device(args.device))
    print(f"\nHeld-out force RMSE: {held_rmse:.4f} eV/Å")

    report = {
        **result.to_dict(),
        "held_out_force_rmse": held_rmse,
        "n_train": len(train),
        "n_test": len(test),
        "reactive_weight": args.reactive_weight,
        "experiment_type": label,
        "dataset": "QO2Mol OOD",
        "train_seed": args.train_seed,
        "test_seed": args.test_seed,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
