"""Reactive-site weighted fine-tuning for ANI-2x (Phase 10, Route B).

Fine-tunes ANI-2x on rMD17 conformers with MENDEL-identified reactive atoms
receiving 3× force loss weight. Validates the functional-group-as-agent
hypothesis: if the fine-tuned model has lower RMSE on the reactive C–O group
while spectator groups are unchanged, MENDEL's decomposition is causally
informative, not just descriptive.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Reactive-atom detection (mirrors compare_pure_vs_mendel_mlip.py)
# ---------------------------------------------------------------------------

def _dist(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def reactive_atom_indices_ethanol(xyz: list[Any]) -> list[int]:
    """Return [alpha_C, O] atom indices for one ethanol conformer."""
    symbols = [row[0] for row in xyz]
    positions = {i: (float(xyz[i][1]), float(xyz[i][2]), float(xyz[i][3])) for i in range(len(xyz))}
    c_idx = [i for i, s in enumerate(symbols) if s == "C"]
    o_idx = next(i for i, s in enumerate(symbols) if s == "O")
    alpha_c = min(c_idx, key=lambda c: _dist(positions[c], positions[o_idx]))
    return sorted([alpha_c, o_idx])


def build_atom_weights(
    n_atoms: int,
    reactive_indices: list[int],
    reactive_weight: float = 3.0,
    spectator_weight: float = 1.0,
) -> list[float]:
    return [reactive_weight if i in reactive_indices else spectator_weight for i in range(n_atoms)]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

@dataclass
class ConformerRecord:
    structure_id: str
    symbols: list[str]
    positions: list[list[float]]   # Å
    energy: float                   # eV
    forces: list[list[float]]       # eV/Å
    atom_weights: list[float]


def load_records(
    reference_path: Path,
    reactive_weight: float = 3.0,
) -> list[ConformerRecord]:
    with open(reference_path) as f:
        data = json.load(f)
    records: list[ConformerRecord] = []
    for r in data["records"]:
        if r.get("reference_energy") is None or r.get("reference_forces") is None:
            continue
        xyz = r["xyz"]
        reactive = reactive_atom_indices_ethanol(xyz)
        weights = build_atom_weights(len(xyz), reactive, reactive_weight=reactive_weight)
        records.append(ConformerRecord(
            structure_id=r["structure_id"],
            symbols=[row[0] for row in xyz],
            positions=[[float(row[1]), float(row[2]), float(row[3])] for row in xyz],
            energy=float(r["reference_energy"]),
            forces=[[float(v) for v in fvec] for fvec in r["reference_forces"]],
            atom_weights=weights,
        ))
    return records


def split_records(
    records: list[ConformerRecord],
    test_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[list[ConformerRecord], list[ConformerRecord]]:
    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)
    n_test = max(1, int(len(shuffled) * test_fraction))
    return shuffled[n_test:], shuffled[:n_test]  # train, test


# ---------------------------------------------------------------------------
# Config and result types
# ---------------------------------------------------------------------------

@dataclass
class FineTuneConfig:
    lr: float = 5e-5
    epochs: int = 30
    force_weight: float = 1.0
    energy_weight: float = 0.01   # small: preserve energy scale
    reactive_loss_weight: float = 3.0
    batch_size: int = 8
    seed: int = 42
    device: str = "cpu"


@dataclass
class EpochLog:
    epoch: int
    train_loss: float
    val_force_rmse: float


@dataclass
class FineTuneResult:
    config: dict[str, Any]
    epochs: list[EpochLog]
    final_train_loss: float
    final_val_force_rmse: float
    reactive_indices_sample: list[int]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config,
            "epochs": [
                {"epoch": e.epoch, "train_loss": e.train_loss, "val_force_rmse": e.val_force_rmse}
                for e in self.epochs
            ],
            "final_train_loss": self.final_train_loss,
            "final_val_force_rmse": self.final_val_force_rmse,
            "reactive_indices_sample": self.reactive_indices_sample,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Element lookup
# ---------------------------------------------------------------------------

_ATOMIC_NUMBER: dict[str, int] = {
    "H": 1, "C": 6, "N": 7, "O": 8, "F": 9, "S": 16, "Cl": 17,
}


def _atomic_numbers(symbols: list[str]) -> list[int]:
    try:
        return [_ATOMIC_NUMBER[s] for s in symbols]
    except KeyError as exc:
        raise ValueError(f"Element {exc} not supported by ANI-2x") from exc


# ---------------------------------------------------------------------------
# Fine-tuning loop
# ---------------------------------------------------------------------------

def finetune_ani2x(
    train: list[ConformerRecord],
    val: list[ConformerRecord],
    config: FineTuneConfig,
) -> tuple[Any, FineTuneResult]:
    """Fine-tune ANI-2x with per-atom weighted force loss.

    Reactive atoms (MENDEL-identified alcohol C–O) receive `reactive_loss_weight`
    in the force MSE. Returns (model, FineTuneResult).
    """
    try:
        import torch
        import torchani
    except ImportError as exc:
        raise ImportError(
            "torch and torchani required. Install: pip install -e '.[ani2x]'"
        ) from exc

    torch.manual_seed(config.seed)
    device = torch.device(config.device)

    model = torchani.models.ANI2x(periodic_table_index=True).to(device)
    # ANI-2x loads with requires_grad=False; unfreeze before optimizer
    for p in model.parameters():
        p.requires_grad_(True)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    # ANI-2x energies are Hartree; forces (dE/dr) are Hartree/Å → convert to eV/Å
    EV_PER_HARTREE = 27.211386

    logs: list[EpochLog] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        random.shuffle(train)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, len(train), config.batch_size):
            batch = train[start : start + config.batch_size]
            optimizer.zero_grad()
            batch_loss = torch.zeros((), device=device)

            for rec in batch:
                species = torch.tensor([_atomic_numbers(rec.symbols)], dtype=torch.long, device=device)
                coords = torch.tensor([rec.positions], dtype=torch.float32, device=device,
                                      requires_grad=True)
                weights = torch.tensor(rec.atom_weights, dtype=torch.float32, device=device)

                result = model((species, coords))
                pred_energy = result.energies[0]                       # Hartree
                pred_forces = -torch.autograd.grad(
                    pred_energy, coords, create_graph=True
                )[0][0] * EV_PER_HARTREE                              # Hartree/Å → eV/Å

                # weighted force MSE
                force_err_sq = ((pred_forces - torch.tensor(
                    rec.forces, dtype=torch.float32, device=device
                )) ** 2).sum(dim=-1)                                   # [N]
                force_loss = (weights * force_err_sq).mean()

                # energy MSE (convert ref to Hartree)
                ref_hartree = torch.tensor(
                    rec.energy / EV_PER_HARTREE, dtype=torch.float32, device=device
                )
                energy_loss = (pred_energy - ref_hartree) ** 2

                batch_loss = batch_loss + (
                    config.force_weight * force_loss + config.energy_weight * energy_loss
                )

            (batch_loss / len(batch)).backward()
            optimizer.step()
            epoch_loss += (batch_loss / len(batch)).item()
            n_batches += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)
        val_rmse = _eval_force_rmse(model, val, device)
        logs.append(EpochLog(epoch=epoch, train_loss=avg_train_loss, val_force_rmse=val_rmse))

        if epoch % 5 == 0 or epoch == 1:
            print(f"  epoch {epoch:3d}  train_loss={avg_train_loss:.6f}  val_rmse={val_rmse:.4f} eV/Å")

    reactive_sample = reactive_atom_indices_ethanol(
        [[train[0].symbols[i]] + train[0].positions[i] for i in range(len(train[0].symbols))]
        if train else []
    )

    return model, FineTuneResult(
        config={
            "lr": config.lr,
            "epochs": config.epochs,
            "force_weight": config.force_weight,
            "energy_weight": config.energy_weight,
            "reactive_loss_weight": config.reactive_loss_weight,
            "batch_size": config.batch_size,
            "device": config.device,
        },
        epochs=logs,
        final_train_loss=logs[-1].train_loss if logs else 0.0,
        final_val_force_rmse=logs[-1].val_force_rmse if logs else 0.0,
        reactive_indices_sample=reactive_sample,
    )


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _eval_force_rmse(model: Any, records: list[ConformerRecord], device: Any) -> float:
    import torch

    sq_sum = 0.0
    n = 0
    for rec in records:
        species = torch.tensor([_atomic_numbers(rec.symbols)], dtype=torch.long, device=device)
        coords = torch.tensor([rec.positions], dtype=torch.float32, device=device, requires_grad=True)
        with torch.enable_grad():
            result = model((species, coords))
            pred_forces = -torch.autograd.grad(result.energies[0], coords)[0][0] * 27.211386
        ref = torch.tensor(rec.forces, dtype=torch.float32, device=device)
        sq_sum += ((pred_forces.detach() - ref) ** 2).sum(dim=-1).sum().item()
        n += len(rec.symbols)
    return math.sqrt(sq_sum / n) if n > 0 else 0.0


def per_group_rmse(
    model: Any,
    records: list[ConformerRecord],
    device: Any,
) -> dict[str, float]:
    """Per-group force RMSE using MENDEL group definitions."""
    import torch

    groups = _build_groups(records[0])
    sq: dict[str, float] = {g: 0.0 for g in groups}
    cnt: dict[str, int] = {g: 0 for g in groups}

    for rec in records:
        species = torch.tensor([_atomic_numbers(rec.symbols)], dtype=torch.long, device=device)
        coords = torch.tensor([rec.positions], dtype=torch.float32, device=device, requires_grad=True)
        with torch.enable_grad():
            result = model((species, coords))
            pred_forces = -torch.autograd.grad(result.energies[0], coords)[0][0] * 27.211386
        ref = torch.tensor(rec.forces, dtype=torch.float32, device=device)
        err = (pred_forces.detach() - ref)  # [N, 3]

        for label, indices in groups.items():
            for idx in indices:
                sq[label] += err[idx].pow(2).sum().item()
                cnt[label] += 1

    return {
        label: math.sqrt(sq[label] / cnt[label]) if cnt[label] > 0 else 0.0
        for label in groups
    }


def _build_groups(rec: ConformerRecord) -> dict[str, list[int]]:
    """Group definitions mirroring compare_pure_vs_mendel_mlip.py."""
    symbols = rec.symbols
    positions = {i: tuple(rec.positions[i]) for i in range(len(symbols))}
    heavy = [i for i, s in enumerate(symbols) if s != "H"]
    h_idx = [i for i, s in enumerate(symbols) if s == "H"]
    c_idx = [i for i, s in enumerate(symbols) if s == "C"]
    o_idx = next(i for i, s in enumerate(symbols) if s == "O")
    alpha_c = min(c_idx, key=lambda c: _dist(positions[c], positions[o_idx]))
    methyl_c = next(c for c in c_idx if c != alpha_c)
    h_parent = {h: min(heavy, key=lambda hv: _dist(positions[h], positions[hv])) for h in h_idx}
    return {
        "alcohol_CO_reactive": sorted([alpha_c, o_idx]),
        "hydroxyl_H":          sorted(h for h, p in h_parent.items() if p == o_idx),
        "alpha_CH":            sorted(h for h, p in h_parent.items() if p == alpha_c),
        "methyl_CH":           sorted(h for h, p in h_parent.items() if p == methyl_c),
        "methyl_C":            [methyl_c],
    }


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------

def save_model(model: Any, path: Path) -> None:
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "model_type": "ani2x_finetuned"}, path)
    print(f"saved: {path}")


def load_finetuned_model(path: Path, device: str = "cpu") -> Any:
    import torch
    import torchani

    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = torchani.models.ANI2x(periodic_table_index=True).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    return model


# ---------------------------------------------------------------------------
# Multi-molecule loading (direct NPZ → ConformerRecord, no JSON intermediary)
# ---------------------------------------------------------------------------

_KCAL_TO_EV = 0.0433641153087705

_ATOMIC_SYMBOL: dict[int, str] = {v: k for k, v in _ATOMIC_NUMBER.items()}

_HETEROATOMS = frozenset({7, 8, 9, 16, 17})  # N, O, F, S, Cl


def reactive_atom_indices_by_heteroatom(
    atomic_numbers: list[int],
    positions: list[tuple[float, float, float]],
    cutoff: float = 1.65,
) -> list[int]:
    """Generic reactive-site detection: heteroatoms + directly bonded heavy atoms.

    Works for any organic molecule whose heteroatoms define the reactive site.
    """
    reactive: set[int] = set()
    for i, z in enumerate(atomic_numbers):
        if z in _HETEROATOMS:
            reactive.add(i)
    for i, zi in enumerate(atomic_numbers):
        if zi == 1:
            continue
        for j in list(reactive):
            if i != j and _dist(positions[i], positions[j]) <= cutoff:
                reactive.add(i)
                break
    return sorted(reactive)


_ANI2X_SUPPORTED = frozenset({1, 6, 7, 8, 9, 16, 17})  # H C N O F S Cl


def reactive_atom_indices_via_mendel_mlp(
    item: dict,
    mlp_checkpoint: "Path",
    confidence_threshold: float = 0.60,
) -> "list[int] | None":
    """Use MENDEL MLP role predictor to identify reactive atoms.

    Returns original-order atom indices for groups predicted as
    nucleophile/electrophile/leaving_group with confidence >= threshold.
    Returns None on any failure so caller can fall back to heuristic.
    """
    try:
        from rdkit import Chem  # type: ignore
        from rdkit.Chem import RWMol  # type: ignore
        from mendel.qo2mol import qo2mol_record_to_rdkit_mol
        from mendel.parser import parse_reaction_smiles
        from mendel.identifier import identify_functional_groups
        from mendel.mlp import MLPRolePredictor
        from mendel.types import ReactionContext, Role

        mol = qo2mol_record_to_rdkit_mol(item)
        if mol is None:
            return None

        # Tag atoms with original index as atom-map number to survive SMILES canonicalization
        rw = RWMol(mol)
        for atom in rw.GetAtoms():
            atom.SetAtomMapNum(atom.GetIdx() + 1)
        smiles = Chem.MolToSmiles(rw)

        parsed = parse_reaction_smiles(f"{smiles}>>{smiles}", context=ReactionContext.ionic)
        groups = identify_functional_groups(parsed)
        if not groups:
            return None

        mlp = MLPRolePredictor.load(mlp_checkpoint, device="cpu")
        preds = mlp.predict_from_reaction(parsed, groups)

        reactive_roles = {Role.reactive_nucleophile, Role.reactive_electrophile, Role.leaving_group}
        reactive_orig: set[int] = set()
        for pred in preds:
            if pred.predicted_role in reactive_roles and pred.confidence >= confidence_threshold:
                grp = next((g for g in groups if g.group_id == pred.group_id), None)
                if grp:
                    for ref in grp.atom_refs:
                        if ref.atom_map_num:
                            reactive_orig.add(ref.atom_map_num - 1)
        return sorted(reactive_orig) if reactive_orig else None
    except Exception:
        return None


def load_qo2mol_pkl_records(
    path: "Path",
    max_records: int = 300,
    reactive_weight: float = 3.0,
    seed: int = 42,
    mlp_checkpoint: "Path | None" = None,
) -> "list[ConformerRecord]":
    """Load QO2Mol pkl into ConformerRecord list.

    Filters for ANI-2x-supported elements. No unit conversion (already eV).
    If mlp_checkpoint is given, uses MENDEL MLP for reactive detection;
    otherwise falls back to heteroatom-proximity heuristic.
    """
    import pickle

    rng = random.Random(seed)
    with open(path, "rb") as fh:
        all_data = pickle.load(fh)  # noqa: S301

    compatible = [
        item for item in all_data
        if all(int(z) in _ANI2X_SUPPORTED for z in item["elements"])
    ]
    sample = rng.sample(compatible, min(max_records, len(compatible)))

    use_mlp = mlp_checkpoint is not None
    mlp_used = mlp_fallback = 0

    records: list[ConformerRecord] = []
    for i, item in enumerate(sample):
        atomic_numbers = [int(z) for z in item["elements"]]
        symbols = [_ATOMIC_SYMBOL[z] for z in atomic_numbers]
        positions = [tuple(float(v) for v in row) for row in item["coordinates"]]
        forces = [[float(v) for v in row] for row in item["forces"]]

        if use_mlp:
            reactive = reactive_atom_indices_via_mendel_mlp(item, mlp_checkpoint)
            if reactive is not None:
                mlp_used += 1
            else:
                reactive = reactive_atom_indices_by_heteroatom(atomic_numbers, positions)
                mlp_fallback += 1
        else:
            reactive = reactive_atom_indices_by_heteroatom(atomic_numbers, positions)

        weights = build_atom_weights(len(symbols), reactive, reactive_weight=reactive_weight)
        records.append(ConformerRecord(
            structure_id=str(item.get("confid", f"qo2mol_{i}")),
            symbols=symbols,
            positions=[list(p) for p in positions],
            energy=float(item["energy"]),
            forces=forces,
            atom_weights=weights,
        ))

    if use_mlp:
        n_reactive = sum(1 for r in records for w in r.atom_weights if w > 1.0)
        n_atoms = sum(len(r.symbols) for r in records)
        print(f"  MLP detection: {mlp_used}/{len(records)} ok, {mlp_fallback} fallback")
        print(f"  reactive atom-slots: {n_reactive}/{n_atoms} ({100*n_reactive/max(1,n_atoms):.1f}%)")
    return records


def load_md17_npz_records(
    path: Path,
    molecule_name: str,
    max_records: int = 500,
    reactive_weight: float = 3.0,
    seed: int = 42,
) -> list[ConformerRecord]:
    """Load rMD17 NPZ directly into ConformerRecord list (no JSON step).

    Energies/forces converted from kcal/mol to eV / eV·Å⁻¹.
    Reactive weights assigned via heteroatom-proximity heuristic.
    """
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError("numpy required: pip install numpy") from exc

    rng = random.Random(seed)
    with np.load(path, allow_pickle=False) as data:
        z_arr = data["nuclear_charges"]  # (n_atoms,)
        coords_arr = data["coords"]      # (n_conf, n_atoms, 3)
        e_arr = data["energies"]         # (n_conf,)
        f_arr = data["forces"]           # (n_conf, n_atoms, 3)

    n_total = int(coords_arr.shape[0])
    indices = list(range(n_total))
    rng.shuffle(indices)
    indices = indices[:max_records]

    atomic_numbers = [int(z) for z in z_arr.tolist()]
    unsupported = {z for z in atomic_numbers if z not in _ATOMIC_SYMBOL}
    if unsupported:
        raise ValueError(f"Molecule {molecule_name!r} has atomic numbers not supported by ANI-2x: {unsupported}")

    symbols = [_ATOMIC_SYMBOL[z] for z in atomic_numbers]
    first_pos = [(float(row[0]), float(row[1]), float(row[2])) for row in coords_arr[0].tolist()]
    reactive = reactive_atom_indices_by_heteroatom(atomic_numbers, first_pos)
    weights = build_atom_weights(len(symbols), reactive, reactive_weight=reactive_weight)

    records: list[ConformerRecord] = []
    for i, idx in enumerate(indices):
        positions = [[float(v) for v in row] for row in coords_arr[idx].tolist()]
        energy_ev = float(e_arr[idx]) * _KCAL_TO_EV
        forces_ev = [[float(v) * _KCAL_TO_EV for v in row] for row in f_arr[idx].tolist()]
        records.append(ConformerRecord(
            structure_id=f"{molecule_name}_{i}",
            symbols=symbols,
            positions=positions,
            energy=energy_ev,
            forces=forces_ev,
            atom_weights=weights,
        ))
    return records
