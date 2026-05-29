"""Phase 8.10: experimental atom-level reaction-center head.

This module predicts whether each reactant atom belongs to the reaction center.
It is a binary atom classifier built from existing MENDELV group descriptors,
role predictions, group membership, and mechanism context. It does not train
MLIP, MACE, energies, forces, transition states, or barriers.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rdkit import Chem

from mendel.benchmark import compute_reaction_center_metrics
from mendel.descriptor import ELECTRONEGATIVITY
from mendel.identifier import identify_functional_groups
from mendel.labels import LabeledReaction
from mendel.parser import parse_reaction_smiles
from mendel.types import FunctionalGroup, FunctionalGroupType, Role

Scalar = str | int | float | bool

_CONTROL_MECHANISMS = frozenset({"control", "ester_control", "nitrile_control", "no_reaction"})
_MECHANISM_FEATURES = [
    "sn2",
    "e2",
    "carbonyl_addition",
    "diels_alder",
    "benzylic_radical_bromination",
    "nitroalkane_deprotonation",
    "aldol",
    "cross_aldol",
    "control",
    "ester_control",
    "nitrile_control",
]


@dataclass
class AtomCenterExample:
    reaction_id: str
    mechanism_type: str
    split: str
    atom_index: int
    molecule_index: int | None
    atomic_number: int | None
    atom_symbol: str | None
    group_ids: list[str]
    group_types: list[str]
    group_roles: list[str]
    role_confidences: list[float]
    is_labeled_center: bool
    features: list[float]
    metadata: dict[str, Scalar] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "reaction_id": self.reaction_id,
            "mechanism_type": self.mechanism_type,
            "split": self.split,
            "atom_index": self.atom_index,
            "molecule_index": self.molecule_index,
            "atomic_number": self.atomic_number,
            "atom_symbol": self.atom_symbol,
            "group_ids": list(self.group_ids),
            "group_types": list(self.group_types),
            "group_roles": list(self.group_roles),
            "role_confidences": list(self.role_confidences),
            "is_labeled_center": self.is_labeled_center,
            "features": list(self.features),
            "metadata": dict(self.metadata),
        }


@dataclass
class AtomCenterPrediction:
    reaction_id: str
    atom_index: int
    molecule_index: int | None
    probability: float
    predicted_center: bool
    threshold: float
    contributing_group_ids: list[str]
    contributing_roles: list[str]
    metadata: dict[str, Scalar] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "reaction_id": self.reaction_id,
            "atom_index": self.atom_index,
            "molecule_index": self.molecule_index,
            "probability": self.probability,
            "predicted_center": self.predicted_center,
            "threshold": self.threshold,
            "contributing_group_ids": list(self.contributing_group_ids),
            "contributing_roles": list(self.contributing_roles),
            "metadata": dict(self.metadata),
        }


@dataclass
class AtomCenterTrainingReport:
    n_examples: int
    n_positive: int
    n_negative: int
    positive_fraction: float
    split_distribution: dict[str, int]
    mechanism_distribution: dict[str, int]
    train_loss_history: list[float]
    train_accuracy: float | None
    val_accuracy: float | None
    val_precision: float | None
    val_recall: float | None
    val_f1: float | None
    threshold: float
    warnings: list[str]
    metadata: dict[str, Scalar] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "n_examples": self.n_examples,
            "n_positive": self.n_positive,
            "n_negative": self.n_negative,
            "positive_fraction": self.positive_fraction,
            "split_distribution": dict(self.split_distribution),
            "mechanism_distribution": dict(self.mechanism_distribution),
            "train_loss_history": list(self.train_loss_history),
            "train_accuracy": self.train_accuracy,
            "val_accuracy": self.val_accuracy,
            "val_precision": self.val_precision,
            "val_recall": self.val_recall,
            "val_f1": self.val_f1,
            "threshold": self.threshold,
            "warnings": list(self.warnings),
            "metadata": dict(self.metadata),
        }


@dataclass
class AtomCenterBenchmarkReport:
    predictor_name: str
    n_reactions: int
    n_atom_examples: int
    atom_accuracy: float
    atom_precision: float | None
    atom_recall: float | None
    atom_f1: float | None
    reaction_center_precision: float | None
    reaction_center_recall: float | None
    reaction_center_f1: float | None
    per_mechanism_f1: dict[str, float | None]
    threshold: float
    metadata: dict[str, Scalar] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "predictor_name": self.predictor_name,
            "n_reactions": self.n_reactions,
            "n_atom_examples": self.n_atom_examples,
            "atom_accuracy": self.atom_accuracy,
            "atom_precision": self.atom_precision,
            "atom_recall": self.atom_recall,
            "atom_f1": self.atom_f1,
            "reaction_center_precision": self.reaction_center_precision,
            "reaction_center_recall": self.reaction_center_recall,
            "reaction_center_f1": self.reaction_center_f1,
            "per_mechanism_f1": dict(self.per_mechanism_f1),
            "threshold": self.threshold,
            "metadata": dict(self.metadata),
        }


def _prediction_role_and_confidence(prediction: Any) -> tuple[str, float]:
    if isinstance(prediction, dict):
        role = prediction.get("predicted_role") or prediction.get("role")
        confidence = prediction.get("confidence")
    else:
        role = getattr(prediction, "predicted_role", getattr(prediction, "role", None))
        confidence = getattr(prediction, "confidence", None)
    role_value = role.value if isinstance(role, Role) else str(role or Role.spectator.value)
    return role_value, float(confidence if confidence is not None else 1.0)


def _roles_for_reaction(
    rxn: LabeledReaction,
    role_predictions_by_reaction: dict[str, list[Any]] | None,
) -> dict[str, tuple[str, float]]:
    if role_predictions_by_reaction and rxn.reaction_id in role_predictions_by_reaction:
        by_group: dict[str, tuple[str, float]] = {}
        for pred in role_predictions_by_reaction[rxn.reaction_id]:
            group_id = (
                pred.get("group_id")
                if isinstance(pred, dict)
                else getattr(pred, "group_id", "")
            )
            by_group[str(group_id)] = _prediction_role_and_confidence(pred)
        return by_group
    return {
        label.group_id: (label.role.value, 1.0)
        for label in rxn.group_roles
    }


def _atom_center_label(atom: Chem.Atom, rxn: LabeledReaction) -> bool:
    atom_map = atom.GetAtomMapNum()
    if atom_map:
        return atom_map in set(rxn.reaction_center_atoms)
    return atom.GetIdx() in set(rxn.reaction_center_atoms)


def _groups_by_atom(groups: list[FunctionalGroup]) -> dict[tuple[int, int], list[FunctionalGroup]]:
    grouped: dict[tuple[int, int], list[FunctionalGroup]] = {}
    for group in groups:
        for ref in group.atom_refs:
            grouped.setdefault((ref.molecule_index, ref.atom_index), []).append(group)
    return grouped


def build_atom_center_examples(
    labeled_reactions: list[LabeledReaction],
    role_predictions_by_reaction: dict[str, list[Any]] | None = None,
    include_hydrogens: bool = False,
) -> list[AtomCenterExample]:
    """Build deterministic atom-level center examples from labeled reactions."""
    examples: list[AtomCenterExample] = []
    for rxn in labeled_reactions:
        parsed = parse_reaction_smiles(rxn.reaction_smiles, context=rxn.context)
        groups = identify_functional_groups(parsed)
        group_lookup = _groups_by_atom(groups)
        role_lookup = _roles_for_reaction(rxn, role_predictions_by_reaction)
        missing_center_label = (
            not rxn.reaction_center_atoms
            and rxn.mechanism_type.lower() not in _CONTROL_MECHANISMS
        )

        for parsed_mol in parsed.reactants:
            mol = Chem.MolFromSmiles(parsed_mol.smiles)
            if mol is None:
                continue
            if include_hydrogens:
                mol = Chem.AddHs(mol)
            for atom in mol.GetAtoms():
                atom_groups = group_lookup.get((parsed_mol.molecule_index, atom.GetIdx()), [])
                group_ids = [group.group_id for group in atom_groups]
                group_types = [group.group_type.value for group in atom_groups]
                roles_and_conf = [
                    role_lookup.get(group.group_id, (Role.spectator.value, 0.0))
                    for group in atom_groups
                ]
                group_roles = [role for role, _ in roles_and_conf]
                role_confidences = [conf for _, conf in roles_and_conf]
                metadata: dict[str, Scalar] = {
                    "atom_map_num": atom.GetAtomMapNum(),
                    "reaction_center_label_kind": "atom_map_num"
                    if atom.GetAtomMapNum()
                    else "atom_index",
                }
                if missing_center_label:
                    metadata["warning"] = "non_control_reaction_without_center_labels"
                example = AtomCenterExample(
                    reaction_id=rxn.reaction_id,
                    mechanism_type=rxn.mechanism_type,
                    split=rxn.split,
                    atom_index=atom.GetIdx(),
                    molecule_index=parsed_mol.molecule_index,
                    atomic_number=atom.GetAtomicNum(),
                    atom_symbol=atom.GetSymbol(),
                    group_ids=group_ids,
                    group_types=group_types,
                    group_roles=group_roles,
                    role_confidences=role_confidences,
                    is_labeled_center=_atom_center_label(atom, rxn),
                    features=[],
                    metadata=metadata,
                )
                example.features = featurize_atom_center_example(example, atom)
                examples.append(example)
    return examples


def featurize_atom_center_example(
    example: AtomCenterExample,
    atom: Chem.Atom | None = None,
) -> list[float]:
    """Return compact, explainable atom-center features."""
    atomic_number = float(example.atomic_number or 0)
    electronegativity = ELECTRONEGATIVITY.get(example.atom_symbol or "", 0.0)
    n_groups = len(example.group_ids)
    confidences = example.role_confidences
    max_conf = max(confidences) if confidences else 0.0
    mean_conf = sum(confidences) / len(confidences) if confidences else 0.0
    role_set = set(example.group_roles)
    group_type_set = set(example.group_types)
    mechanism = example.mechanism_type.lower()
    features = [
        atomic_number / 100.0,
        electronegativity / 4.0,
        1.0 if n_groups else 0.0,
        float(n_groups),
        1.0 if Role.reactive_nucleophile.value in role_set else 0.0,
        1.0 if Role.reactive_electrophile.value in role_set else 0.0,
        1.0 if Role.leaving_group.value in role_set else 0.0,
        1.0 if Role.reactive_radical.value in role_set else 0.0,
        1.0 if Role.spectator.value in role_set else 0.0,
        max_conf,
        mean_conf,
    ]
    features.extend(
        1.0 if group_type.value in group_type_set else 0.0
        for group_type in FunctionalGroupType
    )
    features.extend(1.0 if mechanism == mech else 0.0 for mech in _MECHANISM_FEATURES)
    features.append(1.0 if mechanism in _CONTROL_MECHANISMS else 0.0)
    if atom is None:
        features.extend([0.0, 0.0, 0.0, 0.0])
    else:
        ring_atoms = {idx for ring in atom.GetOwningMol().GetRingInfo().AtomRings() for idx in ring}
        features.extend([
            float(atom.GetDegree()),
            float(atom.GetFormalCharge()),
            1.0 if atom.GetIsAromatic() else 0.0,
            1.0 if atom.GetIdx() in ring_atoms else 0.0,
        ])
    return [float(value) for value in features]


def summarize_atom_center_examples(
    examples: list[AtomCenterExample],
) -> dict[str, object]:
    n = len(examples)
    n_positive = sum(1 for ex in examples if ex.is_labeled_center)
    n_negative = n - n_positive
    split_distribution: dict[str, int] = {}
    mechanism_distribution: dict[str, int] = {}
    for ex in examples:
        split_distribution[ex.split] = split_distribution.get(ex.split, 0) + 1
        mechanism_distribution[ex.mechanism_type] = (
            mechanism_distribution.get(ex.mechanism_type, 0) + 1
        )
    warnings: list[str] = []
    if n_positive == 0:
        warnings.append("no_positive_center_atoms")
    elif n_positive < 10:
        warnings.append("fewer_than_10_positive_center_atoms")
    if n and n_positive / n < 0.1:
        warnings.append("positive_fraction_below_0_10")
    return {
        "n_examples": n,
        "n_positive": n_positive,
        "n_negative": n_negative,
        "positive_fraction": n_positive / n if n else 0.0,
        "mechanism_distribution": mechanism_distribution,
        "split_distribution": split_distribution,
        "warnings": warnings,
    }


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _metric_from_binary(predicted: list[bool], true: list[bool]) -> dict[str, float | None]:
    pairs = list(zip(predicted, true, strict=True))
    tp = sum(1 for p, t in pairs if p and t)
    fp = sum(1 for p, t in pairs if p and not t)
    fn = sum(1 for p, t in pairs if not p and t)
    tn = sum(1 for p, t in pairs if not p and not t)
    total = tp + fp + fn + tn
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall > 0.0
        else None
    )
    return {
        "accuracy": (tp + tn) / total if total else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _make_torch_model(input_dim: int, hidden_dim: int):
    import torch

    return torch.nn.Sequential(
        torch.nn.Linear(input_dim, hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, 1),
    )


def train_atom_center_head(
    examples: list[AtomCenterExample],
    output_path: str | Path,
    report_path: str | Path,
    hidden_dim: int = 32,
    epochs: int = 80,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    threshold: float = 0.5,
    use_class_weights: bool = True,
    seed: int = 42,
    device: str = "auto",
) -> AtomCenterTrainingReport:
    """Train a tiny PyTorch binary MLP for atom-center classification."""
    import torch

    if not examples:
        raise ValueError("Cannot train atom center head with no examples.")
    if not any(ex.is_labeled_center for ex in examples):
        raise ValueError("Cannot train atom center head with no positive center examples.")

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    resolved_device = _resolve_device(device)
    input_dim = len(examples[0].features)
    X = torch.tensor([ex.features for ex in examples], dtype=torch.float32, device=resolved_device)
    y = torch.tensor(
        [[1.0 if ex.is_labeled_center else 0.0] for ex in examples],
        dtype=torch.float32,
        device=resolved_device,
    )
    train_idx = [idx for idx, ex in enumerate(examples) if ex.split != "val"]
    val_idx = [idx for idx, ex in enumerate(examples) if ex.split == "val"]
    if not train_idx:
        train_idx = list(range(len(examples)))
        val_idx = []

    model = _make_torch_model(input_dim, hidden_dim).to(resolved_device)
    positives = sum(1 for ex in examples if ex.is_labeled_center)
    negatives = len(examples) - positives
    pos_weight = None
    if use_class_weights and positives:
        pos_weight = torch.tensor([max(negatives / positives, 1.0)], device=resolved_device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    rng = random.Random(seed)
    loss_history: list[float] = []
    for _ in range(epochs):
        order = list(train_idx)
        rng.shuffle(order)
        batch_losses: list[float] = []
        for start in range(0, len(order), batch_size):
            batch_idx = order[start:start + batch_size]
            xb = X[batch_idx]
            yb = y[batch_idx]
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.item()))
        loss_history.append(sum(batch_losses) / len(batch_losses) if batch_losses else 0.0)

    with torch.no_grad():
        probs = torch.sigmoid(model(X)).squeeze(1).detach().cpu().tolist()
    pred = [prob >= threshold for prob in probs]
    true = [ex.is_labeled_center for ex in examples]
    train_metrics = _metric_from_binary([pred[i] for i in train_idx], [true[i] for i in train_idx])
    val_metrics = (
        _metric_from_binary([pred[i] for i in val_idx], [true[i] for i in val_idx])
        if val_idx
        else {"accuracy": None, "precision": None, "recall": None, "f1": None}
    )
    summary = summarize_atom_center_examples(examples)
    warnings = list(summary["warnings"])  # type: ignore[arg-type]
    if positives < 10:
        warnings.append("positive_examples_below_10")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "threshold": threshold,
        "feature_schema": "phase8_10_atom_center_v1",
    }, out)
    report = AtomCenterTrainingReport(
        n_examples=int(summary["n_examples"]),
        n_positive=int(summary["n_positive"]),
        n_negative=int(summary["n_negative"]),
        positive_fraction=float(summary["positive_fraction"]),
        split_distribution=dict(summary["split_distribution"]),  # type: ignore[arg-type]
        mechanism_distribution=dict(summary["mechanism_distribution"]),  # type: ignore[arg-type]
        train_loss_history=loss_history,
        train_accuracy=train_metrics["accuracy"],
        val_accuracy=val_metrics["accuracy"],
        val_precision=val_metrics["precision"],
        val_recall=val_metrics["recall"],
        val_f1=val_metrics["f1"],
        threshold=threshold,
        warnings=warnings,
        metadata={
            "hidden_dim": hidden_dim,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "use_class_weights": use_class_weights,
            "pos_weight": float(pos_weight.item()) if pos_weight is not None else 1.0,
            "seed": seed,
            "device": resolved_device,
            "scope_note": "Atom reaction-center classifier only; no MLIP or energy/force training.",
        },
    )
    save_atom_center_training_report(report, report_path)
    return report


def load_atom_center_head(checkpoint_path: str | Path, device: str = "cpu"):
    """Load a trained atom-center head checkpoint."""
    import torch

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = _make_torch_model(int(ckpt["input_dim"]), int(ckpt["hidden_dim"]))
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device), ckpt


def predict_atom_centers(
    examples: list[AtomCenterExample],
    checkpoint_path: str | Path,
    threshold: float = 0.5,
    device: str = "cpu",
) -> list[AtomCenterPrediction]:
    """Predict center probabilities for atom examples."""
    import torch

    if not examples:
        return []
    model, ckpt = load_atom_center_head(checkpoint_path, device=device)
    model.eval()
    input_dim = int(ckpt["input_dim"])
    for ex in examples:
        if len(ex.features) != input_dim:
            raise ValueError(
                f"Feature length mismatch for {ex.reaction_id}:{ex.atom_index}: "
                f"expected {input_dim}, got {len(ex.features)}"
            )
    X = torch.tensor([ex.features for ex in examples], dtype=torch.float32, device=device)
    with torch.no_grad():
        probs = torch.sigmoid(model(X)).squeeze(1).detach().cpu().tolist()
    return [
        AtomCenterPrediction(
            reaction_id=ex.reaction_id,
            atom_index=ex.atom_index,
            molecule_index=ex.molecule_index,
            probability=float(prob),
            predicted_center=bool(prob >= threshold),
            threshold=threshold,
            contributing_group_ids=list(ex.group_ids),
            contributing_roles=list(ex.group_roles),
            metadata=dict(ex.metadata),
        )
        for ex, prob in zip(examples, probs, strict=True)
    ]


def _prediction_atom_value(prediction: AtomCenterPrediction) -> int:
    atom_map = prediction.metadata.get("atom_map_num")
    if isinstance(atom_map, int) and atom_map:
        return atom_map
    return prediction.atom_index


def aggregate_atom_predictions_to_reaction_centers(
    predictions: list[AtomCenterPrediction],
) -> dict[str, list[int]]:
    """Aggregate atom predictions into reaction_id -> center atom values."""
    centers: dict[str, list[int]] = {}
    for prediction in predictions:
        centers.setdefault(prediction.reaction_id, [])
        if not prediction.predicted_center:
            continue
        value = _prediction_atom_value(prediction)
        if value not in centers[prediction.reaction_id]:
            centers[prediction.reaction_id].append(value)
    return centers


def benchmark_atom_center_head(
    labeled_reactions: list[LabeledReaction],
    predictions: list[AtomCenterPrediction],
    threshold: float = 0.5,
    predictor_name: str = "atom_center_head",
) -> AtomCenterBenchmarkReport:
    """Compute atom-level and reaction-level center metrics."""
    rxn_by_id = {rxn.reaction_id: rxn for rxn in labeled_reactions}
    pred_flags: list[bool] = []
    true_flags: list[bool] = []
    for prediction in predictions:
        rxn = rxn_by_id.get(prediction.reaction_id)
        if rxn is None:
            continue
        value = _prediction_atom_value(prediction)
        pred_flags.append(prediction.predicted_center)
        true_flags.append(value in set(rxn.reaction_center_atoms))
    atom_metrics = _metric_from_binary(pred_flags, true_flags)
    predicted_centers = aggregate_atom_predictions_to_reaction_centers(predictions)
    rxn_metrics: list[dict[str, float | None]] = []
    by_mechanism: dict[str, list[float | None]] = {}
    failure_counts: dict[str, int] = {
        "exact_match": 0,
        "empty_truth": 0,
        "empty_prediction": 0,
        "partial_overlap": 0,
        "extra_atoms": 0,
        "wrong_nonoverlap": 0,
    }
    for rxn in labeled_reactions:
        predicted = set(predicted_centers.get(rxn.reaction_id, []))
        true = set(rxn.reaction_center_atoms)
        metrics = compute_reaction_center_metrics(
            list(predicted),
            rxn.reaction_center_atoms,
        )
        rxn_metrics.append(metrics)
        by_mechanism.setdefault(rxn.mechanism_type, []).append(metrics["f1"])
        if not true:
            failure_counts["empty_truth"] += 1
        elif predicted == true:
            failure_counts["exact_match"] += 1
        elif not predicted:
            failure_counts["empty_prediction"] += 1
        elif predicted & true:
            failure_counts["partial_overlap"] += 1
        elif predicted - true:
            failure_counts["wrong_nonoverlap"] += 1
        if true and predicted - true:
            failure_counts["extra_atoms"] += 1

    def avg(values: list[float | None]) -> float | None:
        present = [value for value in values if value is not None]
        return sum(present) / len(present) if present else None

    return AtomCenterBenchmarkReport(
        predictor_name=predictor_name,
        n_reactions=len(labeled_reactions),
        n_atom_examples=len(predictions),
        atom_accuracy=float(atom_metrics["accuracy"] or 0.0),
        atom_precision=atom_metrics["precision"],
        atom_recall=atom_metrics["recall"],
        atom_f1=atom_metrics["f1"],
        reaction_center_precision=avg([metrics["precision"] for metrics in rxn_metrics]),
        reaction_center_recall=avg([metrics["recall"] for metrics in rxn_metrics]),
        reaction_center_f1=avg([metrics["f1"] for metrics in rxn_metrics]),
        per_mechanism_f1={
            mechanism: avg(values)
            for mechanism, values in sorted(by_mechanism.items())
        },
        threshold=threshold,
        metadata={
            "scope_note": "Atom reaction-center classifier only; no MLIP.",
            **failure_counts,
        },
    )


def benchmark_atom_center_head_by_split(
    labeled_reactions: list[LabeledReaction],
    predictions: list[AtomCenterPrediction],
    threshold: float = 0.5,
    predictor_name: str = "atom_center_head",
) -> dict[str, dict[str, object]]:
    """Return atom-center benchmark reports for overall/train/val/test splits."""
    by_split = {
        "train": [rxn for rxn in labeled_reactions if rxn.split == "train"],
        "val": [rxn for rxn in labeled_reactions if rxn.split == "val"],
        "test": [rxn for rxn in labeled_reactions if rxn.split == "test"],
    }
    prediction_by_split: dict[str, list[AtomCenterPrediction]] = {split: [] for split in by_split}
    split_by_reaction = {rxn.reaction_id: rxn.split for rxn in labeled_reactions}
    for prediction in predictions:
        split = split_by_reaction.get(prediction.reaction_id)
        if split in prediction_by_split:
            prediction_by_split[split].append(prediction)
    reports = {
        "overall": benchmark_atom_center_head(
            labeled_reactions,
            predictions,
            threshold=threshold,
            predictor_name=predictor_name,
        ).to_dict()
    }
    for split, reactions in by_split.items():
        reports[split] = benchmark_atom_center_head(
            reactions,
            prediction_by_split[split],
            threshold=threshold,
            predictor_name=f"{predictor_name}_{split}",
        ).to_dict()
    return reports


def save_atom_center_training_report(
    report: AtomCenterTrainingReport,
    path: str | Path,
) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")


def save_atom_center_benchmark_report(
    report: AtomCenterBenchmarkReport,
    path: str | Path,
) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
