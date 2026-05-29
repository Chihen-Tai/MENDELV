"""Phase 7: MLP role predictor for functional-group agents.

Trains a small PyTorch MLP to map GroupDescriptor vectors to the five MENDEL
role labels.  This supplements the rule-based predictor (Phase 5) with a
learned model.  Phase 6 negotiation remains responsible for global consistency.

Important scope boundaries:
- Trains role classification only (descriptor → role label).
- Does NOT train MLIP, MACE, or any energy/force model.
- Does NOT predict activation energies, barriers, or transition states.
- Does NOT use Transition1x or any DFT dataset.
"""

from __future__ import annotations

import json
import random
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from mendel.descriptor import (
    FEATURE_SCHEMA_VERSION,
    GroupDescriptor,
    build_descriptors,
    get_feature_names,
)
from mendel.identifier import identify_functional_groups
from mendel.labels import LabeledReaction
from mendel.parser import ParsedReaction, parse_reaction_smiles
from mendel.types import FunctionalGroup, FunctionalGroupType, Role

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

ROLE_TO_INDEX: dict[Role, int] = {role: i for i, role in enumerate(Role)}
INDEX_TO_ROLE: dict[int, Role] = {i: role for role, i in ROLE_TO_INDEX.items()}
DEFAULT_MODEL_VERSION: str = "phase7_mlp_v1"

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class TrainingExample:
    """One labelled functional-group descriptor ready for MLP training.

    Features come from Phase 3 (GroupDescriptor.as_vector()).
    Role label comes from Phase 4 (LabeledGroupRole.role).
    Rule-based predictions are never used as labels.
    """

    reaction_id: str
    group_id: str
    group_type: FunctionalGroupType
    features: list[float]
    role: Role
    split: str = "train"
    metadata: dict[str, str | int | float | bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "reaction_id": self.reaction_id,
            "group_id": self.group_id,
            "group_type": self.group_type.value,
            "features": list(self.features),
            "role": self.role.value,
            "split": self.split,
            "metadata": dict(self.metadata),
        }


@dataclass
class TrainingDatasetSummary:
    """Summary statistics for a collection of TrainingExamples."""

    n_examples: int
    n_features: int
    role_counts: dict[str, int]
    group_type_counts: dict[str, int]
    split_counts: dict[str, int]
    metadata: dict[str, str | int | float | bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_examples": self.n_examples,
            "n_features": self.n_features,
            "role_counts": dict(self.role_counts),
            "group_type_counts": dict(self.group_type_counts),
            "split_counts": dict(self.split_counts),
            "metadata": dict(self.metadata),
        }


@dataclass
class TrainingConfig:
    """Hyperparameter configuration for MLP training."""

    hidden_dim: int = 32
    dropout: float = 0.10
    learning_rate: float = 1e-3
    batch_size: int = 16
    epochs: int = 100
    weight_decay: float = 0.0
    use_class_weights: bool = False
    seed: int = 42
    validation_split: float = 0.2
    early_stopping_patience: int = 15
    device: str = "auto"


@dataclass
class TrainingHistory:
    """Per-epoch loss and accuracy curves recorded during training."""

    train_loss: list[float]
    val_loss: list[float]
    train_accuracy: list[float]
    val_accuracy: list[float]
    metadata: dict[str, str | int | float | bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "train_loss": list(self.train_loss),
            "val_loss": list(self.val_loss),
            "train_accuracy": list(self.train_accuracy),
            "val_accuracy": list(self.val_accuracy),
            "metadata": dict(self.metadata),
        }


@dataclass
class MLPRolePrediction:
    """Role prediction produced by MLPRolePredictor for one functional group."""

    group_id: str
    group_type: FunctionalGroupType
    predicted_role: Role
    confidence: float
    probabilities: dict[str, float]
    metadata: dict[str, str | int | float | bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "group_type": self.group_type.value,
            "predicted_role": self.predicted_role.value,
            "confidence": self.confidence,
            "probabilities": dict(self.probabilities),
            "metadata": dict(self.metadata),
        }


# ---------------------------------------------------------------------------
# Neural network
# ---------------------------------------------------------------------------


class RoleMLP(nn.Module):
    """Small MLP: Linear → ReLU → Dropout → Linear.

    Outputs raw logits over the five MENDEL roles.
    Apply softmax externally for probabilities; use raw logits with
    CrossEntropyLoss during training.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        output_dim: int = 5,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape (batch_size, input_dim).

        Returns:
            Raw logits of shape (batch_size, output_dim).  No softmax applied.
        """
        return self.net(x)


# ---------------------------------------------------------------------------
# Predictor wrapper
# ---------------------------------------------------------------------------


class MLPRolePredictor:
    """Wraps a trained RoleMLP with feature metadata and inference helpers.

    This is the Phase 7 analogue of RuleBasedRolePredictor.  It predicts
    one of the five MENDEL roles for each functional-group agent based on
    its Phase 3 descriptor vector.  It does not perform negotiation.
    """

    def __init__(
        self,
        model: RoleMLP,
        feature_names: list[str],
        device: str = "cpu",
        model_version: str = DEFAULT_MODEL_VERSION,
    ) -> None:
        self.model = model.to(device)
        self.feature_names = list(feature_names)
        self.device = device
        self.model_version = model_version

    def predict_descriptor(self, descriptor: GroupDescriptor) -> MLPRolePrediction:
        """Predict one role from a single GroupDescriptor.

        Runs the model in eval mode.  Applies softmax to produce a probability
        distribution.  Does not mutate the input descriptor.
        """
        x = torch.tensor([descriptor.values], dtype=torch.float32, device=self.device)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(x)  # (1, 5)
            probs = torch.softmax(logits, dim=-1).squeeze(0)  # (5,)
        idx = int(probs.argmax().item())
        role = INDEX_TO_ROLE[idx]
        confidence = float(probs[idx].item())
        probabilities = {INDEX_TO_ROLE[i].value: float(probs[i].item()) for i in range(5)}
        return MLPRolePrediction(
            group_id=descriptor.group_id,
            group_type=descriptor.group_type,
            predicted_role=role,
            confidence=confidence,
            probabilities=probabilities,
            metadata={"model_version": self.model_version},
        )

    def predict_descriptors(
        self,
        descriptors: list[GroupDescriptor],
    ) -> list[MLPRolePrediction]:
        """Predict roles for a list of descriptors, preserving input order."""
        return [self.predict_descriptor(d) for d in descriptors]

    def predict_from_reaction(
        self,
        parsed_reaction: ParsedReaction,
        groups: list[FunctionalGroup],
    ) -> list[MLPRolePrediction]:
        """Build Phase 3 descriptors for *groups* then predict roles.

        Args:
            parsed_reaction: Output of parse_reaction_smiles.
            groups: Output of identify_functional_groups for the same reaction.

        Returns:
            One MLPRolePrediction per group, in input order.
        """
        descriptors = build_descriptors(parsed_reaction, groups)
        return self.predict_descriptors(descriptors)

    def save(self, path: str | Path) -> None:
        """Persist model weights and metadata to *path* using torch.save."""
        checkpoint: dict[str, Any] = {
            "state_dict": self.model.state_dict(),
            "input_dim": self.model.input_dim,
            "hidden_dim": self.model.hidden_dim,
            "output_dim": self.model.output_dim,
            "dropout": self.model.dropout_rate,
            "feature_names": self.feature_names,
            "model_version": self.model_version,
        }
        torch.save(checkpoint, path)

    @staticmethod
    def load(path: str | Path, device: str = "cpu") -> MLPRolePredictor:
        """Load a saved checkpoint and return a ready-to-use MLPRolePredictor.

        Args:
            path: Path to a checkpoint saved by MLPRolePredictor.save.
            device: Device to load the model on.

        Returns:
            Reconstructed MLPRolePredictor.
        """
        ckpt: dict[str, Any] = torch.load(
            path, map_location=device, weights_only=True
        )
        model = RoleMLP(
            input_dim=int(ckpt["input_dim"]),
            hidden_dim=int(ckpt["hidden_dim"]),
            output_dim=int(ckpt.get("output_dim", 5)),
            dropout=float(ckpt.get("dropout", 0.10)),
        )
        model.load_state_dict(ckpt["state_dict"])
        return MLPRolePredictor(
            model=model,
            feature_names=list(ckpt["feature_names"]),
            device=device,
            model_version=str(ckpt.get("model_version", DEFAULT_MODEL_VERSION)),
        )


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def set_random_seed(seed: int) -> None:
    """Seed Python random, PyTorch CPU, and PyTorch CUDA for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_training_examples(
    labeled_reactions: list[LabeledReaction],
    strict_group_matching: bool = True,
    allow_draft_labels: bool = False,
) -> list[TrainingExample]:
    """Convert labeled reactions into TrainingExamples for MLP training.

    For each LabeledReaction:
    1. Parse the reaction SMILES.
    2. Identify functional groups (Phase 2).
    3. Build descriptors (Phase 3).
    4. Match labeled group roles by group_id.
    5. Create a TrainingExample for each matched group.

    Labels come exclusively from LabeledReaction.group_roles.
    Rule-based predictions are never used as labels.

    Args:
        labeled_reactions: Dataset loaded by load_labeled_reactions.
        strict_group_matching: When True, emit a UserWarning for each labeled
            group_id that does not appear in the identified groups.  When False,
            silently skip unmatched labels (partial matching is OK).
        allow_draft_labels: When False (default), skip reactions whose metadata
            has needs_manual_review=True and skip individual labels with
            confidence="draft".  When True, include them and mark
            metadata["draft_label"]=True on each example.

    Returns:
        List of TrainingExamples, one per successfully matched labeled group.
    """
    examples: list[TrainingExample] = []

    for rxn in labeled_reactions:
        # Draft-label guard at reaction level
        if not allow_draft_labels and rxn.metadata.get("needs_manual_review"):
            continue

        try:
            parsed = parse_reaction_smiles(rxn.reaction_smiles, context=rxn.context)
            groups = identify_functional_groups(parsed)
            descriptors = build_descriptors(parsed, groups)
        except Exception as exc:
            warnings.warn(
                f"Skipping reaction '{rxn.reaction_id}': {exc}",
                UserWarning,
                stacklevel=2,
            )
            continue

        desc_by_id: dict[str, GroupDescriptor] = {d.group_id: d for d in descriptors}

        for lgr in rxn.group_roles:
            # Draft-label guard at label level
            is_draft = lgr.confidence == "draft"
            if not allow_draft_labels and is_draft:
                continue

            desc = desc_by_id.get(lgr.group_id)
            if desc is None:
                if strict_group_matching:
                    warnings.warn(
                        f"[strict] Reaction '{rxn.reaction_id}': labeled group "
                        f"'{lgr.group_id}' not found in identified groups "
                        f"{list(desc_by_id.keys())}. Skipping.",
                        UserWarning,
                        stacklevel=2,
                    )
                continue

            schema_ver = desc.metadata.get("schema_version", "")
            meta: dict[str, str | int | float | bool] = {
                "reaction_id": rxn.reaction_id,
                "mechanism_type": rxn.mechanism_type,
                "schema_version": str(schema_ver),
                "strict_group_matching": strict_group_matching,
            }
            if allow_draft_labels and is_draft:
                meta["draft_label"] = True

            examples.append(
                TrainingExample(
                    reaction_id=rxn.reaction_id,
                    group_id=lgr.group_id,
                    group_type=lgr.group_type,
                    features=desc.as_vector(),
                    role=lgr.role,
                    split=rxn.split,
                    metadata=meta,
                )
            )

    return examples


def summarize_training_examples(
    examples: list[TrainingExample],
) -> TrainingDatasetSummary:
    """Compute counts and distributions for a list of TrainingExamples."""
    n = len(examples)
    n_features = len(examples[0].features) if examples else 0
    role_counts: dict[str, int] = {r.value: 0 for r in Role}
    group_type_counts: dict[str, int] = {}
    split_counts: dict[str, int] = {}

    for ex in examples:
        role_counts[ex.role.value] += 1
        gt = ex.group_type.value
        group_type_counts[gt] = group_type_counts.get(gt, 0) + 1
        split_counts[ex.split] = split_counts.get(ex.split, 0) + 1

    counts = list(role_counts.values())
    missing_roles = [role for role, count in role_counts.items() if count == 0]
    roles_below_10 = [role for role, count in role_counts.items() if count < 10]

    return TrainingDatasetSummary(
        n_examples=n,
        n_features=n_features,
        role_counts=role_counts,
        group_type_counts=group_type_counts,
        split_counts=split_counts,
        metadata={
            "schema_version": FEATURE_SCHEMA_VERSION,
            "min_role_count": min(counts) if counts else 0,
            "max_role_count": max(counts) if counts else 0,
            "missing_roles": ",".join(missing_roles),
            "roles_below_10": ",".join(roles_below_10),
        },
    )


def stratified_train_val_split(
    examples: list[TrainingExample],
    validation_split: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Return deterministic train/val indices with role-aware validation picks.

    If a role has at least two examples and the validation budget allows it,
    one example from that role is placed in validation. Remaining validation
    slots are filled from the shuffled remainder. If the dataset is too small,
    this falls back to a deterministic shuffled split.
    """
    n = len(examples)
    if n == 0:
        return [], []
    n_val = max(int(n * validation_split), 1 if n >= 2 else 0)
    if n_val == 0:
        return list(range(n)), []
    n_val = min(n_val, n - 1) if n > 1 else 0
    if n_val == 0:
        return [0], []

    rng = random.Random(seed)
    by_role: dict[Role, list[int]] = {role: [] for role in Role}
    for idx, ex in enumerate(examples):
        by_role[ex.role].append(idx)
    for indices in by_role.values():
        rng.shuffle(indices)

    eligible_roles = [role for role in Role if len(by_role[role]) >= 2]
    if len(eligible_roles) > n_val:
        all_indices = list(range(n))
        rng.shuffle(all_indices)
        val = sorted(all_indices[:n_val])
        train = sorted(all_indices[n_val:])
        return train, val

    val_set: set[int] = set()
    for role in eligible_roles:
        val_set.add(by_role[role][0])

    remaining = [idx for idx in range(n) if idx not in val_set]
    rng.shuffle(remaining)
    for idx in remaining:
        if len(val_set) >= n_val:
            break
        val_set.add(idx)

    train = sorted(idx for idx in range(n) if idx not in val_set)
    val = sorted(val_set)
    if not train or not val:
        all_indices = list(range(n))
        rng.shuffle(all_indices)
        val = sorted(all_indices[:n_val])
        train = sorted(all_indices[n_val:])
    return train, val


def training_examples_to_tensors(
    examples: list[TrainingExample],
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Convert TrainingExamples into tensors suitable for a training loop.

    Returns:
        X: Float tensor of shape (n_examples, n_features).
        y: Long tensor of shape (n_examples,) with integer class indices from
           ROLE_TO_INDEX.
        group_ids: List of group_id strings preserving input order.
    """
    if not examples:
        return torch.zeros((0, 0), dtype=torch.float32), torch.zeros(0, dtype=torch.long), []

    X = torch.tensor([ex.features for ex in examples], dtype=torch.float32)
    y = torch.tensor([ROLE_TO_INDEX[ex.role] for ex in examples], dtype=torch.long)
    group_ids = [ex.group_id for ex in examples]
    return X, y, group_ids


def compute_class_weights(examples: list[TrainingExample]) -> torch.Tensor:
    """Compute inverse-frequency class weights for CrossEntropyLoss.

    Missing classes receive weight 1.0 to avoid division by zero.

    Returns:
        Float tensor of shape (5,) with one weight per role class.
    """
    counts = [0] * 5
    for ex in examples:
        counts[ROLE_TO_INDEX[ex.role]] += 1
    n = len(examples)
    weights = [
        float(n) / (5.0 * c) if c > 0 else 1.0
        for c in counts
    ]
    return torch.tensor(weights, dtype=torch.float32)


def _resolve_device(device: str) -> str:
    """Resolve 'auto' to the best available device string."""
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train_mlp_role_predictor(
    examples: list[TrainingExample],
    config: TrainingConfig | None = None,
) -> tuple[MLPRolePredictor, TrainingHistory]:
    """Train a RoleMLP on the provided examples and return the predictor.

    Training details:
    - Loss: CrossEntropyLoss on raw logits (no pre-softmax applied).
    - Optimizer: Adam.
    - Split: deterministic train/val split shuffled with config.seed.
    - Early stopping: halts when val_loss does not improve for
      config.early_stopping_patience consecutive epochs.
    - Small-dataset fallback: when too few examples exist for a stable
      split, all examples are used for both train and val; this is
      documented in TrainingHistory.metadata["fallback_split"].

    Returns:
        (MLPRolePredictor, TrainingHistory) — predictor is moved to CPU.
    """
    if config is None:
        config = TrainingConfig()
    if not examples:
        raise ValueError("Cannot train with zero examples.")

    set_random_seed(config.seed)

    X, y, _gids = training_examples_to_tensors(examples)
    n, n_features = X.shape[0], int(X.shape[1])
    summary = summarize_training_examples(examples)
    role_counts = dict(summary.role_counts)
    missing_roles = [role for role, count in role_counts.items() if count == 0]
    roles_below_5 = [role for role, count in role_counts.items() if count < 5]
    dataset_warnings: list[str] = []
    if n < 50:
        dataset_warnings.append("n_examples_below_50")
    if missing_roles:
        dataset_warnings.append("missing_roles")
    if roles_below_5:
        dataset_warnings.append("roles_below_5")

    device = _resolve_device(config.device)

    train_idx, val_idx = stratified_train_val_split(
        examples,
        validation_split=config.validation_split,
        seed=config.seed,
    )
    fallback_split = False

    if not train_idx:
        # Degenerate: use all data for both splits
        fallback_split = True
        X_train = X_val = X.to(device)
        y_train = y_val = y.to(device)
        n_train = n
        n_val = n
    else:
        train_tensor = torch.tensor(train_idx, dtype=torch.long)
        X_train = X[train_tensor].to(device)
        y_train = y[train_tensor].to(device)
        n_train = len(train_idx)
        if val_idx:
            val_tensor = torch.tensor(val_idx, dtype=torch.long)
            X_val = X[val_tensor].to(device)
            y_val = y[val_tensor].to(device)
            n_val = len(val_idx)
        else:
            # n == 1: mirror train as val fallback
            fallback_split = True
            X_val, y_val = X_train, y_train
            n_val = n_train

    train_roles = {INDEX_TO_ROLE[int(i)].value for i in y_train.detach().cpu().tolist()}
    val_roles = {INDEX_TO_ROLE[int(i)].value for i in y_val.detach().cpu().tolist()}
    missing_train_classes = [role.value for role in Role if role.value not in train_roles]
    missing_val_classes = [role.value for role in Role if role.value not in val_roles]
    if missing_train_classes:
        dataset_warnings.append("train_split_missing_classes")
    if missing_val_classes:
        dataset_warnings.append("val_split_missing_classes")

    model = RoleMLP(
        input_dim=n_features,
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    if config.use_class_weights:
        criterion: nn.CrossEntropyLoss = nn.CrossEntropyLoss(
            weight=compute_class_weights(examples).to(device)
        )
    else:
        criterion = nn.CrossEntropyLoss()

    history = TrainingHistory(
        train_loss=[],
        val_loss=[],
        train_accuracy=[],
        val_accuracy=[],
        metadata={
            "n_train": n_train,
            "n_val": n_val,
            "n_examples": n,
            "device": device,
            "fallback_split": fallback_split,
            "split_strategy": "stratified_train_val_split",
            "role_counts": role_counts,
            "missing_roles": ",".join(missing_roles),
            "roles_below_5": ",".join(roles_below_5),
            "missing_train_classes": ",".join(missing_train_classes),
            "missing_val_classes": ",".join(missing_val_classes),
            "dataset_warnings": ",".join(sorted(set(dataset_warnings))),
        },
    )

    best_val_loss = float("inf")
    best_state_dict: dict | None = None
    patience_counter = 0
    epochs_run = 0

    for _epoch in range(config.epochs):
        model.train()

        batch_perm = torch.randperm(n_train, device=device)
        total_loss = 0.0
        total_correct = 0

        for start in range(0, n_train, config.batch_size):
            idx = batch_perm[start: start + config.batch_size]
            xb, yb = X_train[idx], y_train[idx]
            optimizer.zero_grad()
            logits = model(xb)  # raw logits — CrossEntropyLoss expects these
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)
            total_correct += int((logits.argmax(dim=1) == yb).sum().item())

        train_loss = total_loss / n_train
        train_acc = total_correct / n_train

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val)
            val_loss = float(criterion(val_logits, y_val).item())
            val_acc = float((val_logits.argmax(dim=1) == y_val).float().mean().item())

        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)
        history.train_accuracy.append(train_acc)
        history.val_accuracy.append(val_acc)
        epochs_run += 1

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.early_stopping_patience:
                break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    history.metadata["epochs_run"] = epochs_run
    history.metadata["stopped_early"] = epochs_run < config.epochs

    predictor = MLPRolePredictor(
        model=model.to("cpu"),
        feature_names=get_feature_names(),
        device="cpu",
        model_version=DEFAULT_MODEL_VERSION,
    )
    return predictor, history


def evaluate_mlp_predictor(
    predictor: MLPRolePredictor,
    examples: list[TrainingExample],
) -> dict[str, Any]:
    """Evaluate a trained predictor against a labelled example set.

    Returns a dict containing:
    - n_examples: int
    - accuracy: float in [0, 1]
    - role_counts: ground-truth role distribution
    - predicted_role_counts: predicted role distribution
    - confusion_matrix: nested dict {true_role: {pred_role: count}}
    - mismatches: list of per-example dicts for incorrect predictions
    """
    role_vals = [r.value for r in Role]
    confusion: dict[str, dict[str, int]] = {r: {r2: 0 for r2 in role_vals} for r in role_vals}
    mismatches: list[dict[str, Any]] = []
    role_counts: dict[str, int] = {r.value: 0 for r in Role}
    predicted_counts: dict[str, int] = {r.value: 0 for r in Role}
    correct = 0

    for ex in examples:
        desc = GroupDescriptor(
            group_id=ex.group_id,
            group_type=ex.group_type,
            feature_names=predictor.feature_names,
            values=list(ex.features),
        )
        pred = predictor.predict_descriptor(desc)
        true_val = ex.role.value
        pred_val = pred.predicted_role.value

        role_counts[true_val] += 1
        predicted_counts[pred_val] += 1
        confusion[true_val][pred_val] += 1

        if true_val == pred_val:
            correct += 1
        else:
            mismatches.append({
                "group_id": ex.group_id,
                "true_role": true_val,
                "predicted_role": pred_val,
                "confidence": pred.confidence,
            })

    n = len(examples)
    return {
        "n_examples": n,
        "accuracy": correct / n if n > 0 else 0.0,
        "role_counts": role_counts,
        "predicted_role_counts": predicted_counts,
        "confusion_matrix": confusion,
        "mismatches": mismatches,
    }


def train_from_labeled_json(
    path: str | Path,
    config: TrainingConfig | None = None,
    strict_group_matching: bool = True,
    allow_draft_labels: bool = False,
) -> tuple[MLPRolePredictor, TrainingHistory, dict[str, Any]]:
    """One-call pipeline: load JSON → build examples → train → evaluate.

    Args:
        path: Path to a labeled reactions JSON file.
        config: Optional TrainingConfig; uses defaults if None.
        strict_group_matching: Passed to build_training_examples.
        allow_draft_labels: When True, include draft-labeled examples (for smoke
            testing only).  When False, skip reactions/labels marked as drafts.

    Returns:
        (predictor, history, evaluation_report)
    """
    from mendel.labels import load_labeled_reactions

    reactions = load_labeled_reactions(path)
    examples = build_training_examples(
        reactions,
        strict_group_matching=strict_group_matching,
        allow_draft_labels=allow_draft_labels,
    )
    predictor, history = train_mlp_role_predictor(examples, config=config)
    report = evaluate_mlp_predictor(predictor, examples)
    return predictor, history, report


def save_training_report(report: dict[str, Any], path: str | Path) -> None:
    """Write an evaluation report dict to a JSON file with readable indentation."""
    Path(path).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
