"""Phase 8: Benchmark evaluators for MENDELV role predictors.

Benchmarks are label-conditioned: ground truth comes only from
LabeledReaction.group_roles. Unlabeled detected groups are ignored for role
accuracy, while labeled groups missing from predictions count as incorrect.

No training, MLIP, MACE, energies, forces, transition states, or barriers are
used here. Functional group = agent remains the benchmark unit.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mendel.descriptor import build_descriptors
from mendel.identifier import identify_functional_groups
from mendel.labels import LabeledGroupRole, LabeledReaction
from mendel.negotiator import NegotiatorConfig, negotiate_predictions
from mendel.parser import parse_reaction_smiles
from mendel.predictor import RolePrediction, predict_roles_for_reaction
from mendel.types import FunctionalGroup, FunctionalGroupType, Role

Scalar = str | int | float | bool


@dataclass
class GroupBenchmarkRecord:
    """One labeled functional-group role comparison."""

    reaction_id: str
    group_id: str
    group_type: FunctionalGroupType
    true_role: Role
    predicted_role: Role
    predicted_confidence: float | None
    predictor_name: str
    correct: bool
    split: str
    mechanism_type: str
    metadata: dict[str, Scalar] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "reaction_id": self.reaction_id,
            "group_id": self.group_id,
            "group_type": self.group_type.value,
            "true_role": self.true_role.value,
            "predicted_role": self.predicted_role.value,
            "predicted_confidence": self.predicted_confidence,
            "predictor_name": self.predictor_name,
            "correct": self.correct,
            "split": self.split,
            "mechanism_type": self.mechanism_type,
            "metadata": dict(self.metadata),
        }


@dataclass
class ReactionBenchmarkRecord:
    """Reaction-level benchmark summary for one predictor and reaction."""

    reaction_id: str
    reaction_smiles: str
    split: str
    mechanism_type: str
    predictor_name: str
    n_labeled_groups: int
    n_correct_roles: int
    role_accuracy: float
    mechanism_hint: str | None
    reaction_center_precision: float | None
    reaction_center_recall: float | None
    reaction_center_f1: float | None
    warnings: list[str]
    metadata: dict[str, Scalar] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "reaction_id": self.reaction_id,
            "reaction_smiles": self.reaction_smiles,
            "split": self.split,
            "mechanism_type": self.mechanism_type,
            "predictor_name": self.predictor_name,
            "n_labeled_groups": self.n_labeled_groups,
            "n_correct_roles": self.n_correct_roles,
            "role_accuracy": self.role_accuracy,
            "mechanism_hint": self.mechanism_hint,
            "reaction_center_precision": self.reaction_center_precision,
            "reaction_center_recall": self.reaction_center_recall,
            "reaction_center_f1": self.reaction_center_f1,
            "warnings": list(self.warnings),
            "metadata": dict(self.metadata),
        }


@dataclass
class BenchmarkReport:
    """Full benchmark report for one predictor."""

    predictor_name: str
    n_reactions: int
    n_group_labels: int
    overall_role_accuracy: float
    role_accuracy_by_role: dict[str, float]
    role_accuracy_by_group_type: dict[str, float]
    role_accuracy_by_mechanism: dict[str, float]
    split_accuracy: dict[str, float]
    confusion_matrix: dict[str, dict[str, int]]
    reaction_center_precision: float | None
    reaction_center_recall: float | None
    reaction_center_f1: float | None
    group_records: list[GroupBenchmarkRecord]
    reaction_records: list[ReactionBenchmarkRecord]
    metadata: dict[str, Scalar] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "predictor_name": self.predictor_name,
            "n_reactions": self.n_reactions,
            "n_group_labels": self.n_group_labels,
            "overall_role_accuracy": self.overall_role_accuracy,
            "role_accuracy_by_role": dict(self.role_accuracy_by_role),
            "role_accuracy_by_group_type": dict(self.role_accuracy_by_group_type),
            "role_accuracy_by_mechanism": dict(self.role_accuracy_by_mechanism),
            "split_accuracy": dict(self.split_accuracy),
            "confusion_matrix": {
                row: dict(cols) for row, cols in self.confusion_matrix.items()
            },
            "reaction_center_precision": self.reaction_center_precision,
            "reaction_center_recall": self.reaction_center_recall,
            "reaction_center_f1": self.reaction_center_f1,
            "group_records": [record.to_dict() for record in self.group_records],
            "reaction_records": [record.to_dict() for record in self.reaction_records],
            "metadata": dict(self.metadata),
        }


def build_confusion_matrix(
    records: list[GroupBenchmarkRecord],
) -> dict[str, dict[str, int]]:
    """Return nested counts {true_role: {predicted_role: count}}."""
    roles = [role.value for role in Role]
    matrix: dict[str, dict[str, int]] = {
        true_role: {pred_role: 0 for pred_role in roles} for true_role in roles
    }
    for record in records:
        matrix[record.true_role.value][record.predicted_role.value] += 1
    return matrix


def _accuracy_by_key(records: list[GroupBenchmarkRecord], key: str) -> dict[str, float]:
    totals: dict[str, int] = {}
    correct: dict[str, int] = {}
    for record in records:
        if key == "role":
            value = record.true_role.value
        elif key == "group_type":
            value = record.group_type.value
        elif key == "mechanism":
            value = record.mechanism_type
        elif key == "split":
            value = record.split
        else:
            raise ValueError(f"Unknown accuracy key: {key}")
        totals[value] = totals.get(value, 0) + 1
        if record.correct:
            correct[value] = correct.get(value, 0) + 1
    return {value: correct.get(value, 0) / total for value, total in sorted(totals.items())}


def compute_accuracy_by_role(records: list[GroupBenchmarkRecord]) -> dict[str, float]:
    """Accuracy grouped by true role denominator."""
    return _accuracy_by_key(records, "role")


def compute_accuracy_by_group_type(records: list[GroupBenchmarkRecord]) -> dict[str, float]:
    """Accuracy grouped by labeled functional group type."""
    return _accuracy_by_key(records, "group_type")


def compute_accuracy_by_mechanism(records: list[GroupBenchmarkRecord]) -> dict[str, float]:
    """Accuracy grouped by LabeledReaction.mechanism_type."""
    return _accuracy_by_key(records, "mechanism")


def compute_split_accuracy(records: list[GroupBenchmarkRecord]) -> dict[str, float]:
    """Accuracy grouped by dataset split."""
    return _accuracy_by_key(records, "split")


def compute_reaction_center_metrics(
    predicted_atoms: list[int],
    true_atoms: list[int],
) -> dict[str, float | None]:
    """Compute deterministic set precision/recall/F1 for reaction-center atoms.

    Empty-case convention:
    - both empty: all metrics are None because there is no labeled center task
    - true empty, predicted non-empty: precision is 0.0, recall/F1 are None
    - predicted empty, true non-empty: precision, recall, and F1 are 0.0
    """
    predicted = set(predicted_atoms)
    true = set(true_atoms)
    if not predicted and not true:
        return {"precision": None, "recall": None, "f1": None}
    if not true:
        return {"precision": 0.0, "recall": None, "f1": None}

    intersection = len(predicted & true)
    precision = intersection / len(predicted) if predicted else 0.0
    recall = intersection / len(true)
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0.0
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def _atom_map_values(groups: list[FunctionalGroup], group_id: str) -> list[int]:
    group = next((g for g in groups if g.group_id == group_id), None)
    if group is None:
        return []
    values: list[int] = []
    for ref in group.atom_refs:
        value = ref.atom_map_num if ref.atom_map_num is not None else ref.atom_index
        if value not in values:
            values.append(value)
    return values


def _reaction_center_values_from_result(result_atoms: list[Any]) -> list[int]:
    values: list[int] = []
    for ref in result_atoms:
        value = ref.atom_map_num if ref.atom_map_num is not None else ref.atom_index
        if value not in values:
            values.append(value)
    return values


def _record_for_missing_group(
    labeled_reaction: LabeledReaction,
    label: LabeledGroupRole,
    predictor_name: str,
    predicted_role: Role = Role.spectator,
) -> GroupBenchmarkRecord:
    return GroupBenchmarkRecord(
        reaction_id=labeled_reaction.reaction_id,
        group_id=label.group_id,
        group_type=label.group_type,
        true_role=label.role,
        predicted_role=predicted_role,
        predicted_confidence=None,
        predictor_name=predictor_name,
        correct=False,
        split=labeled_reaction.split,
        mechanism_type=labeled_reaction.mechanism_type,
        metadata={"missing_group": True},
    )


def _group_records_from_predictions(
    labeled_reaction: LabeledReaction,
    groups: list[FunctionalGroup],
    predictions: dict[str, tuple[Role, float | None, dict[str, Scalar]]],
    predictor_name: str,
) -> list[GroupBenchmarkRecord]:
    records: list[GroupBenchmarkRecord] = []
    for label in labeled_reaction.group_roles:
        pred = predictions.get(label.group_id)
        if pred is None:
            records.append(_record_for_missing_group(labeled_reaction, label, predictor_name))
            continue

        predicted_role, confidence, metadata = pred
        correct = predicted_role == label.role
        record_metadata: dict[str, Scalar] = dict(metadata)
        detected_type = next(
            (group.group_type.value for group in groups if group.group_id == label.group_id),
            None,
        )
        if detected_type is not None:
            record_metadata["detected_group_type"] = detected_type

        records.append(
            GroupBenchmarkRecord(
                reaction_id=labeled_reaction.reaction_id,
                group_id=label.group_id,
                group_type=label.group_type,
                true_role=label.role,
                predicted_role=predicted_role,
                predicted_confidence=confidence,
                predictor_name=predictor_name,
                correct=correct,
                split=labeled_reaction.split,
                mechanism_type=labeled_reaction.mechanism_type,
                metadata=record_metadata,
            )
        )
    return records


def _reaction_record(
    labeled_reaction: LabeledReaction,
    predictor_name: str,
    records: list[GroupBenchmarkRecord],
    mechanism_hint: str | None = None,
    predicted_center_atoms: list[int] | None = None,
    warnings: list[str] | None = None,
    metadata: dict[str, Scalar] | None = None,
) -> ReactionBenchmarkRecord:
    n = len(records)
    n_correct = sum(1 for record in records if record.correct)
    center_metrics = {"precision": None, "recall": None, "f1": None}
    record_metadata: dict[str, Scalar] = dict(metadata or {})
    if predicted_center_atoms is not None or labeled_reaction.reaction_center_atoms:
        record_metadata["true_reaction_center_atoms"] = ",".join(
            str(atom) for atom in labeled_reaction.reaction_center_atoms
        )
        record_metadata["predicted_reaction_center_atoms"] = ",".join(
            str(atom) for atom in (predicted_center_atoms or [])
        )
    if predicted_center_atoms is not None or labeled_reaction.reaction_center_atoms:
        center_metrics = compute_reaction_center_metrics(
            predicted_center_atoms or [], labeled_reaction.reaction_center_atoms
        )

    return ReactionBenchmarkRecord(
        reaction_id=labeled_reaction.reaction_id,
        reaction_smiles=labeled_reaction.reaction_smiles,
        split=labeled_reaction.split,
        mechanism_type=labeled_reaction.mechanism_type,
        predictor_name=predictor_name,
        n_labeled_groups=n,
        n_correct_roles=n_correct,
        role_accuracy=n_correct / n if n else 0.0,
        mechanism_hint=mechanism_hint,
        reaction_center_precision=center_metrics["precision"],
        reaction_center_recall=center_metrics["recall"],
        reaction_center_f1=center_metrics["f1"],
        warnings=warnings or [],
        metadata=record_metadata,
    )


def _average_optional(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return sum(present) / len(present) if present else None


def _build_report(
    predictor_name: str,
    labeled_reactions: list[LabeledReaction],
    group_records: list[GroupBenchmarkRecord],
    reaction_records: list[ReactionBenchmarkRecord],
    metadata: dict[str, Scalar] | None = None,
) -> BenchmarkReport:
    n = len(group_records)
    correct = sum(1 for record in group_records if record.correct)
    return BenchmarkReport(
        predictor_name=predictor_name,
        n_reactions=len(labeled_reactions),
        n_group_labels=n,
        overall_role_accuracy=correct / n if n else 0.0,
        role_accuracy_by_role=compute_accuracy_by_role(group_records),
        role_accuracy_by_group_type=compute_accuracy_by_group_type(group_records),
        role_accuracy_by_mechanism=compute_accuracy_by_mechanism(group_records),
        split_accuracy=compute_split_accuracy(group_records),
        confusion_matrix=build_confusion_matrix(group_records),
        reaction_center_precision=_average_optional(
            [record.reaction_center_precision for record in reaction_records]
        ),
        reaction_center_recall=_average_optional(
            [record.reaction_center_recall for record in reaction_records]
        ),
        reaction_center_f1=_average_optional(
            [record.reaction_center_f1 for record in reaction_records]
        ),
        group_records=group_records,
        reaction_records=reaction_records,
        metadata=metadata or {},
    )


def evaluate_rule_based_predictor(
    labeled_reactions: list[LabeledReaction],
) -> BenchmarkReport:
    """Evaluate the Phase 5 rule-based local role predictor."""
    predictor_name = "rule_based_local"
    all_group_records: list[GroupBenchmarkRecord] = []
    reaction_records: list[ReactionBenchmarkRecord] = []

    for rxn in labeled_reactions:
        parsed = parse_reaction_smiles(rxn.reaction_smiles, context=rxn.context)
        groups = identify_functional_groups(parsed)
        report = predict_roles_for_reaction(parsed, groups)
        predictions = {
            pred.group_id: (
                pred.predicted_role,
                pred.confidence,
                {"reason": pred.reason},
            )
            for pred in report.predictions
        }
        records = _group_records_from_predictions(rxn, groups, predictions, predictor_name)
        all_group_records.extend(records)
        reaction_records.append(
            _reaction_record(
                rxn,
                predictor_name,
                records,
                metadata={"n_detected_groups": len(groups)},
            )
        )

    return _build_report(
        predictor_name,
        labeled_reactions,
        all_group_records,
        reaction_records,
        {"benchmark_level": "group_role_local"},
    )


def evaluate_negotiated_rule_based(
    labeled_reactions: list[LabeledReaction],
) -> BenchmarkReport:
    """Evaluate Phase 5 rule predictions after Phase 6 negotiation."""
    predictor_name = "rule_based_negotiated"
    all_group_records: list[GroupBenchmarkRecord] = []
    reaction_records: list[ReactionBenchmarkRecord] = []

    for rxn in labeled_reactions:
        parsed = parse_reaction_smiles(rxn.reaction_smiles, context=rxn.context)
        groups = identify_functional_groups(parsed)
        local_report = predict_roles_for_reaction(parsed, groups)
        result = negotiate_predictions(parsed, groups, local_report.predictions)
        predictions = {
            assignment.group_id: (
                assignment.final_role,
                assignment.final_confidence,
                {
                    "raw_role": assignment.raw_role.value,
                    "subrole": assignment.subrole or "",
                    "is_reaction_center": assignment.is_reaction_center,
                    "reason": assignment.reason,
                },
            )
            for assignment in result.assignments
        }
        records = _group_records_from_predictions(rxn, groups, predictions, predictor_name)
        all_group_records.extend(records)
        reaction_records.append(
            _reaction_record(
                rxn,
                predictor_name,
                records,
                mechanism_hint=result.mechanism_hint,
                predicted_center_atoms=_reaction_center_values_from_result(
                    result.reaction_center_atoms
                ),
                warnings=[warning.code for warning in result.warnings],
                metadata={"n_detected_groups": len(groups)},
            )
        )

    return _build_report(
        predictor_name,
        labeled_reactions,
        all_group_records,
        reaction_records,
        {"benchmark_level": "group_role_negotiated"},
    )


def evaluate_mlp_predictor(
    labeled_reactions: list[LabeledReaction],
    predictor: Any,
) -> BenchmarkReport:
    """Evaluate an existing MLPRolePredictor without importing torch here."""
    predictor_name = "mlp_local"
    all_group_records: list[GroupBenchmarkRecord] = []
    reaction_records: list[ReactionBenchmarkRecord] = []

    for rxn in labeled_reactions:
        parsed = parse_reaction_smiles(rxn.reaction_smiles, context=rxn.context)
        groups = identify_functional_groups(parsed)
        descriptors = build_descriptors(parsed, groups)
        mlp_predictions = predictor.predict_descriptors(descriptors)
        predictions = {
            pred.group_id: (
                pred.predicted_role,
                pred.confidence,
                {"model_version": str(pred.metadata.get("model_version", ""))},
            )
            for pred in mlp_predictions
        }
        records = _group_records_from_predictions(rxn, groups, predictions, predictor_name)
        all_group_records.extend(records)
        reaction_records.append(
            _reaction_record(
                rxn,
                predictor_name,
                records,
                metadata={"n_detected_groups": len(groups)},
            )
        )

    return _build_report(
        predictor_name,
        labeled_reactions,
        all_group_records,
        reaction_records,
        {"benchmark_level": "group_role_mlp_local"},
    )


def evaluate_negotiated_mlp_predictor(
    labeled_reactions: list[LabeledReaction],
    predictor: Any,
    predictor_name: str = "mlp_negotiated",
    negotiator_config: NegotiatorConfig | None = None,
) -> BenchmarkReport:
    """Evaluate MLP local predictions after Phase 6 negotiation.

    This is feasible because the negotiator consumes RolePrediction objects.
    The MLP still provides only local roles; Phase 6 remains the global
    consistency layer.
    """
    all_group_records: list[GroupBenchmarkRecord] = []
    reaction_records: list[ReactionBenchmarkRecord] = []

    for rxn in labeled_reactions:
        parsed = parse_reaction_smiles(rxn.reaction_smiles, context=rxn.context)
        parsed.metadata["mechanism_type"] = rxn.mechanism_type
        groups = identify_functional_groups(parsed)
        descriptors = build_descriptors(parsed, groups)
        mlp_predictions = predictor.predict_descriptors(descriptors)
        role_predictions = [
            RolePrediction(
                group_id=pred.group_id,
                group_type=pred.group_type,
                predicted_role=pred.predicted_role,
                confidence=pred.confidence,
                reason="MLP local role prediction used as negotiation input",
                scores={},
                metadata={
                    "source_predictor": "mlp_local",
                    "prediction_source": "mlp",
                    "predictor_name": predictor_name,
                    "model_version": str(getattr(pred, "metadata", {}).get("model_version", "")),
                },
            )
            for pred in mlp_predictions
        ]
        result = negotiate_predictions(
            parsed,
            groups,
            role_predictions,
            config=negotiator_config,
        )
        predictions = {
            assignment.group_id: (
                assignment.final_role,
                assignment.final_confidence,
                {
                    "raw_role": assignment.raw_role.value,
                    "subrole": assignment.subrole or "",
                    "is_reaction_center": assignment.is_reaction_center,
                    "reason": assignment.reason,
                },
            )
            for assignment in result.assignments
        }
        records = _group_records_from_predictions(rxn, groups, predictions, predictor_name)
        all_group_records.extend(records)
        reaction_records.append(
            _reaction_record(
                rxn,
                predictor_name,
                records,
                mechanism_hint=result.mechanism_hint,
                predicted_center_atoms=_reaction_center_values_from_result(
                    result.reaction_center_atoms
                ),
                warnings=[warning.code for warning in result.warnings],
                metadata={"n_detected_groups": len(groups)},
            )
        )

    return _build_report(
        predictor_name,
        labeled_reactions,
        all_group_records,
        reaction_records,
        {"benchmark_level": "group_role_mlp_negotiated"},
    )


def evaluate_mlp_checkpoint(
    labeled_reactions: list[LabeledReaction],
    checkpoint_path: str | Path,
    device: str = "cpu",
) -> BenchmarkReport:
    """Load an MLP checkpoint and evaluate it.

    Importing torch remains isolated inside mendel.mlp, so importing
    mendel.benchmark does not require torch.
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"MLP checkpoint does not exist: {path}")

    from mendel.mlp import MLPRolePredictor

    predictor = MLPRolePredictor.load(path, device=device)
    return evaluate_mlp_predictor(labeled_reactions, predictor)


def evaluate_negotiated_mlp_checkpoint(
    labeled_reactions: list[LabeledReaction],
    checkpoint_path: str | Path,
    device: str = "cpu",
    predictor_name: str = "mlp_negotiated",
    negotiator_config: NegotiatorConfig | None = None,
) -> BenchmarkReport:
    """Load an MLP checkpoint and evaluate MLP predictions after negotiation."""
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"MLP checkpoint does not exist: {path}")

    from mendel.mlp import MLPRolePredictor

    predictor = MLPRolePredictor.load(path, device=device)
    return evaluate_negotiated_mlp_predictor(
        labeled_reactions,
        predictor,
        predictor_name=predictor_name,
        negotiator_config=negotiator_config,
    )


def evaluate_mlp_aware_negotiated_checkpoint(
    labeled_reactions: list[LabeledReaction],
    checkpoint_path: str | Path,
    device: str = "cpu",
    predictor_name: str = "new_mlp_aware_negotiated",
) -> BenchmarkReport:
    """Evaluate MLP predictions with Phase 8.9 MLP-aware negotiation."""
    return evaluate_negotiated_mlp_checkpoint(
        labeled_reactions,
        checkpoint_path,
        device=device,
        predictor_name=predictor_name,
        negotiator_config=NegotiatorConfig(mode="mlp_aware"),
    )



def compare_benchmark_reports(
    reports: list[BenchmarkReport],
) -> dict[str, object]:
    """Compare multiple benchmark reports with compact metric tables."""
    predictor_names = [report.predictor_name for report in reports]
    overall = {
        report.predictor_name: report.overall_role_accuracy for report in reports
    }
    per_role = {
        role.value: {
            report.predictor_name: report.role_accuracy_by_role.get(role.value)
            for report in reports
        }
        for role in Role
    }
    mechanisms = sorted({
        mech
        for report in reports
        for mech in report.role_accuracy_by_mechanism
    })
    per_mechanism = {
        mechanism: {
            report.predictor_name: report.role_accuracy_by_mechanism.get(mechanism)
            for report in reports
        }
        for mechanism in mechanisms
    }
    reaction_center_available = any(
        report.reaction_center_precision is not None
        or report.reaction_center_recall is not None
        or report.reaction_center_f1 is not None
        for report in reports
    )
    reaction_center = (
        {
            report.predictor_name: {
                "precision": report.reaction_center_precision,
                "recall": report.reaction_center_recall,
                "f1": report.reaction_center_f1,
            }
            for report in reports
        }
        if reaction_center_available
        else {}
    )
    best = None
    if reports:
        best = max(
            reports,
            key=lambda report: (report.overall_role_accuracy, report.predictor_name),
        ).predictor_name

    notes = [
        "Ground truth comes only from LabeledReaction.group_roles.",
        "Unlabeled detected groups are ignored for role accuracy.",
        "Missing labeled group IDs count as incorrect.",
    ]
    if not reaction_center_available:
        notes.append("No aggregate reaction-center metrics were available.")

    return {
        "predictor_names": predictor_names,
        "overall_role_accuracy": overall,
        "per_role_accuracy": per_role,
        "per_mechanism_accuracy": per_mechanism,
        "reaction_center": reaction_center,
        "best_predictor_by_overall_accuracy": best,
        "notes": notes,
    }


def save_benchmark_report(report: BenchmarkReport, path: str | Path) -> None:
    """Write a BenchmarkReport as readable JSON."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def save_benchmark_comparison(
    comparison: dict[str, object],
    path: str | Path,
) -> None:
    """Write a benchmark comparison dict as readable JSON."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_benchmark_report(path: str | Path) -> dict[str, object]:
    """Load a saved benchmark report JSON into a plain dict."""
    return json.loads(Path(path).read_text(encoding="utf-8"))
