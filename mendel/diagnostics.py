"""Phase 8.8 diagnostics for MLP role and reaction-center failures.

These utilities analyze saved benchmark reports. They do not train models and
do not use MLIP, MACE, energy, force, or transition-state machinery.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mendel.labels import LabeledReaction, load_labeled_reactions

Scalar = str | int | float | bool

REACTIVE_ROLES = {
    "reactive_nucleophile",
    "reactive_electrophile",
    "reactive_radical",
    "leaving_group",
}


@dataclass
class PredictionDisagreementRecord:
    reaction_id: str
    mechanism_type: str
    split: str
    group_id: str
    group_type: str
    true_role: str
    rule_based_role: str | None
    old_mlp_role: str | None
    new_mlp_role: str | None
    rule_based_correct: bool | None
    old_mlp_correct: bool | None
    new_mlp_correct: bool | None
    new_mlp_confidence: float | None
    notes: list[str]
    metadata: dict[str, Scalar] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "reaction_id": self.reaction_id,
            "mechanism_type": self.mechanism_type,
            "split": self.split,
            "group_id": self.group_id,
            "group_type": self.group_type,
            "true_role": self.true_role,
            "rule_based_role": self.rule_based_role,
            "old_mlp_role": self.old_mlp_role,
            "new_mlp_role": self.new_mlp_role,
            "rule_based_correct": self.rule_based_correct,
            "old_mlp_correct": self.old_mlp_correct,
            "new_mlp_correct": self.new_mlp_correct,
            "new_mlp_confidence": self.new_mlp_confidence,
            "notes": list(self.notes),
            "metadata": dict(self.metadata),
        }


@dataclass
class ReactionCenterFailureRecord:
    reaction_id: str
    mechanism_type: str
    split: str
    predictor_name: str
    true_reaction_center_atoms: list[int]
    predicted_reaction_center_atoms: list[int]
    missing_atoms: list[int]
    extra_atoms: list[int]
    precision: float | None
    recall: float | None
    f1: float | None
    role_accuracy_for_reaction: float
    n_labeled_groups: int
    n_correct_roles: int
    failure_type: str
    notes: list[str]
    metadata: dict[str, Scalar] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "reaction_id": self.reaction_id,
            "mechanism_type": self.mechanism_type,
            "split": self.split,
            "predictor_name": self.predictor_name,
            "true_reaction_center_atoms": list(self.true_reaction_center_atoms),
            "predicted_reaction_center_atoms": list(self.predicted_reaction_center_atoms),
            "missing_atoms": list(self.missing_atoms),
            "extra_atoms": list(self.extra_atoms),
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "role_accuracy_for_reaction": self.role_accuracy_for_reaction,
            "n_labeled_groups": self.n_labeled_groups,
            "n_correct_roles": self.n_correct_roles,
            "failure_type": self.failure_type,
            "notes": list(self.notes),
            "metadata": dict(self.metadata),
        }


@dataclass
class MLPCalibrationBin:
    bin_start: float
    bin_end: float
    n: int
    accuracy: float | None
    mean_confidence: float | None
    expected_calibration_error_component: float | None

    def to_dict(self) -> dict[str, object]:
        return {
            "bin_start": self.bin_start,
            "bin_end": self.bin_end,
            "n": self.n,
            "accuracy": self.accuracy,
            "mean_confidence": self.mean_confidence,
            "expected_calibration_error_component": (
                self.expected_calibration_error_component
            ),
        }


@dataclass
class DiagnosticsReport:
    dataset_path: str
    n_reactions: int
    n_group_labels: int
    predictor_names: list[str]
    role_accuracy_summary: dict[str, float]
    reaction_center_summary: dict[str, dict[str, float | None]]
    disagreement_counts: dict[str, int]
    mlp_calibration_bins: list[MLPCalibrationBin]
    reaction_center_failures: list[ReactionCenterFailureRecord]
    disagreement_records: list[PredictionDisagreementRecord]
    top_failure_patterns: list[dict[str, object]]
    recommendations: list[str]
    metadata: dict[str, Scalar] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset_path": self.dataset_path,
            "n_reactions": self.n_reactions,
            "n_group_labels": self.n_group_labels,
            "predictor_names": list(self.predictor_names),
            "role_accuracy_summary": dict(self.role_accuracy_summary),
            "reaction_center_summary": {
                name: dict(metrics)
                for name, metrics in self.reaction_center_summary.items()
            },
            "disagreement_counts": dict(self.disagreement_counts),
            "mlp_calibration_bins": [
                bin_.to_dict() for bin_ in self.mlp_calibration_bins
            ],
            "reaction_center_failures": [
                failure.to_dict() for failure in self.reaction_center_failures
            ],
            "disagreement_records": [
                disagreement.to_dict() for disagreement in self.disagreement_records
            ],
            "top_failure_patterns": list(self.top_failure_patterns),
            "recommendations": list(self.recommendations),
            "metadata": dict(self.metadata),
        }


def load_benchmark_records(report_path: str | Path) -> dict[str, object]:
    data = json.loads(Path(report_path).read_text(encoding="utf-8"))
    has_group = "group_records" in data
    has_reaction = "reaction_records" in data
    if not (has_group or has_reaction):
        raise ValueError(
            f"{report_path} has no detailed benchmark records. Rerun "
            "scripts/benchmark_promoted_mlp.py so predictor JSON files include "
            "group_records and reaction_records."
        )
    return data


def _records_by_group(report: dict[str, object]) -> dict[tuple[str, str], dict[str, object]]:
    return {
        (str(record["reaction_id"]), str(record["group_id"])): record
        for record in report.get("group_records", [])
        if isinstance(record, dict)
    }


def _record_role(record: dict[str, object] | None) -> str | None:
    return str(record["predicted_role"]) if record else None


def _record_correct(record: dict[str, object] | None) -> bool | None:
    return bool(record["correct"]) if record else None


def _disagreement_notes(
    true_role: str,
    rule_record: dict[str, object] | None,
    new_record: dict[str, object],
    old_record: dict[str, object] | None,
) -> list[str]:
    notes: list[str] = []
    rule_correct = _record_correct(rule_record)
    new_correct = _record_correct(new_record)
    old_correct = _record_correct(old_record)
    new_role = str(new_record["predicted_role"])
    confidence = new_record.get("predicted_confidence")
    if rule_correct is True and new_correct is False:
        notes.append("rule_correct_new_mlp_wrong")
    if rule_correct is False and new_correct is True:
        notes.append("rule_wrong_new_mlp_correct")
    if rule_correct is False and new_correct is False:
        notes.append("both_wrong")
    if old_correct is False and new_correct is True:
        notes.append("new_mlp_improved_over_old")
    if true_role == "spectator" and new_role in REACTIVE_ROLES:
        notes.append("spectator_predicted_reactive")
    if true_role in REACTIVE_ROLES and new_role == "spectator":
        notes.append("reactive_predicted_spectator")
    if (
        isinstance(confidence, int | float)
        and float(confidence) >= 0.8
        and new_correct is False
    ):
        notes.append("new_mlp_high_confidence_wrong")
    return notes


def collect_prediction_disagreements(
    labeled_reactions: list[LabeledReaction],
    rule_report: Any,
    new_mlp_report: Any,
    old_mlp_report: Any | None = None,
) -> list[PredictionDisagreementRecord]:
    del labeled_reactions
    rule = rule_report if isinstance(rule_report, dict) else rule_report.to_dict()
    new = new_mlp_report if isinstance(new_mlp_report, dict) else new_mlp_report.to_dict()
    old = (
        old_mlp_report
        if old_mlp_report is None or isinstance(old_mlp_report, dict)
        else old_mlp_report.to_dict()
    )
    rule_by_group = _records_by_group(rule)
    old_by_group = _records_by_group(old or {})
    records: list[PredictionDisagreementRecord] = []
    for key, new_record in _records_by_group(new).items():
        rule_record = rule_by_group.get(key)
        old_record = old_by_group.get(key)
        true_role = str(new_record["true_role"])
        notes = _disagreement_notes(true_role, rule_record, new_record, old_record)
        if not notes:
            continue
        records.append(PredictionDisagreementRecord(
            reaction_id=str(new_record["reaction_id"]),
            mechanism_type=str(new_record["mechanism_type"]),
            split=str(new_record["split"]),
            group_id=str(new_record["group_id"]),
            group_type=str(new_record["group_type"]),
            true_role=true_role,
            rule_based_role=_record_role(rule_record),
            old_mlp_role=_record_role(old_record),
            new_mlp_role=_record_role(new_record),
            rule_based_correct=_record_correct(rule_record),
            old_mlp_correct=_record_correct(old_record),
            new_mlp_correct=_record_correct(new_record),
            new_mlp_confidence=(
                float(new_record["predicted_confidence"])
                if isinstance(new_record.get("predicted_confidence"), int | float)
                else None
            ),
            notes=notes,
            metadata={},
        ))
    return records


def _center_atoms(record: dict[str, object], key: str) -> list[int]:
    metadata = record.get("metadata", {})
    if isinstance(metadata, dict) and key in metadata:
        value = metadata[key]
        if isinstance(value, list):
            return [int(item) for item in value]
        if isinstance(value, str) and value:
            return [int(item) for item in value.split(",") if item]
    return []


def _failure_type(
    true_atoms: list[int],
    predicted_atoms: list[int],
    missing: list[int],
    extra: list[int],
    role_accuracy: float,
) -> str:
    if not true_atoms:
        return "empty_truth"
    if not predicted_atoms:
        return "empty_prediction"
    if missing and extra:
        return "partial_overlap"
    if missing:
        return "role_correct_center_wrong" if role_accuracy >= 1.0 else "partial_overlap"
    if extra:
        return "extra_reaction_center_atoms"
    return "role_wrong_center_wrong" if role_accuracy < 1.0 else "partial_overlap"


def collect_reaction_center_failures(
    benchmark_report: Any,
    predictor_name: str,
    f1_threshold: float = 0.8,
) -> list[ReactionCenterFailureRecord]:
    report = benchmark_report if isinstance(benchmark_report, dict) else benchmark_report.to_dict()
    failures: list[ReactionCenterFailureRecord] = []
    for record in report.get("reaction_records", []):
        if not isinstance(record, dict):
            continue
        f1 = record.get("reaction_center_f1")
        if f1 is not None and float(f1) >= f1_threshold:
            continue
        true_atoms = _center_atoms(record, "true_reaction_center_atoms")
        predicted_atoms = _center_atoms(record, "predicted_reaction_center_atoms")
        missing = sorted(set(true_atoms) - set(predicted_atoms))
        extra = sorted(set(predicted_atoms) - set(true_atoms))
        role_accuracy = float(record.get("role_accuracy", 0.0))
        failure = _failure_type(true_atoms, predicted_atoms, missing, extra, role_accuracy)
        failures.append(ReactionCenterFailureRecord(
            reaction_id=str(record["reaction_id"]),
            mechanism_type=str(record["mechanism_type"]),
            split=str(record["split"]),
            predictor_name=predictor_name,
            true_reaction_center_atoms=true_atoms,
            predicted_reaction_center_atoms=predicted_atoms,
            missing_atoms=missing,
            extra_atoms=extra,
            precision=(
                float(record["reaction_center_precision"])
                if record.get("reaction_center_precision") is not None
                else None
            ),
            recall=(
                float(record["reaction_center_recall"])
                if record.get("reaction_center_recall") is not None
                else None
            ),
            f1=float(f1) if f1 is not None else None,
            role_accuracy_for_reaction=role_accuracy,
            n_labeled_groups=int(record.get("n_labeled_groups", 0)),
            n_correct_roles=int(record.get("n_correct_roles", 0)),
            failure_type=failure,
            notes=[failure],
            metadata={},
        ))
    return failures


def compute_mlp_calibration(
    group_records: list[Any],
    n_bins: int = 10,
) -> list[MLPCalibrationBin]:
    usable = [
        record for record in group_records
        if isinstance(record, dict)
        and isinstance(record.get("predicted_confidence"), int | float)
    ]
    total = len(usable)
    bins: list[MLPCalibrationBin] = []
    for idx in range(n_bins):
        start = idx / n_bins
        end = (idx + 1) / n_bins
        in_bin = [
            record for record in usable
            if start <= float(record["predicted_confidence"]) < end
            or (idx == n_bins - 1 and float(record["predicted_confidence"]) == 1.0)
        ]
        if not in_bin:
            bins.append(MLPCalibrationBin(start, end, 0, None, None, None))
            continue
        accuracy = sum(1 for record in in_bin if record.get("correct") is True) / len(in_bin)
        mean_conf = sum(float(record["predicted_confidence"]) for record in in_bin) / len(in_bin)
        ece = abs(accuracy - mean_conf) * len(in_bin) / total if total else None
        bins.append(MLPCalibrationBin(start, end, len(in_bin), accuracy, mean_conf, ece))
    return bins


def summarize_failure_patterns(
    failures: list[ReactionCenterFailureRecord],
    disagreements: list[PredictionDisagreementRecord],
) -> list[dict[str, object]]:
    patterns: list[dict[str, object]] = []
    for name, counter in (
        ("mechanism_type", Counter(f.mechanism_type for f in failures)),
        ("failure_type", Counter(f.failure_type for f in failures)),
        ("group_type", Counter(d.group_type for d in disagreements)),
        (
            "true_role_vs_new_mlp_role",
            Counter((d.true_role, d.new_mlp_role or "missing") for d in disagreements),
        ),
    ):
        for key, count in counter.most_common(10):
            patterns.append({"pattern_type": name, "pattern": str(key), "count": count})
    high_conf = sum(
        1 for disagreement in disagreements
        if "new_mlp_high_confidence_wrong" in disagreement.notes
    )
    if high_conf:
        patterns.append({
            "pattern_type": "high_confidence_wrong",
            "pattern": "new_mlp_high_confidence_wrong",
            "count": high_conf,
        })
    return patterns


def _disagreement_counts(disagreements: list[PredictionDisagreementRecord]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for disagreement in disagreements:
        for note in disagreement.notes:
            counter[note] += 1
    return dict(sorted(counter.items()))


def generate_diagnostics_recommendations(report: DiagnosticsReport) -> list[str]:
    recommendations: list[str] = []
    rule_f1 = report.reaction_center_summary.get("rule_based_negotiated", {}).get("f1")
    new_f1 = report.reaction_center_summary.get("new_mlp_negotiated", {}).get("f1")
    if rule_f1 is not None and new_f1 is not None and new_f1 < rule_f1:
        recommendations.append(
            "Improve negotiation to use MLP confidence and role provenance "
            "before changing defaults."
        )
    if report.disagreement_counts.get("spectator_predicted_reactive", 0):
        recommendations.append(
            "Add or review control/spectator examples to reduce reactive false positives."
        )
    if report.metadata.get("reactive_radical_labels_below_10"):
        recommendations.append(
            "Add more radical labels; reactive_radical remains underrepresented."
        )
    new_acc = report.role_accuracy_summary.get("new_mlp_negotiated")
    rule_acc = report.role_accuracy_summary.get("rule_based_negotiated")
    if new_acc is not None and rule_acc is not None and new_acc > rule_acc and new_f1 != rule_f1:
        recommendations.append(
            "Consider an atom-level reaction-center head or mapping-aware negotiation layer."
        )
    if report.disagreement_counts.get("new_mlp_high_confidence_wrong", 0):
        recommendations.append(
            "Calibrate MLP confidence with temperature scaling or validation calibration."
        )
    if not recommendations:
        recommendations.append(
            "No dominant diagnostic pattern found; inspect per-reaction failures manually."
        )
    return recommendations


def _role_accuracy(report: dict[str, object]) -> float:
    return float(report.get("overall_role_accuracy", 0.0))


def _rc_summary(report: dict[str, object]) -> dict[str, float | None]:
    return {
        "precision": (
            float(report["reaction_center_precision"])
            if report.get("reaction_center_precision") is not None
            else None
        ),
        "recall": (
            float(report["reaction_center_recall"])
            if report.get("reaction_center_recall") is not None
            else None
        ),
        "f1": (
            float(report["reaction_center_f1"])
            if report.get("reaction_center_f1") is not None
            else None
        ),
    }


def build_diagnostics_report(
    dataset_path: str | Path,
    rule_report_path: str | Path,
    new_mlp_report_path: str | Path,
    old_mlp_report_path: str | Path | None = None,
    f1_threshold: float = 0.8,
) -> DiagnosticsReport:
    reactions = load_labeled_reactions(dataset_path)
    rule_report = load_benchmark_records(rule_report_path)
    new_report = load_benchmark_records(new_mlp_report_path)
    old_report = (
        load_benchmark_records(old_mlp_report_path)
        if old_mlp_report_path is not None and Path(old_mlp_report_path).exists()
        else None
    )
    disagreements = collect_prediction_disagreements(
        reactions,
        rule_report,
        new_report,
        old_report,
    )
    failures = (
        collect_reaction_center_failures(
            rule_report,
            str(rule_report.get("predictor_name", "rule_based_negotiated")),
            f1_threshold,
        )
        + collect_reaction_center_failures(
            new_report,
            str(new_report.get("predictor_name", "new_mlp_negotiated")),
            f1_threshold,
        )
    )
    calibration_bins = compute_mlp_calibration(list(new_report.get("group_records", [])))
    predictor_names = [
        str(rule_report.get("predictor_name", "rule_based_negotiated")),
        str(new_report.get("predictor_name", "new_mlp_negotiated")),
    ]
    if old_report is not None:
        predictor_names.append(str(old_report.get("predictor_name", "old_mlp_negotiated")))
    role_summary = {
        predictor_names[0]: _role_accuracy(rule_report),
        predictor_names[1]: _role_accuracy(new_report),
    }
    rc_summary = {
        predictor_names[0]: _rc_summary(rule_report),
        predictor_names[1]: _rc_summary(new_report),
    }
    if old_report is not None:
        role_summary[predictor_names[2]] = _role_accuracy(old_report)
        rc_summary[predictor_names[2]] = _rc_summary(old_report)
    n_labels = sum(len(reaction.group_roles) for reaction in reactions)
    radical_labels = sum(
        1 for reaction in reactions for role in reaction.group_roles
        if role.role.value == "reactive_radical"
    )
    report = DiagnosticsReport(
        dataset_path=str(dataset_path),
        n_reactions=len(reactions),
        n_group_labels=n_labels,
        predictor_names=predictor_names,
        role_accuracy_summary=role_summary,
        reaction_center_summary=rc_summary,
        disagreement_counts=_disagreement_counts(disagreements),
        mlp_calibration_bins=calibration_bins,
        reaction_center_failures=failures,
        disagreement_records=disagreements,
        top_failure_patterns=summarize_failure_patterns(failures, disagreements),
        recommendations=[],
        metadata={
            "f1_threshold": f1_threshold,
            "reactive_radical_label_count": radical_labels,
            "reactive_radical_labels_below_10": radical_labels < 10,
        },
    )
    report.recommendations = generate_diagnostics_recommendations(report)
    return report


def save_diagnostics_report(report: DiagnosticsReport, path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
