"""Phase 5: Rule-based role predictor for functional-group agents.

Each FunctionalGroup agent receives exactly one of the five MENDEL roles using
deterministic threshold rules on Phase 3 descriptor scores.  No MLP, no training,
no negotiation.

Rule priority order (highest to lowest):
  1. Radical-context rules
  2. Leaving-group rules
  3. Pericyclic-context rules
  4. Ionic nucleophile / electrophile rules
  5. Spectator fallback
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mendel.descriptor import GroupDescriptor, build_descriptors, feature_index
from mendel.labels import LabeledReaction
from mendel.parser import ParsedReaction
from mendel.types import FunctionalGroup, FunctionalGroupType, ReactionContext, Role

# ---------------------------------------------------------------------------
# Public helper: safe feature lookup
# ---------------------------------------------------------------------------


def get_feature_value(
    descriptor: GroupDescriptor,
    feature_name: str,
    default: float = 0.0,
) -> float:
    """Return a feature value by name; return *default* when absent.  Never raises."""
    # Fast path: standard schema uses the shared O(1) index map.
    idx = feature_index(feature_name)
    if 0 <= idx < len(descriptor.values) and descriptor.feature_names[idx] == feature_name:
        return descriptor.values[idx]
    # Fallback for non-standard / reordered feature_names.
    try:
        idx = descriptor.feature_names.index(feature_name)
        return descriptor.values[idx]
    except (ValueError, IndexError):
        return default


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class RuleBasedPredictorConfig:
    """Threshold configuration for the rule-based predictor.

    All thresholds are inclusive (score >= threshold triggers the rule).
    """

    nucleophile_threshold: float = 0.55
    electrophile_threshold: float = 0.55
    leaving_group_threshold: float = 0.50
    acidity_threshold: float = 0.45
    radical_threshold: float = 0.55
    spectator_confidence: float = 0.50


# ---------------------------------------------------------------------------
# Output dataclasses
# ---------------------------------------------------------------------------


@dataclass
class RolePrediction:
    """Role prediction for a single functional-group agent.

    scores holds the five mechanistic heuristic scores from the descriptor.
    reason is a human-readable explanation of which rule fired.
    confidence is in [0.0, 1.0].
    """

    group_id: str
    group_type: FunctionalGroupType
    predicted_role: Role
    confidence: float
    reason: str
    scores: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, str | int | float | bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict."""
        return {
            "group_id": self.group_id,
            "group_type": self.group_type.value,
            "predicted_role": self.predicted_role.value,
            "confidence": self.confidence,
            "reason": self.reason,
            "scores": dict(self.scores),
            "metadata": dict(self.metadata),
        }


@dataclass
class PredictionReport:
    """Full prediction output for a single reaction.

    Contains one RolePrediction per functional-group agent, in the same order
    as the input descriptor list.
    """

    reaction_smiles: str
    context: ReactionContext
    predictions: list[RolePrediction] = field(default_factory=list)
    metadata: dict[str, str | int | float | bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict."""
        return {
            "reaction_smiles": self.reaction_smiles,
            "context": self.context.value,
            "predictions": [p.to_dict() for p in self.predictions],
            "metadata": dict(self.metadata),
        }


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------


class RuleBasedRolePredictor:
    """Deterministic rule-based predictor.

    Assigns exactly one MENDEL role per functional-group descriptor using
    threshold rules on heuristic scores.  No learning, no negotiation.

    Instantiate once and call predict() or predict_from_reaction() repeatedly.
    """

    def __init__(self, config: RuleBasedPredictorConfig | None = None) -> None:
        self.config: RuleBasedPredictorConfig = config or RuleBasedPredictorConfig()

    # ------------------------------------------------------------------
    # Core single-group prediction
    # ------------------------------------------------------------------

    def predict_group(
        self,
        descriptor: GroupDescriptor,
        context: ReactionContext,
    ) -> RolePrediction:
        """Assign a role to one functional-group agent.

        Rules are applied in strict priority order (see module docstring).
        The returned confidence is derived from the triggering score and is
        always in [0.0, 1.0].
        """
        cfg = self.config
        gt = descriptor.group_type
        gid = descriptor.group_id

        nuc = get_feature_value(descriptor, "nucleophilicity_score")
        elec = get_feature_value(descriptor, "electrophilicity_score")
        lg = get_feature_value(descriptor, "leaving_group_score")
        acid = get_feature_value(descriptor, "acidity_score")
        rad = get_feature_value(descriptor, "radical_stability_score")
        fc = get_feature_value(descriptor, "total_formal_charge")
        has_pi = get_feature_value(descriptor, "has_pi_bond")
        is_alpha = get_feature_value(descriptor, "is_alpha_carbon")
        is_benz = get_feature_value(descriptor, "is_benzylic_site")

        scores: dict[str, float] = {
            "nucleophilicity_score": nuc,
            "electrophilicity_score": elec,
            "leaving_group_score": lg,
            "acidity_score": acid,
            "radical_stability_score": rad,
        }

        def _make(
            role: Role,
            conf: float,
            reason: str,
            extra: dict[str, str | int | float | bool] | None = None,
        ) -> RolePrediction:
            meta: dict[str, str | int | float | bool] = {"group_type": gt.value}
            if extra:
                meta.update(extra)
            return RolePrediction(
                group_id=gid,
                group_type=gt,
                predicted_role=role,
                confidence=min(1.0, max(0.0, conf)),
                reason=reason,
                scores=dict(scores),
                metadata=meta,
            )

        # ------------------------------------------------------------------
        # 1. Radical-context rules
        # ------------------------------------------------------------------
        if context == ReactionContext.radical:
            if is_benz > 0.5:
                return _make(
                    Role.reactive_radical,
                    max(rad, 0.80),
                    f"benzylic_site in radical context: resonance-stabilised radical "
                    f"(radical_stability_score={rad:.2f})",
                    {"v0.1_note": "benzylic radical by group-type rule"},
                )
            if rad >= cfg.radical_threshold:
                return _make(
                    Role.reactive_radical,
                    rad,
                    f"radical_stability_score={rad:.2f} >= threshold {cfg.radical_threshold} "
                    f"in radical context",
                )
            if gt == FunctionalGroupType.halide and lg >= cfg.leaving_group_threshold:
                return _make(
                    Role.leaving_group,
                    lg,
                    f"halide leaving_group_score={lg:.2f} >= threshold "
                    f"{cfg.leaving_group_threshold} in radical context",
                )
            return _make(
                Role.spectator,
                cfg.spectator_confidence,
                f"radical context: no score exceeded threshold "
                f"(radical_stability_score={rad:.2f}, threshold={cfg.radical_threshold})",
            )

        # ------------------------------------------------------------------
        # 2. Leaving-group rules
        # ------------------------------------------------------------------
        if lg >= cfg.leaving_group_threshold:
            return _make(
                Role.leaving_group,
                lg,
                f"leaving_group_score={lg:.2f} >= threshold {cfg.leaving_group_threshold}",
                {"v0.1_note": "one-role-per-group: substrate carbon electrophilicity not separately modelled"},
            )

        # ------------------------------------------------------------------
        # 3. Pericyclic-context rules
        # ------------------------------------------------------------------
        if context == ReactionContext.pericyclic:
            if has_pi > 0.5:
                if nuc >= elec:
                    return _make(
                        Role.reactive_nucleophile,
                        min(1.0, nuc + 0.10),
                        f"pericyclic context: pi-system assigned reactive_nucleophile "
                        f"(nucleophilicity_score={nuc:.2f} >= electrophilicity_score={elec:.2f}); "
                        f"v0.1 flat taxonomy represents pericyclic partners as nucleophile/electrophile",
                    )
                return _make(
                    Role.reactive_electrophile,
                    min(1.0, elec + 0.10),
                    f"pericyclic context: pi-system assigned reactive_electrophile "
                    f"(electrophilicity_score={elec:.2f} > nucleophilicity_score={nuc:.2f}); "
                    f"v0.1 flat taxonomy represents pericyclic partners as nucleophile/electrophile",
                )
            return _make(
                Role.spectator,
                cfg.spectator_confidence,
                f"pericyclic context: no pi_bond detected (has_pi_bond={has_pi:.1f})",
            )

        # ------------------------------------------------------------------
        # 4. Ionic (and unknown) context rules
        # ------------------------------------------------------------------

        # 4a. Alpha-carbon enolate donor
        if is_alpha > 0.5 and acid >= cfg.acidity_threshold:
            return _make(
                Role.reactive_nucleophile,
                min(1.0, acid),
                f"alpha carbon represented as nucleophile after deprotonation in v0.1 flat taxonomy "
                f"(acidity_score={acid:.2f} >= threshold {cfg.acidity_threshold})",
                {"v0.1_note": "base not explicitly present in descriptor; deprotonation assumed"},
            )

        # 4b. Negatively charged nucleophile
        if fc < 0 and nuc >= 0.40:
            return _make(
                Role.reactive_nucleophile,
                min(1.0, nuc + abs(fc) * 0.10),
                f"negative formal_charge={fc:.0f} boosts nucleophilicity "
                f"(nucleophilicity_score={nuc:.2f} >= 0.40)",
            )

        # 4c. Both thresholds exceeded — choose stronger
        if nuc >= cfg.nucleophile_threshold and elec >= cfg.electrophile_threshold:
            if nuc >= elec:
                return _make(
                    Role.reactive_nucleophile,
                    nuc,
                    f"nucleophilicity_score={nuc:.2f} >= electrophilicity_score={elec:.2f}; "
                    f"both above threshold — stronger score wins",
                )
            return _make(
                Role.reactive_electrophile,
                elec,
                f"electrophilicity_score={elec:.2f} > nucleophilicity_score={nuc:.2f}; "
                f"both above threshold — stronger score wins",
            )

        # 4d. Single threshold
        if nuc >= cfg.nucleophile_threshold:
            return _make(
                Role.reactive_nucleophile,
                nuc,
                f"nucleophilicity_score={nuc:.2f} >= threshold {cfg.nucleophile_threshold}",
            )

        if elec >= cfg.electrophile_threshold:
            return _make(
                Role.reactive_electrophile,
                elec,
                f"electrophilicity_score={elec:.2f} >= threshold {cfg.electrophile_threshold}",
            )

        # ------------------------------------------------------------------
        # 5. Spectator fallback
        # ------------------------------------------------------------------
        return _make(
            Role.spectator,
            cfg.spectator_confidence,
            f"no heuristic score exceeded threshold "
            f"(nuc={nuc:.2f}, elec={elec:.2f}, lg={lg:.2f}, acid={acid:.2f}, rad={rad:.2f})",
        )

    # ------------------------------------------------------------------
    # Batch prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        descriptors: list[GroupDescriptor],
        context: ReactionContext,
    ) -> list[RolePrediction]:
        """Predict roles for a list of descriptors.  Preserves input order."""
        return [self.predict_group(d, context) for d in descriptors]

    # ------------------------------------------------------------------
    # Reaction-level prediction
    # ------------------------------------------------------------------

    def predict_from_reaction(
        self,
        parsed_reaction: ParsedReaction,
        groups: list[FunctionalGroup],
    ) -> PredictionReport:
        """Build descriptors and predict roles for every group in a reaction."""
        descriptors = build_descriptors(parsed_reaction, groups)
        predictions = self.predict(descriptors, parsed_reaction.context)
        return PredictionReport(
            reaction_smiles=parsed_reaction.reaction_smiles,
            context=parsed_reaction.context,
            predictions=predictions,
            metadata={"n_groups": len(groups), "predictor": "RuleBasedRolePredictor_v0.1"},
        )


# ---------------------------------------------------------------------------
# Public helper functions
# ---------------------------------------------------------------------------


def predict_roles(
    descriptors: list[GroupDescriptor],
    context: ReactionContext,
    config: RuleBasedPredictorConfig | None = None,
) -> list[RolePrediction]:
    """Convenience wrapper: predict roles for all descriptors."""
    return RuleBasedRolePredictor(config).predict(descriptors, context)


def predict_roles_for_reaction(
    parsed_reaction: ParsedReaction,
    groups: list[FunctionalGroup],
    config: RuleBasedPredictorConfig | None = None,
) -> PredictionReport:
    """Convenience wrapper: build descriptors and predict roles for all groups."""
    return RuleBasedRolePredictor(config).predict_from_reaction(parsed_reaction, groups)


def summarize_predictions(predictions: list[RolePrediction]) -> dict[str, Any]:
    """Return a count/distribution summary of a prediction list."""
    role_counts: dict[str, int] = {}
    group_type_counts: dict[str, int] = {}
    low_confidence: list[str] = []
    total_conf = 0.0

    for p in predictions:
        role_counts[p.predicted_role.value] = role_counts.get(p.predicted_role.value, 0) + 1
        group_type_counts[p.group_type.value] = group_type_counts.get(p.group_type.value, 0) + 1
        total_conf += p.confidence
        if p.confidence < 0.60:
            low_confidence.append(p.group_id)

    return {
        "n_predictions": len(predictions),
        "role_counts": role_counts,
        "group_type_counts": group_type_counts,
        "average_confidence": total_conf / len(predictions) if predictions else 0.0,
        "low_confidence_group_ids": low_confidence,
    }


def compare_predictions_to_labels(
    predictions: list[RolePrediction],
    labeled_reaction: LabeledReaction,
) -> dict[str, Any]:
    """Lightweight sanity check: compare predictions to ground-truth labels by group_id.

    Not a full benchmark report — that belongs to a later phase.
    """
    pred_by_id: dict[str, RolePrediction] = {p.group_id: p for p in predictions}
    n_labeled = len(labeled_reaction.group_roles)
    n_matched = 0
    n_correct = 0
    mismatches: list[dict[str, Any]] = []

    for lgr in labeled_reaction.group_roles:
        pred = pred_by_id.get(lgr.group_id)
        if pred is None:
            continue
        n_matched += 1
        if pred.predicted_role == lgr.role:
            n_correct += 1
        else:
            mismatches.append({
                "group_id": lgr.group_id,
                "predicted_role": pred.predicted_role.value,
                "labeled_role": lgr.role.value,
                "confidence": pred.confidence,
            })

    return {
        "n_labeled": n_labeled,
        "n_matched": n_matched,
        "n_correct": n_correct,
        "accuracy": n_correct / n_matched if n_matched > 0 else 0.0,
        "mismatches": mismatches,
    }
