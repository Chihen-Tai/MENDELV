"""RuleBasedNegotiator orchestrator (slim dispatcher)."""

from __future__ import annotations

from typing import Any

from mendel.negotiation.mechanism_hints import infer_mechanism_hint
from mendel.negotiation.mlp_aware import negotiate_mlp_aware
from mendel.negotiation.reaction_center import infer_reaction_center_atoms
from mendel.negotiation.strategies import (
    is_symmetric_self_reaction,
    negotiate_aldol,
    negotiate_click,
    negotiate_diels_alder,
    negotiate_ester_hydrolysis,
    negotiate_ionic_addition,
    negotiate_michael,
    negotiate_radical,
    negotiate_radical_addition,
    negotiate_sn2_e2,
    negotiate_unknown,
)
from mendel.negotiation.types import (
    NegotiatedRoleAssignment,
    NegotiationResult,
    NegotiationWarning,
    NegotiatorConfig,
)
from mendel.parser import ParsedReaction
from mendel.predictor import RolePrediction
from mendel.types import (
    FunctionalGroup,
)


class RuleBasedNegotiator:
    """Global consistency coordinator for MENDEL Phase 6.

    Takes raw per-group role predictions (Phase 5) and produces final
    coordinated role assignments, mechanism hints, reaction center atoms,
    and warnings.  Fully deterministic, no ML.
    """

    def __init__(self, config: NegotiatorConfig | None = None) -> None:
        self.config: NegotiatorConfig = config or NegotiatorConfig()

    def negotiate(
        self,
        parsed_reaction: ParsedReaction,
        groups: list[FunctionalGroup],
        predictions: list[RolePrediction],
    ) -> NegotiationResult:
        """Negotiate raw Phase 5 predictions into final coordinated assignments.

        Does not mutate input groups or predictions.
        """
        warnings: list[NegotiationWarning] = []

        # Build initial assignments from raw predictions (copies metadata, no mutation)
        assignments: list[NegotiatedRoleAssignment] = []
        for pred in predictions:
            meta: dict[str, str | int | float | bool] = dict(pred.metadata)
            meta.setdefault("prediction_source", str(pred.metadata.get("prediction_source", "")))
            meta.setdefault("predictor_name", str(pred.metadata.get("predictor_name", "")))
            meta["confidence"] = pred.confidence
            meta["original_role"] = pred.predicted_role.value
            assignments.append(
                NegotiatedRoleAssignment(
                    group_id=pred.group_id,
                    group_type=pred.group_type,
                    raw_role=pred.predicted_role,
                    final_role=pred.predicted_role,
                    raw_confidence=pred.confidence,
                    final_confidence=pred.confidence,
                    reason=pred.reason,
                    subrole=None,
                    is_reaction_center=False,
                    metadata=meta,
                )
            )

        assign_by_id: dict[str, NegotiatedRoleAssignment] = {
            a.group_id: a for a in assignments
        }

        mechanism_hint = infer_mechanism_hint(parsed_reaction, groups, predictions)

        if self.config.mode == "mlp_aware":
            mechanism_hint = negotiate_mlp_aware(self.config, 
                parsed_reaction,
                groups,
                predictions,
                assign_by_id,
                warnings,
                mechanism_hint,
            )
        elif mechanism_hint == "sn2_or_e2_like":
            negotiate_sn2_e2(self.config, groups, predictions, assign_by_id, warnings)
        elif mechanism_hint == "michael_like":
            negotiate_michael(self.config, groups, predictions, assign_by_id, warnings)
        elif mechanism_hint == "aldol_like":
            negotiate_aldol(self.config,
                groups,
                predictions,
                assign_by_id,
                warnings,
                symmetric_self=is_symmetric_self_reaction(parsed_reaction),
            )
        elif mechanism_hint == "click_like":
            negotiate_click(self.config, groups, predictions, assign_by_id, warnings)
        elif mechanism_hint == "diels_alder_like":
            negotiate_diels_alder(self.config, groups, predictions, assign_by_id, warnings)
        elif mechanism_hint == "giese_like":
            negotiate_radical_addition(
                self.config, groups, predictions, assign_by_id, warnings, variant="giese"
            )
        elif mechanism_hint == "minisci_like":
            negotiate_radical_addition(
                self.config, groups, predictions, assign_by_id, warnings, variant="minisci"
            )
        elif mechanism_hint == "radical_addition_like":
            negotiate_radical_addition(
                self.config, groups, predictions, assign_by_id, warnings,
                variant="radical_addition",
            )
        elif mechanism_hint == "radical_bromination_like":
            negotiate_radical(self.config, groups, predictions, assign_by_id, warnings)
        elif mechanism_hint == "ester_hydrolysis_like":
            negotiate_ester_hydrolysis(groups, predictions, assign_by_id, warnings)
        elif mechanism_hint == "ionic_addition_like":
            negotiate_ionic_addition(groups, predictions, assign_by_id, warnings)
        else:
            negotiate_unknown(
                self.config, groups, predictions, assign_by_id, warnings,
                parsed_reaction=parsed_reaction,
            )

        final_assignments = [assign_by_id[p.group_id] for p in predictions]
        reaction_center_atoms = infer_reaction_center_atoms(
            parsed_reaction, groups, final_assignments
        )

        return NegotiationResult(
            reaction_smiles=parsed_reaction.reaction_smiles,
            context=parsed_reaction.context,
            mechanism_hint=mechanism_hint,
            assignments=final_assignments,
            reaction_center_atoms=reaction_center_atoms,
            warnings=warnings,
            metadata={
                "n_groups": len(groups),
                "negotiator": "RuleBasedNegotiator_v0.1",
                "mode": self.config.mode,
            },
        )

    def summarize_result(self, result: NegotiationResult) -> dict[str, Any]:
        """Return a high-level summary of a NegotiationResult."""
        final_role_counts: dict[str, int] = {}
        subrole_counts: dict[str, int] = {}
        warning_counts: dict[str, int] = {}
        total_conf = 0.0

        for a in result.assignments:
            key = a.final_role.value
            final_role_counts[key] = final_role_counts.get(key, 0) + 1
            total_conf += a.final_confidence
            if a.subrole:
                subrole_counts[a.subrole] = subrole_counts.get(a.subrole, 0) + 1

        for w in result.warnings:
            warning_counts[w.severity] = warning_counts.get(w.severity, 0) + 1

        n = len(result.assignments)
        return {
            "mechanism_hint": result.mechanism_hint,
            "n_assignments": n,
            "n_reaction_center_atoms": len(result.reaction_center_atoms),
            "final_role_counts": final_role_counts,
            "subrole_counts": subrole_counts,
            "warning_counts": warning_counts,
            "average_final_confidence": total_conf / n if n > 0 else 0.0,
        }
