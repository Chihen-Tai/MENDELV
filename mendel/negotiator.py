"""Phase 6: Negotiation layer for MENDEL.

Coordinates raw per-group role predictions (Phase 5) into a globally
consistent reaction-level interpretation.  Fully deterministic, no ML.

The central abstraction is preserved: each FunctionalGroup is an agent
whose local prediction is negotiated against its peers to produce a
chemically plausible reaction-level assignment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mendel.identifier import identify_functional_groups
from mendel.parser import ParsedReaction, parse_reaction_smiles
from mendel.predictor import RolePrediction, predict_roles_for_reaction
from mendel.types import AtomRef, FunctionalGroup, FunctionalGroupType, ReactionContext, Role

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PI_GROUP_TYPES: frozenset[FunctionalGroupType] = frozenset({
    FunctionalGroupType.alkene,
    FunctionalGroupType.alkyne,
    FunctionalGroupType.aromatic,
})


# ---------------------------------------------------------------------------
# NegotiationWarning
# ---------------------------------------------------------------------------


@dataclass
class NegotiationWarning:
    """A warning produced during negotiation.

    severity: "info" | "warning" | "error"
    """

    code: str
    message: str
    severity: str
    metadata: dict[str, str | int | float | bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict."""
        return {
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
            "metadata": dict(self.metadata),
        }


# ---------------------------------------------------------------------------
# NegotiatedRoleAssignment
# ---------------------------------------------------------------------------


@dataclass
class NegotiatedRoleAssignment:
    """Final role assignment for a single functional-group agent after negotiation.

    raw_role / raw_confidence: values from Phase 5 predictor (never mutated).
    final_role / final_confidence: values after global consistency rules.
    subrole: optional fine-grained label; does not introduce a new Role enum value.
    is_reaction_center: True when this group's atoms belong to the reaction center.
    """

    group_id: str
    group_type: FunctionalGroupType
    raw_role: Role
    final_role: Role
    raw_confidence: float
    final_confidence: float
    reason: str
    subrole: str | None = None
    is_reaction_center: bool = False
    metadata: dict[str, str | int | float | bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict."""
        return {
            "group_id": self.group_id,
            "group_type": self.group_type.value,
            "raw_role": self.raw_role.value,
            "final_role": self.final_role.value,
            "raw_confidence": self.raw_confidence,
            "final_confidence": self.final_confidence,
            "reason": self.reason,
            "subrole": self.subrole,
            "is_reaction_center": self.is_reaction_center,
            "metadata": dict(self.metadata),
        }


# ---------------------------------------------------------------------------
# NegotiationResult
# ---------------------------------------------------------------------------


@dataclass
class NegotiationResult:
    """Full negotiation output for a single reaction."""

    reaction_smiles: str
    context: ReactionContext
    mechanism_hint: str
    assignments: list[NegotiatedRoleAssignment]
    reaction_center_atoms: list[AtomRef]
    warnings: list[NegotiationWarning]
    metadata: dict[str, str | int | float | bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict."""
        return {
            "reaction_smiles": self.reaction_smiles,
            "context": self.context.value,
            "mechanism_hint": self.mechanism_hint,
            "assignments": [a.to_dict() for a in self.assignments],
            "reaction_center_atoms": [a.to_dict() for a in self.reaction_center_atoms],
            "warnings": [w.to_dict() for w in self.warnings],
            "metadata": dict(self.metadata),
        }


# ---------------------------------------------------------------------------
# NegotiatorConfig
# ---------------------------------------------------------------------------


@dataclass
class NegotiatorConfig:
    """Configuration for the rule-based negotiator."""

    mode: str = "rule_based"
    use_confidence: bool = True
    spectator_confidence_threshold: float = 0.70
    reactive_confidence_threshold: float = 0.50
    suppress_control_centers: bool = True
    prefer_mechanism_specific_centers: bool = True
    metadata: dict[str, str | int | float | bool] = field(default_factory=dict)
    min_center_confidence: float = 0.45
    prefer_high_confidence_candidates: bool = True
    allow_role_downgrade_to_spectator: bool = True
    require_ionic_pair: bool = True
    require_pericyclic_partners: bool = True
    require_radical_center: bool = True


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _warn(
    code: str,
    message: str,
    severity: str,
    meta: dict[str, str | int | float | bool] | None = None,
) -> NegotiationWarning:
    return NegotiationWarning(
        code=code,
        message=message,
        severity=severity,
        metadata=meta or {},
    )


def _has_group_type(groups: list[FunctionalGroup], gt: FunctionalGroupType) -> bool:
    return any(g.group_type == gt for g in groups)


def _groups_of_type(
    groups: list[FunctionalGroup], gt: FunctionalGroupType
) -> list[FunctionalGroup]:
    return [g for g in groups if g.group_type == gt]


def _group_by_id(groups: list[FunctionalGroup]) -> dict[str, FunctionalGroup]:
    return {g.group_id: g for g in groups}


def _pred_by_id(predictions: list[RolePrediction]) -> dict[str, RolePrediction]:
    return {p.group_id: p for p in predictions}


def _deduplicate_atom_refs(atom_refs: list[AtomRef]) -> list[AtomRef]:
    """Deduplicate AtomRefs by (molecule_index, atom_index), stable order."""
    seen: set[tuple[int, int]] = set()
    result: list[AtomRef] = []
    for ref in atom_refs:
        key = (ref.molecule_index, ref.atom_index)
        if key not in seen:
            seen.add(key)
            result.append(ref)
    return result


def _is_reactive_role(role: Role) -> bool:
    return role in {
        Role.reactive_nucleophile,
        Role.reactive_electrophile,
        Role.reactive_radical,
        Role.leaving_group,
    }


# ---------------------------------------------------------------------------
# RuleBasedNegotiator
# ---------------------------------------------------------------------------


class RuleBasedNegotiator:
    """Global consistency coordinator for MENDEL Phase 6.

    Takes raw per-group role predictions (Phase 5) and produces final
    coordinated role assignments, mechanism hints, reaction center atoms,
    and warnings.  Fully deterministic, no ML.
    """

    def __init__(self, config: NegotiatorConfig | None = None) -> None:
        self.config: NegotiatorConfig = config or NegotiatorConfig()

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def infer_mechanism_hint(
        self,
        parsed_reaction: ParsedReaction,
        groups: list[FunctionalGroup],
        predictions: list[RolePrediction],
    ) -> str:
        """Infer the most likely mechanism type.

        Returns one of:
          "sn2_or_e2_like", "aldol_like", "diels_alder_like",
          "radical_bromination_like", "ionic_addition_like", "unknown"
        """
        ctx = parsed_reaction.context

        if ctx == ReactionContext.radical:
            return "radical_bromination_like"

        if ctx == ReactionContext.pericyclic:
            return "diels_alder_like"

        if ctx == ReactionContext.ionic:
            has_carbonyl = _has_group_type(groups, FunctionalGroupType.carbonyl)
            has_alpha = _has_group_type(groups, FunctionalGroupType.alpha_carbon)
            has_halide = _has_group_type(groups, FunctionalGroupType.halide)
            has_leaving_pred = any(
                p.predicted_role == Role.leaving_group for p in predictions
            )
            has_nuc = any(
                p.predicted_role == Role.reactive_nucleophile for p in predictions
            )
            has_elec = any(
                p.predicted_role == Role.reactive_electrophile for p in predictions
            )

            # Aldol checked before SN2 — carbonyl+alpha_carbon signature is specific
            if has_carbonyl and has_alpha:
                return "aldol_like"

            if has_halide or has_leaving_pred:
                return "sn2_or_e2_like"

            if has_nuc and has_elec:
                return "ionic_addition_like"

        return "unknown"

    def infer_reaction_center_atoms(
        self,
        parsed_reaction: ParsedReaction,
        groups: list[FunctionalGroup],
        assignments: list[NegotiatedRoleAssignment],
    ) -> list[AtomRef]:
        """Collect atom refs from groups marked as reaction center.

        Deduplicates by (molecule_index, atom_index) in stable input order.
        """
        group_lookup = _group_by_id(groups)
        all_refs: list[AtomRef] = []
        for assignment in assignments:
            if not assignment.is_reaction_center:
                continue
            group = group_lookup.get(assignment.group_id)
            if group is None:
                continue
            all_refs.extend(group.atom_refs)
        return _deduplicate_atom_refs(all_refs)

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

        mechanism_hint = self.infer_mechanism_hint(parsed_reaction, groups, predictions)

        if self.config.mode == "mlp_aware":
            mechanism_hint = self._negotiate_mlp_aware(
                parsed_reaction,
                groups,
                predictions,
                assign_by_id,
                warnings,
                mechanism_hint,
            )
        elif mechanism_hint == "sn2_or_e2_like":
            self._negotiate_sn2_e2(groups, predictions, assign_by_id, warnings)
        elif mechanism_hint == "aldol_like":
            self._negotiate_aldol(groups, predictions, assign_by_id, warnings)
        elif mechanism_hint == "diels_alder_like":
            self._negotiate_diels_alder(groups, predictions, assign_by_id, warnings)
        elif mechanism_hint == "radical_bromination_like":
            self._negotiate_radical(groups, predictions, assign_by_id, warnings)
        elif mechanism_hint == "ionic_addition_like":
            self._negotiate_ionic_addition(groups, predictions, assign_by_id, warnings)
        else:
            self._negotiate_unknown(groups, predictions, assign_by_id, warnings)

        final_assignments = [assign_by_id[p.group_id] for p in predictions]
        reaction_center_atoms = self.infer_reaction_center_atoms(
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

    # ------------------------------------------------------------------
    # Mechanism-specific negotiation helpers
    # ------------------------------------------------------------------

    def _mechanism_from_metadata(self, parsed_reaction: ParsedReaction, fallback: str) -> str:
        raw = str(parsed_reaction.metadata.get("mechanism_type", "")).strip().lower()
        if not raw:
            return fallback
        aliases = {
            "sn2": "sn2",
            "e2": "e2",
            "control": "control",
            "ester_control": "ester_control",
            "nitrile_control": "nitrile_control",
            "no_reaction": "control",
            "carbonyl_addition": "carbonyl_addition",
            "aldol": "aldol",
            "cross_aldol": "cross_aldol",
            "diels_alder": "diels_alder",
            "benzylic_radical_bromination": "benzylic_radical_bromination",
            "radical_bromination": "benzylic_radical_bromination",
            "nitroalkane_deprotonation": "nitroalkane_deprotonation",
        }
        return aliases.get(raw, fallback)

    def _clear_centers(self, assign_by_id: dict[str, NegotiatedRoleAssignment]) -> None:
        for assignment in assign_by_id.values():
            assignment.is_reaction_center = False
            assignment.metadata["center_selection_reason"] = ""

    def _mark_center(
        self,
        assign_by_id: dict[str, NegotiatedRoleAssignment],
        group_id: str,
        reason: str,
    ) -> None:
        assignment = assign_by_id.get(group_id)
        if assignment is None:
            return
        assignment.is_reaction_center = True
        assignment.metadata["center_selection_reason"] = reason
        assignment.metadata["final_role"] = assignment.final_role.value
        assignment.metadata["role_changed_by_negotiation"] = (
            assignment.raw_role != assignment.final_role
        )

    def _high_conf_spectator(self, assignment: NegotiatedRoleAssignment) -> bool:
        return (
            assignment.final_role == Role.spectator
            and assignment.final_confidence >= self.config.spectator_confidence_threshold
        )

    def _confident_reactive(self, assignment: NegotiatedRoleAssignment) -> bool:
        return (
            _is_reactive_role(assignment.final_role)
            and assignment.final_confidence >= self.config.reactive_confidence_threshold
        )

    def _negotiate_mlp_aware(
        self,
        parsed_reaction: ParsedReaction,
        groups: list[FunctionalGroup],
        predictions: list[RolePrediction],
        assign_by_id: dict[str, NegotiatedRoleAssignment],
        warnings: list[NegotiationWarning],
        fallback_hint: str,
    ) -> str:
        mechanism = self._mechanism_from_metadata(parsed_reaction, fallback_hint)
        if mechanism in {"sn2", "e2", "sn2_or_e2_like"} or fallback_hint == "sn2_or_e2_like":
            self._negotiate_sn2_e2(groups, predictions, assign_by_id, warnings)
        elif mechanism in {"aldol", "cross_aldol", "aldol_like"}:
            self._negotiate_aldol(groups, predictions, assign_by_id, warnings)
        elif mechanism in {"diels_alder", "diels_alder_like"}:
            self._negotiate_diels_alder(groups, predictions, assign_by_id, warnings)
        elif mechanism in {"benzylic_radical_bromination", "radical_bromination_like"}:
            self._negotiate_radical(groups, predictions, assign_by_id, warnings)
        self._clear_centers(assign_by_id)

        if mechanism in {"control", "ester_control", "nitrile_control"}:
            return self._mlp_aware_control(mechanism, assign_by_id, warnings)
        if mechanism in {"sn2", "e2", "sn2_or_e2_like"}:
            self._mlp_aware_sn2_e2(mechanism, groups, assign_by_id, warnings)
            return "sn2_or_e2_like"
        if mechanism == "carbonyl_addition":
            self._mlp_aware_carbonyl_addition(groups, assign_by_id)
            return "ionic_addition_like"
        if mechanism in {"aldol", "cross_aldol", "aldol_like"}:
            self._mlp_aware_aldol(groups, assign_by_id, warnings)
            return "aldol_like"
        if mechanism in {"diels_alder", "diels_alder_like"}:
            self._mlp_aware_diels_alder(groups, assign_by_id)
            return "diels_alder_like"
        if mechanism in {"benzylic_radical_bromination", "radical_bromination_like"}:
            self._mlp_aware_radical(groups, assign_by_id)
            return "radical_bromination_like"
        if mechanism == "nitroalkane_deprotonation":
            self._mlp_aware_nitroalkane(groups, assign_by_id)
            return "ionic_addition_like"

        for assignment in assign_by_id.values():
            if self._confident_reactive(assignment):
                self._mark_center(assign_by_id, assignment.group_id, "mlp_aware_unknown_reactive")
        return fallback_hint

    def _mlp_aware_control(
        self,
        mechanism: str,
        assign_by_id: dict[str, NegotiatedRoleAssignment],
        warnings: list[NegotiationWarning],
    ) -> str:
        high_spectators = [a for a in assign_by_id.values() if self._high_conf_spectator(a)]
        reactive = [a for a in assign_by_id.values() if _is_reactive_role(a.final_role)]
        if reactive:
            warnings.append(_warn(
                "control_reactive_prediction_suppressed",
                "Control reaction has reactive predictions; possible false positive.",
                "warning",
                {"mechanism": mechanism},
            ))
        if self.config.suppress_control_centers and high_spectators:
            for assignment in assign_by_id.values():
                assignment.is_reaction_center = False
                assignment.metadata["center_selection_reason"] = "control_suppressed"
            return "control_like"
        for assignment in reactive:
            if self._confident_reactive(assignment):
                self._mark_center(assign_by_id, assignment.group_id, "control_reactive_low_trust")
        return "control_like"

    def _mlp_aware_sn2_e2(
        self,
        mechanism: str,
        groups: list[FunctionalGroup],
        assign_by_id: dict[str, NegotiatedRoleAssignment],
        warnings: list[NegotiationWarning],
    ) -> None:
        for group in groups:
            assignment = assign_by_id.get(group.group_id)
            if assignment is None:
                continue
            if group.group_type == FunctionalGroupType.halide and (
                assignment.final_role == Role.leaving_group
                or self._confident_reactive(assignment)
            ):
                self._mark_center(assign_by_id, group.group_id, "mlp_aware_halide_center")
                assignment.subrole = "leaving_group_site"
        if mechanism == "e2":
            warnings.append(_warn(
                "beta_center_not_fully_represented",
                "beta center not fully represented in v0.1 schema.",
                "info",
                {"mechanism": "e2"},
            ))

    def _mlp_aware_carbonyl_addition(
        self,
        groups: list[FunctionalGroup],
        assign_by_id: dict[str, NegotiatedRoleAssignment],
    ) -> None:
        for group in groups:
            assignment = assign_by_id.get(group.group_id)
            if assignment is None or self._high_conf_spectator(assignment):
                continue
            if (
                group.group_type == FunctionalGroupType.carbonyl
                and assignment.final_role == Role.reactive_electrophile
                and self._confident_reactive(assignment)
            ):
                self._mark_center(assign_by_id, group.group_id, "mlp_aware_carbonyl_addition")

    def _mlp_aware_aldol(
        self,
        groups: list[FunctionalGroup],
        assign_by_id: dict[str, NegotiatedRoleAssignment],
        warnings: list[NegotiationWarning],
    ) -> None:
        donors = [
            a for a in assign_by_id.values()
            if a.group_type == FunctionalGroupType.alpha_carbon
            and a.final_role == Role.reactive_nucleophile
            and self._confident_reactive(a)
        ]
        acceptors = [
            a for a in assign_by_id.values()
            if a.group_type == FunctionalGroupType.carbonyl
            and a.final_role == Role.reactive_electrophile
            and self._confident_reactive(a)
        ]
        if len(donors) != 1 or len(acceptors) != 1:
            warnings.append(_warn(
                "ambiguous_aldol_donor_acceptor_center",
                "ambiguous aldol donor/acceptor center",
                "warning",
            ))
            if donors:
                donor = max(donors, key=lambda a: a.final_confidence)
                self._mark_center(assign_by_id, donor.group_id, "mlp_aware_aldol_fallback_donor")
            if acceptors:
                acceptor = max(acceptors, key=lambda a: a.final_confidence)
                self._mark_center(
                    assign_by_id,
                    acceptor.group_id,
                    "mlp_aware_aldol_fallback_acceptor",
                )
            return
        self._mark_center(assign_by_id, donors[0].group_id, "mlp_aware_aldol_donor")
        self._mark_center(assign_by_id, acceptors[0].group_id, "mlp_aware_aldol_acceptor")

    def _mlp_aware_diels_alder(
        self,
        groups: list[FunctionalGroup],
        assign_by_id: dict[str, NegotiatedRoleAssignment],
    ) -> None:
        for group in groups:
            assignment = assign_by_id.get(group.group_id)
            if assignment is None:
                continue
            if (
                group.group_type == FunctionalGroupType.alkene
                and self._confident_reactive(assignment)
            ):
                self._mark_center(assign_by_id, group.group_id, "mlp_aware_diels_alder_pi_partner")
                if assignment.final_role == Role.reactive_nucleophile:
                    assignment.subrole = "diene_like"
                elif assignment.final_role == Role.reactive_electrophile:
                    assignment.subrole = "dienophile_like"

    def _mlp_aware_radical(
        self,
        groups: list[FunctionalGroup],
        assign_by_id: dict[str, NegotiatedRoleAssignment],
    ) -> None:
        for group in groups:
            assignment = assign_by_id.get(group.group_id)
            if assignment is None:
                continue
            if (
                group.group_type == FunctionalGroupType.benzylic_site
                and assignment.final_role == Role.reactive_radical
                and self._confident_reactive(assignment)
            ):
                self._mark_center(assign_by_id, group.group_id, "mlp_aware_benzylic_radical")

    def _mlp_aware_nitroalkane(
        self,
        groups: list[FunctionalGroup],
        assign_by_id: dict[str, NegotiatedRoleAssignment],
    ) -> None:
        for group in groups:
            assignment = assign_by_id.get(group.group_id)
            if assignment is None:
                continue
            if group.group_type == FunctionalGroupType.alpha_carbon:
                self._mark_center(assign_by_id, group.group_id, "mlp_aware_nitronate_alpha")
                assignment.metadata["center_selection_note"] = (
                    "alpha carbon represents nitronate-like center in v0.1"
                )

    def _negotiate_sn2_e2(
        self,
        groups: list[FunctionalGroup],
        predictions: list[RolePrediction],
        assign_by_id: dict[str, NegotiatedRoleAssignment],
        warnings: list[NegotiationWarning],
    ) -> None:
        """SN2/E2-like: mark halide as leaving group and reaction center."""
        halide_groups = _groups_of_type(groups, FunctionalGroupType.halide)
        pred_lookup = _pred_by_id(predictions)

        for group in halide_groups:
            gid = group.group_id
            if gid not in assign_by_id:
                continue
            a = assign_by_id[gid]
            pred = pred_lookup.get(gid)

            # If locally predicted as reactive_electrophile but leaving_group_score is high,
            # convert — can happen for unusual halides or descriptor edge cases.
            if a.raw_role == Role.reactive_electrophile and pred is not None:
                lg_score = pred.scores.get("leaving_group_score", 0.0)
                if lg_score >= 0.50:
                    a.final_role = Role.leaving_group
                    a.final_confidence = min(1.0, lg_score)
                    a.reason = (
                        f"halide re-assigned reactive_electrophile → leaving_group "
                        f"by SN2/E2 negotiation "
                        f"(leaving_group_score={lg_score:.2f}); original: {a.reason}"
                    )

            a.is_reaction_center = True
            a.subrole = "leaving_group_site"
            a.metadata["v0.1_note"] = (
                "alkyl halide C–X represented as one group; "
                "electrophilic carbon not separately modelled in v0.1"
            )

        has_nucleophile = any(
            a.final_role == Role.reactive_nucleophile for a in assign_by_id.values()
        )
        if not has_nucleophile:
            warnings.append(_warn(
                "missing_nucleophile",
                "No nucleophile detected; hydroxide or other nucleophile may not be "
                "represented as a functional group in v0.1.",
                "warning",
                {"mechanism": "sn2_or_e2_like"},
            ))

        if not halide_groups:
            warnings.append(_warn(
                "missing_leaving_group",
                "No halide group detected in SN2/E2-like context.",
                "warning",
                {"mechanism": "sn2_or_e2_like"},
            ))

        warnings.append(_warn(
            "coarse_group_granularity",
            "Alkyl halide C–X bond is one group in v0.1; "
            "electrophilic carbon and leaving halide share one group_id.",
            "info",
            {"v0.1_limitation": True},
        ))

    def _negotiate_aldol(
        self,
        groups: list[FunctionalGroup],
        predictions: list[RolePrediction],
        assign_by_id: dict[str, NegotiatedRoleAssignment],
        warnings: list[NegotiationWarning],
    ) -> None:
        """Aldol-like: select primary donor alpha_carbon and acceptor carbonyl."""
        cfg = self.config
        alpha_preds = [
            p for p in predictions
            if p.group_type == FunctionalGroupType.alpha_carbon
            and p.predicted_role == Role.reactive_nucleophile
        ]
        carbonyl_preds = [
            p for p in predictions
            if p.group_type == FunctionalGroupType.carbonyl
            and p.predicted_role == Role.reactive_electrophile
        ]

        warnings.append(_warn(
            "heuristic_donor_acceptor_assignment",
            "Aldol donor/acceptor selection is heuristic (highest-confidence pick); "
            "without atom mapping the assignment may be ambiguous.",
            "info",
            {"mechanism": "aldol_like"},
        ))

        if not alpha_preds:
            warnings.append(_warn(
                "missing_nucleophile",
                "No alpha_carbon reactive_nucleophile found; aldol donor is absent.",
                "warning",
                {"mechanism": "aldol_like"},
            ))
        if not carbonyl_preds:
            warnings.append(_warn(
                "missing_electrophile",
                "No carbonyl reactive_electrophile found; aldol acceptor is absent.",
                "warning",
                {"mechanism": "aldol_like"},
            ))

        primary_donor: RolePrediction | None = None
        if alpha_preds:
            primary_donor = (
                max(alpha_preds, key=lambda p: p.confidence)
                if cfg.prefer_high_confidence_candidates
                else alpha_preds[0]
            )

        primary_acceptor: RolePrediction | None = None
        if carbonyl_preds:
            primary_acceptor = (
                max(carbonyl_preds, key=lambda p: p.confidence)
                if cfg.prefer_high_confidence_candidates
                else carbonyl_preds[0]
            )

        if primary_donor and primary_donor.group_id in assign_by_id:
            a = assign_by_id[primary_donor.group_id]
            a.subrole = "aldol_donor_alpha_carbon"
            a.is_reaction_center = True
            a.reason = (
                f"selected as primary aldol donor (alpha_carbon nucleophile, "
                f"confidence={primary_donor.confidence:.2f}); {a.reason}"
            )

        if primary_acceptor and primary_acceptor.group_id in assign_by_id:
            a = assign_by_id[primary_acceptor.group_id]
            a.subrole = "aldol_acceptor_carbonyl"
            a.is_reaction_center = True
            a.reason = (
                f"selected as primary aldol acceptor (carbonyl electrophile, "
                f"confidence={primary_acceptor.confidence:.2f}); {a.reason}"
            )

        # Downgrade non-selected alpha_carbons if configured
        if cfg.allow_role_downgrade_to_spectator and primary_donor is not None:
            for pred in alpha_preds:
                if pred.group_id == primary_donor.group_id:
                    continue
                if pred.group_id in assign_by_id:
                    a = assign_by_id[pred.group_id]
                    a.final_role = Role.spectator
                    a.final_confidence = 0.50
                    a.subrole = "secondary_alpha_candidate"
                    a.reason = (
                        f"downgraded reactive_nucleophile → spectator by global "
                        f"aldol disambiguation; primary donor is {primary_donor.group_id}"
                    )

    def _negotiate_diels_alder(
        self,
        groups: list[FunctionalGroup],
        predictions: list[RolePrediction],
        assign_by_id: dict[str, NegotiatedRoleAssignment],
        warnings: list[NegotiationWarning],
    ) -> None:
        """Diels-Alder-like: assign diene_like and dienophile_like subroles."""
        cfg = self.config
        pi_preds = [p for p in predictions if p.group_type in _PI_GROUP_TYPES]

        if cfg.require_pericyclic_partners and len(pi_preds) < 2:
            warnings.append(_warn(
                "missing_pericyclic_partner",
                f"Fewer than 2 pi partners detected ({len(pi_preds)}); "
                "Diels-Alder requires diene + dienophile.",
                "warning",
                {"mechanism": "diels_alder_like", "n_pi_groups": len(pi_preds)},
            ))

        if not pi_preds:
            return

        all_nucleophile = all(
            p.predicted_role == Role.reactive_nucleophile for p in pi_preds
        )

        if all_nucleophile and len(pi_preds) >= 2:
            group_lookup = _group_by_id(groups)
            by_mol: dict[int, list[RolePrediction]] = {}
            for pred in pi_preds:
                grp = group_lookup.get(pred.group_id)
                mol_idx = grp.atom_refs[0].molecule_index if (grp and grp.atom_refs) else 0
                by_mol.setdefault(mol_idx, []).append(pred)

            mol_indices = sorted(by_mol.keys())

            if len(mol_indices) >= 2:
                dienophile_mol = min(mol_indices, key=lambda m: len(by_mol[m]))
                diene_mol = max(mol_indices, key=lambda m: len(by_mol[m]))

                if diene_mol == dienophile_mol:
                    # Tied: pick lowest-confidence as dienophile
                    dienophile_pred = min(pi_preds, key=lambda p: p.confidence)
                    diene_preds = [p for p in pi_preds if p.group_id != dienophile_pred.group_id]
                else:
                    diene_preds = by_mol[diene_mol]
                    dienophile_pred = by_mol[dienophile_mol][0]
            else:
                dienophile_pred = min(pi_preds, key=lambda p: p.confidence)
                diene_preds = [p for p in pi_preds if p.group_id != dienophile_pred.group_id]

            if dienophile_pred.group_id in assign_by_id:
                a = assign_by_id[dienophile_pred.group_id]
                a.final_role = Role.reactive_electrophile
                a.final_confidence = min(1.0, a.raw_confidence + 0.05)
                a.subrole = "dienophile_like"
                a.is_reaction_center = True
                a.reason = (
                    "reassigned reactive_nucleophile → reactive_electrophile as dienophile "
                    "by global Diels-Alder negotiation; "
                    "avoids all-nucleophile assignment from Phase 5 flat taxonomy"
                )

            for pred in diene_preds:
                if pred.group_id in assign_by_id:
                    a = assign_by_id[pred.group_id]
                    a.subrole = "diene_like"
                    a.is_reaction_center = True
                    a.reason = (
                        f"confirmed as diene partner (reactive_nucleophile, "
                        f"confidence={pred.confidence:.2f}); {a.reason}"
                    )

            marked = {dienophile_pred.group_id} | {p.group_id for p in diene_preds}
            for pred in pi_preds:
                if pred.group_id not in marked and pred.group_id in assign_by_id:
                    assign_by_id[pred.group_id].subrole = "secondary_pi_partner"

        else:
            # Some already have different roles — assign subroles from existing roles
            for pred in pi_preds:
                if pred.group_id not in assign_by_id:
                    continue
                a = assign_by_id[pred.group_id]
                if a.subrole is not None:
                    continue
                if a.final_role == Role.reactive_nucleophile:
                    a.subrole = "diene_like"
                    a.is_reaction_center = True
                elif a.final_role == Role.reactive_electrophile:
                    a.subrole = "dienophile_like"
                    a.is_reaction_center = True

    def _negotiate_radical(
        self,
        groups: list[FunctionalGroup],
        predictions: list[RolePrediction],
        assign_by_id: dict[str, NegotiatedRoleAssignment],
        warnings: list[NegotiationWarning],
    ) -> None:
        """Radical bromination-like: promote benzylic_site to reactive_radical."""
        cfg = self.config
        benzylic_groups = _groups_of_type(groups, FunctionalGroupType.benzylic_site)
        found_radical_center = False

        for group in benzylic_groups:
            gid = group.group_id
            if gid not in assign_by_id:
                continue
            a = assign_by_id[gid]
            if a.final_role != Role.reactive_radical:
                a.final_role = Role.reactive_radical
                a.final_confidence = max(a.raw_confidence, 0.75)
                a.reason = (
                    "promoted to reactive_radical by global radical negotiation "
                    "(benzylic_site provides resonance-stabilised radical); "
                    f"original: {a.reason}"
                )
            a.is_reaction_center = True
            found_radical_center = True

        if not found_radical_center:
            for a in assign_by_id.values():
                if a.final_role == Role.reactive_radical:
                    a.is_reaction_center = True
                    found_radical_center = True
                    break

        if not found_radical_center and cfg.require_radical_center:
            warnings.append(_warn(
                "missing_radical_center",
                "No radical center candidate found (no benzylic_site or reactive_radical).",
                "warning",
                {"mechanism": "radical_bromination_like"},
            ))

        warnings.append(_warn(
            "unsupported_radical_source",
            "Radical source (e.g. Br2, AIBN) is not a supported functional group in v0.1 "
            "and cannot be detected by Phase 2.",
            "info",
            {"mechanism": "radical_bromination_like", "v0.1_limitation": True},
        ))

    def _negotiate_ionic_addition(
        self,
        groups: list[FunctionalGroup],
        predictions: list[RolePrediction],
        assign_by_id: dict[str, NegotiatedRoleAssignment],
        warnings: list[NegotiationWarning],
    ) -> None:
        """Ionic addition-like: mark top nucleophile and electrophile as reaction center."""
        nuc_preds = [p for p in predictions if p.predicted_role == Role.reactive_nucleophile]
        elec_preds = [p for p in predictions if p.predicted_role == Role.reactive_electrophile]

        if nuc_preds:
            top = max(nuc_preds, key=lambda p: p.confidence)
            if top.group_id in assign_by_id:
                a = assign_by_id[top.group_id]
                a.is_reaction_center = True
                a.subrole = "ionic_nucleophile_candidate"
        else:
            warnings.append(_warn(
                "missing_nucleophile",
                "No nucleophile detected in ionic addition context.",
                "warning",
                {"mechanism": "ionic_addition_like"},
            ))

        if elec_preds:
            top = max(elec_preds, key=lambda p: p.confidence)
            if top.group_id in assign_by_id:
                a = assign_by_id[top.group_id]
                a.is_reaction_center = True
                a.subrole = "ionic_electrophile_candidate"
        else:
            warnings.append(_warn(
                "missing_electrophile",
                "No electrophile detected in ionic addition context.",
                "warning",
                {"mechanism": "ionic_addition_like"},
            ))

    def _negotiate_unknown(
        self,
        groups: list[FunctionalGroup],
        predictions: list[RolePrediction],
        assign_by_id: dict[str, NegotiatedRoleAssignment],
        warnings: list[NegotiationWarning],
    ) -> None:
        """Unknown mechanism: preserve raw roles, mark high-confidence non-spectators."""
        warnings.append(_warn(
            "unknown_mechanism",
            "No specific mechanism rule matched; raw role assignments preserved. "
            "High-confidence non-spectator groups are marked as possible reaction centers.",
            "info",
        ))
        for a in assign_by_id.values():
            if (
                a.final_role != Role.spectator
                and a.final_confidence >= self.config.min_center_confidence
            ):
                a.is_reaction_center = True


# ---------------------------------------------------------------------------
# Public helper functions
# ---------------------------------------------------------------------------


def negotiate_predictions(
    parsed_reaction: ParsedReaction,
    groups: list[FunctionalGroup],
    predictions: list[RolePrediction],
    config: NegotiatorConfig | None = None,
) -> NegotiationResult:
    """Convenience wrapper around RuleBasedNegotiator.negotiate."""
    return RuleBasedNegotiator(config).negotiate(parsed_reaction, groups, predictions)


def run_full_rule_pipeline(
    reaction_smiles: str,
    context: ReactionContext | str = ReactionContext.unknown,
    config: NegotiatorConfig | None = None,
) -> NegotiationResult:
    """One-call public pipeline: parser → identifier → predictor → negotiator.

    This is the first true one-call public interface for MENDEL.  It chains
    all implemented phases and returns a fully negotiated NegotiationResult.

    Args:
        reaction_smiles: Full reaction SMILES string (reactants>>products).
        context: Mechanistic category; accepts ReactionContext enum or plain string.
        config: Optional negotiator configuration.

    Returns:
        NegotiationResult with final coordinated assignments, mechanism hint,
        reaction center atoms, and warnings.
    """
    if isinstance(context, str):
        try:
            context = ReactionContext(context)
        except ValueError:
            context = ReactionContext.unknown

    parsed = parse_reaction_smiles(reaction_smiles, context=context)
    groups = identify_functional_groups(parsed)
    report = predict_roles_for_reaction(parsed, groups)
    return negotiate_predictions(parsed, groups, report.predictions, config)


def run_pipeline_with_mlp(
    reaction_smiles: str,
    mlp_checkpoint: str | Path,
    context: ReactionContext | str = ReactionContext.unknown,
    config: NegotiatorConfig | None = None,
    device: str = "cpu",
) -> NegotiationResult:
    """One-call pipeline using the Phase 7 MLP predictor instead of rule-based.

    parser → identifier → MLP predictor → negotiator

    Args:
        reaction_smiles: Full reaction SMILES (reactants>>products).
        mlp_checkpoint: Path to a checkpoint saved by MLPRolePredictor.save.
        context: Mechanistic category string or ReactionContext enum.
        config: Optional negotiator configuration.
        device: Torch device string ('cpu', 'cuda', 'mps').

    Returns:
        NegotiationResult identical in shape to run_full_rule_pipeline output.
    """
    from pathlib import Path as _Path

    from mendel.mlp import MLPRolePredictor

    if isinstance(context, str):
        try:
            context = ReactionContext(context)
        except ValueError:
            context = ReactionContext.unknown

    parsed = parse_reaction_smiles(reaction_smiles, context=context)
    groups = identify_functional_groups(parsed)
    mlp = MLPRolePredictor.load(_Path(mlp_checkpoint), device=device)
    mlp_preds = mlp.predict_from_reaction(parsed, groups)

    group_type_by_id = {g.group_id: g.group_type for g in groups}
    predictions = [
        RolePrediction(
            group_id=p.group_id,
            group_type=group_type_by_id[p.group_id],
            predicted_role=p.predicted_role,
            confidence=p.confidence,
            reason="mlp_prediction",
            metadata={"prediction_source": "mlp", "predictor_name": "phase7_mlp"},
        )
        for p in mlp_preds
    ]
    return negotiate_predictions(parsed, groups, predictions, config)


def summarize_negotiation_result(result: NegotiationResult) -> dict[str, Any]:
    """Convenience wrapper for RuleBasedNegotiator.summarize_result."""
    return RuleBasedNegotiator().summarize_result(result)


def get_final_role_counts(result: NegotiationResult) -> dict[str, int]:
    """Return count of each final role across all assignments."""
    counts: dict[str, int] = {}
    for a in result.assignments:
        key = a.final_role.value
        counts[key] = counts.get(key, 0) + 1
    return counts


def get_reaction_center_group_ids(result: NegotiationResult) -> list[str]:
    """Return group_ids for assignments marked as reaction center."""
    return [a.group_id for a in result.assignments if a.is_reaction_center]
