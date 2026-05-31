"""MLP-aware negotiation dispatch."""

from __future__ import annotations

from mendel.negotiation.mechanism_hints import mechanism_from_metadata
from mendel.negotiation.reaction_center import mark_center
from mendel.negotiation.strategies import (
    confident_reactive,
    high_conf_spectator,
    is_symmetric_self_reaction,
    negotiate_aldol,
    negotiate_diels_alder,
    negotiate_ester_hydrolysis,
    negotiate_radical,
    negotiate_sn2_e2,
)
from mendel.negotiation.types import (
    NegotiatedRoleAssignment,
    NegotiationWarning,
    NegotiatorConfig,
    _is_reactive_role,
    _warn,
)
from mendel.parser import ParsedReaction
from mendel.predictor import RolePrediction
from mendel.types import (
    FunctionalGroup,
    FunctionalGroupType,
    Role,
)


def negotiate_mlp_aware(
    cfg: NegotiatorConfig,
    parsed_reaction: ParsedReaction,
    groups: list[FunctionalGroup],
    predictions: list[RolePrediction],
    assign_by_id: dict[str, NegotiatedRoleAssignment],
    warnings: list[NegotiationWarning],
    fallback_hint: str,
) -> str:
    mechanism = mechanism_from_metadata(parsed_reaction, fallback_hint)

    if mechanism in {"control", "ester_control", "nitrile_control"}:
        return mlp_aware_control(cfg, mechanism, assign_by_id, warnings)
    if mechanism in {"sn2", "e2", "sn2_or_e2_like"}:
        negotiate_sn2_e2(cfg, 
            groups,
            predictions,
            assign_by_id,
            warnings,
            confidence_aware=True,
            mechanism=mechanism,
        )
        return "sn2_or_e2_like"
    if mechanism == "carbonyl_addition":
        mlp_aware_carbonyl_addition(cfg, groups, assign_by_id)
        return "ionic_addition_like"
    if mechanism in {"aldol", "cross_aldol", "aldol_like"}:
        negotiate_aldol(cfg, 
            groups,
            predictions,
            assign_by_id,
            warnings,
            confidence_aware=True,
            symmetric_self=is_symmetric_self_reaction(parsed_reaction),
        )
        return "aldol_like"
    if mechanism in {"diels_alder", "diels_alder_like"}:
        negotiate_diels_alder(cfg, 
            groups,
            predictions,
            assign_by_id,
            warnings,
            confidence_aware=True,
        )
        return "diels_alder_like"
    if mechanism in {"benzylic_radical_bromination", "radical_bromination_like"}:
        negotiate_radical(cfg, 
            groups,
            predictions,
            assign_by_id,
            warnings,
            confidence_aware=True,
        )
        return "radical_bromination_like"
    if mechanism == "nitroalkane_deprotonation":
        mlp_aware_nitroalkane(groups, assign_by_id)
        return "ionic_addition_like"
    if mechanism in {"ester_hydrolysis", "amide_hydrolysis", "ester_hydrolysis_like"} \
            or fallback_hint == "ester_hydrolysis_like":
        negotiate_ester_hydrolysis(groups, predictions, assign_by_id, warnings)
        return "ester_hydrolysis_like"

    for assignment in assign_by_id.values():
        if confident_reactive(cfg, assignment):
            mark_center(assign_by_id, assignment.group_id, "mlp_aware_unknown_reactive")
    return fallback_hint


def mlp_aware_control(
    cfg: NegotiatorConfig,
    mechanism: str,
    assign_by_id: dict[str, NegotiatedRoleAssignment],
    warnings: list[NegotiationWarning],
) -> str:
    high_spectators = [a for a in assign_by_id.values() if high_conf_spectator(cfg, a)]
    reactive = [a for a in assign_by_id.values() if _is_reactive_role(a.final_role)]
    if reactive:
        warnings.append(_warn(
            "control_reactive_prediction_suppressed",
            "Control reaction has reactive predictions; possible false positive.",
            "warning",
            {"mechanism": mechanism},
        ))
    if cfg.suppress_control_centers and high_spectators:
        for assignment in assign_by_id.values():
            assignment.is_reaction_center = False
            assignment.metadata["center_selection_reason"] = "control_suppressed"
        return "control_like"
    for assignment in reactive:
        if confident_reactive(cfg, assignment):
            mark_center(assign_by_id, assignment.group_id, "control_reactive_low_trust")
    return "control_like"


def mlp_aware_carbonyl_addition(
    cfg: NegotiatorConfig,
    groups: list[FunctionalGroup],
    assign_by_id: dict[str, NegotiatedRoleAssignment],
) -> None:
    for group in groups:
        assignment = assign_by_id.get(group.group_id)
        if assignment is None or high_conf_spectator(cfg, assignment):
            continue
        if (
            group.group_type == FunctionalGroupType.carbonyl
            and assignment.final_role == Role.reactive_electrophile
            and confident_reactive(cfg, assignment)
        ):
            mark_center(assign_by_id, group.group_id, "mlp_aware_carbonyl_addition")


def mlp_aware_nitroalkane(
    groups: list[FunctionalGroup],
    assign_by_id: dict[str, NegotiatedRoleAssignment],
) -> None:
    for group in groups:
        assignment = assign_by_id.get(group.group_id)
        if assignment is None:
            continue
        if group.group_type == FunctionalGroupType.alpha_carbon:
            mark_center(assign_by_id, group.group_id, "mlp_aware_nitronate_alpha")
            assignment.metadata["center_selection_note"] = (
                "alpha carbon represents nitronate-like center in v0.1"
            )
