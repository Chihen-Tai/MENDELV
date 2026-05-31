"""Mechanism-hint inference."""

from __future__ import annotations

from mendel.negotiation.types import (
    _has_group_type,
)
from mendel.parser import ParsedReaction
from mendel.predictor import RolePrediction
from mendel.types import (
    FunctionalGroup,
    FunctionalGroupType,
    ReactionContext,
    Role,
)


def _has_michael_acceptor(groups: list[FunctionalGroup]) -> bool:
    """True when any alkene group was tagged as a Michael acceptor (enone, acrylate…)."""
    return any(
        g.group_type == FunctionalGroupType.alkene
        and bool(g.metadata.get("is_michael_acceptor"))
        for g in groups
    )


def _has_heteroaromatic_n(groups: list[FunctionalGroup]) -> bool:
    """True when any aromatic ring contains an aromatic nitrogen (pyridine-like)."""
    return any(
        g.group_type == FunctionalGroupType.aromatic
        and bool(g.metadata.get("heteroaromatic_n"))
        for g in groups
    )


def infer_mechanism_hint(
    parsed_reaction: ParsedReaction,
    groups: list[FunctionalGroup],
    predictions: list[RolePrediction],
) -> str:
    """Infer the most likely mechanism type.

    Returns one of:
      "sn2_or_e2_like", "aldol_like", "michael_like", "diels_alder_like",
      "click_like", "radical_bromination_like", "radical_addition_like",
      "giese_like", "minisci_like", "ester_hydrolysis_like",
      "ionic_addition_like", "unknown"
    """
    ctx = parsed_reaction.context

    # Azide + alkyne (Huisgen / click) is distinctive enough to win over the
    # generic pericyclic Diels-Alder default, regardless of declared context.
    if _has_group_type(groups, FunctionalGroupType.azide) and _has_group_type(
        groups, FunctionalGroupType.alkyne
    ):
        return "click_like"

    if ctx == ReactionContext.radical:
        # Giese (radical conjugate addition) before generic radical addition,
        # then Minisci (heteroaromatic), then plain alkene addition, then the
        # historical bromination default.
        if _has_michael_acceptor(groups):
            return "giese_like"
        if _has_heteroaromatic_n(groups):
            return "minisci_like"
        if _has_group_type(groups, FunctionalGroupType.alkene):
            return "radical_addition_like"
        return "radical_bromination_like"

    if ctx == ReactionContext.pericyclic:
        return "diels_alder_like"

    if ctx == ReactionContext.ionic:
        # Michael / conjugate addition must be checked before aldol: an enone
        # presents both a carbonyl and an alpha_carbon, but the 1,4-addition at
        # the beta carbon is the real event.
        if _has_michael_acceptor(groups):
            return "michael_like"

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

        # Aldol checked before SN2 — carbonyl+alpha_carbon signature is specific.
        # Exception: if an alkene is also predicted reactive_electrophile (Michael
        # acceptor), fall through to ionic_addition_like instead.
        if has_carbonyl and has_alpha:
            alkene_gids = {g.group_id for g in groups if g.group_type == FunctionalGroupType.alkene}
            has_alkene_elec = any(
                p.predicted_role == Role.reactive_electrophile and p.group_id in alkene_gids
                for p in predictions
            )
            if not has_alkene_elec:
                return "aldol_like"

        if has_halide or has_leaving_pred:
            return "sn2_or_e2_like"

        # Hydrolysis: acyl group (ester/amide/carboxylic_acid) + nucleophilic
        # solvent partner (alcohol/water).  Checked before ionic_addition so the
        # weaker "nuc+elec present" signal doesn't swallow it.
        has_acyl = (
            _has_group_type(groups, FunctionalGroupType.ester)
            or _has_group_type(groups, FunctionalGroupType.amide)
            or _has_group_type(groups, FunctionalGroupType.carboxylic_acid)
        )
        has_alcohol = _has_group_type(groups, FunctionalGroupType.alcohol)
        if has_acyl and has_alcohol:
            return "ester_hydrolysis_like"

        if has_nuc and has_elec:
            return "ionic_addition_like"

    return "unknown"


def mechanism_from_metadata(parsed_reaction: ParsedReaction, fallback: str) -> str:
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
