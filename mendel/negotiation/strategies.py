"""Rule-based mechanism strategies and confidence predicates."""

from __future__ import annotations

import re

from mendel.negotiation.reaction_center import mark_center
from mendel.negotiation.types import (
    _PI_GROUP_TYPES,
    NegotiatedRoleAssignment,
    NegotiationWarning,
    NegotiatorConfig,
    _group_by_id,
    _groups_of_type,
    _is_reactive_role,
    _pred_by_id,
    _warn,
)
from mendel.parser import ParsedReaction
from mendel.predictor import RolePrediction
from mendel.types import (
    FunctionalGroup,
    FunctionalGroupType,
    Role,
)


def confident_reactive(cfg: NegotiatorConfig, assignment: NegotiatedRoleAssignment) -> bool:
    return (
        _is_reactive_role(assignment.final_role)
        and assignment.final_confidence >= cfg.reactive_confidence_threshold
    )


def high_conf_spectator(cfg: NegotiatorConfig, assignment: NegotiatedRoleAssignment) -> bool:
    return (
        assignment.final_role == Role.spectator
        and assignment.final_confidence >= cfg.spectator_confidence_threshold
    )


def is_symmetric_self_reaction(parsed_reaction: ParsedReaction) -> bool:
    """True when two or more reactant molecules are identical.

    For such self-reactions (e.g. acetone + acetone aldol) the donor/acceptor
    assignment is not determinable from the reactant features, so the helpers
    must not impose an asymmetric single-donor convention.
    """
    normalized = [
        re.sub(r":\d+", "", mol.smiles) for mol in parsed_reaction.reactants
    ]
    return len(normalized) >= 2 and len(set(normalized)) < len(normalized)


def negotiate_sn2_e2(
    cfg: NegotiatorConfig,
    groups: list[FunctionalGroup],
    predictions: list[RolePrediction],
    assign_by_id: dict[str, NegotiatedRoleAssignment],
    warnings: list[NegotiationWarning],
    *,
    confidence_aware: bool = False,
    mechanism: str | None = None,
) -> None:
    """SN2/E2-like: mark halide as leaving group and reaction center.

    ``confidence_aware`` selects the center-selection policy: rule mode marks
    every halide as a center; mlp-aware mode marks only halides that are
    leaving groups or confidently reactive.
    """
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

        a.subrole = "leaving_group_site"
        a.metadata["v0.1_note"] = (
            "alkyl halide C–X represented as one group; "
            "electrophilic carbon not separately modelled in v0.1"
        )
        if confidence_aware:
            if a.final_role == Role.leaving_group or confident_reactive(cfg, a):
                mark_center(assign_by_id, gid, "mlp_aware_halide_center")
        else:
            a.is_reaction_center = True

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

    if confidence_aware and mechanism == "e2":
        warnings.append(_warn(
            "beta_center_not_fully_represented",
            "beta center not fully represented in v0.1 schema.",
            "info",
            {"mechanism": "e2"},
        ))


def negotiate_aldol(
    cfg: NegotiatorConfig,
    groups: list[FunctionalGroup],
    predictions: list[RolePrediction],
    assign_by_id: dict[str, NegotiatedRoleAssignment],
    warnings: list[NegotiationWarning],
    *,
    confidence_aware: bool = False,
    symmetric_self: bool = False,
) -> None:
    """Aldol-like: select primary donor alpha_carbon and acceptor carbonyl.

    ``confidence_aware`` selects the center-selection policy: rule mode marks
    the heuristically-chosen primary donor/acceptor; mlp-aware mode re-selects
    centers from confidently-reactive donors/acceptors.
    """
    cfg = cfg
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
        if not confidence_aware:
            a.is_reaction_center = True
        a.reason = (
            f"selected as primary aldol donor (alpha_carbon nucleophile, "
            f"confidence={primary_donor.confidence:.2f}); {a.reason}"
        )

    if primary_acceptor and primary_acceptor.group_id in assign_by_id:
        a = assign_by_id[primary_acceptor.group_id]
        a.subrole = "aldol_acceptor_carbonyl"
        if not confidence_aware:
            a.is_reaction_center = True
        a.reason = (
            f"selected as primary aldol acceptor (carbonyl electrophile, "
            f"confidence={primary_acceptor.confidence:.2f}); {a.reason}"
        )

    # Downgrade non-selected alpha_carbons if configured. Skipped for
    # symmetric self-reactions, where every alpha_carbon is an equally valid
    # donor and forcing a single one would contradict the symmetric labels.
    if (
        cfg.allow_role_downgrade_to_spectator
        and primary_donor is not None
        and not symmetric_self
    ):
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

    if not confidence_aware:
        return

    donors = [
        a for a in assign_by_id.values()
        if a.group_type == FunctionalGroupType.alpha_carbon
        and a.final_role == Role.reactive_nucleophile
        and confident_reactive(cfg, a)
    ]
    acceptors = [
        a for a in assign_by_id.values()
        if a.group_type == FunctionalGroupType.carbonyl
        and a.final_role == Role.reactive_electrophile
        and confident_reactive(cfg, a)
    ]
    if len(donors) != 1 or len(acceptors) != 1:
        warnings.append(_warn(
            "ambiguous_aldol_donor_acceptor_center",
            "ambiguous aldol donor/acceptor center",
            "warning",
        ))
        if donors:
            donor = max(donors, key=lambda a: a.final_confidence)
            mark_center(assign_by_id, donor.group_id, "mlp_aware_aldol_fallback_donor")
        if acceptors:
            acceptor = max(acceptors, key=lambda a: a.final_confidence)
            mark_center(
                assign_by_id,
                acceptor.group_id,
                "mlp_aware_aldol_fallback_acceptor",
            )
        return
    mark_center(assign_by_id, donors[0].group_id, "mlp_aware_aldol_donor")
    mark_center(assign_by_id, acceptors[0].group_id, "mlp_aware_aldol_acceptor")


def negotiate_diels_alder(
    cfg: NegotiatorConfig,
    groups: list[FunctionalGroup],
    predictions: list[RolePrediction],
    assign_by_id: dict[str, NegotiatedRoleAssignment],
    warnings: list[NegotiationWarning],
    *,
    confidence_aware: bool = False,
) -> None:
    """Diels-Alder-like: assign diene_like and dienophile_like subroles.

    ``confidence_aware`` selects the center-selection policy: rule mode marks
    the negotiated diene/dienophile partners; mlp-aware mode marks confidently
    reactive alkene pi-partners.
    """
    cfg = cfg
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
            if not confidence_aware:
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
                if not confidence_aware:
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
                if not confidence_aware:
                    a.is_reaction_center = True
            elif a.final_role == Role.reactive_electrophile:
                a.subrole = "dienophile_like"
                if not confidence_aware:
                    a.is_reaction_center = True

    if confidence_aware:
        for group in groups:
            assignment = assign_by_id.get(group.group_id)
            if assignment is None:
                continue
            if (
                group.group_type == FunctionalGroupType.alkene
                and confident_reactive(cfg, assignment)
            ):
                mark_center(
                    assign_by_id, group.group_id, "mlp_aware_diels_alder_pi_partner"
                )
                if assignment.final_role == Role.reactive_nucleophile:
                    assignment.subrole = "diene_like"
                elif assignment.final_role == Role.reactive_electrophile:
                    assignment.subrole = "dienophile_like"


def negotiate_radical(
    cfg: NegotiatorConfig,
    groups: list[FunctionalGroup],
    predictions: list[RolePrediction],
    assign_by_id: dict[str, NegotiatedRoleAssignment],
    warnings: list[NegotiationWarning],
    *,
    confidence_aware: bool = False,
) -> None:
    """Radical bromination-like: promote benzylic_site to reactive_radical.

    ``confidence_aware`` selects the center-selection policy: rule mode marks
    every benzylic (or a fallback radical) as a center; mlp-aware mode marks
    only confidently-reactive benzylic radicals.
    """
    cfg = cfg
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
        if not confidence_aware:
            a.is_reaction_center = True
        found_radical_center = True

    if not found_radical_center:
        for a in assign_by_id.values():
            if a.final_role == Role.reactive_radical:
                if not confidence_aware:
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

    if confidence_aware:
        for group in groups:
            assignment = assign_by_id.get(group.group_id)
            if assignment is None:
                continue
            if (
                group.group_type == FunctionalGroupType.benzylic_site
                and assignment.final_role == Role.reactive_radical
                and confident_reactive(cfg, assignment)
            ):
                mark_center(
                    assign_by_id, group.group_id, "mlp_aware_benzylic_radical"
                )


def negotiate_ester_hydrolysis(
    groups: list[FunctionalGroup],
    predictions: list[RolePrediction],
    assign_by_id: dict[str, NegotiatedRoleAssignment],
    warnings: list[NegotiationWarning],
) -> None:
    """Hydrolysis-like: force acyl group → electrophile, alcohol/water → nucleophile."""
    _ACYL_TYPES = {
        FunctionalGroupType.ester,
        FunctionalGroupType.amide,
        FunctionalGroupType.carboxylic_acid,
    }
    group_by_id = _group_by_id(groups)

    # Pick acyl electrophile — prefer ester, then amide, then carboxylic_acid
    acyl_pred: RolePrediction | None = None
    for gt in (FunctionalGroupType.ester, FunctionalGroupType.amide,
               FunctionalGroupType.carboxylic_acid):
        candidates = [p for p in predictions if p.group_type == gt]
        if candidates:
            acyl_pred = max(candidates, key=lambda p: p.confidence)
            break

    # Pick nucleophile — alcohol/water in a partner reactant molecule preferred
    nuc_candidates = [
        p for p in predictions if p.group_type == FunctionalGroupType.alcohol
    ]
    # Prefer the one on a different molecule than the acyl group
    if acyl_pred and nuc_candidates:
        acyl_group = group_by_id.get(acyl_pred.group_id)
        if acyl_group and acyl_group.atom_refs:
            acyl_mol = acyl_group.atom_refs[0].molecule_index
            cross_mol = [
                p for p in nuc_candidates
                if (g := group_by_id.get(p.group_id)) and g.atom_refs
                and g.atom_refs[0].molecule_index != acyl_mol
            ]
            if cross_mol:
                nuc_candidates = cross_mol
    nuc_pred = max(nuc_candidates, key=lambda p: p.confidence) if nuc_candidates else None

    if acyl_pred and acyl_pred.group_id in assign_by_id:
        a = assign_by_id[acyl_pred.group_id]
        a.final_role = Role.reactive_electrophile
        a.final_confidence = acyl_pred.confidence
        a.is_reaction_center = True
        a.subrole = "hydrolysis_acyl_electrophile"
        a.reason = (
            f"forced to reactive_electrophile by ester_hydrolysis rule "
            f"(acyl group + nucleophilic partner detected); {a.reason}"
        )
    else:
        warnings.append(_warn(
            "hydrolysis_missing_acyl",
            "No ester/amide/carboxylic_acid group found for hydrolysis.",
            "warning",
            {"mechanism": "ester_hydrolysis_like"},
        ))

    if nuc_pred and nuc_pred.group_id in assign_by_id:
        a = assign_by_id[nuc_pred.group_id]
        a.final_role = Role.reactive_nucleophile
        a.final_confidence = nuc_pred.confidence
        a.is_reaction_center = True
        a.subrole = "hydrolysis_nucleophile"
        a.reason = (
            f"forced to reactive_nucleophile by ester_hydrolysis rule; {a.reason}"
        )
    else:
        warnings.append(_warn(
            "hydrolysis_missing_nucleophile",
            "No alcohol/water nucleophile found for hydrolysis.",
            "warning",
            {"mechanism": "ester_hydrolysis_like"},
        ))


def negotiate_ionic_addition(
    groups: list[FunctionalGroup],
    predictions: list[RolePrediction],
    assign_by_id: dict[str, NegotiatedRoleAssignment],
    warnings: list[NegotiationWarning],
) -> None:
    """Ionic addition-like: mark top nucleophile and electrophile as reaction center."""
    nuc_preds = [p for p in predictions if p.predicted_role == Role.reactive_nucleophile]
    elec_preds = [p for p in predictions if p.predicted_role == Role.reactive_electrophile]

    # Michael acceptor disambiguation: when a molecule has both an alkene and a
    # carbonyl predicted as reactive_electrophile, the alkene is the 1,4-addition
    # site and the carbonyl is a spectator activating group.
    group_type_by_id = {g.group_id: g.group_type for g in groups}
    mol_idx_by_id: dict[str, int] = {}
    for g in groups:
        if g.atom_refs:
            mol_idx_by_id[g.group_id] = g.atom_refs[0].molecule_index

    elec_by_mol: dict[int, list[RolePrediction]] = {}
    for p in elec_preds:
        mol_idx = mol_idx_by_id.get(p.group_id)
        if mol_idx is not None:
            elec_by_mol.setdefault(mol_idx, []).append(p)

    demoted: set[str] = set()
    for mol_elec in elec_by_mol.values():
        types = {p.group_id: group_type_by_id.get(p.group_id) for p in mol_elec}
        has_alkene = any(t == FunctionalGroupType.alkene for t in types.values())
        has_carbonyl = any(
            t in (FunctionalGroupType.carbonyl, FunctionalGroupType.ester)
            for t in types.values()
        )
        if has_alkene and has_carbonyl:
            for p in mol_elec:
                if types.get(p.group_id) in (
                    FunctionalGroupType.carbonyl,
                    FunctionalGroupType.ester,
                ):
                    if p.group_id in assign_by_id:
                        assign_by_id[p.group_id].final_role = Role.spectator
                        assign_by_id[p.group_id].subrole = "michael_acceptor_activating_group"
                        demoted.add(p.group_id)
    elec_preds = [p for p in elec_preds if p.group_id not in demoted]

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


_EWG_ACTIVATING_TYPES: frozenset[FunctionalGroupType] = frozenset({
    FunctionalGroupType.carbonyl,
    FunctionalGroupType.ester,
    FunctionalGroupType.carboxylic_acid,
    FunctionalGroupType.amide,
    FunctionalGroupType.nitrile,
    FunctionalGroupType.nitro,
})


def _michael_acceptor_groups(groups: list[FunctionalGroup]) -> list[FunctionalGroup]:
    return [
        g for g in groups
        if g.group_type == FunctionalGroupType.alkene
        and bool(g.metadata.get("is_michael_acceptor"))
    ]


def _demote_activating_ewg(
    acceptor: FunctionalGroup,
    groups: list[FunctionalGroup],
    assign_by_id: dict[str, NegotiatedRoleAssignment],
) -> None:
    """Demote the conjugated EWG on the acceptor's molecule to a spectator activator."""
    if not acceptor.atom_refs:
        return
    mol_idx = acceptor.atom_refs[0].molecule_index
    for other in groups:
        if other.group_id == acceptor.group_id or not other.atom_refs:
            continue
        if (
            other.atom_refs[0].molecule_index == mol_idx
            and other.group_type in _EWG_ACTIVATING_TYPES
        ):
            oa = assign_by_id.get(other.group_id)
            if oa is not None and oa.final_role == Role.reactive_electrophile:
                oa.final_role = Role.spectator
                oa.final_confidence = 0.55
                oa.subrole = "michael_activating_group"
                oa.is_reaction_center = False
                oa.reason = (
                    "demoted to conjugated activating group (carbonyl/EWG provides "
                    f"electron withdrawal, not the reaction center); {oa.reason}"
                )


def negotiate_michael(
    cfg: NegotiatorConfig,
    groups: list[FunctionalGroup],
    predictions: list[RolePrediction],
    assign_by_id: dict[str, NegotiatedRoleAssignment],
    warnings: list[NegotiationWarning],
) -> None:
    """Michael / conjugate addition: beta-carbon of the enone is the electrophilic center.

    The conjugated carbonyl/EWG is demoted to an activating spectator; the highest
    -confidence nucleophile (if any) is marked as the donor center.
    """
    acceptors = _michael_acceptor_groups(groups)
    if not acceptors:
        warnings.append(_warn(
            "missing_michael_acceptor",
            "No Michael acceptor alkene detected in michael_like context.",
            "warning",
            {"mechanism": "michael_like"},
        ))

    for grp in acceptors:
        a = assign_by_id.get(grp.group_id)
        if a is None:
            continue
        a.final_role = Role.reactive_electrophile
        a.final_confidence = max(a.raw_confidence, 0.70)
        a.subrole = "michael_acceptor_beta_carbon"
        a.is_reaction_center = True
        a.metadata["activating_group"] = str(grp.metadata.get("activating_group", ""))
        beta = grp.metadata.get("beta_carbon_atom_index")
        if isinstance(beta, int):
            a.metadata["beta_carbon_atom_index"] = beta
        a.reason = (
            "alkene marked Michael-acceptor beta-carbon (electrophilic 1,4-addition "
            f"site) by michael negotiation; {a.reason}"
        )
        _demote_activating_ewg(grp, groups, assign_by_id)

    nuc_preds = [p for p in predictions if p.predicted_role == Role.reactive_nucleophile]
    if nuc_preds:
        top = max(nuc_preds, key=lambda p: p.confidence)
        na = assign_by_id.get(top.group_id)
        if na is not None and na.final_role == Role.reactive_nucleophile:
            na.is_reaction_center = True
            na.subrole = na.subrole or "michael_donor_nucleophile"
    else:
        warnings.append(_warn(
            "missing_nucleophile",
            "No nucleophile detected for Michael addition; donor may be unsupported "
            "in v0.1 (e.g. thiol).",
            "warning",
            {"mechanism": "michael_like"},
        ))


def negotiate_click(
    cfg: NegotiatorConfig,
    groups: list[FunctionalGroup],
    predictions: list[RolePrediction],
    assign_by_id: dict[str, NegotiatedRoleAssignment],
    warnings: list[NegotiationWarning],
) -> None:
    """Azide-alkyne (Huisgen / click) cycloaddition: both partners are centers."""
    azides = _groups_of_type(groups, FunctionalGroupType.azide)
    alkynes = _groups_of_type(groups, FunctionalGroupType.alkyne)

    if not azides:
        warnings.append(_warn(
            "missing_click_dipole",
            "No azide 1,3-dipole detected in click_like context.",
            "warning",
            {"mechanism": "click_like"},
        ))
    if not alkynes:
        warnings.append(_warn(
            "missing_click_dipolarophile",
            "No alkyne dipolarophile detected in click_like context.",
            "warning",
            {"mechanism": "click_like"},
        ))

    for grp in azides:
        a = assign_by_id.get(grp.group_id)
        if a is None:
            continue
        if a.final_role == Role.spectator:
            a.final_role = Role.reactive_nucleophile
            a.final_confidence = max(a.raw_confidence, 0.60)
        a.subrole = "click_dipole_azide"
        a.is_reaction_center = True
        a.reason = f"azide 1,3-dipole in click/Huisgen cycloaddition; {a.reason}"

    for grp in alkynes:
        a = assign_by_id.get(grp.group_id)
        if a is None:
            continue
        if a.final_role == Role.spectator:
            a.final_role = Role.reactive_electrophile
            a.final_confidence = max(a.raw_confidence, 0.60)
        a.subrole = "click_dipolarophile_alkyne"
        a.is_reaction_center = True
        a.reason = f"alkyne dipolarophile in click/Huisgen cycloaddition; {a.reason}"


def negotiate_radical_addition(
    cfg: NegotiatorConfig,
    groups: list[FunctionalGroup],
    predictions: list[RolePrediction],
    assign_by_id: dict[str, NegotiatedRoleAssignment],
    warnings: list[NegotiationWarning],
    *,
    variant: str,
) -> None:
    """Radical addition family: giese (conjugate), minisci (heteroarene), or plain alkene.

    ``variant`` is one of "giese", "minisci", "radical_addition".
    """
    found = False

    if variant == "giese":
        for grp in _michael_acceptor_groups(groups):
            a = assign_by_id.get(grp.group_id)
            if a is None:
                continue
            a.final_role = Role.reactive_electrophile
            a.final_confidence = max(a.raw_confidence, 0.65)
            a.subrole = "giese_acceptor_beta_carbon"
            a.is_reaction_center = True
            a.reason = (
                "Michael-acceptor beta-carbon as Giese radical acceptor "
                f"(SOMO-LUMO addition); {a.reason}"
            )
            _demote_activating_ewg(grp, groups, assign_by_id)
            found = True
    elif variant == "minisci":
        for grp in _groups_of_type(groups, FunctionalGroupType.aromatic):
            if not grp.metadata.get("heteroaromatic_n"):
                continue
            a = assign_by_id.get(grp.group_id)
            if a is None:
                continue
            if a.final_role == Role.spectator:
                a.final_role = Role.reactive_electrophile
                a.final_confidence = max(a.raw_confidence, 0.60)
            a.subrole = "minisci_heteroarene"
            a.is_reaction_center = True
            a.reason = (
                "protonated heteroarene as Minisci radical acceptor "
                f"(electron-poor C adjacent to N); {a.reason}"
            )
            found = True
    else:  # radical_addition
        for grp in _groups_of_type(groups, FunctionalGroupType.alkene):
            a = assign_by_id.get(grp.group_id)
            if a is None:
                continue
            a.subrole = "radical_acceptor_alkene"
            a.is_reaction_center = True
            a.reason = f"alkene as radical-addition acceptor; {a.reason}"
            found = True

    # Any explicit radical carbon is the chain carrier / reaction center.
    for a in assign_by_id.values():
        if a.final_role == Role.reactive_radical:
            a.is_reaction_center = True
            found = True

    if not found:
        warnings.append(_warn(
            "missing_radical_acceptor",
            f"No radical acceptor detected for {variant}.",
            "warning",
            {"mechanism": f"{variant}_like"},
        ))

    warnings.append(_warn(
        "unsupported_radical_source",
        "Radical source/initiator (e.g. AIBN, Br2, R•) is not a supported functional "
        "group in v0.1 and cannot be detected by Phase 2.",
        "info",
        {"mechanism": f"{variant}_like", "v0.1_limitation": True},
    ))


def _is_noop_reaction(parsed_reaction: ParsedReaction | None) -> bool:
    """True when products are identical to reactants (a no-reaction control).

    Compares atom-map-stripped SMILES multisets. Used to avoid fabricating a
    reaction center for benign 'no reaction' inputs.
    """
    if parsed_reaction is None or not parsed_reaction.products:
        return False
    react = sorted(re.sub(r":\d+", "", m.smiles) for m in parsed_reaction.reactants)
    prod = sorted(re.sub(r":\d+", "", m.smiles) for m in parsed_reaction.products)
    return bool(react) and react == prod


def negotiate_unknown(
    cfg: NegotiatorConfig,
    groups: list[FunctionalGroup],
    predictions: list[RolePrediction],
    assign_by_id: dict[str, NegotiatedRoleAssignment],
    warnings: list[NegotiationWarning],
    *,
    parsed_reaction: ParsedReaction | None = None,
) -> None:
    """Unknown mechanism: preserve raw roles, mark high-confidence non-spectators.

    For a literal no-op reaction (products == reactants) no reaction center is
    marked — a benign 'no reaction' control must not invent a reactive center.
    """
    if _is_noop_reaction(parsed_reaction):
        warnings.append(_warn(
            "no_reaction_no_center",
            "Products are identical to reactants (no-reaction control); no reaction "
            "center marked.",
            "info",
        ))
        return

    warnings.append(_warn(
        "unknown_mechanism",
        "No specific mechanism rule matched; raw role assignments preserved. "
        "High-confidence non-spectator groups are marked as possible reaction centers.",
        "info",
    ))
    for a in assign_by_id.values():
        if (
            a.final_role != Role.spectator
            and a.final_confidence >= cfg.min_center_confidence
        ):
            a.is_reaction_center = True
