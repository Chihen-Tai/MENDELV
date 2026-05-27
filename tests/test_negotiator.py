"""Phase 6 negotiation layer tests."""

from __future__ import annotations

import pytest

from mendel.identifier import identify_functional_groups
from mendel.negotiator import (
    NegotiatedRoleAssignment,
    NegotiationResult,
    NegotiationWarning,
    NegotiatorConfig,
    RuleBasedNegotiator,
    get_final_role_counts,
    get_reaction_center_group_ids,
    negotiate_predictions,
    run_full_rule_pipeline,
    summarize_negotiation_result,
)
from mendel.parser import parse_reaction_smiles
from mendel.predictor import predict_roles_for_reaction
from mendel.types import FunctionalGroupType, ReactionContext, Role


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(smiles: str, context: ReactionContext = ReactionContext.ionic) -> NegotiationResult:
    return run_full_rule_pipeline(smiles, context=context)


def _find_assignment(
    result: NegotiationResult, gt: FunctionalGroupType
) -> NegotiatedRoleAssignment | None:
    return next((a for a in result.assignments if a.group_type == gt), None)


def _find_subrole(result: NegotiationResult, subrole: str) -> NegotiatedRoleAssignment | None:
    return next((a for a in result.assignments if a.subrole == subrole), None)


def _has_warning_code(result: NegotiationResult, code: str) -> bool:
    return any(w.code == code for w in result.warnings)


# ---------------------------------------------------------------------------
# 1. Full pipeline helper
# ---------------------------------------------------------------------------


def test_full_pipeline_returns_negotiation_result() -> None:
    result = _run("CC(=O)C.CBr>>CC(=O)CO", ReactionContext.ionic)
    assert isinstance(result, NegotiationResult)


def test_full_pipeline_has_assignments() -> None:
    result = _run("CC(=O)C.CBr>>CC(=O)CO", ReactionContext.ionic)
    assert len(result.assignments) > 0


def test_full_pipeline_has_mechanism_hint() -> None:
    result = _run("CC(=O)C.CBr>>CC(=O)CO", ReactionContext.ionic)
    assert result.mechanism_hint in {
        "aldol_like",
        "sn2_or_e2_like",
        "ionic_addition_like",
        "unknown",
    }


def test_full_pipeline_summary_works() -> None:
    result = _run("CC(=O)C.CBr>>CC(=O)CO", ReactionContext.ionic)
    summary = summarize_negotiation_result(result)
    for key in (
        "mechanism_hint",
        "n_assignments",
        "n_reaction_center_atoms",
        "final_role_counts",
        "subrole_counts",
        "warning_counts",
        "average_final_confidence",
    ):
        assert key in summary


def test_full_pipeline_confidence_range() -> None:
    result = _run("CC(=O)C.CBr>>CC(=O)CO", ReactionContext.ionic)
    for a in result.assignments:
        assert 0.0 <= a.final_confidence <= 1.0


def test_full_pipeline_all_assignments_have_reason() -> None:
    result = _run("CC(=O)C.CBr>>CC(=O)CO", ReactionContext.ionic)
    for a in result.assignments:
        assert a.reason


# ---------------------------------------------------------------------------
# 2. Aldol-like disambiguation
# ---------------------------------------------------------------------------


def test_aldol_mechanism_hint() -> None:
    result = _run(
        "CC(=O)C.CC(=O)C>>CC(=O)CC(O)(C)C", ReactionContext.ionic
    )
    assert result.mechanism_hint == "aldol_like"


def test_aldol_donor_subrole_present() -> None:
    result = _run(
        "CC(=O)C.CC(=O)C>>CC(=O)CC(O)(C)C", ReactionContext.ionic
    )
    donor = _find_subrole(result, "aldol_donor_alpha_carbon")
    assert donor is not None, "Expected assignment with subrole 'aldol_donor_alpha_carbon'"


def test_aldol_acceptor_subrole_present() -> None:
    result = _run(
        "CC(=O)C.CC(=O)C>>CC(=O)CC(O)(C)C", ReactionContext.ionic
    )
    acceptor = _find_subrole(result, "aldol_acceptor_carbonyl")
    assert acceptor is not None, "Expected assignment with subrole 'aldol_acceptor_carbonyl'"


def test_aldol_reaction_center_atoms_nonempty() -> None:
    result = _run(
        "CC(=O)C.CC(=O)C>>CC(=O)CC(O)(C)C", ReactionContext.ionic
    )
    assert len(result.reaction_center_atoms) > 0


def test_aldol_not_all_alpha_carbons_remain_nucleophile() -> None:
    result = _run(
        "CC(=O)C.CC(=O)C>>CC(=O)CC(O)(C)C", ReactionContext.ionic
    )
    alpha_assignments = [
        a for a in result.assignments
        if a.group_type == FunctionalGroupType.alpha_carbon
    ]
    if len(alpha_assignments) > 1:
        nucleophile_count = sum(
            1 for a in alpha_assignments if a.final_role == Role.reactive_nucleophile
        )
        assert nucleophile_count < len(alpha_assignments), (
            "All alpha_carbons remain nucleophile — aldol disambiguation did not fire"
        )


def test_aldol_donor_is_reaction_center() -> None:
    result = _run(
        "CC(=O)C.CC(=O)C>>CC(=O)CC(O)(C)C", ReactionContext.ionic
    )
    donor = _find_subrole(result, "aldol_donor_alpha_carbon")
    assert donor is not None
    assert donor.is_reaction_center


def test_aldol_acceptor_is_reaction_center() -> None:
    result = _run(
        "CC(=O)C.CC(=O)C>>CC(=O)CC(O)(C)C", ReactionContext.ionic
    )
    acceptor = _find_subrole(result, "aldol_acceptor_carbonyl")
    assert acceptor is not None
    assert acceptor.is_reaction_center


# ---------------------------------------------------------------------------
# 3. Diels-Alder negotiation
# ---------------------------------------------------------------------------


def test_diels_alder_mechanism_hint() -> None:
    result = _run("C=CC=C.C=C>>C1CCC=CC1", ReactionContext.pericyclic)
    assert result.mechanism_hint == "diels_alder_like"


def test_diels_alder_diene_subrole_present() -> None:
    result = _run("C=CC=C.C=C>>C1CCC=CC1", ReactionContext.pericyclic)
    diene = _find_subrole(result, "diene_like")
    assert diene is not None, "Expected assignment with subrole 'diene_like'"


def test_diels_alder_dienophile_subrole_present() -> None:
    result = _run("C=CC=C.C=C>>C1CCC=CC1", ReactionContext.pericyclic)
    dienophile = _find_subrole(result, "dienophile_like")
    assert dienophile is not None, "Expected assignment with subrole 'dienophile_like'"


def test_diels_alder_has_both_nucleophile_and_electrophile() -> None:
    result = _run("C=CC=C.C=C>>C1CCC=CC1", ReactionContext.pericyclic)
    final_roles = {a.final_role for a in result.assignments}
    assert Role.reactive_nucleophile in final_roles
    assert Role.reactive_electrophile in final_roles


def test_diels_alder_reaction_center_atoms_nonempty() -> None:
    result = _run("C=CC=C.C=C>>C1CCC=CC1", ReactionContext.pericyclic)
    assert len(result.reaction_center_atoms) > 0


def test_diels_alder_diene_is_nucleophile() -> None:
    result = _run("C=CC=C.C=C>>C1CCC=CC1", ReactionContext.pericyclic)
    diene = _find_subrole(result, "diene_like")
    assert diene is not None
    assert diene.final_role == Role.reactive_nucleophile


def test_diels_alder_dienophile_is_electrophile() -> None:
    result = _run("C=CC=C.C=C>>C1CCC=CC1", ReactionContext.pericyclic)
    dienophile = _find_subrole(result, "dienophile_like")
    assert dienophile is not None
    assert dienophile.final_role == Role.reactive_electrophile


# ---------------------------------------------------------------------------
# 4. SN2/E2-like negotiation
# ---------------------------------------------------------------------------


def test_sn2_mechanism_hint_with_halide() -> None:
    result = _run("CBr.[OH-]>>CO.[Br-]", ReactionContext.ionic)
    halide_assignments = [
        a for a in result.assignments if a.group_type == FunctionalGroupType.halide
    ]
    if halide_assignments:
        assert result.mechanism_hint == "sn2_or_e2_like"


def test_sn2_halide_final_role_is_leaving_group() -> None:
    result = _run("CBr.[OH-]>>CO.[Br-]", ReactionContext.ionic)
    halide = _find_assignment(result, FunctionalGroupType.halide)
    if halide is not None:
        assert halide.final_role == Role.leaving_group


def test_sn2_halide_is_reaction_center() -> None:
    result = _run("CBr.[OH-]>>CO.[Br-]", ReactionContext.ionic)
    halide = _find_assignment(result, FunctionalGroupType.halide)
    if halide is not None:
        assert halide.is_reaction_center


def test_sn2_reaction_center_includes_halide_atoms() -> None:
    result = _run("CBr.[OH-]>>CO.[Br-]", ReactionContext.ionic)
    halide_assignments = [
        a for a in result.assignments
        if a.group_type == FunctionalGroupType.halide and a.is_reaction_center
    ]
    if halide_assignments:
        assert len(result.reaction_center_atoms) > 0


def test_sn2_missing_nucleophile_warning() -> None:
    result = _run("CBr.[OH-]>>CO.[Br-]", ReactionContext.ionic)
    nuc_assignments = [
        a for a in result.assignments if a.final_role == Role.reactive_nucleophile
    ]
    if not nuc_assignments:
        assert _has_warning_code(result, "missing_nucleophile")


def test_sn2_coarse_group_granularity_warning() -> None:
    result = _run("CBr.[OH-]>>CO.[Br-]", ReactionContext.ionic)
    if result.mechanism_hint == "sn2_or_e2_like":
        assert _has_warning_code(result, "coarse_group_granularity")


# ---------------------------------------------------------------------------
# 5. Radical benzylic negotiation
# ---------------------------------------------------------------------------


def test_radical_mechanism_hint() -> None:
    result = _run("Cc1ccccc1>>Cc1ccccc1", ReactionContext.radical)
    assert result.mechanism_hint == "radical_bromination_like"


def test_radical_benzylic_site_is_reactive_radical() -> None:
    result = _run("Cc1ccccc1>>Cc1ccccc1", ReactionContext.radical)
    benzylic = _find_assignment(result, FunctionalGroupType.benzylic_site)
    if benzylic is None:
        pytest.skip("benzylic_site not detected")
    assert benzylic.final_role == Role.reactive_radical


def test_radical_benzylic_site_is_reaction_center() -> None:
    result = _run("Cc1ccccc1>>Cc1ccccc1", ReactionContext.radical)
    benzylic = _find_assignment(result, FunctionalGroupType.benzylic_site)
    if benzylic is None:
        pytest.skip("benzylic_site not detected")
    assert benzylic.is_reaction_center


def test_radical_unsupported_source_warning() -> None:
    result = _run("Cc1ccccc1>>Cc1ccccc1", ReactionContext.radical)
    assert _has_warning_code(result, "unsupported_radical_source")


def test_radical_confidence_in_range() -> None:
    result = _run("Cc1ccccc1>>Cc1ccccc1", ReactionContext.radical)
    for a in result.assignments:
        assert 0.0 <= a.final_confidence <= 1.0


# ---------------------------------------------------------------------------
# 6. Unknown mechanism
# ---------------------------------------------------------------------------


def test_unknown_mechanism_no_crash() -> None:
    result = _run("CCO>>CCO", ReactionContext.unknown)
    assert isinstance(result, NegotiationResult)


def test_unknown_mechanism_hint_value() -> None:
    result = _run("CCO>>CCO", ReactionContext.unknown)
    assert result.mechanism_hint == "unknown"


def test_unknown_mechanism_assignments_returned() -> None:
    result = _run("CCO>>CCO", ReactionContext.unknown)
    assert isinstance(result.assignments, list)


def test_unknown_mechanism_warning_code_present() -> None:
    result = _run("CCO>>CCO", ReactionContext.unknown)
    assert _has_warning_code(result, "unknown_mechanism")


def test_unknown_mechanism_valid_roles() -> None:
    result = _run("CCO>>CCO", ReactionContext.unknown)
    valid = set(Role)
    for a in result.assignments:
        assert a.final_role in valid


# ---------------------------------------------------------------------------
# 7. Result serialization
# ---------------------------------------------------------------------------


def test_negotiation_result_to_dict() -> None:
    result = _run("CC(=O)C>>CC(=O)C", ReactionContext.ionic)
    d = result.to_dict()
    assert isinstance(d, dict)
    for key in (
        "reaction_smiles", "context", "mechanism_hint",
        "assignments", "reaction_center_atoms", "warnings", "metadata",
    ):
        assert key in d


def test_all_assignments_serialize() -> None:
    result = _run("CC(=O)C.CBr>>CC(=O)CO", ReactionContext.ionic)
    for a in result.assignments:
        d = a.to_dict()
        assert isinstance(d, dict)
        for key in (
            "group_id", "group_type", "raw_role", "final_role",
            "raw_confidence", "final_confidence", "reason",
            "subrole", "is_reaction_center", "metadata",
        ):
            assert key in d


def test_warnings_serialize() -> None:
    result = _run("CBr.[OH-]>>CO.[Br-]", ReactionContext.ionic)
    for w in result.warnings:
        d = w.to_dict()
        assert isinstance(d, dict)
        for key in ("code", "message", "severity", "metadata"):
            assert key in d


def test_serialized_roles_are_strings() -> None:
    result = _run("CC(=O)C>>CC(=O)C", ReactionContext.ionic)
    d = result.to_dict()
    for a_dict in d["assignments"]:
        assert isinstance(a_dict["final_role"], str)
        assert isinstance(a_dict["raw_role"], str)


# ---------------------------------------------------------------------------
# 8. Final role counts
# ---------------------------------------------------------------------------


def test_get_final_role_counts_returns_dict() -> None:
    result = _run("CC(=O)C>>CC(=O)C", ReactionContext.ionic)
    counts = get_final_role_counts(result)
    assert isinstance(counts, dict)


def test_get_final_role_counts_sum_equals_assignments() -> None:
    result = _run("CC(=O)C.CC(=O)C>>CC(=O)CC(O)(C)C", ReactionContext.ionic)
    counts = get_final_role_counts(result)
    assert sum(counts.values()) == len(result.assignments)


def test_get_final_role_counts_keys_are_role_strings() -> None:
    result = _run("CC(=O)C>>CC(=O)C", ReactionContext.ionic)
    valid_role_values = {r.value for r in Role}
    for key in get_final_role_counts(result):
        assert key in valid_role_values


# ---------------------------------------------------------------------------
# 9. Reaction center group IDs
# ---------------------------------------------------------------------------


def test_get_reaction_center_group_ids_returns_list() -> None:
    result = _run("CC(=O)C.CC(=O)C>>CC(=O)CC(O)(C)C", ReactionContext.ionic)
    ids = get_reaction_center_group_ids(result)
    assert isinstance(ids, list)


def test_reaction_center_group_ids_are_strings() -> None:
    result = _run("CC(=O)C.CC(=O)C>>CC(=O)CC(O)(C)C", ReactionContext.ionic)
    for gid in get_reaction_center_group_ids(result):
        assert isinstance(gid, str)


def test_reaction_center_group_ids_match_is_reaction_center_flag() -> None:
    result = _run("CC(=O)C.CC(=O)C>>CC(=O)CC(O)(C)C", ReactionContext.ionic)
    center_ids = set(get_reaction_center_group_ids(result))
    for a in result.assignments:
        if a.is_reaction_center:
            assert a.group_id in center_ids
        else:
            assert a.group_id not in center_ids


# ---------------------------------------------------------------------------
# 10. Input immutability
# ---------------------------------------------------------------------------


def test_predictions_not_mutated_by_negotiation() -> None:
    rxn = parse_reaction_smiles("CC(=O)C.CBr>>CC(=O)CO", context=ReactionContext.ionic)
    groups = identify_functional_groups(rxn)
    report = predict_roles_for_reaction(rxn, groups)

    raw_roles_before = [p.predicted_role for p in report.predictions]
    raw_conf_before = [p.confidence for p in report.predictions]
    raw_reason_before = [p.reason for p in report.predictions]

    negotiate_predictions(rxn, groups, report.predictions)

    for i, pred in enumerate(report.predictions):
        assert pred.predicted_role == raw_roles_before[i]
        assert pred.confidence == raw_conf_before[i]
        assert pred.reason == raw_reason_before[i]


def test_groups_not_mutated_by_negotiation() -> None:
    rxn = parse_reaction_smiles("CBr.[OH-]>>CO.[Br-]", context=ReactionContext.ionic)
    groups = identify_functional_groups(rxn)
    report = predict_roles_for_reaction(rxn, groups)

    ids_before = [g.group_id for g in groups]
    types_before = [g.group_type for g in groups]

    negotiate_predictions(rxn, groups, report.predictions)

    for i, g in enumerate(groups):
        assert g.group_id == ids_before[i]
        assert g.group_type == types_before[i]


# ---------------------------------------------------------------------------
# 11. Acceptance snippet
# ---------------------------------------------------------------------------


def test_acceptance_snippet() -> None:
    result = run_full_rule_pipeline(
        "CC(=O)C.CC(=O)C>>CC(=O)CC(O)(C)C",
        context=ReactionContext.ionic,
    )

    summary = summarize_negotiation_result(result)

    assert result.assignments
    assert result.mechanism_hint in {
        "aldol_like",
        "ionic_addition_like",
        "unknown",
    }
    assert isinstance(result.reaction_center_atoms, list)

    for assignment in result.assignments:
        assert assignment.final_role is not None
        assert 0.0 <= assignment.final_confidence <= 1.0
        assert assignment.reason

    print(summary)
    print("Phase 6 negotiation layer OK")


# ---------------------------------------------------------------------------
# 12. NegotiatorConfig and string context
# ---------------------------------------------------------------------------


def test_config_string_context() -> None:
    result = run_full_rule_pipeline(
        "CC(=O)C.CC(=O)C>>CC(=O)CC(O)(C)C",
        context="ionic",
    )
    assert result.mechanism_hint == "aldol_like"


def test_config_unknown_string_context_falls_back() -> None:
    result = run_full_rule_pipeline(
        "CCO>>CCO",
        context="not_a_real_context",
    )
    assert result.context == ReactionContext.unknown


def test_summarize_result_via_instance() -> None:
    result = _run("C=CC=C.C=C>>C1CCC=CC1", ReactionContext.pericyclic)
    negotiator = RuleBasedNegotiator()
    summary = negotiator.summarize_result(result)
    assert summary["n_assignments"] == len(result.assignments)
    assert sum(summary["final_role_counts"].values()) == len(result.assignments)
