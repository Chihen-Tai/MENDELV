"""Phase 5 rule-based predictor tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from mendel.descriptor import build_descriptors, get_feature_names
from mendel.identifier import identify_functional_groups
from mendel.labels import load_labeled_reactions
from mendel.parser import parse_reaction_smiles
from mendel.predictor import (
    PredictionReport,
    RolePrediction,
    RuleBasedPredictorConfig,
    RuleBasedRolePredictor,
    compare_predictions_to_labels,
    get_feature_value,
    predict_roles,
    predict_roles_for_reaction,
    summarize_predictions,
)
from mendel.types import FunctionalGroupType, ReactionContext, Role

_MINIMAL = Path(__file__).parent.parent / "data" / "reactions.minimal.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _predict(smiles: str, context: ReactionContext = ReactionContext.ionic):
    rxn = parse_reaction_smiles(smiles, context=context)
    groups = identify_functional_groups(rxn)
    return rxn, groups, predict_roles_for_reaction(rxn, groups)


def _find_pred(report: PredictionReport, gt: FunctionalGroupType) -> RolePrediction | None:
    return next((p for p in report.predictions if p.group_type == gt), None)


# ---------------------------------------------------------------------------
# 1. get_feature_value
# ---------------------------------------------------------------------------


def test_get_feature_value_existing() -> None:
    rxn = parse_reaction_smiles("CBr>>CBr", context=ReactionContext.ionic)
    groups = identify_functional_groups(rxn)
    descs = build_descriptors(rxn, groups)
    assert len(descs) >= 1
    d = descs[0]
    name = get_feature_names()[0]
    assert get_feature_value(d, name) == d.values[0]


def test_get_feature_value_missing_returns_default() -> None:
    rxn = parse_reaction_smiles("CBr>>CBr", context=ReactionContext.ionic)
    groups = identify_functional_groups(rxn)
    descs = build_descriptors(rxn, groups)
    d = descs[0]
    assert get_feature_value(d, "__nonexistent__") == 0.0
    assert get_feature_value(d, "__nonexistent__", default=99.0) == 99.0


def test_get_feature_value_never_raises() -> None:
    rxn = parse_reaction_smiles("CC>>CC", context=ReactionContext.ionic)
    groups = identify_functional_groups(rxn)
    descs = build_descriptors(rxn, groups)
    if descs:
        get_feature_value(descs[0], "")  # must not raise


# ---------------------------------------------------------------------------
# 2. SN2-like prediction
# ---------------------------------------------------------------------------


def test_sn2_at_least_one_prediction() -> None:
    _, _, report = _predict("CBr.[OH-]>>CO.[Br-]")
    assert len(report.predictions) >= 1


def test_sn2_halide_role() -> None:
    _, _, report = _predict("CBr.[OH-]>>CO.[Br-]")
    p = _find_pred(report, FunctionalGroupType.halide)
    assert p is not None
    assert p.predicted_role in (Role.leaving_group, Role.reactive_electrophile)


def test_sn2_reason_nonempty() -> None:
    _, _, report = _predict("CBr.[OH-]>>CO.[Br-]")
    for pred in report.predictions:
        assert pred.reason != ""


def test_sn2_confidence_range() -> None:
    _, _, report = _predict("CBr.[OH-]>>CO.[Br-]")
    for pred in report.predictions:
        assert 0.0 <= pred.confidence <= 1.0


# ---------------------------------------------------------------------------
# 3. Carbonyl electrophile
# ---------------------------------------------------------------------------


def test_carbonyl_electrophile() -> None:
    _, _, report = _predict("CC(=O)C>>CC(=O)C")
    p = _find_pred(report, FunctionalGroupType.carbonyl)
    assert p is not None
    assert p.predicted_role == Role.reactive_electrophile


# ---------------------------------------------------------------------------
# 4. Alpha carbon nucleophile
# ---------------------------------------------------------------------------


def test_alpha_carbon_nucleophile() -> None:
    _, _, report = _predict("CC(=O)C>>CC(=O)C")
    p = _find_pred(report, FunctionalGroupType.alpha_carbon)
    assert p is not None
    assert p.predicted_role == Role.reactive_nucleophile


def test_alpha_carbon_reason_mentions_alpha() -> None:
    _, _, report = _predict("CC(=O)C>>CC(=O)C")
    p = _find_pred(report, FunctionalGroupType.alpha_carbon)
    assert p is not None
    reason_lower = p.reason.lower()
    assert "alpha" in reason_lower or "deprotonation" in reason_lower or "acidity" in reason_lower


# ---------------------------------------------------------------------------
# 5. Phenol / alcohol — no crash, valid role
# ---------------------------------------------------------------------------


def test_phenol_no_crash() -> None:
    _, _, report = _predict("c1ccccc1O>>c1ccccc1O")
    assert isinstance(report, PredictionReport)


def test_phenol_valid_roles() -> None:
    _, _, report = _predict("c1ccccc1O>>c1ccccc1O")
    valid_roles = set(Role)
    for pred in report.predictions:
        assert pred.predicted_role in valid_roles


def test_alcohol_no_crash() -> None:
    _, _, report = _predict("CCO>>CCO")
    assert isinstance(report, PredictionReport)


# ---------------------------------------------------------------------------
# 6. Pericyclic flat-role handling
# ---------------------------------------------------------------------------


def test_pericyclic_alkene_role() -> None:
    _, _, report = _predict(
        "C=CC=C.C=C>>C1CCC=CC1",
        context=ReactionContext.pericyclic,
    )
    p = _find_pred(report, FunctionalGroupType.alkene)
    assert p is not None
    assert p.predicted_role in (Role.reactive_nucleophile, Role.reactive_electrophile)


def test_pericyclic_reason_mentions_context() -> None:
    _, _, report = _predict(
        "C=CC=C.C=C>>C1CCC=CC1",
        context=ReactionContext.pericyclic,
    )
    p = _find_pred(report, FunctionalGroupType.alkene)
    assert p is not None
    reason_lower = p.reason.lower()
    assert "pericyclic" in reason_lower or "flat" in reason_lower or "taxonomy" in reason_lower


# ---------------------------------------------------------------------------
# 7. Radical context — benzylic site
# ---------------------------------------------------------------------------


def test_benzylic_radical_role() -> None:
    _, _, report = _predict("Cc1ccccc1>>Cc1ccccc1", context=ReactionContext.radical)
    p = _find_pred(report, FunctionalGroupType.benzylic_site)
    if p is None:
        pytest.skip("benzylic_site not detected")
    assert p.predicted_role == Role.reactive_radical


def test_benzylic_radical_confidence() -> None:
    _, _, report = _predict("Cc1ccccc1>>Cc1ccccc1", context=ReactionContext.radical)
    p = _find_pred(report, FunctionalGroupType.benzylic_site)
    if p is None:
        pytest.skip("benzylic_site not detected")
    assert p.confidence > 0.5


# ---------------------------------------------------------------------------
# 8. summarize_predictions
# ---------------------------------------------------------------------------


def test_summarize_required_keys() -> None:
    _, _, report = _predict("CC(=O)C>>CC(=O)C")
    s = summarize_predictions(report.predictions)
    for key in ("n_predictions", "role_counts", "average_confidence",
                 "group_type_counts", "low_confidence_group_ids"):
        assert key in s


def test_summarize_counts_consistent() -> None:
    _, _, report = _predict("CC(=O)C>>CC(=O)C")
    s = summarize_predictions(report.predictions)
    assert s["n_predictions"] == len(report.predictions)
    assert sum(s["role_counts"].values()) == len(report.predictions)


def test_summarize_empty() -> None:
    s = summarize_predictions([])
    assert s["n_predictions"] == 0
    assert s["average_confidence"] == 0.0


# ---------------------------------------------------------------------------
# 9. PredictionReport
# ---------------------------------------------------------------------------


def test_prediction_report_type() -> None:
    rxn = parse_reaction_smiles("CC(=O)C.CBr>>CC(=O)CO", context=ReactionContext.ionic)
    groups = identify_functional_groups(rxn)
    report = predict_roles_for_reaction(rxn, groups)
    assert isinstance(report, PredictionReport)


def test_prediction_report_length_matches_groups() -> None:
    rxn = parse_reaction_smiles("CC(=O)C.CBr>>CC(=O)CO", context=ReactionContext.ionic)
    groups = identify_functional_groups(rxn)
    report = predict_roles_for_reaction(rxn, groups)
    assert len(report.predictions) == len(groups)


def test_prediction_report_to_dict() -> None:
    _, _, report = _predict("CBr>>CBr")
    d = report.to_dict()
    assert "reaction_smiles" in d
    assert "context" in d
    assert "predictions" in d


# ---------------------------------------------------------------------------
# 10. compare_predictions_to_labels
# ---------------------------------------------------------------------------


def test_compare_required_keys() -> None:
    rxns = load_labeled_reactions(_MINIMAL)
    labeled = next(r for r in rxns if r.reaction_id == "minimal_sn2")
    rxn = parse_reaction_smiles(labeled.reaction_smiles, context=labeled.context)
    groups = identify_functional_groups(rxn)
    report = predict_roles_for_reaction(rxn, groups)
    result = compare_predictions_to_labels(report.predictions, labeled)
    for key in ("n_labeled", "n_matched", "accuracy", "mismatches", "n_correct"):
        assert key in result


def test_compare_accuracy_in_range() -> None:
    rxns = load_labeled_reactions(_MINIMAL)
    labeled = next(r for r in rxns if r.reaction_id == "minimal_sn2")
    rxn = parse_reaction_smiles(labeled.reaction_smiles, context=labeled.context)
    groups = identify_functional_groups(rxn)
    report = predict_roles_for_reaction(rxn, groups)
    result = compare_predictions_to_labels(report.predictions, labeled)
    assert 0.0 <= result["accuracy"] <= 1.0


def test_compare_empty_predictions() -> None:
    rxns = load_labeled_reactions(_MINIMAL)
    result = compare_predictions_to_labels([], rxns[0])
    assert result["n_matched"] == 0
    assert result["accuracy"] == 0.0


# ---------------------------------------------------------------------------
# 11. No invalid roles
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("smiles,ctx", [
    ("CBr.[OH-]>>CO.[Br-]", ReactionContext.ionic),
    ("CC(=O)C>>CC(=O)C", ReactionContext.ionic),
    ("C=CC=C.C=C>>C1CCC=CC1", ReactionContext.pericyclic),
    ("Cc1ccccc1>>Cc1ccccc1", ReactionContext.radical),
    ("c1ccccc1O>>c1ccccc1O", ReactionContext.ionic),
])
def test_all_roles_valid_enum(smiles: str, ctx: ReactionContext) -> None:
    _, _, report = _predict(smiles, ctx)
    for pred in report.predictions:
        assert isinstance(pred.predicted_role, Role)
        assert pred.predicted_role is not None


@pytest.mark.parametrize("smiles,ctx", [
    ("CBr.[OH-]>>CO.[Br-]", ReactionContext.ionic),
    ("CC(=O)C>>CC(=O)C", ReactionContext.ionic),
    ("C=CC=C.C=C>>C1CCC=CC1", ReactionContext.pericyclic),
    ("Cc1ccccc1>>Cc1ccccc1", ReactionContext.radical),
])
def test_confidence_always_in_range(smiles: str, ctx: ReactionContext) -> None:
    _, _, report = _predict(smiles, ctx)
    for pred in report.predictions:
        assert 0.0 <= pred.confidence <= 1.0


# ---------------------------------------------------------------------------
# 12. Acceptance snippet
# ---------------------------------------------------------------------------


def test_acceptance_snippet() -> None:
    rxn = parse_reaction_smiles(
        "CC(=O)C.CBr>>CC(=O)CO",
        context=ReactionContext.ionic,
    )
    groups = identify_functional_groups(rxn)
    report = predict_roles_for_reaction(rxn, groups)
    summary = summarize_predictions(report.predictions)

    assert len(report.predictions) == len(groups)
    assert summary["n_predictions"] == len(groups)

    for pred in report.predictions:
        assert pred.predicted_role is not None
        assert 0.0 <= pred.confidence <= 1.0
        assert pred.reason

    print(summary)
    print("Phase 5 rule-based predictor OK")
