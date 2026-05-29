"""Tests for Phase 8.9 MLP-aware negotiation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mendel.negotiator import (
    NegotiatorConfig,
    negotiate_predictions,
    run_full_rule_pipeline,
)
from mendel.parser import parse_reaction_smiles
from mendel.predictor import RolePrediction
from mendel.types import AtomRef, FunctionalGroup, FunctionalGroupType, ReactionContext, Role


def _group(
    group_id: str,
    group_type: FunctionalGroupType,
    atom_indices: list[int],
    molecule_index: int = 0,
) -> FunctionalGroup:
    return FunctionalGroup(
        group_id=group_id,
        group_type=group_type,
        atom_refs=[
            AtomRef(molecule_index=molecule_index, atom_index=idx, atom_map_num=idx + 1)
            for idx in atom_indices
        ],
    )


def _pred(
    group: FunctionalGroup,
    role: Role,
    confidence: float,
) -> RolePrediction:
    return RolePrediction(
        group_id=group.group_id,
        group_type=group.group_type,
        predicted_role=role,
        confidence=confidence,
        reason="test MLP prediction",
        metadata={"prediction_source": "mlp", "predictor_name": "test_mlp"},
    )


def _run_mlp_aware(
    mechanism_type: str,
    groups: list[FunctionalGroup],
    predictions: list[RolePrediction],
    context: ReactionContext = ReactionContext.ionic,
):
    parsed = parse_reaction_smiles("CBr.[OH-]>>CO.[Br-]", context=context)
    parsed.metadata["mechanism_type"] = mechanism_type
    return negotiate_predictions(
        parsed,
        groups,
        predictions,
        NegotiatorConfig(mode="mlp_aware"),
    )


def _center_group_ids(result) -> set[str]:
    return {
        assignment.group_id
        for assignment in result.assignments
        if assignment.is_reaction_center
    }


def test_control_high_confidence_spectators_return_empty_center() -> None:
    group = _group("mol0_aromatic_0", FunctionalGroupType.aromatic, [0, 1, 2])

    result = _run_mlp_aware("control", [group], [_pred(group, Role.spectator, 0.95)])

    assert result.reaction_center_atoms == []
    assert result.mechanism_hint == "control_like"


def test_control_low_confidence_reactive_with_spectator_dominance_warns_and_suppresses() -> None:
    spectator = _group("mol0_aromatic_0", FunctionalGroupType.aromatic, [0, 1, 2])
    reactive = _group("mol0_halide_0", FunctionalGroupType.halide, [3])

    result = _run_mlp_aware(
        "control",
        [spectator, reactive],
        [_pred(spectator, Role.spectator, 0.95), _pred(reactive, Role.leaving_group, 0.40)],
    )

    assert result.reaction_center_atoms == []
    assert any(w.code == "control_reactive_prediction_suppressed" for w in result.warnings)


def test_carbonyl_addition_includes_carbonyl_excludes_alpha_spectator() -> None:
    carbonyl = _group("mol0_carbonyl_0", FunctionalGroupType.carbonyl, [1, 2])
    alpha = _group("mol0_alpha_carbon_0", FunctionalGroupType.alpha_carbon, [0])

    result = _run_mlp_aware(
        "carbonyl_addition",
        [carbonyl, alpha],
        [
            _pred(carbonyl, Role.reactive_electrophile, 0.90),
            _pred(alpha, Role.spectator, 0.90),
        ],
    )

    assert "mol0_carbonyl_0" in _center_group_ids(result)
    assert "mol0_alpha_carbon_0" not in _center_group_ids(result)


def test_diels_alder_includes_alkenes_excludes_substituent_spectators() -> None:
    diene_a = _group("mol0_alkene_0", FunctionalGroupType.alkene, [0, 1], 0)
    diene_b = _group("mol0_alkene_1", FunctionalGroupType.alkene, [2, 3], 0)
    dienophile = _group("mol1_alkene_0", FunctionalGroupType.alkene, [0, 1], 1)
    nitrile = _group("mol1_nitrile_0", FunctionalGroupType.nitrile, [2, 3], 1)

    result = _run_mlp_aware(
        "diels_alder",
        [diene_a, diene_b, dienophile, nitrile],
        [
            _pred(diene_a, Role.reactive_nucleophile, 0.80),
            _pred(diene_b, Role.reactive_nucleophile, 0.80),
            _pred(dienophile, Role.reactive_electrophile, 0.80),
            _pred(nitrile, Role.spectator, 0.90),
        ],
        context=ReactionContext.pericyclic,
    )

    center_ids = _center_group_ids(result)
    assert {"mol0_alkene_0", "mol0_alkene_1", "mol1_alkene_0"} <= center_ids
    assert "mol1_nitrile_0" not in center_ids


def test_sn2_halide_includes_attached_carbon_when_available() -> None:
    halide = _group("mol0_halide_0", FunctionalGroupType.halide, [0, 1])

    result = _run_mlp_aware("sn2", [halide], [_pred(halide, Role.leaving_group, 0.90)])

    atoms = {(ref.atom_index, ref.atom_map_num) for ref in result.reaction_center_atoms}
    assert (0, 1) in atoms
    assert (1, 2) in atoms


def test_e2_includes_lg_and_attached_carbon_and_warns_for_missing_beta() -> None:
    halide = _group("mol0_halide_0", FunctionalGroupType.halide, [0, 1])

    result = _run_mlp_aware("e2", [halide], [_pred(halide, Role.leaving_group, 0.90)])

    assert len(result.reaction_center_atoms) == 2
    assert any(w.code == "beta_center_not_fully_represented" for w in result.warnings)


def test_radical_bromination_includes_benzylic_excludes_aromatic() -> None:
    aromatic = _group("mol0_aromatic_0", FunctionalGroupType.aromatic, [0, 1, 2])
    benzylic = _group("mol0_benzylic_site_0", FunctionalGroupType.benzylic_site, [3])

    result = _run_mlp_aware(
        "benzylic_radical_bromination",
        [aromatic, benzylic],
        [_pred(aromatic, Role.spectator, 0.90), _pred(benzylic, Role.reactive_radical, 0.90)],
        context=ReactionContext.radical,
    )

    assert "mol0_benzylic_site_0" in _center_group_ids(result)
    assert "mol0_aromatic_0" not in _center_group_ids(result)


def test_low_confidence_reactive_does_not_dominate_center_selection() -> None:
    group = _group("mol0_carbonyl_0", FunctionalGroupType.carbonyl, [1, 2])

    result = _run_mlp_aware(
        "carbonyl_addition",
        [group],
        [_pred(group, Role.reactive_electrophile, 0.30)],
    )

    assert result.reaction_center_atoms == []


def test_default_negotiation_backward_compatibility_simple_sn2() -> None:
    before = run_full_rule_pipeline("CBr.[OH-]>>CO.[Br-]", context=ReactionContext.ionic)
    after = run_full_rule_pipeline("CBr.[OH-]>>CO.[Br-]", context=ReactionContext.ionic)

    assert [a.to_dict() for a in before.assignments] == [a.to_dict() for a in after.assignments]
    assert [r.to_dict() for r in before.reaction_center_atoms] == [
        r.to_dict() for r in after.reaction_center_atoms
    ]


_CHECKPOINT = Path(__file__).parent.parent / "models" / "role_mlp.pt"


@pytest.mark.skipif(not _CHECKPOINT.exists(), reason="role_mlp.pt not yet trained")
def test_run_pipeline_with_mlp_sn2() -> None:
    from mendel.negotiator import run_pipeline_with_mlp

    result = run_pipeline_with_mlp(
        "CBr.[OH-]>>CO.[Br-]",
        mlp_checkpoint=_CHECKPOINT,
        context="ionic",
    )

    group_ids = {a.group_id for a in result.assignments}
    roles = {a.group_id: a.final_role.value for a in result.assignments}
    assert len(result.assignments) > 0
    assert any("halide" in gid or "leaving" in gid for gid in group_ids)
    assert "leaving_group" in roles.values() or "reactive_nucleophile" in roles.values()


def test_mlp_aware_serialization_preserves_provenance() -> None:
    group = _group("mol0_halide_0", FunctionalGroupType.halide, [0, 1])

    result = _run_mlp_aware("sn2", [group], [_pred(group, Role.leaving_group, 0.90)])
    payload = result.to_dict()

    assert json.dumps(payload)
    metadata = payload["assignments"][0]["metadata"]
    assert metadata["prediction_source"] == "mlp"
    assert metadata["center_selection_reason"]
