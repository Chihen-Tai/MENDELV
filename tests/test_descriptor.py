"""Phase 3 descriptor tests — all 9 required cases."""

from __future__ import annotations

import math

import pytest

from mendel.descriptor import (
    FEATURE_SCHEMA_VERSION,
    GroupDescriptor,
    build_descriptors,
    descriptor_matrix,
    get_feature_names,
    summarize_descriptors,
    validate_descriptor_schema,
)
from mendel.identifier import get_group_summary, identify_functional_groups
from mendel.parser import parse_reaction_smiles
from mendel.types import FunctionalGroupType, ReactionContext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rxn(reactant: str, context: str = "ionic"):
    return parse_reaction_smiles(f"{reactant}>>C", context=context)


def _build(reactant: str, context: str = "ionic"):
    rxn = _rxn(reactant, context)
    groups = identify_functional_groups(rxn)
    return rxn, groups, build_descriptors(rxn, groups)


def _feature(desc: GroupDescriptor, name: str) -> float:
    return desc.values[desc.feature_names.index(name)]


def _desc_for(descs: list[GroupDescriptor], gt: FunctionalGroupType) -> GroupDescriptor | None:
    return next((d for d in descs if d.group_type == gt), None)


# ---------------------------------------------------------------------------
# 1. Feature schema is deterministic
# ---------------------------------------------------------------------------


def test_feature_names_deterministic() -> None:
    assert get_feature_names() == get_feature_names()


def test_feature_names_no_duplicates() -> None:
    names = get_feature_names()
    assert len(names) == len(set(names))


def test_feature_names_nonempty() -> None:
    assert len(get_feature_names()) > 0


# ---------------------------------------------------------------------------
# 2. Build descriptors for SN2
# ---------------------------------------------------------------------------


def test_sn2_descriptors_exist() -> None:
    rxn = parse_reaction_smiles("CBr.[OH-]>>CO.[Br-]", context=ReactionContext.ionic)
    groups = identify_functional_groups(rxn)
    assert len(build_descriptors(rxn, groups)) >= 1


def test_sn2_all_same_feature_names() -> None:
    rxn = parse_reaction_smiles("CBr.[OH-]>>CO.[Br-]", context=ReactionContext.ionic)
    groups = identify_functional_groups(rxn)
    expected = get_feature_names()
    for d in build_descriptors(rxn, groups):
        assert d.feature_names == expected


def test_sn2_schema_valid() -> None:
    rxn = parse_reaction_smiles("CBr.[OH-]>>CO.[Br-]", context=ReactionContext.ionic)
    groups = identify_functional_groups(rxn)
    assert validate_descriptor_schema(build_descriptors(rxn, groups))


def test_sn2_context_ionic() -> None:
    rxn = parse_reaction_smiles("CBr.[OH-]>>CO.[Br-]", context=ReactionContext.ionic)
    groups = identify_functional_groups(rxn)
    for d in build_descriptors(rxn, groups):
        assert _feature(d, "context_ionic") == 1.0


def test_sn2_context_radical_zero() -> None:
    rxn = parse_reaction_smiles("CBr.[OH-]>>CO.[Br-]", context=ReactionContext.ionic)
    groups = identify_functional_groups(rxn)
    for d in build_descriptors(rxn, groups):
        assert _feature(d, "context_radical") == 0.0


# ---------------------------------------------------------------------------
# 3. Halide leaving group score > 0.5
# ---------------------------------------------------------------------------


def test_halide_leaving_group_score() -> None:
    _, _, descs = _build("CBr")
    d = _desc_for(descs, FunctionalGroupType.halide)
    assert d is not None
    assert _feature(d, "leaving_group_score") > 0.5


def test_iodo_leaving_group_gt_bromo() -> None:
    _, _, descs_br = _build("CBr")
    _, _, descs_i = _build("CI")
    br_lg = _feature(_desc_for(descs_br, FunctionalGroupType.halide), "leaving_group_score")
    i_lg = _feature(_desc_for(descs_i, FunctionalGroupType.halide), "leaving_group_score")
    assert i_lg > br_lg


# ---------------------------------------------------------------------------
# 4. Carbonyl electrophilicity > 0.5
# ---------------------------------------------------------------------------


def test_carbonyl_electrophilicity() -> None:
    _, _, descs = _build("CC(=O)C")
    d = _desc_for(descs, FunctionalGroupType.carbonyl)
    assert d is not None
    assert _feature(d, "electrophilicity_score") > 0.5


# ---------------------------------------------------------------------------
# 5. Alpha carbon acidity > 0.4
# ---------------------------------------------------------------------------


def test_alpha_carbon_exists_in_ketone() -> None:
    _, _, descs = _build("CC(=O)C")
    assert _desc_for(descs, FunctionalGroupType.alpha_carbon) is not None


def test_alpha_carbon_acidity() -> None:
    _, _, descs = _build("CC(=O)C")
    d = _desc_for(descs, FunctionalGroupType.alpha_carbon)
    assert _feature(d, "acidity_score") > 0.4


# ---------------------------------------------------------------------------
# 6. Phenol acidity > alcohol acidity
# ---------------------------------------------------------------------------


def test_phenol_acidity_gt_alcohol() -> None:
    _, _, phenol_descs = _build("c1ccccc1O")
    _, _, alcohol_descs = _build("CCO")
    pd = _desc_for(phenol_descs, FunctionalGroupType.phenol)
    ad = _desc_for(alcohol_descs, FunctionalGroupType.alcohol)
    assert pd is not None and ad is not None
    assert _feature(pd, "acidity_score") > _feature(ad, "acidity_score")


# ---------------------------------------------------------------------------
# 7. Benzylic radical stability > 0.5
# ---------------------------------------------------------------------------


def test_benzylic_site_exists() -> None:
    _, _, descs = _build("Cc1ccccc1")
    assert _desc_for(descs, FunctionalGroupType.benzylic_site) is not None


def test_benzylic_radical_stability() -> None:
    _, _, descs = _build("Cc1ccccc1")
    d = _desc_for(descs, FunctionalGroupType.benzylic_site)
    assert _feature(d, "radical_stability_score") > 0.5


# ---------------------------------------------------------------------------
# 8. Descriptor matrix shape
# ---------------------------------------------------------------------------


def test_descriptor_matrix_shape() -> None:
    _, groups, descs = _build("CC(=O)C.CBr")
    names, matrix = descriptor_matrix(descs)
    assert len(matrix) == len(descs)
    assert all(len(row) == len(names) for row in matrix)
    assert names == get_feature_names()


def test_descriptor_matrix_empty() -> None:
    names, matrix = descriptor_matrix([])
    assert matrix == []
    assert names == get_feature_names()


# ---------------------------------------------------------------------------
# 9. No NaN
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("smiles", ["C=C", "C#N", "C[N+](=O)[O-]", "CC(=O)OC"])
def test_no_nan_or_inf(smiles: str) -> None:
    _, _, descs = _build(smiles)
    for d in descs:
        for v in d.values:
            assert not math.isnan(v), f"NaN in {d.group_id}"
            assert not math.isinf(v), f"Inf in {d.group_id}"


# ---------------------------------------------------------------------------
# Additional correctness checks
# ---------------------------------------------------------------------------


def test_feature_vector_length_matches_schema() -> None:
    _, _, descs = _build("CC(=O)C")
    n = len(get_feature_names())
    for d in descs:
        assert len(d.values) == n


def test_one_hot_sums_to_one() -> None:
    _, _, descs = _build("CCO")
    for d in descs:
        oh_names = [n for n in d.feature_names if n.startswith("is_")]
        assert sum(_feature(d, n) for n in oh_names) == 1.0


def test_in_reactant_flag() -> None:
    rxn = parse_reaction_smiles("CBr.[OH-]>>CO.[Br-]", context=ReactionContext.ionic)
    groups = identify_functional_groups(rxn)
    for d in build_descriptors(rxn, groups):
        assert _feature(d, "in_reactant") == 1.0
        assert _feature(d, "in_product") == 0.0


def test_context_radical_reaction() -> None:
    rxn = parse_reaction_smiles("C.BrBr>>CBr.Br", context=ReactionContext.radical)
    groups = identify_functional_groups(rxn)
    for d in build_descriptors(rxn, groups):
        assert _feature(d, "context_radical") == 1.0
        assert _feature(d, "context_ionic") == 0.0


def test_summarize_descriptors() -> None:
    _, _, descs = _build("CC(=O)C")
    s = summarize_descriptors(descs)
    for key in ("n_descriptors", "n_features", "group_ids", "group_types", "schema_version"):
        assert key in s
    assert s["schema_version"] == FEATURE_SCHEMA_VERSION


def test_to_dict_structure() -> None:
    _, _, descs = _build("CBr")
    d = descs[0].to_dict()
    assert "features" in d and "group_id" in d and "group_type" in d


def test_acceptance_snippet() -> None:
    rxn = parse_reaction_smiles("CC(=O)C.CBr>>CC(=O)CO", context=ReactionContext.ionic)
    groups = identify_functional_groups(rxn)
    descs = build_descriptors(rxn, groups)
    assert validate_descriptor_schema(descs)
    names, matrix = descriptor_matrix(descs)
    summary = summarize_descriptors(descs)
    assert len(matrix) == len(groups)
    assert len(names) == len(matrix[0])
    print(get_group_summary(groups))
    print(summary)
    print("Phase 3 descriptor builder OK")
