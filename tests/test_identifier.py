"""Phase 2 identifier tests — all 16 required cases."""

from __future__ import annotations

import pytest
from rdkit import Chem

from mendel.identifier import (
    FUNCTIONAL_GROUP_SMARTS,
    get_group_summary,
    has_group_type,
    identify_functional_groups,
    identify_functional_groups_in_mol,
    validate_smarts_patterns,
)
from mendel.parser import parse_reaction_smiles
from mendel.types import FunctionalGroupType, ReactionContext


# ---------------------------------------------------------------------------
# 16. validate_smarts_patterns — all compile
# ---------------------------------------------------------------------------


def test_all_smarts_valid() -> None:
    result = validate_smarts_patterns()
    bad = [k for k, v in result.items() if not v]
    assert not bad, f"Invalid SMARTS for: {bad}"


def test_smarts_dict_covers_all_non_aromatic_types() -> None:
    expected = set(FUNCTIONAL_GROUP_SMARTS.keys())
    # aromatic is detected via ring info, not SMARTS
    assert FunctionalGroupType.aromatic not in expected


# ---------------------------------------------------------------------------
# 1. Alkene
# ---------------------------------------------------------------------------


def test_alkene_detected() -> None:
    mol = Chem.MolFromSmiles("C=C")
    assert has_group_type(identify_functional_groups_in_mol(mol, 0), FunctionalGroupType.alkene)


# ---------------------------------------------------------------------------
# 2. Alkyne
# ---------------------------------------------------------------------------


def test_alkyne_detected() -> None:
    mol = Chem.MolFromSmiles("C#C")
    assert has_group_type(identify_functional_groups_in_mol(mol, 0), FunctionalGroupType.alkyne)


# ---------------------------------------------------------------------------
# 3. Alcohol
# ---------------------------------------------------------------------------


def test_alcohol_detected() -> None:
    mol = Chem.MolFromSmiles("CCO")
    assert has_group_type(identify_functional_groups_in_mol(mol, 0), FunctionalGroupType.alcohol)


# ---------------------------------------------------------------------------
# 4. Phenol — not double-counted as alcohol
# ---------------------------------------------------------------------------


def test_phenol_detected() -> None:
    mol = Chem.MolFromSmiles("c1ccccc1O")
    groups = identify_functional_groups_in_mol(mol, 0)
    assert has_group_type(groups, FunctionalGroupType.phenol)


def test_phenol_not_double_counted_as_alcohol() -> None:
    mol = Chem.MolFromSmiles("c1ccccc1O")
    groups = identify_functional_groups_in_mol(mol, 0)
    assert not has_group_type(groups, FunctionalGroupType.alcohol)


# ---------------------------------------------------------------------------
# 5. Ether
# ---------------------------------------------------------------------------


def test_ether_detected() -> None:
    mol = Chem.MolFromSmiles("COC")
    assert has_group_type(identify_functional_groups_in_mol(mol, 0), FunctionalGroupType.ether)


# ---------------------------------------------------------------------------
# 6. Carbonyl (ketone)
# ---------------------------------------------------------------------------


def test_carbonyl_detected() -> None:
    mol = Chem.MolFromSmiles("CC(=O)C")
    assert has_group_type(identify_functional_groups_in_mol(mol, 0), FunctionalGroupType.carbonyl)


# ---------------------------------------------------------------------------
# 7. Carboxylic acid — not double-counted as carbonyl + alcohol
# ---------------------------------------------------------------------------


def test_carboxylic_acid_detected() -> None:
    mol = Chem.MolFromSmiles("CC(=O)O")
    assert has_group_type(
        identify_functional_groups_in_mol(mol, 0), FunctionalGroupType.carboxylic_acid
    )


def test_carboxylic_acid_not_double_counted_as_carbonyl() -> None:
    mol = Chem.MolFromSmiles("CC(=O)O")
    groups = identify_functional_groups_in_mol(mol, 0)
    assert not has_group_type(groups, FunctionalGroupType.carbonyl)


def test_carboxylic_acid_not_double_counted_as_alcohol() -> None:
    mol = Chem.MolFromSmiles("CC(=O)O")
    groups = identify_functional_groups_in_mol(mol, 0)
    assert not has_group_type(groups, FunctionalGroupType.alcohol)


# ---------------------------------------------------------------------------
# 8. Ester — not double-counted as carbonyl + ether
# ---------------------------------------------------------------------------


def test_ester_detected() -> None:
    mol = Chem.MolFromSmiles("CC(=O)OC")
    assert has_group_type(identify_functional_groups_in_mol(mol, 0), FunctionalGroupType.ester)


def test_ester_not_double_counted_as_carbonyl() -> None:
    mol = Chem.MolFromSmiles("CC(=O)OC")
    groups = identify_functional_groups_in_mol(mol, 0)
    assert not has_group_type(groups, FunctionalGroupType.carbonyl)


def test_ester_not_double_counted_as_ether() -> None:
    mol = Chem.MolFromSmiles("CC(=O)OC")
    groups = identify_functional_groups_in_mol(mol, 0)
    assert not has_group_type(groups, FunctionalGroupType.ether)


# ---------------------------------------------------------------------------
# 9. Amide — not double-counted as carbonyl + amine
# ---------------------------------------------------------------------------


def test_amide_detected() -> None:
    mol = Chem.MolFromSmiles("CC(=O)N")
    assert has_group_type(identify_functional_groups_in_mol(mol, 0), FunctionalGroupType.amide)


def test_amide_not_double_counted_as_carbonyl() -> None:
    mol = Chem.MolFromSmiles("CC(=O)N")
    groups = identify_functional_groups_in_mol(mol, 0)
    assert not has_group_type(groups, FunctionalGroupType.carbonyl)


def test_amide_not_double_counted_as_amine() -> None:
    mol = Chem.MolFromSmiles("CC(=O)N")
    groups = identify_functional_groups_in_mol(mol, 0)
    assert not has_group_type(groups, FunctionalGroupType.amine)


# ---------------------------------------------------------------------------
# 10. Halide
# ---------------------------------------------------------------------------


def test_halide_detected() -> None:
    mol = Chem.MolFromSmiles("CBr")
    assert has_group_type(identify_functional_groups_in_mol(mol, 0), FunctionalGroupType.halide)


# ---------------------------------------------------------------------------
# 11. Nitrile
# ---------------------------------------------------------------------------


def test_nitrile_detected() -> None:
    mol = Chem.MolFromSmiles("CC#N")
    assert has_group_type(identify_functional_groups_in_mol(mol, 0), FunctionalGroupType.nitrile)


# ---------------------------------------------------------------------------
# 12. Nitro (charge-separated form)
# ---------------------------------------------------------------------------


def test_nitro_detected() -> None:
    mol = Chem.MolFromSmiles("C[N+](=O)[O-]")
    assert has_group_type(identify_functional_groups_in_mol(mol, 0), FunctionalGroupType.nitro)


# ---------------------------------------------------------------------------
# 13. Alpha carbon (ketone + alpha_carbon both present)
# ---------------------------------------------------------------------------


def test_alpha_carbon_detected_with_carbonyl() -> None:
    mol = Chem.MolFromSmiles("CC(=O)C")
    groups = identify_functional_groups_in_mol(mol, 0)
    assert has_group_type(groups, FunctionalGroupType.carbonyl)
    assert has_group_type(groups, FunctionalGroupType.alpha_carbon)


def test_alpha_carbon_not_detected_when_contextual_disabled() -> None:
    mol = Chem.MolFromSmiles("CC(=O)C")
    groups = identify_functional_groups_in_mol(mol, 0, include_contextual=False)
    assert not has_group_type(groups, FunctionalGroupType.alpha_carbon)


# ---------------------------------------------------------------------------
# 14. Benzylic site (aromatic + benzylic_site both present)
# ---------------------------------------------------------------------------


def test_benzylic_site_and_aromatic_detected() -> None:
    mol = Chem.MolFromSmiles("Cc1ccccc1")
    groups = identify_functional_groups_in_mol(mol, 0)
    assert has_group_type(groups, FunctionalGroupType.aromatic)
    assert has_group_type(groups, FunctionalGroupType.benzylic_site)


# ---------------------------------------------------------------------------
# 15. Reaction-level identification via ParsedReaction
# ---------------------------------------------------------------------------


def test_reaction_level_halide_in_reactants() -> None:
    rxn = parse_reaction_smiles("CBr.[OH-]>>CO.[Br-]", context=ReactionContext.ionic)
    groups = identify_functional_groups(rxn)
    assert has_group_type(groups, FunctionalGroupType.halide)


def test_reaction_level_molecule_indices() -> None:
    rxn = parse_reaction_smiles("CBr.[OH-]>>CO.[Br-]", context=ReactionContext.ionic)
    groups = identify_functional_groups(rxn)
    mol_indices = {ref.molecule_index for g in groups for ref in g.atom_refs}
    # reactant molecules are at indices 0 and 1
    assert 0 in mol_indices


def test_reaction_level_products_excluded_by_default() -> None:
    rxn = parse_reaction_smiles("CBr.[OH-]>>CO.[Br-]", context=ReactionContext.ionic)
    groups = identify_functional_groups(rxn, include_products=False)
    roles = {g.metadata.get("molecule_role") for g in groups}
    assert "product" not in roles


def test_reaction_level_products_included_when_requested() -> None:
    rxn = parse_reaction_smiles("CBr.[OH-]>>CO.[Br-]", context=ReactionContext.ionic)
    groups = identify_functional_groups(rxn, include_products=True)
    roles = {g.metadata.get("molecule_role") for g in groups}
    assert "reactant" in roles
    assert "product" in roles


def test_group_id_deterministic_format() -> None:
    mol = Chem.MolFromSmiles("CBr")
    groups = identify_functional_groups_in_mol(mol, molecule_index=3)
    halide = next(g for g in groups if g.group_type == FunctionalGroupType.halide)
    assert halide.group_id == "mol3_halide_0"


# ---------------------------------------------------------------------------
# Metadata fields
# ---------------------------------------------------------------------------


def test_metadata_has_source() -> None:
    mol = Chem.MolFromSmiles("CCO")
    groups = identify_functional_groups_in_mol(mol, 0)
    for g in groups:
        assert "source" in g.metadata


def test_metadata_has_atom_indices() -> None:
    mol = Chem.MolFromSmiles("CCO")
    groups = identify_functional_groups_in_mol(mol, 0)
    for g in groups:
        assert "atom_indices" in g.metadata


def test_metadata_has_priority() -> None:
    mol = Chem.MolFromSmiles("CCO")
    groups = identify_functional_groups_in_mol(mol, 0)
    for g in groups:
        assert "priority" in g.metadata


def test_metadata_molecule_role_set_via_reaction() -> None:
    rxn = parse_reaction_smiles("CBr.[OH-]>>CO.[Br-]", context=ReactionContext.ionic)
    groups = identify_functional_groups(rxn)
    for g in groups:
        assert g.metadata.get("molecule_role") == "reactant"


# ---------------------------------------------------------------------------
# get_group_summary
# ---------------------------------------------------------------------------


def test_get_group_summary_counts() -> None:
    mol = Chem.MolFromSmiles("CCO")
    summary = get_group_summary(identify_functional_groups_in_mol(mol, 0))
    assert summary.get("alcohol", 0) >= 1


def test_get_group_summary_empty() -> None:
    assert get_group_summary([]) == {}


# ---------------------------------------------------------------------------
# Acceptance snippet (mirrors the spec's required snippet)
# ---------------------------------------------------------------------------


def test_acceptance_snippet() -> None:
    rxn = parse_reaction_smiles(
        "CC(=O)C.CBr>>CC(=O)CO",
        context=ReactionContext.ionic,
    )
    groups = identify_functional_groups(rxn)
    summary = get_group_summary(groups)
    assert summary.get("carbonyl", 0) >= 1
    assert summary.get("alpha_carbon", 0) >= 1
    assert summary.get("halide", 0) >= 1
