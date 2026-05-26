"""Phase 1 parser tests.

Covers parse_reaction_smiles, validate_reaction_smiles, get_reaction_summary,
parse_reaction_record, and ReactionParseError for all required cases.
"""

from __future__ import annotations

import pytest

from mendel.parser import (
    ReactionParseError,
    get_reaction_summary,
    parse_reaction_record,
    parse_reaction_smiles,
    validate_reaction_smiles,
)
from mendel.types import ReactionContext, ReactionRecord


# ---------------------------------------------------------------------------
# SN2
# ---------------------------------------------------------------------------


def test_sn2_reactant_product_count() -> None:
    rxn = parse_reaction_smiles("CBr.[OH-]>>CO.[Br-]", context=ReactionContext.ionic)
    assert len(rxn.reactants) == 2
    assert len(rxn.products) == 2


def test_sn2_context() -> None:
    rxn = parse_reaction_smiles("CBr.[OH-]>>CO.[Br-]", context=ReactionContext.ionic)
    assert rxn.context == ReactionContext.ionic


def test_sn2_charges() -> None:
    rxn = parse_reaction_smiles("CBr.[OH-]>>CO.[Br-]", context=ReactionContext.ionic)
    assert rxn.total_charge_reactants == -1
    assert rxn.total_charge_products == -1


def test_sn2_summary() -> None:
    rxn = parse_reaction_smiles("CBr.[OH-]>>CO.[Br-]", context=ReactionContext.ionic)
    s = get_reaction_summary(rxn)
    assert s["n_reactants"] == 2
    assert s["n_products"] == 2
    assert s["total_charge_reactants"] == -1
    assert s["total_charge_products"] == -1


def test_sn2_molecule_roles() -> None:
    rxn = parse_reaction_smiles("CBr.[OH-]>>CO.[Br-]", context=ReactionContext.ionic)
    assert all(m.role == "reactant" for m in rxn.reactants)
    assert all(m.role == "product" for m in rxn.products)


# ---------------------------------------------------------------------------
# Diels-Alder
# ---------------------------------------------------------------------------


def test_diels_alder_counts() -> None:
    rxn = parse_reaction_smiles("C=CC=C.C=C>>C1CCC=CC1", context=ReactionContext.pericyclic)
    assert len(rxn.reactants) == 2
    assert len(rxn.products) == 1


def test_diels_alder_context() -> None:
    rxn = parse_reaction_smiles("C=CC=C.C=C>>C1CCC=CC1", context=ReactionContext.pericyclic)
    assert rxn.context == ReactionContext.pericyclic


def test_diels_alder_neutral() -> None:
    rxn = parse_reaction_smiles("C=CC=C.C=C>>C1CCC=CC1", context=ReactionContext.pericyclic)
    assert rxn.total_charge_reactants == 0
    assert rxn.total_charge_products == 0


# ---------------------------------------------------------------------------
# Atom-mapped reaction
# ---------------------------------------------------------------------------


_MAPPED = "[CH3:1][Br:2].[OH-:3]>>[CH3:1][OH:3].[Br-:2]"


def test_atom_mapping_detected() -> None:
    rxn = parse_reaction_smiles(_MAPPED, context="ionic")
    assert rxn.has_atom_mapping is True


def test_atom_mapping_numbers_extracted() -> None:
    rxn = parse_reaction_smiles(_MAPPED, context="ionic")
    all_map_values: set[int] = set()
    for mol in rxn.reactants + rxn.products:
        all_map_values.update(mol.atom_map_nums.values())
    assert 1 in all_map_values
    assert 2 in all_map_values
    assert 3 in all_map_values


def test_no_atom_mapping_when_absent() -> None:
    rxn = parse_reaction_smiles("CBr.[OH-]>>CO.[Br-]", context="ionic")
    assert rxn.has_atom_mapping is False


# ---------------------------------------------------------------------------
# String context coercion
# ---------------------------------------------------------------------------


def test_string_context_ionic() -> None:
    rxn = parse_reaction_smiles("CBr.[OH-]>>CO.[Br-]", context="ionic")
    assert rxn.context == ReactionContext.ionic


def test_string_context_unknown_fallback() -> None:
    rxn = parse_reaction_smiles("CBr.[OH-]>>CO.[Br-]", context="gibberish")
    assert rxn.context == ReactionContext.unknown


# ---------------------------------------------------------------------------
# validate_reaction_smiles
# ---------------------------------------------------------------------------


def test_validate_returns_true_for_valid() -> None:
    assert validate_reaction_smiles("CBr.[OH-]>>CO.[Br-]") is True


def test_validate_returns_false_for_missing_arrow() -> None:
    assert validate_reaction_smiles("CBr.[OH-]CO.[Br-]") is False


def test_validate_returns_false_for_empty_string() -> None:
    assert validate_reaction_smiles("") is False


def test_validate_returns_false_for_invalid_smiles() -> None:
    assert validate_reaction_smiles("NOTASMILES>>C") is False


def test_validate_never_raises() -> None:
    for bad in ["", ">>>", "C>>", ">>C", "xyz>>abc"]:
        result = validate_reaction_smiles(bad)
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# ReactionParseError cases
# ---------------------------------------------------------------------------


def test_error_missing_arrow() -> None:
    with pytest.raises(ReactionParseError, match=">>"):
        parse_reaction_smiles("CBr.[OH-]CO.[Br-]")


def test_error_empty_reactants() -> None:
    with pytest.raises(ReactionParseError, match="[Rr]eactant"):
        parse_reaction_smiles(">>CO.[Br-]")


def test_error_empty_products() -> None:
    with pytest.raises(ReactionParseError, match="[Pp]roduct"):
        parse_reaction_smiles("CBr.[OH-]>>")


def test_error_invalid_molecule_smiles() -> None:
    with pytest.raises(ReactionParseError):
        parse_reaction_smiles("NOTASMILES>>CO")


# ---------------------------------------------------------------------------
# parse_reaction_record
# ---------------------------------------------------------------------------


def test_parse_reaction_record_sn2() -> None:
    record = ReactionRecord(
        reaction_id="sn2_test",
        reaction_smiles="CBr.[OH-]>>CO.[Br-]",
        context=ReactionContext.ionic,
    )
    rxn = parse_reaction_record(record)
    assert rxn.context == ReactionContext.ionic
    assert len(rxn.reactants) == 2
    assert len(rxn.products) == 2


def test_parse_reaction_record_preserves_smiles() -> None:
    smiles = "CBr.[OH-]>>CO.[Br-]"
    record = ReactionRecord(
        reaction_id="test",
        reaction_smiles=smiles,
        context=ReactionContext.ionic,
    )
    rxn = parse_reaction_record(record)
    assert rxn.reaction_smiles == smiles


# ---------------------------------------------------------------------------
# get_reaction_summary keys
# ---------------------------------------------------------------------------


def test_summary_has_all_keys() -> None:
    rxn = parse_reaction_smiles("CBr.[OH-]>>CO.[Br-]", context="ionic")
    s = get_reaction_summary(rxn)
    expected_keys = {
        "n_reactants", "n_products",
        "total_charge_reactants", "total_charge_products",
        "has_atom_mapping", "has_radicals",
    }
    assert expected_keys == set(s.keys())


# ---------------------------------------------------------------------------
# Molecule-level fields
# ---------------------------------------------------------------------------


def test_molecule_index_assigned() -> None:
    rxn = parse_reaction_smiles("CBr.[OH-]>>CO.[Br-]", context="ionic")
    assert rxn.reactants[0].molecule_index == 0
    assert rxn.reactants[1].molecule_index == 1
    assert rxn.products[0].molecule_index == 0
    assert rxn.products[1].molecule_index == 1


def test_num_atoms_positive() -> None:
    rxn = parse_reaction_smiles("CBr.[OH-]>>CO.[Br-]", context="ionic")
    for mol in rxn.reactants + rxn.products:
        assert mol.num_atoms > 0


def test_reaction_smiles_preserved() -> None:
    smiles = "CBr.[OH-]>>CO.[Br-]"
    rxn = parse_reaction_smiles(smiles, context="ionic")
    assert rxn.reaction_smiles == smiles
