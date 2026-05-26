"""Phase 4 label schema tests."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from mendel.labels import (
    LabelValidationError,
    LabeledGroupRole,
    LabeledReaction,
    labels_to_training_rows,
    load_labeled_reactions,
    save_labeled_reactions,
    summarize_labeled_dataset,
    validate_labeled_dataset,
    validate_labeled_reaction,
)
from mendel.types import FunctionalGroupType, ReactionContext, Role

_MINIMAL = Path(__file__).parent.parent / "data" / "reactions.minimal.json"
_FULL = Path(__file__).parent.parent / "data" / "reactions.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_rxn(
    reaction_id: str = "test_rxn",
    smiles: str = "CBr.[OH-]>>CO.[Br-]",
    split: str = "train",
    roles: list[LabeledGroupRole] | None = None,
) -> LabeledReaction:
    return LabeledReaction(
        reaction_id=reaction_id,
        reaction_smiles=smiles,
        context=ReactionContext.ionic,
        mechanism_type="SN2",
        split=split,
        group_roles=roles or [],
    )


def _make_lgr(
    group_id: str = "mol0_halide_0",
    role: Role = Role.leaving_group,
) -> LabeledGroupRole:
    return LabeledGroupRole(
        group_id=group_id,
        molecule_index=0,
        group_type=FunctionalGroupType.halide,
        atom_indices=[0, 1],
        role=role,
    )


# ---------------------------------------------------------------------------
# 1. Load minimal dataset
# ---------------------------------------------------------------------------


def test_load_minimal_returns_list() -> None:
    rxns = load_labeled_reactions(_MINIMAL)
    assert isinstance(rxns, list)
    assert len(rxns) >= 1


def test_load_minimal_reaction_ids() -> None:
    rxns = load_labeled_reactions(_MINIMAL)
    ids = {r.reaction_id for r in rxns}
    assert "minimal_sn2" in ids


def test_load_minimal_group_roles_typed() -> None:
    rxns = load_labeled_reactions(_MINIMAL)
    sn2 = next(r for r in rxns if r.reaction_id == "minimal_sn2")
    assert len(sn2.group_roles) == 1
    lgr = sn2.group_roles[0]
    assert isinstance(lgr.role, Role)
    assert isinstance(lgr.group_type, FunctionalGroupType)
    assert lgr.group_id == "mol0_halide_0"
    assert lgr.role == Role.leaving_group


# ---------------------------------------------------------------------------
# 2. Load full benchmark dataset
# ---------------------------------------------------------------------------


def test_load_full_contains_benchmark_reactions() -> None:
    rxns = load_labeled_reactions(_FULL)
    ids = {r.reaction_id for r in rxns}
    for expected in (
        "sn2_methyl_bromide_oh",
        "e2_ethyl_bromide_oh",
        "diels_alder_butadiene_ethylene",
        "aldol_acetone_self",
        "radical_bromination_methane",
    ):
        assert expected in ids


def test_load_full_schema_types() -> None:
    rxns = load_labeled_reactions(_FULL)
    for rxn in rxns:
        assert isinstance(rxn.context, ReactionContext)
        assert rxn.split in {"train", "val", "test"}
        for lgr in rxn.group_roles:
            assert isinstance(lgr.role, Role)
            assert isinstance(lgr.group_type, FunctionalGroupType)
            assert len(lgr.atom_indices) > 0


# ---------------------------------------------------------------------------
# 3. Roundtrip save / load
# ---------------------------------------------------------------------------


def test_roundtrip_save_load() -> None:
    rxns = [_make_rxn(roles=[_make_lgr()])]
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmp = Path(f.name)
    save_labeled_reactions(rxns, tmp)
    loaded = load_labeled_reactions(tmp)
    tmp.unlink()
    assert len(loaded) == 1
    assert loaded[0].reaction_id == "test_rxn"
    assert loaded[0].group_roles[0].role == Role.leaving_group


def test_roundtrip_preserves_all_fields() -> None:
    lgr = LabeledGroupRole(
        group_id="mol0_carbonyl_0",
        molecule_index=0,
        group_type=FunctionalGroupType.carbonyl,
        atom_indices=[1, 2],
        role=Role.reactive_electrophile,
        confidence="manual",
        notes="test note",
    )
    rxn = LabeledReaction(
        reaction_id="roundtrip_test",
        reaction_smiles="CC(=O)C>>CC(=O)CO",
        context=ReactionContext.ionic,
        mechanism_type="Aldol",
        split="val",
        group_roles=[lgr],
        reaction_center_atoms=[2, 3],
        metadata={"source": "test"},
    )
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmp = Path(f.name)
    save_labeled_reactions([rxn], tmp)
    loaded = load_labeled_reactions(tmp)[0]
    tmp.unlink()
    assert loaded.reaction_center_atoms == [2, 3]
    assert loaded.metadata["source"] == "test"
    assert loaded.group_roles[0].notes == "test note"


# ---------------------------------------------------------------------------
# 4. Validation — valid reactions pass
# ---------------------------------------------------------------------------


def test_validate_single_reaction_passes() -> None:
    rxn = _make_rxn(roles=[_make_lgr()])
    assert validate_labeled_reaction(rxn) is True


def test_validate_dataset_passes() -> None:
    rxns = load_labeled_reactions(_FULL)
    assert validate_labeled_dataset(rxns) is True


def test_validate_minimal_passes() -> None:
    rxns = load_labeled_reactions(_MINIMAL)
    assert validate_labeled_dataset(rxns) is True


# ---------------------------------------------------------------------------
# 5. Validation — invalid reactions raise LabelValidationError
# ---------------------------------------------------------------------------


def test_validate_bad_split_raises() -> None:
    rxn = _make_rxn(split="holdout")
    with pytest.raises(LabelValidationError, match="split"):
        validate_labeled_reaction(rxn)


def test_validate_missing_arrow_raises() -> None:
    rxn = _make_rxn(smiles="CBr.[OH-]")
    with pytest.raises(LabelValidationError, match=">>"):
        validate_labeled_reaction(rxn)


def test_validate_duplicate_group_id_raises() -> None:
    lgr1 = _make_lgr("mol0_halide_0")
    lgr2 = _make_lgr("mol0_halide_0")
    rxn = _make_rxn(roles=[lgr1, lgr2])
    with pytest.raises(LabelValidationError, match="duplicate"):
        validate_labeled_reaction(rxn)


def test_validate_empty_atom_indices_raises() -> None:
    lgr = LabeledGroupRole(
        group_id="mol0_halide_0",
        molecule_index=0,
        group_type=FunctionalGroupType.halide,
        atom_indices=[],
        role=Role.leaving_group,
    )
    rxn = _make_rxn(roles=[lgr])
    with pytest.raises(LabelValidationError, match="atom_indices"):
        validate_labeled_reaction(rxn)


def test_validate_duplicate_reaction_id_across_dataset_raises() -> None:
    rxn1 = _make_rxn("dup")
    rxn2 = _make_rxn("dup")
    with pytest.raises(LabelValidationError, match="Duplicate"):
        validate_labeled_dataset([rxn1, rxn2])


# ---------------------------------------------------------------------------
# 6. Summarise dataset
# ---------------------------------------------------------------------------


def test_summarize_keys() -> None:
    rxns = load_labeled_reactions(_MINIMAL)
    s = summarize_labeled_dataset(rxns)
    for key in (
        "n_reactions",
        "n_labels",
        "role_distribution",
        "mechanism_distribution",
        "split_distribution",
    ):
        assert key in s


def test_summarize_counts() -> None:
    rxns = load_labeled_reactions(_MINIMAL)
    s = summarize_labeled_dataset(rxns)
    assert s["n_reactions"] == 2
    assert s["n_labels"] >= 1


def test_summarize_role_distribution() -> None:
    rxns = load_labeled_reactions(_FULL)
    s = summarize_labeled_dataset(rxns)
    assert "leaving_group" in s["role_distribution"]
    assert s["role_distribution"]["leaving_group"] >= 1


# ---------------------------------------------------------------------------
# 7. labels_to_training_rows
# ---------------------------------------------------------------------------


def test_training_rows_count() -> None:
    rxns = load_labeled_reactions(_MINIMAL)
    rows = labels_to_training_rows(rxns)
    total_labels = sum(len(r.group_roles) for r in rxns)
    assert len(rows) == total_labels


def test_training_rows_fields() -> None:
    rxns = load_labeled_reactions(_MINIMAL)
    rows = labels_to_training_rows(rxns)
    for row in rows:
        assert "reaction_id" in row
        assert "group_id" in row
        assert "role" in row
        assert "atom_indices" in row


def test_training_rows_empty_group_roles() -> None:
    rows = labels_to_training_rows([_make_rxn(roles=[])])
    assert rows == []


# ---------------------------------------------------------------------------
# 8. to_dict roundtrip
# ---------------------------------------------------------------------------


def test_lgr_to_dict() -> None:
    d = _make_lgr().to_dict()
    assert d["group_id"] == "mol0_halide_0"
    assert d["role"] == "leaving_group"
    assert d["group_type"] == "halide"
    assert isinstance(d["atom_indices"], list)


def test_lr_to_dict() -> None:
    d = _make_rxn(roles=[_make_lgr()]).to_dict()
    assert d["reaction_id"] == "test_rxn"
    assert d["group_roles"][0]["group_id"] == "mol0_halide_0"


# ---------------------------------------------------------------------------
# 9. Acceptance snippet
# ---------------------------------------------------------------------------


def test_acceptance_snippet() -> None:
    rxns = load_labeled_reactions(_FULL)
    assert validate_labeled_dataset(rxns)
    summary = summarize_labeled_dataset(rxns)
    rows = labels_to_training_rows(rxns)
    assert summary["n_reactions"] >= 5
    assert len(rows) >= 1
    print(f"Phase 4 labels OK: {summary['n_reactions']} reactions, {summary['n_labels']} labels")
