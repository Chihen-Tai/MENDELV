"""Phase 0 scaffold tests.

Verifies that the project installs correctly, enums are complete,
dataclasses construct without errors, and the example data file is valid.
"""

from __future__ import annotations

import json
import pathlib


def test_import_mendel() -> None:
    import mendel  # noqa: F401


def test_version_is_string() -> None:
    import mendel

    assert isinstance(mendel.__version__, str)
    assert len(mendel.__version__) > 0


def test_reaction_context_enum_members() -> None:
    from mendel.types import ReactionContext

    expected = {"ionic", "radical", "pericyclic", "unknown"}
    actual = {m.value for m in ReactionContext}
    assert actual == expected


def test_role_enum_members() -> None:
    from mendel.types import Role

    expected = {
        "reactive_nucleophile",
        "reactive_electrophile",
        "reactive_radical",
        "leaving_group",
        "spectator",
    }
    actual = {m.value for m in Role}
    assert actual == expected


def test_functional_group_type_enum_members() -> None:
    from mendel.types import FunctionalGroupType

    expected = {
        "alkene", "alkyne", "aromatic", "alcohol", "phenol", "ether",
        "carbonyl", "carboxylic_acid", "ester", "amine", "amide",
        "halide", "nitrile", "nitro", "alpha_carbon", "benzylic_site", "unknown",
    }
    actual = {m.value for m in FunctionalGroupType}
    assert actual == expected


def test_reaction_record_construction() -> None:
    from mendel.types import ReactionContext, ReactionRecord

    record = ReactionRecord(
        reaction_id="sn2_demo",
        reaction_smiles="CBr.[OH-]>>CO.[Br-]",
        context=ReactionContext.ionic,
    )

    assert record.reaction_id == "sn2_demo"
    assert record.reaction_smiles == "CBr.[OH-]>>CO.[Br-]"
    assert record.context == ReactionContext.ionic
    assert record.expected_roles == []
    assert record.expected_reaction_center == []
    assert record.metadata == {}


def test_reaction_record_to_dict() -> None:
    from mendel.types import ReactionContext, ReactionRecord

    record = ReactionRecord(
        reaction_id="test_id",
        reaction_smiles="C>>C",
        context=ReactionContext.unknown,
    )
    d = record.to_dict()

    assert d["reaction_id"] == "test_id"
    assert d["context"] == "unknown"
    assert isinstance(d["expected_roles"], list)


def test_role_assignment_construction() -> None:
    from mendel.types import Role, RoleAssignment

    ra = RoleAssignment(group_id="mol0_halide_0", role=Role.leaving_group)
    assert ra.group_id == "mol0_halide_0"
    assert ra.role == Role.leaving_group
    assert ra.confidence is None
    assert ra.reason is None


def test_atom_ref_construction() -> None:
    from mendel.types import AtomRef

    ref = AtomRef(molecule_index=0, atom_index=1, atom_map_num=5)
    assert ref.molecule_index == 0
    assert ref.atom_index == 1
    assert ref.atom_map_num == 5
    assert ref.to_dict() == {"molecule_index": 0, "atom_index": 1, "atom_map_num": 5}


def test_example_json_exists() -> None:
    path = pathlib.Path(__file__).parent.parent / "data" / "reactions.example.json"
    assert path.exists(), f"Expected {path} to exist"


def test_example_json_is_valid_json() -> None:
    path = pathlib.Path(__file__).parent.parent / "data" / "reactions.example.json"
    with path.open() as fh:
        data = json.load(fh)
    assert isinstance(data, list)
    assert len(data) >= 1


def test_example_json_required_fields() -> None:
    path = pathlib.Path(__file__).parent.parent / "data" / "reactions.example.json"
    with path.open() as fh:
        records = json.load(fh)

    for record in records:
        assert "reaction_id" in record, f"Missing reaction_id in {record}"
        assert "reaction_smiles" in record, f"Missing reaction_smiles in {record}"
        assert "context" in record, f"Missing context in {record}"


def test_constants_derived_from_enums() -> None:
    from mendel.constants import SUPPORTED_CONTEXTS, SUPPORTED_FUNCTIONAL_GROUPS, SUPPORTED_ROLES
    from mendel.types import FunctionalGroupType, ReactionContext, Role

    assert frozenset(Role) == SUPPORTED_ROLES
    assert frozenset(ReactionContext) == SUPPORTED_CONTEXTS
    assert frozenset(FunctionalGroupType) == SUPPORTED_FUNCTIONAL_GROUPS
