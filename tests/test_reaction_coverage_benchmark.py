"""Tests for the Phase 12 reaction-coverage benchmark and MLP-compatibility contract.

Two groups of tests:

* Coverage tests — exercise representative reactions through the rule pipeline via
  the benchmark runner (``scripts/reaction_coverage_benchmark.py``).
* Descriptor / MLP-compatibility regression tests — guard that adding the new
  functional-group types (isocyanide / imine / azide) and the Michael-acceptor
  metadata did NOT change the 65-dim descriptor schema, so models/role_mlp.pt
  stays loadable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mendel.descriptor import (  # noqa: E402
    FEATURE_SCHEMA_VERSION,
    build_descriptors,
    build_group_descriptor,
    get_feature_names,
)
from mendel.identifier import identify_functional_groups  # noqa: E402
from mendel.negotiation import run_full_rule_pipeline  # noqa: E402
from mendel.parser import parse_reaction_smiles  # noqa: E402
from mendel.types import FunctionalGroupType  # noqa: E402
from scripts.reaction_coverage_benchmark import (  # noqa: E402
    evaluate_benchmark,
    evaluate_reaction,
    load_benchmark,
)

_EXPECTED_DIM = 65


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pipeline(smiles: str, context: str):
    parsed = parse_reaction_smiles(smiles, context=context)
    groups = identify_functional_groups(parsed)
    result = run_full_rule_pipeline(smiles, context=context)
    return parsed, groups, result


def _detected_types(groups) -> set[str]:
    return {g.group_type.value for g in groups}


def _center_types(result) -> set[str]:
    return {
        a.group_type.value
        for a in result.assignments
        if getattr(a, "is_reaction_center", False)
    }


# ---------------------------------------------------------------------------
# 1. Coverage tests (the seven explicitly-requested cases)
# ---------------------------------------------------------------------------


def test_diels_alder_standard_passes() -> None:
    _, _, result = _pipeline("C=CC=C.C=C>>C1=CCCCC1", "pericyclic")
    assert result.mechanism_hint == "diels_alder_like"


def test_sn2_standard_passes() -> None:
    _, _, result = _pipeline("CBr.[OH-]>>CO.[Br-]", "ionic")
    assert result.mechanism_hint == "sn2_or_e2_like"


def test_aldol_standard_passes() -> None:
    _, _, result = _pipeline("CC(C)=O.CC=O>>CC(O)CC(C)=O", "ionic")
    assert result.mechanism_hint == "aldol_like"


def test_michael_addition_is_michael_with_alkene_center() -> None:
    _, groups, result = _pipeline(
        "C=CC(=O)C.C[N+](=O)[O-]>>CC(=O)CC[N+](=O)[O-]", "ionic"
    )
    assert result.mechanism_hint in {"michael_like", "conjugate_addition_like"}
    # The enone alkene (beta carbon) must be a reaction center, not a spectator.
    assert "alkene" in _center_types(result)
    michael_alkenes = [
        g for g in groups
        if g.group_type == FunctionalGroupType.alkene
        and g.metadata.get("is_michael_acceptor")
    ]
    assert michael_alkenes, "enone alkene should carry is_michael_acceptor metadata"


def test_michael_carbonyl_not_marked_center() -> None:
    # The conjugated carbonyl is an activating spectator, not the center.
    _, _, result = _pipeline(
        "C=CC(=O)C.C[N+](=O)[O-]>>CC(=O)CC[N+](=O)[O-]", "ionic"
    )
    carbonyl_centers = [
        a for a in result.assignments
        if a.group_type == FunctionalGroupType.carbonyl and a.is_reaction_center
    ]
    assert not carbonyl_centers


def test_isocyanide_detected() -> None:
    _, groups, _ = _pipeline("C[N+]#[C-].CI>>CC#[N+]C.[I-]", "ionic")
    assert FunctionalGroupType.isocyanide.value in _detected_types(groups)


def test_azide_alkyne_is_click_not_diels_alder() -> None:
    _, groups, result = _pipeline("CN=[N+]=[N-].C#CC>>Cc1cn(C)nn1", "pericyclic")
    assert result.mechanism_hint in {"click_like", "huisgen_like"}
    assert result.mechanism_hint != "diels_alder_like"
    assert {"azide", "alkyne"} <= _center_types(result)


def test_negative_control_invents_no_strong_reactive_center() -> None:
    # A benign ester that does not react must not fabricate a reactive center.
    _, _, result = _pipeline("CCOC(C)=O>>CCOC(C)=O", "unknown")
    strong_centers = [
        a for a in result.assignments
        if a.is_reaction_center and a.final_role.value != "spectator"
    ]
    assert not strong_centers, f"unexpected reactive centers: {strong_centers}"


# ---------------------------------------------------------------------------
# 2. New mechanism-hint coverage (radical family)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "smiles,context,expected",
    [
        ("[CH3].C=C>>CCC", "radical", "radical_addition_like"),
        ("[CH3].C=CC(=O)OC>>CCC(=O)OC", "radical", "giese_like"),
        ("c1ccncc1.[CH3]>>Cc1ccncc1", "radical", "minisci_like"),
        ("Cc1ccccc1.BrBr>>BrCc1ccccc1.[Br]", "radical", "radical_bromination_like"),
    ],
)
def test_radical_family_hints_are_distinct(smiles, context, expected) -> None:
    _, _, result = _pipeline(smiles, context)
    assert result.mechanism_hint == expected


# ---------------------------------------------------------------------------
# 3. Whole-benchmark gate: no hard failures
# ---------------------------------------------------------------------------


def test_full_benchmark_has_no_hard_failures() -> None:
    cases = load_benchmark()
    assert len(cases) >= 30
    results, summary = evaluate_benchmark(cases)
    assert summary["parse_failures"] == 0, [
        r["name"] for r in results if r["status"] == "parse_failure"
    ]
    assert summary["failed"] == 0, [
        (r["name"], r["reasons"]) for r in results if r["status"] == "fail"
    ]
    assert summary["ci_ok"] is True


def test_every_benchmark_smiles_is_rdkit_parseable() -> None:
    for case in load_benchmark():
        row = evaluate_reaction(case)
        assert row["parse_ok"], f"{case['name']} failed to parse: {row['error']}"


# ---------------------------------------------------------------------------
# 4. Descriptor / MLP-compatibility regression tests
# ---------------------------------------------------------------------------


def test_feature_schema_version_unchanged() -> None:
    assert FEATURE_SCHEMA_VERSION == "phase6_6_v1"


def test_descriptor_dimension_is_65() -> None:
    assert len(get_feature_names()) == _EXPECTED_DIM


def test_descriptor_values_length_is_65_for_classic_groups() -> None:
    parsed = parse_reaction_smiles("CBr.[OH-]>>CO.[Br-]", context="ionic")
    groups = identify_functional_groups(parsed)
    for g in groups:
        d = build_group_descriptor(parsed, g, all_groups=groups)
        assert len(d.values) == _EXPECTED_DIM


def test_descriptor_feature_names_length_equals_values_length() -> None:
    parsed = parse_reaction_smiles(
        "C=CC(=O)C.C[N+](=O)[O-]>>CC(=O)CC[N+](=O)[O-]", context="ionic"
    )
    groups = identify_functional_groups(parsed)
    for d in build_descriptors(parsed, groups):
        assert len(d.feature_names) == len(d.values) == _EXPECTED_DIM
        assert d.feature_names == get_feature_names()


@pytest.mark.parametrize(
    "smiles,new_type",
    [
        ("C[N+]#[C-].CI>>CC#[N+]C.[I-]", "isocyanide"),
        ("CC=NC.C[N+]#[C-]>>CC(NC)C#[N+]C", "imine"),
        ("CN=[N+]=[N-].C#CC>>Cc1cn(C)nn1", "azide"),
    ],
)
def test_new_types_detected_without_changing_descriptor_dim(smiles, new_type) -> None:
    parsed = parse_reaction_smiles(smiles, context="ionic")
    groups = identify_functional_groups(parsed)
    assert new_type in {g.group_type.value for g in groups}, (
        f"{new_type} silently dropped from identifier output"
    )
    # Descriptors for the molecule containing the new type stay 65-dim.
    for d in build_descriptors(parsed, groups):
        assert len(d.values) == _EXPECTED_DIM


def test_new_types_not_silently_dropped_from_identifier() -> None:
    # Each new type, in isolation, must appear in identifier output.
    cases = {
        "isocyanide": "C[N+]#[C-]>>C[N+]#[C-]",
        "imine": "CC=NC>>CC=NC",
        "azide": "CN=[N+]=[N-]>>CN=[N+]=[N-]",
    }
    for new_type, smiles in cases.items():
        parsed = parse_reaction_smiles(smiles, context="ionic")
        groups = identify_functional_groups(parsed)
        assert new_type in {g.group_type.value for g in groups}


def test_michael_metadata_does_not_alter_descriptor_schema() -> None:
    # An enone carries Michael-acceptor metadata, but the descriptor schema/dim
    # is identical to a plain alkene's descriptor.
    enone = parse_reaction_smiles("C=CC(=O)C>>CCC(=O)C", context="ionic")
    enone_groups = identify_functional_groups(enone)
    michael_alkenes = [
        g for g in enone_groups
        if g.group_type == FunctionalGroupType.alkene
        and g.metadata.get("is_michael_acceptor")
    ]
    assert michael_alkenes, "expected an is_michael_acceptor alkene in the enone"

    for d in build_descriptors(enone, enone_groups):
        assert len(d.values) == _EXPECTED_DIM
        assert d.feature_names == get_feature_names()

    # A plain (non-Michael) alkene yields the same schema and dimension.
    plain = parse_reaction_smiles("C=CC>>CCC", context="ionic")
    plain_groups = identify_functional_groups(plain)
    plain_alkene = next(
        g for g in plain_groups if g.group_type == FunctionalGroupType.alkene
    )
    plain_desc = build_group_descriptor(plain, plain_alkene, all_groups=plain_groups)
    assert len(plain_desc.values) == _EXPECTED_DIM
    assert plain_desc.feature_names == get_feature_names()


def test_role_mlp_checkpoint_still_loadable_if_present() -> None:
    ckpt = _REPO_ROOT / "models" / "role_mlp.pt"
    if not ckpt.exists():
        pytest.skip("models/role_mlp.pt not present")
    torch = pytest.importorskip("torch")
    blob = torch.load(str(ckpt), map_location="cpu", weights_only=True)
    assert blob["input_dim"] == _EXPECTED_DIM
    assert len(blob["feature_names"]) == _EXPECTED_DIM

    from mendel.mlp import MLPRolePredictor

    mlp = MLPRolePredictor.load(ckpt, device="cpu")
    # End-to-end MLP pipeline still runs against the frozen 65-dim descriptor.
    from mendel.negotiation import run_pipeline_with_mlp

    result = run_pipeline_with_mlp("CBr.[OH-]>>CO.[Br-]", ckpt, context="ionic")
    assert result.assignments
    assert mlp is not None
