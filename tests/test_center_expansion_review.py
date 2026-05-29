"""Tests for Phase 8.13 center-expansion conservative review."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from mendel.center_expansion_review import (
    CenterExpansionPromotionRecord,
    apply_mechanism_center_policy,
    merge_center_expansion_with_cleaned_base,
    promote_center_expansion_reactions,
    save_center_expansion_promotion_report,
)
from mendel.labels import LabeledGroupRole, LabeledReaction, load_labeled_reactions
from mendel.types import FunctionalGroupType, ReactionContext, Role

_ROOT = Path(__file__).parent.parent
_SCRIPT = _ROOT / "scripts" / "promote_center_expansion.py"


def _role(
    group_id: str, group_type: FunctionalGroupType, atoms: list[int], role: Role
) -> LabeledGroupRole:
    return LabeledGroupRole(group_id, 0, group_type, atoms, role, confidence="draft")


def _rxn(
    reaction_id: str,
    smiles: str,
    mechanism: str,
    roles: list[LabeledGroupRole],
    center: list[int] | None = None,
    context: ReactionContext = ReactionContext.ionic,
) -> LabeledReaction:
    return LabeledReaction(
        reaction_id=reaction_id,
        reaction_smiles=smiles,
        context=context,
        mechanism_type=mechanism,
        split="draft",
        group_roles=roles,
        reaction_center_atoms=center or [],
        metadata={"source": "local_template_center_expansion", "template_name": reaction_id},
    )


def test_sn2_promotion_sets_halide_leaving_group_and_center() -> None:
    rxn = _rxn(
        "sn2",
        "[CH3:1][Br:2].[I-:3]>>[CH3:1][I:3].[Br-:2]",
        "sn2",
        [_role("mol0_halide_0", FunctionalGroupType.halide, [0, 1], Role.spectator)],
    )

    promoted, corrections, policy = apply_mechanism_center_policy(rxn)

    assert promoted is not None
    assert promoted.group_roles[0].role is Role.leaving_group
    assert promoted.reaction_center_atoms == [1, 2]
    assert "SN2" in policy


def test_e2_promotion_includes_halide_center() -> None:
    rxn = _rxn(
        "e2",
        "[CH3:1][CH:2]([Br:3])[CH3:4]>>[CH2:1]=[CH:2][CH3:4].[Br-:3]",
        "e2",
        [_role("mol0_halide_0", FunctionalGroupType.halide, [1, 2], Role.spectator)],
    )

    promoted, _, _ = apply_mechanism_center_policy(rxn)

    assert promoted is not None
    assert 3 in promoted.reaction_center_atoms


def test_diels_alder_excludes_nitrile_substituent_atoms() -> None:
    rxn = _rxn(
        "da",
        "[CH2:1]=[CH:2][C:3]#[N:4].[CH2:5]=[CH2:6]>>[CH2:1][CH:2][CH2:5][CH2:6]",
        "diels_alder",
        [
            _role("mol0_alkene_0", FunctionalGroupType.alkene, [0, 1], Role.spectator),
            _role(
                "mol0_nitrile_0", FunctionalGroupType.nitrile, [2, 3], Role.reactive_electrophile
            ),
            _role("mol1_alkene_0", FunctionalGroupType.alkene, [0, 1], Role.spectator),
        ],
        context=ReactionContext.pericyclic,
    )

    promoted, _, _ = apply_mechanism_center_policy(rxn)

    assert promoted is not None
    assert promoted.reaction_center_atoms == [1, 2, 5, 6]
    assert all(
        role.role is Role.spectator
        for role in promoted.group_roles
        if role.group_type is FunctionalGroupType.nitrile
    )


def test_carbonyl_addition_promotes_carbonyl_and_excludes_alpha() -> None:
    rxn = _rxn(
        "carbonyl",
        "[CH3:1][C:2](=[O:3])[CH3:4]>>[CH3:1][C:2]([O-:3])[CH3:4]",
        "carbonyl_addition",
        [
            _role("mol0_carbonyl_0", FunctionalGroupType.carbonyl, [1, 2], Role.spectator),
            _role(
                "mol0_alpha_carbon_0",
                FunctionalGroupType.alpha_carbon,
                [0],
                Role.reactive_nucleophile,
            ),
        ],
    )

    promoted, _, _ = apply_mechanism_center_policy(rxn)

    assert promoted is not None
    assert promoted.reaction_center_atoms == [2, 3]
    assert promoted.group_roles[0].role is Role.reactive_electrophile
    assert promoted.group_roles[1].role is Role.spectator


def test_control_promotion_sets_spectators_and_empty_center() -> None:
    rxn = _rxn(
        "control",
        "[CH3:1][CH3:2]>>[CH3:1][CH3:2]",
        "control",
        [_role("mol0_alkene_0", FunctionalGroupType.alkene, [0, 1], Role.reactive_nucleophile)],
        [1],
        ReactionContext.unknown,
    )

    promoted, _, _ = apply_mechanism_center_policy(rxn)

    assert promoted is not None
    assert promoted.reaction_center_atoms == []
    assert all(role.role is Role.spectator for role in promoted.group_roles)


def test_ambiguous_aldol_is_skipped() -> None:
    rxn = _rxn(
        "aldol",
        "[CH3:1][C:2](=[O:3])[CH3:4].[CH3:5][C:6](=[O:7])[CH3:8]>>[CH3:1][C:2](=[O:3])[CH2:4][C:6]([O-:7])[CH3:5]",
        "aldol",
        [
            _role("mol0_alpha_carbon_0", FunctionalGroupType.alpha_carbon, [0], Role.spectator),
            _role("mol0_alpha_carbon_1", FunctionalGroupType.alpha_carbon, [2], Role.spectator),
        ],
    )

    promoted, _, reason = apply_mechanism_center_policy(rxn)

    assert promoted is None
    assert "ambiguous" in reason


def test_merge_avoids_duplicate_reaction_ids() -> None:
    base = [_rxn("dup", "[CH3:1][CH3:2]>>[CH3:1][CH3:2]", "control", [])]
    promoted = [_rxn("dup", "[CH3:1][Br:2]>>[CH3:1][OH:3]", "sn2", [])]

    merged = merge_center_expansion_with_cleaned_base(base, promoted)

    assert [rxn.reaction_id for rxn in merged] == ["dup", "dup_center_expansion"]


def test_report_serialization(tmp_path: Path) -> None:
    record = CenterExpansionPromotionRecord(
        "r",
        "control",
        True,
        None,
        [],
        [],
        [1],
        [],
        "empty center",
        ["control"],
        {},
    )
    out = tmp_path / "report.json"
    _, report = promote_center_expansion_reactions(
        [_rxn("control", "[CH3:1][CH3:2]>>[CH3:1][CH3:2]", "control", [])]
    )
    report.promotion_records.append(record)

    save_center_expansion_promotion_report(report, out)

    assert json.loads(out.read_text(encoding="utf-8"))["promotion_records"]


def test_cli_smoke_creates_promoted_output_and_report(tmp_path: Path) -> None:
    inp = tmp_path / "draft.json"
    base = tmp_path / "base.json"
    output = tmp_path / "promoted.json"
    merged = tmp_path / "merged.json"
    report = tmp_path / "report.json"
    rxn = _rxn("control", "[CH3:1][CH3:2]>>[CH3:1][CH3:2]", "control", [])
    inp.write_text(json.dumps({"reactions": [rxn.to_dict()]}), encoding="utf-8")
    base.write_text(json.dumps({"reactions": []}), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--input",
            str(inp),
            "--base",
            str(base),
            "--output",
            str(output),
            "--merged-output",
            str(merged),
            "--report",
            str(report),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert load_labeled_reactions(output)[0].metadata["center_expansion_promoted"] is True
    assert report.exists()


def test_dry_run_does_not_write_outputs(tmp_path: Path) -> None:
    inp = tmp_path / "draft.json"
    base = tmp_path / "base.json"
    output = tmp_path / "promoted.json"
    merged = tmp_path / "merged.json"
    report = tmp_path / "report.json"
    rxn = _rxn("control", "[CH3:1][CH3:2]>>[CH3:1][CH3:2]", "control", [])
    inp.write_text(json.dumps({"reactions": [rxn.to_dict()]}), encoding="utf-8")
    base.write_text(json.dumps({"reactions": []}), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--input",
            str(inp),
            "--base",
            str(base),
            "--output",
            str(output),
            "--merged-output",
            str(merged),
            "--report",
            str(report),
            "--dry-run",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert not output.exists()
    assert not merged.exists()
    assert not report.exists()


def test_no_mlip_training_invoked() -> None:
    text = _SCRIPT.read_text(encoding="utf-8").lower()
    for token in ("mace", "transition1x", "dft", "forces", "barrier"):
        assert token not in text
