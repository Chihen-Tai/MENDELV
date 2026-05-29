"""Tests for Phase 8.12 center-label cleanup."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from mendel.center_cleanup import (
    CenterCleanupReport,
    CenterLabelCorrection,
    cleanup_center_labels,
    save_center_cleanup_report,
)
from mendel.labels import LabeledGroupRole, LabeledReaction, load_labeled_reactions
from mendel.types import FunctionalGroupType, ReactionContext, Role

_ROOT = Path(__file__).parent.parent
_SCRIPT = _ROOT / "scripts" / "cleanup_center_labels.py"


def _rxn(
    reaction_id: str,
    smiles: str,
    mechanism: str,
    roles: list[LabeledGroupRole],
    center: list[int],
    context: ReactionContext = ReactionContext.ionic,
) -> LabeledReaction:
    return LabeledReaction(
        reaction_id=reaction_id,
        reaction_smiles=smiles,
        context=context,
        mechanism_type=mechanism,
        split="train",
        group_roles=roles,
        reaction_center_atoms=center,
        metadata={"template_name": reaction_id},
    )


def _lgr(
    group_id: str, group_type: FunctionalGroupType, atoms: list[int], role: Role
) -> LabeledGroupRole:
    return LabeledGroupRole(group_id, 0, group_type, atoms, role)


def test_control_non_empty_center_corrected_to_empty() -> None:
    rxn = _rxn("control", "[CH3:1][CH3:2]>>[CH3:1][CH3:2]", "control", [], [1])

    cleaned, corrections = cleanup_center_labels([rxn])

    assert cleaned[0].reaction_center_atoms == []
    assert corrections[0].correction_type == "control_empty_center"


def test_spectator_only_center_corrected_to_empty() -> None:
    rxn = _rxn(
        "spectator",
        "[CH3:1][CH3:2]>>[CH3:1][CH3:2]",
        "ester_control",
        [_lgr("mol0_ester_0", FunctionalGroupType.ester, [0, 1], Role.spectator)],
        [1],
    )

    cleaned, corrections = cleanup_center_labels([rxn])

    assert cleaned[0].reaction_center_atoms == []
    assert any(c.correction_type == "control_empty_center" for c in corrections)


def test_diels_alder_substituent_centers_removed_while_alkene_remains() -> None:
    rxn = _rxn(
        "da",
        "[CH2:1]=[CH:2][C:3]#[N:4].[CH2:5]=[CH2:6]>>[CH2:1][CH:2][CH2:5][CH2:6]",
        "diels_alder",
        [
            _lgr("mol0_alkene_0", FunctionalGroupType.alkene, [0, 1], Role.reactive_nucleophile),
            _lgr("mol0_nitrile_0", FunctionalGroupType.nitrile, [2, 3], Role.spectator),
            _lgr("mol1_alkene_0", FunctionalGroupType.alkene, [0, 1], Role.reactive_electrophile),
        ],
        [1, 2, 3, 4, 5, 6],
        ReactionContext.pericyclic,
    )

    cleaned, corrections = cleanup_center_labels([rxn])

    assert cleaned[0].reaction_center_atoms == [1, 2, 5, 6]
    assert corrections[0].correction_type == "diels_alder_substituent_center_cleanup"


def test_carbonyl_addition_alpha_spectator_removed() -> None:
    rxn = _rxn(
        "carbonyl",
        "[CH3:1][C:2](=[O:3])[CH3:4]>>[CH3:1][C:2]([O-:3])[CH3:4]",
        "carbonyl_addition",
        [
            _lgr(
                "mol0_carbonyl_0", FunctionalGroupType.carbonyl, [1, 2], Role.reactive_electrophile
            ),
            _lgr("mol0_alpha_carbon_0", FunctionalGroupType.alpha_carbon, [0], Role.spectator),
        ],
        [1, 2, 3],
    )

    cleaned, _ = cleanup_center_labels([rxn])

    assert cleaned[0].reaction_center_atoms == [2, 3]


def test_sn2_cleanup_adds_missing_halide_atom() -> None:
    rxn = _rxn(
        "sn2",
        "[CH3:1][Br:2]>>[CH3:1][OH:3]",
        "sn2",
        [_lgr("mol0_halide_0", FunctionalGroupType.halide, [0, 1], Role.leaving_group)],
        [1],
    )

    cleaned, _ = cleanup_center_labels([rxn])

    assert cleaned[0].reaction_center_atoms == [1, 2]


def test_radical_cleanup_keeps_benzylic_and_removes_aromatic() -> None:
    rxn = _rxn(
        "rad",
        "[cH:1]1[cH:2][cH:3][cH:4][cH:5][c:6]1[CH3:7]>>[cH:1]1[cH:2][cH:3][cH:4][cH:5][c:6]1[CH2:7]",
        "radical_bromination",
        [
            _lgr(
                "mol0_aromatic_0", FunctionalGroupType.aromatic, [0, 1, 2, 3, 4, 5], Role.spectator
            ),
            _lgr(
                "mol0_benzylic_site_0",
                FunctionalGroupType.benzylic_site,
                [6, 5],
                Role.reactive_radical,
            ),
        ],
        [1, 2, 3, 4, 5, 6, 7],
        ReactionContext.radical,
    )

    cleaned, _ = cleanup_center_labels([rxn])

    assert cleaned[0].reaction_center_atoms == [6, 7]


def test_cleanup_does_not_mutate_input() -> None:
    rxn = _rxn("control", "[CH3:1][CH3:2]>>[CH3:1][CH3:2]", "control", [], [1])

    cleaned, _ = cleanup_center_labels([rxn])

    assert rxn.reaction_center_atoms == [1]
    assert cleaned[0] is not rxn


def test_report_serialization(tmp_path: Path) -> None:
    report = CenterCleanupReport(
        input_path="in.json",
        output_path="out.json",
        n_reactions=1,
        n_corrected_reactions=1,
        n_skipped_reactions=0,
        n_corrections=1,
        corrections_by_type={"control_empty_center": 1},
        remaining_issues_by_severity={},
        remaining_issues_by_type={},
        corrections=[
            CenterLabelCorrection(
                "r", "control", "control_empty_center", "error", [1], [], "x", "manual", {}
            )
        ],
        skipped_reactions=[],
        recommendations=[],
        metadata={},
    )
    out = tmp_path / "report.json"

    save_center_cleanup_report(report, out)

    assert json.loads(out.read_text(encoding="utf-8"))["n_corrections"] == 1


def test_cleanup_cli_smoke_writes_output_and_report(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.json"
    output = tmp_path / "cleaned.json"
    report = tmp_path / "report.json"
    rxn = _rxn("control", "[CH3:1][CH3:2]>>[CH3:1][CH3:2]", "control", [], [1])
    dataset.write_text(json.dumps({"reactions": [rxn.to_dict()]}), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--input",
            str(dataset),
            "--output",
            str(output),
            "--report",
            str(report),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert load_labeled_reactions(output)[0].reaction_center_atoms == []
    assert report.exists()


def test_no_mlip_training_invoked() -> None:
    text = _SCRIPT.read_text(encoding="utf-8").lower()
    for token in ("mace", "transition1x", "dft", "forces", "barrier"):
        assert token not in text
