"""Tests for Phase 8.6 conservative review-queue promotion."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from mendel.labels import (
    LabeledGroupRole,
    LabeledReaction,
    load_labeled_reactions,
    validate_labeled_dataset,
)
from mendel.promotion import (
    build_promotion_report,
    merge_promoted_with_base,
    promote_review_queue,
    promote_review_queue_reaction,
)
from mendel.types import FunctionalGroupType, ReactionContext, Role

_ROOT = Path(__file__).parent.parent
_SCRIPT = _ROOT / "scripts" / "promote_review_queue.py"


def _role(
    group_id: str,
    group_type: FunctionalGroupType,
    role: Role,
    molecule_index: int = 0,
) -> LabeledGroupRole:
    return LabeledGroupRole(
        group_id=group_id,
        molecule_index=molecule_index,
        group_type=group_type,
        atom_indices=[0],
        role=role,
        confidence="draft",
        notes="draft note",
    )


def _reaction(
    mechanism_type: str,
    group_roles: list[LabeledGroupRole],
    context: ReactionContext = ReactionContext.ionic,
    reaction_id: str = "rxn",
) -> LabeledReaction:
    return LabeledReaction(
        reaction_id=reaction_id,
        reaction_smiles="CBr.[OH-]>>CO.[Br-]",
        context=context,
        mechanism_type=mechanism_type,
        split="draft",
        group_roles=group_roles,
        reaction_center_atoms=[0],
        metadata={
            "source": "phase6_5_draft",
            "needs_manual_review": True,
            "exclude_from_ground_truth_until_review": True,
        },
    )


def _roles_by_type(reaction: LabeledReaction) -> dict[str, list[Role]]:
    roles: dict[str, list[Role]] = {}
    for group_role in reaction.group_roles:
        roles.setdefault(group_role.group_type.value, []).append(group_role.role)
    return roles


def test_promote_tiny_sn2_halide_becomes_manual_leaving_group() -> None:
    reaction = _reaction(
        "sn2",
        [_role("mol0_halide_0", FunctionalGroupType.halide, Role.spectator)],
        reaction_id="sn2",
    )

    promoted, skip_reason, corrections, _ = promote_review_queue_reaction(reaction)

    assert skip_reason is None
    assert promoted is not None
    assert promoted.group_roles[0].role == Role.leaving_group
    assert promoted.group_roles[0].confidence == "manual"
    assert promoted.metadata["review_status"] == "promoted_manual_review"
    assert promoted.metadata["needs_manual_review"] is False
    assert corrections[0]["old_role"] == "spectator"
    assert corrections[0]["new_role"] == "leaving_group"


def test_carbonyl_addition_corrects_alpha_carbon_to_spectator() -> None:
    reaction = _reaction(
        "carbonyl_addition",
        [
            _role("mol0_carbonyl_0", FunctionalGroupType.carbonyl, Role.reactive_electrophile),
            _role(
                "mol0_alpha_carbon_0",
                FunctionalGroupType.alpha_carbon,
                Role.reactive_nucleophile,
            ),
        ],
        reaction_id="carbonyl_addition",
    )

    promoted, _, corrections, _ = promote_review_queue_reaction(reaction)

    assert promoted is not None
    assert _roles_by_type(promoted)["carbonyl"] == [Role.reactive_electrophile]
    assert _roles_by_type(promoted)["alpha_carbon"] == [Role.spectator]
    assert any(correction["group_type"] == "alpha_carbon" for correction in corrections)


def test_diels_alder_promotes_pi_partners_and_corrects_substituents() -> None:
    reaction = _reaction(
        "diels_alder",
        [
            _role("mol0_alkene_0", FunctionalGroupType.alkene, Role.reactive_nucleophile, 0),
            _role("mol0_alkene_1", FunctionalGroupType.alkene, Role.reactive_nucleophile, 0),
            _role("mol1_alkene_0", FunctionalGroupType.alkene, Role.reactive_electrophile, 1),
            _role("mol1_nitrile_0", FunctionalGroupType.nitrile, Role.reactive_electrophile, 1),
            _role("mol1_ester_0", FunctionalGroupType.ester, Role.reactive_electrophile, 1),
            _role("mol1_carbonyl_0", FunctionalGroupType.carbonyl, Role.reactive_electrophile, 1),
        ],
        context=ReactionContext.pericyclic,
        reaction_id="diels_alder",
    )

    promoted, _, corrections, _ = promote_review_queue_reaction(reaction)

    assert promoted is not None
    by_id = {role.group_id: role.role for role in promoted.group_roles}
    assert by_id["mol0_alkene_0"] == Role.reactive_nucleophile
    assert by_id["mol0_alkene_1"] == Role.reactive_nucleophile
    assert by_id["mol1_alkene_0"] == Role.reactive_electrophile
    assert by_id["mol1_nitrile_0"] == Role.spectator
    assert by_id["mol1_ester_0"] == Role.spectator
    assert by_id["mol1_carbonyl_0"] == Role.spectator
    assert {correction["group_type"] for correction in corrections} >= {
        "nitrile",
        "ester",
        "carbonyl",
    }


def test_ester_control_all_groups_become_spectator() -> None:
    reaction = _reaction(
        "ester_control",
        [
            _role("mol0_ester_0", FunctionalGroupType.ester, Role.reactive_electrophile),
            _role(
                "mol0_alpha_carbon_0",
                FunctionalGroupType.alpha_carbon,
                Role.reactive_nucleophile,
            ),
        ],
        reaction_id="ester_control",
    )

    promoted, _, _, _ = promote_review_queue_reaction(reaction)

    assert promoted is not None
    assert {role.role for role in promoted.group_roles} == {Role.spectator}


def test_nitrile_control_all_groups_become_spectator() -> None:
    reaction = _reaction(
        "nitrile_control",
        [
            _role("mol0_nitrile_0", FunctionalGroupType.nitrile, Role.reactive_electrophile),
            _role(
                "mol0_alpha_carbon_0",
                FunctionalGroupType.alpha_carbon,
                Role.reactive_nucleophile,
            ),
        ],
        reaction_id="nitrile_control",
    )

    promoted, _, _, _ = promote_review_queue_reaction(reaction)

    assert promoted is not None
    assert {role.role for role in promoted.group_roles} == {Role.spectator}


def test_general_control_all_groups_become_spectator() -> None:
    reaction = _reaction(
        "control",
        [
            _role("mol0_aromatic_0", FunctionalGroupType.aromatic, Role.reactive_electrophile),
            _role("mol0_benzylic_site_0", FunctionalGroupType.benzylic_site, Role.reactive_radical),
        ],
        reaction_id="general_control",
    )

    promoted, _, _, _ = promote_review_queue_reaction(reaction)

    assert promoted is not None
    assert {role.role for role in promoted.group_roles} == {Role.spectator}


def test_nitroalkane_deprotonation_promotes_alpha_and_spectator_nitro() -> None:
    reaction = _reaction(
        "nitroalkane_deprotonation",
        [
            _role("mol0_nitro_0", FunctionalGroupType.nitro, Role.reactive_electrophile),
            _role("mol0_alpha_carbon_0", FunctionalGroupType.alpha_carbon, Role.spectator),
        ],
        reaction_id="nitroalkane",
    )

    promoted, _, _, _ = promote_review_queue_reaction(reaction)

    assert promoted is not None
    assert _roles_by_type(promoted)["nitro"] == [Role.spectator]
    assert _roles_by_type(promoted)["alpha_carbon"] == [Role.reactive_nucleophile]


def test_aldol_skipped_by_default() -> None:
    reaction = _reaction(
        "aldol",
        [
            _role(
                "mol0_alpha_carbon_0",
                FunctionalGroupType.alpha_carbon,
                Role.reactive_nucleophile,
            ),
            _role("mol1_carbonyl_0", FunctionalGroupType.carbonyl, Role.reactive_electrophile, 1),
        ],
        reaction_id="aldol",
    )

    promoted, skip_reason, _, _ = promote_review_queue_reaction(reaction)

    assert promoted is None
    assert skip_reason == "aldol_skipped_by_default"


def test_include_aldol_only_promotes_if_unambiguous() -> None:
    clear = _reaction(
        "aldol",
        [
            _role(
                "mol0_alpha_carbon_0",
                FunctionalGroupType.alpha_carbon,
                Role.reactive_nucleophile,
            ),
            _role("mol1_carbonyl_0", FunctionalGroupType.carbonyl, Role.reactive_electrophile, 1),
            _role("mol1_aromatic_0", FunctionalGroupType.aromatic, Role.spectator, 1),
        ],
        reaction_id="aldol_clear",
    )
    ambiguous = _reaction(
        "aldol",
        [
            _role(
                "mol0_alpha_carbon_0",
                FunctionalGroupType.alpha_carbon,
                Role.reactive_nucleophile,
            ),
            _role("mol1_carbonyl_0", FunctionalGroupType.carbonyl, Role.reactive_electrophile, 1),
            _role("mol2_carbonyl_0", FunctionalGroupType.carbonyl, Role.reactive_electrophile, 2),
        ],
        reaction_id="aldol_ambiguous",
    )

    promoted, clear_skip, _, _ = promote_review_queue_reaction(clear, include_aldol=True)
    ambiguous_promoted, ambiguous_skip, _, _ = promote_review_queue_reaction(
        ambiguous,
        include_aldol=True,
    )

    assert promoted is not None
    assert clear_skip is None
    assert ambiguous_promoted is None
    assert ambiguous_skip == "aldol_ambiguous_donor_acceptor"


def test_output_validates_after_merge() -> None:
    base = [
        _reaction(
            "sn2",
            [_role("base_halide", FunctionalGroupType.halide, Role.leaving_group)],
            reaction_id="base",
        )
    ]
    base[0].split = "train"
    base[0].metadata["needs_manual_review"] = False
    promoted, skipped, corrections, warnings = promote_review_queue(
        [
            _reaction(
                "sn2",
                [_role("new_halide", FunctionalGroupType.halide, Role.spectator)],
                reaction_id="new",
            )
        ]
    )

    merged, merge_warnings = merge_promoted_with_base(base, promoted)
    report = build_promotion_report(
        input_reactions=base + promoted,
        promoted_reactions=promoted,
        skipped_reactions=skipped,
        corrected_labels=corrections,
        warnings=warnings + merge_warnings,
        output_paths={"output": "out.json"},
    )

    assert validate_labeled_dataset(promoted)
    assert validate_labeled_dataset(merged)
    assert report["skipped_reactions"] == []
    assert report["corrected_labels"]
    assert report["n_corrected_labels"] == len(corrections)


def test_report_contains_skipped_reactions_and_corrected_labels() -> None:
    promoted, skipped, corrections, warnings = promote_review_queue(
        [
            _reaction(
                "sn2",
                [_role("halide", FunctionalGroupType.halide, Role.spectator)],
                reaction_id="promote",
            ),
            _reaction(
                "aldol",
                [
                    _role(
                        "alpha",
                        FunctionalGroupType.alpha_carbon,
                        Role.reactive_nucleophile,
                    )
                ],
                reaction_id="skip",
            ),
        ]
    )

    report = build_promotion_report(promoted + [], promoted, skipped, corrections, warnings, {})

    assert report["skipped_reactions"]
    assert report["corrected_labels"]


def test_cli_dry_run_does_not_write_output_files(tmp_path: Path) -> None:
    input_path = tmp_path / "review.json"
    base_path = tmp_path / "base.json"
    output_path = tmp_path / "promoted.json"
    merged_path = tmp_path / "merged.json"
    report_path = tmp_path / "report.json"
    payload = {"reactions": [_reaction(
        "sn2",
        [_role("mol0_halide_0", FunctionalGroupType.halide, Role.spectator)],
        reaction_id="cli_sn2",
    ).to_dict()]}
    input_path.write_text(json.dumps(payload), encoding="utf-8")
    base_path.write_text(json.dumps({"reactions": []}), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--input",
            str(input_path),
            "--base",
            str(base_path),
            "--output",
            str(output_path),
            "--merged-output",
            str(merged_path),
            "--report",
            str(report_path),
            "--dry-run",
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert not output_path.exists()
    assert not merged_path.exists()
    assert not report_path.exists()


def test_cli_writes_valid_outputs(tmp_path: Path) -> None:
    input_path = tmp_path / "review.json"
    base_path = tmp_path / "base.json"
    output_path = tmp_path / "promoted.json"
    merged_path = tmp_path / "merged.json"
    report_path = tmp_path / "report.json"
    payload = {"reactions": [_reaction(
        "sn2",
        [_role("mol0_halide_0", FunctionalGroupType.halide, Role.spectator)],
        reaction_id="cli_sn2",
    ).to_dict()]}
    input_path.write_text(json.dumps(payload), encoding="utf-8")
    base_path.write_text(json.dumps({"reactions": []}), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--input",
            str(input_path),
            "--base",
            str(base_path),
            "--output",
            str(output_path),
            "--merged-output",
            str(merged_path),
            "--report",
            str(report_path),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert validate_labeled_dataset(load_labeled_reactions(output_path))
    assert validate_labeled_dataset(load_labeled_reactions(merged_path))
    assert json.loads(report_path.read_text(encoding="utf-8"))["n_promoted_reactions"] == 1


def test_no_mlp_training_invoked() -> None:
    text = (_ROOT / "mendel" / "promotion.py").read_text(encoding="utf-8")
    if _SCRIPT.exists():
        text += _SCRIPT.read_text(encoding="utf-8")
    assert "train_mlp" not in text
    assert "mendel.mlp" not in text
