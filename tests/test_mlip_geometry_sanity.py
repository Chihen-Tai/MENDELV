"""Tests for Phase 9.2 MLIP geometry sanity guardrails."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from mendel.mlip import (
    GeometrySanityReport,
    MLIPResult,
    attach_geometry_sanity_to_mlip_result,
    check_geometry_sanity,
    compute_interatomic_distances,
    detect_reaction_smiles_charge_risk,
)

_ROOT = Path(__file__).parent.parent
_SCRIPT = _ROOT / "scripts" / "mlip_singlepoint.py"


class FakeAtoms:
    def __init__(self, positions: list[list[float]]) -> None:
        self._positions = positions

    def __len__(self) -> int:
        return len(self._positions)

    def get_positions(self) -> list[list[float]]:
        return self._positions


def test_geometry_sanity_report_serializes() -> None:
    report = GeometrySanityReport(
        n_atoms=2,
        min_interatomic_distance=0.5,
        min_distance_atom_pair=[0, 1],
        max_interatomic_distance=0.5,
        n_fragments=None,
        total_formal_charge=None,
        has_disconnected_fragments=False,
        has_charged_fragments=False,
        has_disconnected_charged_fragments=False,
        mean_force_norm=None,
        max_force_norm=None,
        mean_force_threshold=100.0,
        max_force_threshold=1000.0,
        min_distance_threshold=0.6,
        status="fail",
        warnings=["too close"],
        metadata={"phase": "9.2"},
    )

    assert report.to_dict()["status"] == "fail"
    assert report.to_dict()["min_distance_atom_pair"] == [0, 1]


def test_compute_interatomic_distances_detects_short_pair() -> None:
    result = compute_interatomic_distances(FakeAtoms([[0.0, 0.0, 0.0], [0.4, 0.0, 0.0]]))

    assert result["n_atoms"] == 2
    assert result["min_interatomic_distance"] == 0.4
    assert result["min_distance_atom_pair"] == [0, 1]


def test_check_geometry_sanity_detects_huge_mean_force_norm() -> None:
    report = check_geometry_sanity(
        FakeAtoms([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        forces=[[200.0, 0.0, 0.0], [200.0, 0.0, 0.0]],
        mean_force_threshold=100.0,
    )

    assert report.status == "fail"
    assert report.mean_force_norm == 200.0
    assert any("mean force norm" in warning for warning in report.warnings)


def test_check_geometry_sanity_detects_huge_max_force_norm() -> None:
    report = check_geometry_sanity(
        FakeAtoms([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        forces=[[10.0, 0.0, 0.0], [1500.0, 0.0, 0.0]],
        max_force_threshold=1000.0,
    )

    assert report.status == "fail"
    assert report.max_force_norm == 1500.0
    assert any("max force norm" in warning for warning in report.warnings)


def test_detect_reaction_smiles_charge_risk_flags_disconnected_charged_reactants() -> None:
    risk = detect_reaction_smiles_charge_risk("CBr.[OH-]>>CO.[Br-]")

    assert risk["has_disconnected_charged_reactants"] is True
    assert "[OH-]" in risk["charged_reactant_fragments"]


def test_detect_reaction_smiles_charge_risk_does_not_flag_neutral_molecule() -> None:
    risk = detect_reaction_smiles_charge_risk("CC(=O)C")

    assert risk["has_disconnected_charged_reactants"] is False
    assert risk["has_charged_reactants"] is False


def test_check_geometry_sanity_warns_for_disconnected_charged_reaction_smiles() -> None:
    report = check_geometry_sanity(
        FakeAtoms([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        reaction_smiles="CBr.[OH-]>>CO.[Br-]",
    )

    assert report.status == "warning"
    assert report.has_disconnected_charged_fragments is True
    assert any("Disconnected charged reaction SMILES" in warning for warning in report.warnings)


def test_attach_geometry_sanity_preserves_success_and_adds_warnings() -> None:
    result = MLIPResult(
        energy=1.0,
        energy_unit="eV",
        forces=[[1.0, 0.0, 0.0]],
        force_unit="eV/Angstrom",
        n_atoms=1,
        backend_name="mace",
        model_name="small",
        device="cpu",
        success=True,
        warnings=[],
        metadata={},
    )
    report = check_geometry_sanity(
        FakeAtoms([[0.0, 0.0, 0.0]]),
        forces=[[200.0, 0.0, 0.0]],
        mean_force_threshold=100.0,
    )

    updated = attach_geometry_sanity_to_mlip_result(result, report)

    assert updated.success is True
    assert updated.metadata["geometry_sanity_status"] == "fail"
    assert updated.metadata["geometry_sanity_failed"] is True
    assert updated.warnings


def test_cli_help_includes_geometry_sanity_options() -> None:
    result = subprocess.run(
        [sys.executable, str(_SCRIPT), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--fail-on-geometry-sanity" in result.stdout
    assert "--mean-force-threshold" in result.stdout


def test_no_training_invoked() -> None:
    text = _SCRIPT.read_text(encoding="utf-8").lower()
    for token in ("train_mlp", "fit(", "neb", "irc", "transition1x"):
        assert token not in text
