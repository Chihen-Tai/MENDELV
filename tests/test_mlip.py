"""Tests for optional Phase 9 MLIP single-point utilities."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from mendel.mlip import (
    ANI2X_SUPPORTED_ELEMENTS,
    GeometrySanityReport,
    MENDELVGuidedMLIPResult,
    MLIPConfig,
    MLIPResult,
    ReactionCenterForceSummary,
    attach_geometry_sanity_to_mlip_result,
    compute_force_norms,
    diagnose_ani2x,
    diagnose_mace_calculators,
    normalize_mace_model_config,
    optional_import_ase,
    optional_import_mace,
    optional_import_torchani,
    resolve_device,
    smiles_to_ase_atoms,
    validate_ani2x_elements,
    summarize_reaction_center_forces,
)

_ROOT = Path(__file__).parent.parent
_SCRIPT = _ROOT / "scripts" / "mlip_singlepoint.py"


def test_mlip_config_serializes() -> None:
    config = MLIPConfig(device="cpu", metadata={"phase": 9})

    assert config.to_dict()["backend_name"] == "mace"
    assert config.to_dict()["model_family"] == "mace-off"
    assert config.to_dict()["model_name"] == "mace-off-small"
    assert config.to_dict()["metadata"]["phase"] == 9


def test_normalize_mace_model_config_accepts_aliases() -> None:
    assert normalize_mace_model_config(MLIPConfig(model_name="mace-off-small")) == {
        "model_family": "mace-off",
        "model_name": "small",
        "mace_off_model": "small",
        "model_path": "",
        "is_local_path": False,
    }
    assert normalize_mace_model_config(MLIPConfig(model_name="mace-off-medium"))[
        "model_name"
    ] == "medium"
    assert normalize_mace_model_config(MLIPConfig(model_name="small"))["model_family"] == "mace-off"


def test_normalize_mace_model_config_supports_local_model_path(tmp_path: Path) -> None:
    model_path = tmp_path / "local.model"
    normalized = normalize_mace_model_config(MLIPConfig(model_name=str(model_path)))

    assert normalized["model_family"] == "local"
    assert normalized["is_local_path"] is True
    assert normalized["model_path"].endswith("local.model")


def test_create_mlip_calculator_prefers_mace_off(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, Any] = {}

    def fake_mace_off(**kwargs: Any) -> dict[str, Any]:
        calls["mace_off"] = kwargs
        return {"calculator": "mace_off", "kwargs": kwargs}

    def fake_mace_mp(**kwargs: Any) -> dict[str, Any]:
        calls["mace_mp"] = kwargs
        return {"calculator": "mace_mp", "kwargs": kwargs}

    import mendel.mlip as mlip

    monkeypatch.setattr(
        mlip,
        "optional_import_mace",
        lambda: {
            "module": None,
            "mace_off": fake_mace_off,
            "mace_mp": fake_mace_mp,
            "MACECalculator": None,
        },
    )

    calculator = mlip.create_mlip_calculator(MLIPConfig(model_name="mace-off-small", device="cpu"))

    assert calculator["calculator"] == "mace_off"
    assert calls["mace_off"]["model"] == "small"
    assert calls["mace_off"]["device"] == "cpu"
    assert "mace_mp" not in calls


def test_create_mlip_calculator_uses_local_model_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: dict[str, Any] = {}

    class FakeMACECalculator:
        def __init__(self, **kwargs: Any) -> None:
            calls["MACECalculator"] = kwargs

    import mendel.mlip as mlip

    monkeypatch.setattr(
        mlip,
        "optional_import_mace",
        lambda: {
            "module": None,
            "mace_off": None,
            "mace_mp": None,
            "MACECalculator": FakeMACECalculator,
        },
    )
    model_path = tmp_path / "model.pt"

    mlip.create_mlip_calculator(MLIPConfig(model_name=str(model_path), device="cpu"))

    assert calls["MACECalculator"]["model_paths"].endswith("model.pt")


def test_mlip_result_serializes() -> None:
    result = MLIPResult(
        energy=-1.0,
        energy_unit="eV",
        forces=[[1.0, 0.0, 0.0]],
        force_unit="eV/Angstrom",
        n_atoms=1,
        backend_name="mace",
        model_name="test",
        device="cpu",
        success=True,
        warnings=["approximate"],
        metadata={},
    )

    assert result.to_dict()["energy"] == -1.0
    assert result.to_dict()["forces"] == [[1.0, 0.0, 0.0]]


def test_mlip_result_serializes_geometry_sanity_metadata() -> None:
    result = MLIPResult(
        energy=-1.0,
        energy_unit="eV",
        forces=[[1.0, 0.0, 0.0]],
        force_unit="eV/Angstrom",
        n_atoms=1,
        backend_name="mace",
        model_name="test",
        device="cpu",
        success=True,
        warnings=[],
        metadata={},
    )
    report = GeometrySanityReport(
        n_atoms=1,
        min_interatomic_distance=None,
        min_distance_atom_pair=None,
        max_interatomic_distance=None,
        n_fragments=None,
        total_formal_charge=None,
        has_disconnected_fragments=False,
        has_charged_fragments=False,
        has_disconnected_charged_fragments=False,
        mean_force_norm=1.0,
        max_force_norm=1.0,
        mean_force_threshold=100.0,
        max_force_threshold=1000.0,
        min_distance_threshold=0.6,
        status="pass",
        warnings=[],
        metadata={},
    )

    payload = attach_geometry_sanity_to_mlip_result(result, report).to_dict()

    assert payload["metadata"]["geometry_sanity_status"] == "pass"
    assert payload["metadata"]["geometry_sanity"]["n_atoms"] == 1


def test_force_summary_serializes() -> None:
    summary = ReactionCenterForceSummary(
        reaction_center_atoms=[0],
        n_center_atoms=1,
        mean_center_force_norm=1.0,
        max_center_force_norm=1.0,
        mean_all_atom_force_norm=0.5,
        max_all_atom_force_norm=1.0,
        center_to_all_mean_force_ratio=2.0,
        metadata={"x": True},
    )

    assert summary.to_dict()["center_to_all_mean_force_ratio"] == 2.0


def test_guided_result_serializes() -> None:
    mlip_result = MLIPResult(None, "eV", None, "eV/Angstrom", 0, "mace", "m", "cpu", False, [], {})
    force_summary = ReactionCenterForceSummary([], 0, None, None, None, None, None, {})
    result = MENDELVGuidedMLIPResult(
        reaction_smiles="C>>C",
        context="unknown",
        mechanism_hint=None,
        role_assignments=[],
        reaction_center_atoms=[],
        center_source="negotiated",
        mlip_result=mlip_result,
        force_summary=force_summary,
        warnings=["single-point only"],
        metadata={},
    )

    assert result.to_dict()["center_source"] == "negotiated"


def test_compute_force_norms() -> None:
    assert compute_force_norms([[3.0, 4.0, 0.0], [0.0, 0.0, 0.0]]) == [5.0, 0.0]


def test_summarize_reaction_center_forces() -> None:
    result = MLIPResult(
        energy=0.0,
        energy_unit="eV",
        forces=[[3.0, 4.0, 0.0], [0.0, 0.0, 2.0]],
        force_unit="eV/Angstrom",
        n_atoms=2,
        backend_name="mace",
        model_name="test",
        device="cpu",
        success=True,
        warnings=[],
        metadata={},
    )

    summary = summarize_reaction_center_forces(result, [0])

    assert summary.n_center_atoms == 1
    assert summary.mean_center_force_norm == 5.0
    assert summary.mean_all_atom_force_norm == 3.5


def test_summarize_handles_empty_center() -> None:
    result = MLIPResult(
        0.0, "eV", [[1.0, 0.0, 0.0]], "eV/Angstrom", 1, "mace", "m", "cpu", True, [], {}
    )

    summary = summarize_reaction_center_forces(result, [])

    assert summary.n_center_atoms == 0
    assert summary.mean_center_force_norm is None
    assert "empty_reaction_center" in summary.metadata


def test_summarize_handles_out_of_range_center_atom() -> None:
    result = MLIPResult(
        0.0, "eV", [[1.0, 0.0, 0.0]], "eV/Angstrom", 1, "mace", "m", "cpu", True, [], {}
    )

    summary = summarize_reaction_center_forces(result, [9])

    assert summary.n_center_atoms == 0
    assert summary.metadata["out_of_range_center_atoms"] == "9"


def test_optional_import_ase_clear_error_if_missing() -> None:
    try:
        optional_import_ase()
    except ImportError as exc:
        assert "pip install -e '.[mlip]'" in str(exc)


def test_optional_import_mace_clear_error_if_missing() -> None:
    try:
        optional_import_mace()
    except ImportError as exc:
        assert "pip install -e '.[mlip]'" in str(exc)


def test_create_mlip_calculator_missing_mace_clear_error() -> None:
    import mendel.mlip as mlip

    try:
        mlip.create_mlip_calculator(MLIPConfig(model_name="mace-off-small", device="cpu"))
    except ImportError as exc:
        assert "mace-torch is required" in str(exc)
        assert "pip install -e '.[mlip]'" in str(exc)


def test_resolve_device_returns_supported_value() -> None:
    assert resolve_device("auto") in {"cpu", "cuda", "mps"}
    assert resolve_device("cpu") == "cpu"


def test_cli_help_runs() -> None:
    result = subprocess.run(
        [sys.executable, str(_SCRIPT), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "single-point" in result.stdout.lower()


def test_mace_diagnostic_is_safe_without_mace() -> None:
    report = diagnose_mace_calculators()

    assert "installed" in report
    assert "calculators" in report


def test_no_training_invoked() -> None:
    text = _SCRIPT.read_text(encoding="utf-8").lower()
    for token in ("train_mlp", "fit(", "neb", "irc", "transition1x"):
        assert token not in text


@pytest.mark.mlip
@pytest.mark.optional
def test_smiles_to_ase_atoms_if_ase_installed() -> None:
    pytest.importorskip("ase")

    atoms = smiles_to_ase_atoms("CC(=O)C", seed=1)

    assert len(atoms) > 0


# ---------------------------------------------------------------------------
# ANI-2x backend tests
# ---------------------------------------------------------------------------


def test_ani2x_supported_elements_constant() -> None:
    assert "H" in ANI2X_SUPPORTED_ELEMENTS
    assert "C" in ANI2X_SUPPORTED_ELEMENTS
    assert "Br" not in ANI2X_SUPPORTED_ELEMENTS
    assert "I" not in ANI2X_SUPPORTED_ELEMENTS
    assert len(ANI2X_SUPPORTED_ELEMENTS) == 7


def test_optional_import_torchani_missing_gives_clear_error() -> None:
    import importlib
    import sys

    saved = sys.modules.pop("torchani", None)
    try:
        with pytest.raises(ImportError, match="torchani is required"):
            optional_import_torchani()
    finally:
        if saved is not None:
            sys.modules["torchani"] = saved
        else:
            importlib.invalidate_caches()


def test_diagnose_ani2x_safe_without_torchani() -> None:
    report = diagnose_ani2x()

    assert "installed" in report


def test_validate_ani2x_elements_detects_unsupported() -> None:
    class FakeAtoms:
        def get_chemical_symbols(self) -> list[str]:
            return ["C", "H", "Br", "I"]

    unsupported = validate_ani2x_elements(FakeAtoms())

    assert "Br" in unsupported
    assert "I" in unsupported
    assert "C" not in unsupported
    assert "H" not in unsupported


def test_validate_ani2x_elements_all_supported() -> None:
    class FakeAtoms:
        def get_chemical_symbols(self) -> list[str]:
            return ["C", "H", "N", "O", "F", "S", "Cl"]

    assert validate_ani2x_elements(FakeAtoms()) == []


def test_create_mlip_calculator_dispatches_ani2x(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, Any] = {}

    class FakeModel:
        def to(self, device: str) -> "FakeModel":
            calls["device"] = device
            return self

        def ase(self) -> dict[str, str]:
            calls["ase_called"] = True
            return {"calculator": "ani2x"}

    class FakeTorchANI:
        class models:
            @staticmethod
            def ANI2x(periodic_table_index: bool = False) -> "FakeModel":
                calls["periodic_table_index"] = periodic_table_index
                return FakeModel()

    import mendel.mlip as mlip

    monkeypatch.setattr(mlip, "optional_import_torchani", lambda: {"torchani": FakeTorchANI})

    result = mlip.create_mlip_calculator(MLIPConfig(backend_name="ani2x", device="cpu"))

    assert result == {"calculator": "ani2x"}
    assert calls["periodic_table_index"] is True
    assert calls["ase_called"] is True
    assert calls["device"] == "cpu"


def test_create_mlip_calculator_ani2x_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    import mendel.mlip as mlip

    dispatched: list[str] = []

    def fake_create(config: MLIPConfig) -> str:
        dispatched.append(config.backend_name)
        return "fake_calc"

    monkeypatch.setattr(mlip, "create_ani2x_calculator", fake_create)

    mlip.create_mlip_calculator(MLIPConfig(backend_name="ani-2x", device="cpu"))
    mlip.create_mlip_calculator(MLIPConfig(backend_name="ani", device="cpu"))

    assert len(dispatched) == 2


def test_create_mlip_calculator_rejects_unknown_backend() -> None:
    import mendel.mlip as mlip

    with pytest.raises(ValueError, match="Unsupported MLIP backend"):
        mlip.create_mlip_calculator(MLIPConfig(backend_name="gfn2xtb", device="cpu"))


def test_compute_mlip_singlepoint_warns_unsupported_ani2x_elements(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mendel.mlip as mlip

    class FakeAtoms:
        def get_chemical_symbols(self) -> list[str]:
            return ["C", "Br"]

        def get_potential_energy(self) -> float:
            return -1.0

        def get_forces(self) -> Any:
            import types

            obj = types.SimpleNamespace()
            obj.tolist = lambda: [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
            return obj

        def __len__(self) -> int:
            return 2

    monkeypatch.setattr(mlip, "create_mlip_calculator", lambda cfg: "fake_calc")
    monkeypatch.setattr(mlip, "check_geometry_sanity", lambda *a, **kw: mlip.GeometrySanityReport(
        n_atoms=2, min_interatomic_distance=1.5, min_distance_atom_pair=[0, 1],
        max_interatomic_distance=1.5, n_fragments=1, total_formal_charge=0,
        has_disconnected_fragments=False, has_charged_fragments=False,
        has_disconnected_charged_fragments=False, mean_force_norm=0.0,
        max_force_norm=0.0, mean_force_threshold=100.0, max_force_threshold=1000.0,
        min_distance_threshold=0.6, status="pass", warnings=[],
    ))

    fake_atoms = FakeAtoms()
    fake_atoms.calc = None  # type: ignore[attr-defined]

    config = MLIPConfig(backend_name="ani2x", device="cpu")
    result = mlip.compute_mlip_singlepoint(fake_atoms, config)  # type: ignore[arg-type]

    assert any("Br" in w for w in result.warnings)
