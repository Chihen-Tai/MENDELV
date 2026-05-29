"""Phase 7 MLP role predictor tests."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.phase7, pytest.mark.ml]

torch = pytest.importorskip("torch")

from mendel.descriptor import build_descriptors
from mendel.identifier import identify_functional_groups
from mendel.labels import load_labeled_reactions
from mendel.mlp import (
    DEFAULT_MODEL_VERSION,
    INDEX_TO_ROLE,
    ROLE_TO_INDEX,
    MLPRolePrediction,
    MLPRolePredictor,
    RoleMLP,
    TrainingConfig,
    TrainingHistory,
    build_training_examples,
    evaluate_mlp_predictor,
    stratified_train_val_split,
    summarize_training_examples,
    train_mlp_role_predictor,
    training_examples_to_tensors,
)
from mendel.parser import parse_reaction_smiles
from mendel.types import ReactionContext, Role

_MINIMAL = Path(__file__).parent.parent / "data" / "reactions.minimal.json"
_SCRIPT = Path(__file__).parent.parent / "scripts" / "train_mlp.py"

_TINY_CFG = TrainingConfig(
    hidden_dim=8,
    epochs=3,
    batch_size=4,
    learning_rate=1e-2,
    seed=0,
    early_stopping_patience=100,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def minimal_examples():
    reactions = load_labeled_reactions(_MINIMAL)
    return build_training_examples(reactions, strict_group_matching=False)


@pytest.fixture(scope="module")
def tiny_predictor_and_history(minimal_examples):
    return train_mlp_role_predictor(minimal_examples, config=_TINY_CFG)


# ---------------------------------------------------------------------------
# 1. Role index mapping
# ---------------------------------------------------------------------------


def test_role_to_index_has_five_entries() -> None:
    assert len(ROLE_TO_INDEX) == 5
    assert set(ROLE_TO_INDEX.keys()) == set(Role)


def test_index_to_role_reverses_mapping() -> None:
    assert len(INDEX_TO_ROLE) == 5
    for role, idx in ROLE_TO_INDEX.items():
        assert INDEX_TO_ROLE[idx] is role


def test_role_indices_are_zero_to_four() -> None:
    assert sorted(INDEX_TO_ROLE.keys()) == list(range(5))


# ---------------------------------------------------------------------------
# 2. RoleMLP forward shape
# ---------------------------------------------------------------------------


def test_rolemlp_forward_shape() -> None:
    model = RoleMLP(input_dim=10)
    x = torch.zeros(4, 10)
    out = model(x)
    assert out.shape == (4, 5), f"Expected (4, 5), got {out.shape}"


def test_rolemlp_stores_dims() -> None:
    model = RoleMLP(input_dim=55, hidden_dim=16)
    assert model.input_dim == 55
    assert model.hidden_dim == 16
    assert model.output_dim == 5


# ---------------------------------------------------------------------------
# 3. Build training examples
# ---------------------------------------------------------------------------


def test_build_training_examples_returns_nonempty(minimal_examples) -> None:
    assert len(minimal_examples) > 0


def test_training_example_has_features_and_role(minimal_examples) -> None:
    for ex in minimal_examples:
        assert len(ex.features) > 0
        assert isinstance(ex.role, Role)


def test_training_example_to_dict(minimal_examples) -> None:
    d = minimal_examples[0].to_dict()
    assert "reaction_id" in d
    assert "features" in d
    assert "role" in d
    assert isinstance(d["features"], list)


def test_training_summary_contains_imbalance_metadata(minimal_examples) -> None:
    summary = summarize_training_examples(minimal_examples)
    metadata = summary.metadata
    assert "min_role_count" in metadata
    assert "max_role_count" in metadata
    assert "missing_roles" in metadata
    assert "roles_below_10" in metadata


def test_stratified_split_returns_disjoint_indices(minimal_examples) -> None:
    train_idx, val_idx = stratified_train_val_split(
        minimal_examples,
        validation_split=0.5,
        seed=0,
    )
    assert set(train_idx).isdisjoint(set(val_idx))
    assert sorted(train_idx + val_idx) == list(range(len(minimal_examples)))


# ---------------------------------------------------------------------------
# 4. Tensor conversion
# ---------------------------------------------------------------------------


def test_training_examples_to_tensors(minimal_examples) -> None:
    X, y, group_ids = training_examples_to_tensors(minimal_examples)
    n = len(minimal_examples)
    assert X.shape[0] == n
    assert y.shape[0] == n
    assert X.shape[1] > 0
    assert len(group_ids) == n


def test_tensor_y_values_are_valid_indices(minimal_examples) -> None:
    _, y, _ = training_examples_to_tensors(minimal_examples)
    assert y.dtype == torch.long
    assert int(y.min().item()) >= 0
    assert int(y.max().item()) <= 4


def test_empty_examples_return_zero_tensors() -> None:
    X, y, gids = training_examples_to_tensors([])
    assert X.shape[0] == 0
    assert y.shape[0] == 0
    assert gids == []


# ---------------------------------------------------------------------------
# 5. Train tiny model
# ---------------------------------------------------------------------------


def test_train_returns_predictor_and_history(tiny_predictor_and_history) -> None:
    predictor, history = tiny_predictor_and_history
    assert isinstance(predictor, MLPRolePredictor)
    assert isinstance(history, TrainingHistory)


def test_training_history_has_train_loss(tiny_predictor_and_history) -> None:
    _, history = tiny_predictor_and_history
    assert len(history.train_loss) > 0


def test_training_history_lengths_match(tiny_predictor_and_history) -> None:
    _, history = tiny_predictor_and_history
    n = len(history.train_loss)
    assert len(history.val_loss) == n
    assert len(history.train_accuracy) == n
    assert len(history.val_accuracy) == n


def test_training_history_to_dict(tiny_predictor_and_history) -> None:
    _, history = tiny_predictor_and_history
    d = history.to_dict()
    assert "train_loss" in d
    assert "val_loss" in d
    assert isinstance(d["train_loss"], list)


def test_training_history_records_dataset_warnings(tiny_predictor_and_history) -> None:
    _, history = tiny_predictor_and_history
    metadata = history.metadata
    assert "dataset_warnings" in metadata
    assert "role_counts" in metadata
    assert metadata["n_examples"] > 0


# ---------------------------------------------------------------------------
# 6. Predict descriptor
# ---------------------------------------------------------------------------


def test_predict_descriptor_returns_prediction(tiny_predictor_and_history) -> None:
    predictor, _ = tiny_predictor_and_history
    rxn = parse_reaction_smiles("CBr.[OH-]>>CO.[Br-]", context=ReactionContext.ionic)
    groups = identify_functional_groups(rxn)
    assert groups, "No functional groups identified."
    descs = build_descriptors(rxn, groups)
    pred = predictor.predict_descriptor(descs[0])
    assert isinstance(pred, MLPRolePrediction)


def test_predicted_role_is_valid(tiny_predictor_and_history) -> None:
    predictor, _ = tiny_predictor_and_history
    rxn = parse_reaction_smiles("CBr.[OH-]>>CO.[Br-]", context=ReactionContext.ionic)
    groups = identify_functional_groups(rxn)
    descs = build_descriptors(rxn, groups)
    pred = predictor.predict_descriptor(descs[0])
    assert pred.predicted_role in set(Role)


def test_confidence_in_unit_interval(tiny_predictor_and_history) -> None:
    predictor, _ = tiny_predictor_and_history
    rxn = parse_reaction_smiles("CBr.[OH-]>>CO.[Br-]", context=ReactionContext.ionic)
    groups = identify_functional_groups(rxn)
    descs = build_descriptors(rxn, groups)
    pred = predictor.predict_descriptor(descs[0])
    assert 0.0 <= pred.confidence <= 1.0


def test_probabilities_sum_to_one(tiny_predictor_and_history) -> None:
    predictor, _ = tiny_predictor_and_history
    rxn = parse_reaction_smiles("CBr.[OH-]>>CO.[Br-]", context=ReactionContext.ionic)
    groups = identify_functional_groups(rxn)
    descs = build_descriptors(rxn, groups)
    pred = predictor.predict_descriptor(descs[0])
    total = sum(pred.probabilities.values())
    assert abs(total - 1.0) < 1e-5, f"Probabilities sum to {total}"


# ---------------------------------------------------------------------------
# 7. Save and load
# ---------------------------------------------------------------------------


def test_save_creates_file(tmp_path, tiny_predictor_and_history) -> None:
    predictor, _ = tiny_predictor_and_history
    ckpt = tmp_path / "model.pt"
    predictor.save(ckpt)
    assert ckpt.exists()


def test_load_returns_predictor(tmp_path, tiny_predictor_and_history) -> None:
    predictor, _ = tiny_predictor_and_history
    ckpt = tmp_path / "load_test.pt"
    predictor.save(ckpt)
    loaded = MLPRolePredictor.load(ckpt)
    assert isinstance(loaded, MLPRolePredictor)
    assert loaded.model_version == DEFAULT_MODEL_VERSION
    assert loaded.feature_names == predictor.feature_names


def test_loaded_predictor_can_predict(tmp_path, tiny_predictor_and_history) -> None:
    predictor, _ = tiny_predictor_and_history
    ckpt = tmp_path / "reload.pt"
    predictor.save(ckpt)
    loaded = MLPRolePredictor.load(ckpt)

    rxn = parse_reaction_smiles("CBr.[OH-]>>CO.[Br-]", context=ReactionContext.ionic)
    groups = identify_functional_groups(rxn)
    descs = build_descriptors(rxn, groups)
    pred = loaded.predict_descriptor(descs[0])
    assert pred.predicted_role in set(Role)


# ---------------------------------------------------------------------------
# 8. Evaluate
# ---------------------------------------------------------------------------


def test_evaluate_returns_required_keys(tiny_predictor_and_history, minimal_examples) -> None:
    predictor, _ = tiny_predictor_and_history
    report = evaluate_mlp_predictor(predictor, minimal_examples)
    assert "n_examples" in report
    assert "accuracy" in report
    assert "confusion_matrix" in report
    assert "mismatches" in report


def test_evaluate_n_examples_matches(tiny_predictor_and_history, minimal_examples) -> None:
    predictor, _ = tiny_predictor_and_history
    report = evaluate_mlp_predictor(predictor, minimal_examples)
    assert report["n_examples"] == len(minimal_examples)


def test_evaluate_accuracy_in_unit_interval(tiny_predictor_and_history, minimal_examples) -> None:
    predictor, _ = tiny_predictor_and_history
    report = evaluate_mlp_predictor(predictor, minimal_examples)
    assert 0.0 <= report["accuracy"] <= 1.0


def test_evaluate_confusion_matrix_structure(tiny_predictor_and_history, minimal_examples) -> None:
    predictor, _ = tiny_predictor_and_history
    report = evaluate_mlp_predictor(predictor, minimal_examples)
    cm = report["confusion_matrix"]
    role_vals = {r.value for r in Role}
    assert set(cm.keys()) == role_vals
    for row in cm.values():
        assert set(row.keys()) == role_vals


def test_evaluate_empty_examples(tiny_predictor_and_history) -> None:
    predictor, _ = tiny_predictor_and_history
    report = evaluate_mlp_predictor(predictor, [])
    assert report["n_examples"] == 0
    assert report["accuracy"] == 0.0


# ---------------------------------------------------------------------------
# 9. CLI smoke test
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _SCRIPT.exists(), reason="scripts/train_mlp.py not found")
def test_cli_smoke(tmp_path) -> None:
    ckpt = tmp_path / "smoke_model.pt"
    rep = tmp_path / "smoke_report.json"

    result = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--data", str(_MINIMAL),
            "--output", str(ckpt),
            "--report", str(rep),
            "--epochs", "2",
            "--hidden-dim", "8",
            "--batch-size", "4",
            "--no-strict-group-matching",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"CLI exited with {result.returncode}.\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert ckpt.exists(), "Checkpoint not created."
    assert rep.exists(), "Report not created."
