"""Tests for Phase 6.5: dataset curation and draft-label generation."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from mendel.curation import (
    DraftLabelConfig,
    DraftReactionInput,
    create_core_draft_inputs,
    create_extended_draft_inputs,
    draft_labeled_reaction,
    draft_labeled_reactions,
    load_draft_inputs,
    merge_labeled_reactions,
    save_draft_inputs,
    save_draft_labeled_reactions,
    summarize_draft_labels,
)
from mendel.labels import LabeledReaction, load_labeled_reactions
from mendel.types import ReactionContext

_SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"

_SIMPLE_SN2 = DraftReactionInput(
    reaction_id="test_sn2_simple",
    reaction_smiles="[CH3:1][Br:2].[OH-:3]>>[CH3:1][OH:3].[Br-:2]",
    context=ReactionContext.ionic,
    mechanism_type="SN2",
)

_ALDOL_SIMPLE = DraftReactionInput(
    reaction_id="test_aldol_simple",
    reaction_smiles="CC(=O)C.CBr>>CC(=O)CO",
    context=ReactionContext.ionic,
    mechanism_type="ionic_addition_or_aldol_like",
)


# ---------------------------------------------------------------------------
# A. Draft input creation
# ---------------------------------------------------------------------------


def test_create_core_draft_inputs_count() -> None:
    inputs = create_core_draft_inputs()
    assert len(inputs) >= 5


def test_create_extended_draft_inputs_count() -> None:
    inputs = create_extended_draft_inputs()
    assert len(inputs) >= 5


def test_core_draft_inputs_required_fields() -> None:
    for inp in create_core_draft_inputs():
        assert inp.reaction_id
        assert ">>" in inp.reaction_smiles
        assert isinstance(inp.context, ReactionContext)
        assert inp.mechanism_type


def test_extended_draft_inputs_required_fields() -> None:
    for inp in create_extended_draft_inputs():
        assert inp.reaction_id
        assert ">>" in inp.reaction_smiles
        assert isinstance(inp.context, ReactionContext)
        assert inp.mechanism_type


def test_core_draft_inputs_unique_ids() -> None:
    inputs = create_core_draft_inputs()
    ids = [inp.reaction_id for inp in inputs]
    assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# B. Single draft label
# ---------------------------------------------------------------------------


def test_draft_labeled_reaction_returns_labeled_reaction() -> None:
    result = draft_labeled_reaction(_ALDOL_SIMPLE)
    assert isinstance(result, LabeledReaction)


def test_draft_labeled_reaction_needs_manual_review() -> None:
    result = draft_labeled_reaction(_ALDOL_SIMPLE)
    assert result.metadata.get("needs_manual_review") is True


def test_draft_labeled_reaction_group_roles_is_list() -> None:
    result = draft_labeled_reaction(_ALDOL_SIMPLE)
    assert isinstance(result.group_roles, list)


def test_draft_labeled_reaction_preserves_smiles() -> None:
    result = draft_labeled_reaction(_ALDOL_SIMPLE)
    assert result.reaction_smiles == _ALDOL_SIMPLE.reaction_smiles


def test_draft_labeled_reaction_preserves_reaction_id() -> None:
    result = draft_labeled_reaction(_ALDOL_SIMPLE)
    assert result.reaction_id == _ALDOL_SIMPLE.reaction_id


def test_draft_labeled_reaction_confidence_is_draft() -> None:
    result = draft_labeled_reaction(_ALDOL_SIMPLE)
    for lgr in result.group_roles:
        assert lgr.confidence == "draft"


def test_draft_labeled_reaction_source_tag() -> None:
    cfg = DraftLabelConfig(source_tag="test_tag")
    result = draft_labeled_reaction(_ALDOL_SIMPLE, cfg)
    assert result.metadata.get("source") == "test_tag"


def test_draft_labeled_reaction_sn2_roundtrip() -> None:
    result = draft_labeled_reaction(_SIMPLE_SN2)
    assert result.reaction_id == "test_sn2_simple"
    assert result.metadata.get("needs_manual_review") is True


def test_draft_labeled_reaction_include_spectators() -> None:
    cfg_with = DraftLabelConfig(include_spectators=True)
    cfg_without = DraftLabelConfig(include_spectators=False)
    result_with = draft_labeled_reaction(_ALDOL_SIMPLE, cfg_with)
    result_without = draft_labeled_reaction(_ALDOL_SIMPLE, cfg_without)
    assert len(result_with.group_roles) >= len(result_without.group_roles)


def test_draft_labeled_reaction_use_raw_roles() -> None:
    cfg = DraftLabelConfig(use_negotiated_roles=False, include_spectators=True)
    result = draft_labeled_reaction(_SIMPLE_SN2, cfg)
    assert isinstance(result, LabeledReaction)


# ---------------------------------------------------------------------------
# C. Multiple draft labels
# ---------------------------------------------------------------------------


def test_draft_labeled_reactions_batch() -> None:
    inputs = [_SIMPLE_SN2, _ALDOL_SIMPLE]
    reactions, report = draft_labeled_reactions(inputs)
    assert report.n_inputs == 2
    assert report.n_outputs == 2
    assert len(reactions) == 2


def test_draft_labeled_reactions_report_fields() -> None:
    _, report = draft_labeled_reactions([_SIMPLE_SN2])
    assert hasattr(report, "n_inputs")
    assert hasattr(report, "n_outputs")
    assert hasattr(report, "n_group_roles")
    assert hasattr(report, "skipped")


def test_draft_labeled_reactions_continues_on_failure() -> None:
    bad = DraftReactionInput(
        reaction_id="test_bad",
        reaction_smiles="NOT_VALID_SMILES>>ALSO_BAD",
        context=ReactionContext.ionic,
        mechanism_type="unknown",
    )
    reactions, report = draft_labeled_reactions([_SIMPLE_SN2, bad])
    assert report.n_inputs == 2
    assert len(reactions) >= 1
    assert len(report.skipped) >= 1
    assert report.skipped[0]["reaction_id"] == "test_bad"


def test_draft_labeled_reactions_skipped_has_error_field() -> None:
    bad = DraftReactionInput(
        reaction_id="test_bad2",
        reaction_smiles=">>",
        context=ReactionContext.ionic,
        mechanism_type="unknown",
    )
    _, report = draft_labeled_reactions([bad])
    if report.skipped:
        assert "error" in report.skipped[0]


# ---------------------------------------------------------------------------
# D. Save / load draft inputs
# ---------------------------------------------------------------------------


def test_save_load_draft_inputs_roundtrip(tmp_path: Path) -> None:
    inputs = create_core_draft_inputs()
    path = tmp_path / "draft_inputs.json"
    save_draft_inputs(inputs, path)
    loaded = load_draft_inputs(path)
    assert len(loaded) == len(inputs)
    for orig, reloaded in zip(inputs, loaded, strict=True):
        assert orig.reaction_id == reloaded.reaction_id
        assert orig.reaction_smiles == reloaded.reaction_smiles
        assert orig.context == reloaded.context
        assert orig.mechanism_type == reloaded.mechanism_type


def test_load_draft_inputs_missing_field(tmp_path: Path) -> None:
    bad_json = json.dumps([{"reaction_id": "x", "reaction_smiles": "C>>C"}])
    path = tmp_path / "bad.json"
    path.write_text(bad_json, encoding="utf-8")
    with pytest.raises(ValueError, match="missing required field"):
        load_draft_inputs(path)


def test_load_draft_inputs_bad_context(tmp_path: Path) -> None:
    bad_json = json.dumps([{
        "reaction_id": "x",
        "reaction_smiles": "C>>C",
        "context": "NOT_A_CONTEXT",
        "mechanism_type": "SN2",
    }])
    path = tmp_path / "bad_ctx.json"
    path.write_text(bad_json, encoding="utf-8")
    with pytest.raises(ValueError, match="unknown context"):
        load_draft_inputs(path)


def test_load_draft_inputs_not_a_list(tmp_path: Path) -> None:
    path = tmp_path / "not_list.json"
    path.write_text(json.dumps({"reactions": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="Expected a JSON array"):
        load_draft_inputs(path)


# ---------------------------------------------------------------------------
# E. Save draft labeled reactions
# ---------------------------------------------------------------------------


def test_save_draft_labeled_reactions_valid_json(tmp_path: Path) -> None:
    reactions, _ = draft_labeled_reactions([_SIMPLE_SN2])
    path = tmp_path / "out.json"
    save_draft_labeled_reactions(reactions, path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert "reactions" in data
    assert isinstance(data["reactions"], list)


def test_save_draft_labeled_reactions_loadable(tmp_path: Path) -> None:
    reactions, _ = draft_labeled_reactions([_SIMPLE_SN2])
    path = tmp_path / "out.json"
    save_draft_labeled_reactions(reactions, path)
    loaded = load_labeled_reactions(path)
    assert len(loaded) == len(reactions)
    assert loaded[0].reaction_id == reactions[0].reaction_id


# ---------------------------------------------------------------------------
# F. Merge
# ---------------------------------------------------------------------------


def test_merge_keeps_existing_no_overwrite() -> None:
    existing_r, _ = draft_labeled_reactions([_SIMPLE_SN2])
    draft_r, _ = draft_labeled_reactions([_SIMPLE_SN2])
    merged = merge_labeled_reactions(existing_r, draft_r, overwrite=False)
    assert len(merged) == 1
    assert merged[0] is existing_r[0]


def test_merge_overwrites_with_flag() -> None:
    existing_r, _ = draft_labeled_reactions([_SIMPLE_SN2])
    draft_r, _ = draft_labeled_reactions([_SIMPLE_SN2])
    merged = merge_labeled_reactions(existing_r, draft_r, overwrite=True)
    assert len(merged) == 1
    assert merged[0] is draft_r[0]


def test_merge_appends_new_drafts() -> None:
    existing_r, _ = draft_labeled_reactions([_SIMPLE_SN2])
    new_draft, _ = draft_labeled_reactions([_ALDOL_SIMPLE])
    merged = merge_labeled_reactions(existing_r, new_draft, overwrite=False)
    assert len(merged) == 2
    ids = {r.reaction_id for r in merged}
    assert "test_sn2_simple" in ids
    assert "test_aldol_simple" in ids


def test_merge_deterministic_order() -> None:
    existing_r, _ = draft_labeled_reactions([_SIMPLE_SN2])
    new_r, _ = draft_labeled_reactions([_ALDOL_SIMPLE])
    merged = merge_labeled_reactions(existing_r, new_r)
    assert merged[0].reaction_id == "test_sn2_simple"
    assert merged[1].reaction_id == "test_aldol_simple"


# ---------------------------------------------------------------------------
# G. Summary
# ---------------------------------------------------------------------------


def test_summarize_draft_labels_keys() -> None:
    reactions, _ = draft_labeled_reactions([_SIMPLE_SN2, _ALDOL_SIMPLE])
    summary = summarize_draft_labels(reactions)
    for key in ("n_reactions", "n_group_roles", "role_counts", "group_type_counts",
                "mechanism_counts", "split_counts", "needs_manual_review_count"):
        assert key in summary, f"Missing key: {key}"


def test_summarize_draft_labels_counts() -> None:
    reactions, _ = draft_labeled_reactions([_SIMPLE_SN2])
    summary = summarize_draft_labels(reactions)
    assert summary["n_reactions"] == 1
    assert summary["needs_manual_review_count"] == 1


def test_summarize_empty_list() -> None:
    summary = summarize_draft_labels([])
    assert summary["n_reactions"] == 0
    assert summary["n_group_roles"] == 0


# ---------------------------------------------------------------------------
# H. CLI smoke test
# ---------------------------------------------------------------------------


def test_cli_core_creates_output(tmp_path: Path) -> None:
    out = tmp_path / "draft_core.json"
    report = tmp_path / "draft_core_report.json"
    result = subprocess.run(
        [
            sys.executable,
            str(_SCRIPTS_DIR / "draft_labels.py"),
            "--core",
            "--output", str(out),
            "--report", str(report),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"CLI failed:\n{result.stdout}\n{result.stderr}"
    assert out.exists(), "Output file not created"
    assert report.exists(), "Report file not created"

    data = json.loads(out.read_text(encoding="utf-8"))
    assert "reactions" in data
    assert len(data["reactions"]) >= 5

    report_data = json.loads(report.read_text(encoding="utf-8"))
    assert report_data["n_inputs"] >= 5


def test_cli_all_have_needs_manual_review(tmp_path: Path) -> None:
    out = tmp_path / "draft.json"
    report = tmp_path / "report.json"
    subprocess.run(
        [
            sys.executable,
            str(_SCRIPTS_DIR / "draft_labels.py"),
            "--core",
            "--output", str(out),
            "--report", str(report),
        ],
        check=True,
        capture_output=True,
    )
    data = json.loads(out.read_text(encoding="utf-8"))
    for rxn in data["reactions"]:
        assert rxn["metadata"].get("needs_manual_review") is True, (
            f"Reaction {rxn['reaction_id']} missing needs_manual_review=true"
        )
