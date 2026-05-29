"""Characterization (golden) test locking negotiator behavior before the Phase 11
consolidation refactor.

For every reaction in the canonical 166-reaction dataset this snapshots the full
output of ``negotiate()`` in BOTH modes:

* rule mode      (``NegotiatorConfig()`` default) — exercises ``_negotiate_<mechanism>``
* mlp_aware mode (``NegotiatorConfig(mode="mlp_aware")``) — exercises the
  ``_negotiate_<mechanism>`` + center-selection layer

Predictions are produced by the deterministic rule-based predictor so the snapshot
needs no torch/checkpoint. The point is to lock the negotiator's transformation of
``(parsed, groups, predictions, mode)`` into roles/subroles/centers/warnings, so the
refactor that folds the center-selection helpers into the mechanism helpers cannot
silently change behavior.

The golden file is generated once from pre-refactor code via::

    python tests/test_negotiator_characterization.py --regenerate

Do NOT regenerate during the refactor — that would mask behavior changes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from mendel.identifier import identify_functional_groups
from mendel.negotiator import NegotiatorConfig, negotiate_predictions
from mendel.parser import parse_reaction_smiles
from mendel.predictor import predict_roles_for_reaction
from mendel.types import ReactionContext

_DATASET = Path(__file__).parent.parent / "data" / "reactions.center_balanced.cleaned.json"
_GOLDEN = Path(__file__).parent / "data" / "negotiator_characterization_golden.json"


def _load_records() -> list[dict[str, Any]]:
    raw = json.loads(_DATASET.read_text())
    records = raw if isinstance(raw, list) else raw.get("reactions", raw)
    return list(records.values()) if isinstance(records, dict) else list(records)


def _to_context(value: str | None) -> ReactionContext:
    try:
        return ReactionContext(value) if value else ReactionContext.unknown
    except ValueError:
        return ReactionContext.unknown


def _snapshot_one(
    reaction_smiles: str,
    context: ReactionContext,
    mechanism_type: str,
    mode: str,
) -> dict[str, Any]:
    parsed = parse_reaction_smiles(reaction_smiles, context=context)
    groups = identify_functional_groups(parsed)
    report = predict_roles_for_reaction(parsed, groups)

    if mode == "mlp_aware":
        parsed.metadata["mechanism_type"] = mechanism_type
        config = NegotiatorConfig(mode="mlp_aware")
    else:
        config = NegotiatorConfig()

    result = negotiate_predictions(parsed, groups, report.predictions, config)
    return {
        "mechanism_hint": result.mechanism_hint,
        "assignments": [
            {
                "group_id": a.group_id,
                "raw_role": a.raw_role.value,
                "final_role": a.final_role.value,
                "subrole": a.subrole,
                "is_reaction_center": a.is_reaction_center,
            }
            for a in result.assignments
        ],
        "center_group_ids": sorted(
            a.group_id for a in result.assignments if a.is_reaction_center
        ),
        "warning_codes": sorted(w.code for w in result.warnings),
    }


def _build_snapshot(record: dict[str, Any]) -> dict[str, Any]:
    smiles = record["reaction_smiles"]
    context = _to_context(record.get("context"))
    mechanism_type = record.get("mechanism_type") or ""
    return {
        "rule": _snapshot_one(smiles, context, mechanism_type, "rule"),
        "mlp_aware": _snapshot_one(smiles, context, mechanism_type, "mlp_aware"),
    }


def _build_all() -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    for record in _load_records():
        key = record.get("reaction_id") or record["reaction_smiles"]
        snapshot[key] = _build_snapshot(record)
    return snapshot


def _golden_cases() -> list[tuple[str, dict[str, Any]]]:
    if not _GOLDEN.exists():
        return []
    return sorted(json.loads(_GOLDEN.read_text()).items())


@pytest.mark.skipif(not _GOLDEN.exists(), reason="golden snapshot not generated yet")
@pytest.mark.parametrize(
    "key,expected", _golden_cases(), ids=lambda c: c if isinstance(c, str) else ""
)
def test_negotiator_matches_golden(key: str, expected: dict[str, Any]) -> None:
    records = {r.get("reaction_id") or r["reaction_smiles"]: r for r in _load_records()}
    assert key in records, f"golden key {key!r} no longer in dataset"
    actual = _build_snapshot(records[key])
    assert actual == expected, f"negotiator behavior changed for {key!r}"


def test_golden_snapshot_present_and_covers_dataset() -> None:
    assert _GOLDEN.exists(), "run `python tests/test_negotiator_characterization.py --regenerate`"
    golden = json.loads(_GOLDEN.read_text())
    assert len(golden) == len(_load_records())


if __name__ == "__main__":
    import sys

    if "--regenerate" in sys.argv:
        _GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        _GOLDEN.write_text(json.dumps(_build_all(), indent=2, sort_keys=True) + "\n")
        print(f"wrote {_GOLDEN} ({len(json.loads(_GOLDEN.read_text()))} reactions)")
    else:
        print("pass --regenerate to (re)write the golden snapshot")
