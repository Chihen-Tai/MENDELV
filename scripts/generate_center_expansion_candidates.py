"""Generate local center-label-focused candidate reactions for Phase 8.12."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

MECHANISMS = [
    "sn2",
    "e2",
    "diels_alder",
    "carbonyl_addition",
    "aldol",
    "cross_aldol",
    "benzylic_radical_bromination",
    "radical_bromination",
    "nitroalkane_deprotonation",
    "ester_control",
    "nitrile_control",
    "control",
]

TEMPLATES: dict[str, tuple[str, str, str]] = {
    "sn2": ("[CH3:1][Br:2].[I-:3]>>[CH3:1][I:3].[Br-:2]", "ionic", "halide plus attached carbon"),
    "e2": (
        "[CH3:1][CH:2]([Br:3])[CH3:4].[OH-:5]>>[CH2:1]=[CH:2][CH3:4].[Br-:3].[OH2:5]",
        "ionic",
        "leaving group and alkene-forming carbons",
    ),
    "diels_alder": (
        "[CH2:1]=[CH:2][CH:3]=[CH2:4].[CH2:5]=[CH:6][C:7]#[N:8]>>[CH2:1][CH:2][CH:3][CH2:4][CH2:5][CH:6][C:7]#[N:8]",
        "pericyclic",
        "alkene pi atoms only; nitrile spectator",
    ),
    "carbonyl_addition": (
        "[CH3:1][C:2](=[O:3])[CH3:4].[C-:5]#[N:6]>>[CH3:1][C:2]([O-:3])([C:5]#[N:6])[CH3:4]",
        "ionic",
        "carbonyl carbon and oxygen",
    ),
    "aldol": (
        "[CH3:1][C:2](=[O:3])[CH3:4].[CH3:5][C:6](=[O:7])[CH3:8]>>[CH3:1][C:2](=[O:3])[CH2:4][C:6]([O-:7])([CH3:5])[CH3:8]",
        "ionic",
        "donor alpha carbon and acceptor carbonyl",
    ),
    "cross_aldol": (
        "[CH3:1][C:2](=[O:3])[CH3:4].[CH3:5][C:6](=[O:7])[H:8]>>[CH3:1][C:2](=[O:3])[CH2:4][C:6]([O-:7])([H:8])[CH3:5]",
        "ionic",
        "cross aldol donor and acceptor",
    ),
    "benzylic_radical_bromination": (
        "[cH:1]1[cH:2][cH:3][cH:4][cH:5][c:6]1[CH3:7]>>[cH:1]1[cH:2][cH:3][cH:4][cH:5][c:6]1[CH2:7]",
        "radical",
        "benzylic atom only",
    ),
    "radical_bromination": (
        "[CH3:1][CH2:2][CH3:3]>>[CH3:1][CH:2][CH3:3]",
        "radical",
        "radical carbon",
    ),
    "nitroalkane_deprotonation": (
        "[CH3:1][N+:2](=[O:3])[O-:4].[OH-:5]>>[CH2-:1][N+:2](=[O:3])[O-:4].[OH2:5]",
        "ionic",
        "alpha carbon nitronate center",
    ),
    "ester_control": (
        "[CH3:1][C:2](=[O:3])[O:4][CH3:5]>>[CH3:1][C:2](=[O:3])[O:4][CH3:5]",
        "ionic",
        "empty center control",
    ),
    "nitrile_control": ("[CH3:1][C:2]#[N:3]>>[CH3:1][C:2]#[N:3]", "ionic", "empty center control"),
    "control": ("[CH3:1][CH3:2]>>[CH3:1][CH3:2]", "unknown", "empty center control"),
}


def _candidate(mechanism: str, idx: int) -> dict[str, Any]:
    smiles, context, policy = TEMPLATES[mechanism]
    return {
        "reaction_id": f"center_expansion_{mechanism}_{idx:02d}",
        "reaction_smiles": smiles,
        "context": context,
        "mechanism_type": mechanism,
        "split": "draft",
        "metadata": {
            "source": "local_template_center_expansion",
            "source_type": "textbook_template",
            "license_note": "locally generated generic textbook-style template; no external source",
            "generation_method": "center_expansion_template_v1",
            "template_name": f"center_expansion_{mechanism}",
            "mechanism_type": mechanism,
            "needs_manual_review": True,
            "center_label_focus": True,
            "expected_center_policy": policy,
            "exclude_from_ground_truth_until_review": True,
        },
    }


def generate_center_expansion_candidates(max_count: int = 72) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    idx = 0
    while len(candidates) < max_count:
        mechanism = MECHANISMS[idx % len(MECHANISMS)]
        candidates.append(_candidate(mechanism, idx // len(MECHANISMS)))
        idx += 1
    return candidates


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate local Phase 8.12 center expansion candidates."
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/draft_inputs.center_expansion.json")
    )
    parser.add_argument("--max-count", type=int, default=72)
    args = parser.parse_args(argv)
    candidates = generate_center_expansion_candidates(args.max_count)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"candidates": candidates}, indent=2), encoding="utf-8")
    print(f"Generated {len(candidates)} center expansion candidates: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
