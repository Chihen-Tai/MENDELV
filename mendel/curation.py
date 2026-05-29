"""Phase 6.5: Dataset curation and draft-label generation utilities.

Runs the Phase 0–6 rule-based pipeline over new reaction SMILES and exports
draft LabeledReaction records for manual review.  Draft labels are NOT ground
truth.  Every generated record is marked needs_manual_review=true and must be
inspected and corrected by a chemist before use in Phase 7 training.

No PyTorch, MACE, ASE, or any ML dependency is imported here.
"""

from __future__ import annotations

import json
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mendel.descriptor import build_descriptors
from mendel.identifier import identify_functional_groups
from mendel.labels import LabeledGroupRole, LabeledReaction
from mendel.negotiator import negotiate_predictions
from mendel.parser import parse_reaction_smiles
from mendel.predictor import predict_roles_for_reaction
from mendel.types import FunctionalGroup, ReactionContext, Role

# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass
class DraftReactionInput:
    """Input descriptor for generating one draft-labeled reaction.

    reaction_id: stable human-readable identifier, unique within a run.
    reaction_smiles: full reaction SMILES (reactants>>products).
    context: broad mechanistic category.
    mechanism_type: fine-grained label (e.g. 'SN2', 'Aldol').
    split: partition tag; defaults to 'draft'.
    metadata: arbitrary annotations (e.g. notes about SMILES limitations).
    """

    reaction_id: str
    reaction_smiles: str
    context: ReactionContext
    mechanism_type: str
    split: str = "draft"
    metadata: dict[str, str | int | float | bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "reaction_id": self.reaction_id,
            "reaction_smiles": self.reaction_smiles,
            "context": self.context.value,
            "mechanism_type": self.mechanism_type,
            "split": self.split,
            "metadata": dict(self.metadata),
        }


@dataclass
class DraftLabelConfig:
    """Configuration controlling how draft labels are generated.

    use_negotiated_roles: use final_role from Phase 6 negotiator.
        False means use raw Phase 5 role.
    include_spectators: include spectator groups in group_roles.
    include_low_confidence: include assignments below min_confidence.
    min_confidence: threshold; assignments below this are excluded when
        include_low_confidence=False.
    mark_needs_manual_review: always set needs_manual_review=True in metadata.
    source_tag: written to metadata['source'] on every generated record.
    """

    use_negotiated_roles: bool = True
    include_spectators: bool = False
    include_low_confidence: bool = True
    min_confidence: float = 0.0
    mark_needs_manual_review: bool = True
    source_tag: str = "phase6_5_draft"


@dataclass
class DraftLabelReport:
    """Summary of a batch draft-labeling run.

    n_inputs: number of inputs submitted.
    n_outputs: number of LabeledReaction records successfully generated.
    n_group_roles: total group role entries across all outputs.
    skipped: list of error records for inputs that failed.
    warnings: pipeline warning records (from negotiation).
    metadata: arbitrary annotations about the run.
    """

    n_inputs: int
    n_outputs: int
    n_group_roles: int
    skipped: list[dict[str, object]] = field(default_factory=list)
    warnings: list[dict[str, object]] = field(default_factory=list)
    metadata: dict[str, str | int | float | bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "n_inputs": self.n_inputs,
            "n_outputs": self.n_outputs,
            "n_group_roles": self.n_group_roles,
            "skipped": list(self.skipped),
            "warnings": list(self.warnings),
            "metadata": dict(self.metadata),
        }


# ---------------------------------------------------------------------------
# Internal pipeline helper
# ---------------------------------------------------------------------------


def _run_pipeline_with_groups(
    reaction_smiles: str,
    context: ReactionContext,
) -> tuple[Any, list[FunctionalGroup]]:
    """Run phases 1–6 and return (NegotiationResult, groups).

    Keeps intermediate `groups` accessible so atom_indices can be extracted
    for LabeledGroupRole construction.  run_full_rule_pipeline discards groups.
    """
    parsed = parse_reaction_smiles(reaction_smiles, context=context)
    groups = identify_functional_groups(parsed)
    build_descriptors(parsed, groups)
    report = predict_roles_for_reaction(parsed, groups)
    result = negotiate_predictions(parsed, groups, report.predictions)
    return result, groups


# ---------------------------------------------------------------------------
# Single-reaction draft labeling
# ---------------------------------------------------------------------------


def draft_labeled_reaction(
    draft_input: DraftReactionInput,
    config: DraftLabelConfig | None = None,
) -> LabeledReaction:
    """Generate a single draft-labeled reaction from the Phase 0–6 pipeline.

    Runs parser → identifier → descriptor → predictor → negotiator and
    converts negotiated assignments into LabeledGroupRole entries.

    Draft labels are marked confidence='draft' and must not be treated as
    ground truth.  The returned LabeledReaction always has
    metadata['needs_manual_review'] = True.

    Raises ValueError or RuntimeError if the SMILES cannot be parsed or the
    pipeline fails.
    """
    cfg = config or DraftLabelConfig()
    result, groups = _run_pipeline_with_groups(
        draft_input.reaction_smiles,
        draft_input.context,
    )

    group_map: dict[str, FunctionalGroup] = {g.group_id: g for g in groups}

    group_roles: list[LabeledGroupRole] = []
    for assignment in result.assignments:
        role = assignment.final_role if cfg.use_negotiated_roles else assignment.raw_role

        if not cfg.include_spectators and role == Role.spectator:
            continue

        if not cfg.include_low_confidence and assignment.final_confidence < cfg.min_confidence:
            continue

        group = group_map.get(assignment.group_id)
        if group is None:
            continue

        molecule_index = group.atom_refs[0].molecule_index if group.atom_refs else 0
        atom_indices = [ref.atom_index for ref in group.atom_refs]

        notes = (
            f"draft_source: {cfg.source_tag}; "
            f"final_confidence={assignment.final_confidence:.3f}; "
            f"subrole={assignment.subrole or 'none'}; "
            f"reason: {assignment.reason}"
        )

        group_roles.append(LabeledGroupRole(
            group_id=assignment.group_id,
            molecule_index=molecule_index,
            group_type=assignment.group_type,
            atom_indices=atom_indices,
            role=role,
            confidence="draft",
            notes=notes,
        ))

    # Prefer atom_map_num; fall back to atom_index for unmapped reactions
    reaction_center_atoms: list[int] = []
    for ref in result.reaction_center_atoms:
        val = ref.atom_map_num if ref.atom_map_num is not None else ref.atom_index
        if val not in reaction_center_atoms:
            reaction_center_atoms.append(val)

    metadata: dict[str, Any] = {
        "source": cfg.source_tag,
        "needs_manual_review": True,
        "mechanism_hint": result.mechanism_hint,
        "generated_by": "MENDEL Phase 6.5",
        "warning_count": len(result.warnings),
    }
    metadata.update(draft_input.metadata)

    return LabeledReaction(
        reaction_id=draft_input.reaction_id,
        reaction_smiles=draft_input.reaction_smiles,
        context=draft_input.context,
        mechanism_type=draft_input.mechanism_type,
        split=draft_input.split,
        group_roles=group_roles,
        reaction_center_atoms=reaction_center_atoms,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Batch draft labeling
# ---------------------------------------------------------------------------


def draft_labeled_reactions(
    inputs: list[DraftReactionInput],
    config: DraftLabelConfig | None = None,
) -> tuple[list[LabeledReaction], DraftLabelReport]:
    """Generate draft labels for a list of DraftReactionInputs.

    Continues on failure; failed inputs are recorded in DraftLabelReport.skipped.
    Returns (labeled_reactions, report).
    """
    reactions: list[LabeledReaction] = []
    skipped: list[dict[str, object]] = []
    pipeline_warnings: list[dict[str, object]] = []

    for inp in inputs:
        try:
            labeled = draft_labeled_reaction(inp, config)
            reactions.append(labeled)
            wcount = labeled.metadata.get("warning_count", 0)
            if wcount:
                pipeline_warnings.append({
                    "reaction_id": inp.reaction_id,
                    "warning_count": wcount,
                })
        except Exception as exc:
            skipped.append({
                "reaction_id": inp.reaction_id,
                "reaction_smiles": inp.reaction_smiles,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            })

    report = DraftLabelReport(
        n_inputs=len(inputs),
        n_outputs=len(reactions),
        n_group_roles=sum(len(r.group_roles) for r in reactions),
        skipped=skipped,
        warnings=pipeline_warnings,
    )
    return reactions, report


# ---------------------------------------------------------------------------
# I/O utilities
# ---------------------------------------------------------------------------


def load_draft_inputs(path: str | Path) -> list[DraftReactionInput]:
    """Load a list of DraftReactionInput from a JSON file.

    Expected format: a top-level JSON array with objects containing
    reaction_id, reaction_smiles, context, mechanism_type, and optionally
    split and metadata.

    Raises ValueError for missing required fields or unknown context values.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(
            f"Expected a JSON array at top level in {path}, got {type(raw).__name__}"
        )

    inputs: list[DraftReactionInput] = []
    for i, item in enumerate(raw):
        for key in ("reaction_id", "reaction_smiles", "context", "mechanism_type"):
            if key not in item:
                raise ValueError(f"Item {i}: missing required field '{key}'")

        try:
            context = ReactionContext(item["context"])
        except ValueError as err:
            raise ValueError(
                f"Item {i} ({item.get('reaction_id', '?')}): "
                f"unknown context '{item['context']}'. "
                f"Expected one of: {[c.value for c in ReactionContext]}"
            ) from err

        inputs.append(DraftReactionInput(
            reaction_id=item["reaction_id"],
            reaction_smiles=item["reaction_smiles"],
            context=context,
            mechanism_type=item["mechanism_type"],
            split=item.get("split", "draft"),
            metadata={k: v for k, v in item.get("metadata", {}).items()},
        ))
    return inputs


def save_draft_inputs(inputs: list[DraftReactionInput], path: str | Path) -> None:
    """Save a list of DraftReactionInput to a JSON file (top-level array)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = [inp.to_dict() for inp in inputs]
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def save_draft_labeled_reactions(
    reactions: list[LabeledReaction],
    path: str | Path,
) -> None:
    """Save draft labeled reactions in the same schema as data/reactions.json.

    Uses the top-level 'reactions' wrapper.  Enum values are serialised as strings.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {"reactions": [r.to_dict() for r in reactions]}
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------


def merge_labeled_reactions(
    existing: list[LabeledReaction],
    drafts: list[LabeledReaction],
    overwrite: bool = False,
) -> list[LabeledReaction]:
    """Merge draft records into an existing list, matched by reaction_id.

    overwrite=False (default): keep existing curated records; skip duplicates.
    overwrite=True: replace existing records with draft versions.

    Ordering: existing records first (original order), then new drafts.
    """
    existing_by_id: dict[str, LabeledReaction] = {r.reaction_id: r for r in existing}
    result: list[LabeledReaction] = []

    if overwrite:
        draft_by_id: dict[str, LabeledReaction] = {r.reaction_id: r for r in drafts}
        for rxn in existing:
            result.append(draft_by_id.get(rxn.reaction_id, rxn))
        for draft in drafts:
            if draft.reaction_id not in existing_by_id:
                result.append(draft)
    else:
        result.extend(existing)
        for draft in drafts:
            if draft.reaction_id not in existing_by_id:
                result.append(draft)

    return result


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def summarize_draft_labels(reactions: list[LabeledReaction]) -> dict[str, object]:
    """Return counts and distributions for a list of LabeledReactions.

    Keys: n_reactions, n_group_roles, role_counts, group_type_counts,
    mechanism_counts, split_counts, needs_manual_review_count.
    """
    role_counts: dict[str, int] = {}
    group_type_counts: dict[str, int] = {}
    mechanism_counts: dict[str, int] = {}
    split_counts: dict[str, int] = {}
    needs_manual_review_count = 0
    n_group_roles = 0

    for rxn in reactions:
        mechanism_counts[rxn.mechanism_type] = mechanism_counts.get(rxn.mechanism_type, 0) + 1
        split_counts[rxn.split] = split_counts.get(rxn.split, 0) + 1
        if rxn.metadata.get("needs_manual_review"):
            needs_manual_review_count += 1
        for lgr in rxn.group_roles:
            role_counts[lgr.role.value] = role_counts.get(lgr.role.value, 0) + 1
            group_type_counts[lgr.group_type.value] = (
                group_type_counts.get(lgr.group_type.value, 0) + 1
            )
            n_group_roles += 1

    return {
        "n_reactions": len(reactions),
        "n_group_roles": n_group_roles,
        "role_counts": role_counts,
        "group_type_counts": group_type_counts,
        "mechanism_counts": mechanism_counts,
        "split_counts": split_counts,
        "needs_manual_review_count": needs_manual_review_count,
    }


# ---------------------------------------------------------------------------
# Starter input sets
# ---------------------------------------------------------------------------


def create_core_draft_inputs() -> list[DraftReactionInput]:
    """Return starter inputs for the five core benchmark reactions.

    Mirrors the benchmark set in BENCHMARK.md: SN2, E2, Diels-Alder,
    Aldol, Radical bromination.  Atom-mapped SMILES used where stable.
    """
    return [
        DraftReactionInput(
            reaction_id="draft_sn2_methyl_bromide_oh",
            reaction_smiles="[CH3:1][Br:2].[OH-:3]>>[CH3:1][OH:3].[Br-:2]",
            context=ReactionContext.ionic,
            mechanism_type="SN2",
            metadata={"benchmark": True, "note": "core benchmark SN2"},
        ),
        DraftReactionInput(
            reaction_id="draft_e2_ethyl_bromide_oh",
            reaction_smiles=(
                "[CH3:1][CH2:2][Br:3].[OH-:4]"
                ">>[CH2:1]=[CH2:2].[Br-:3].[OH2:4]"
            ),
            context=ReactionContext.ionic,
            mechanism_type="E2",
            metadata={"benchmark": True, "note": "core benchmark E2"},
        ),
        DraftReactionInput(
            reaction_id="draft_diels_alder_butadiene_ethylene",
            reaction_smiles=(
                "[CH2:1]=[CH:2][CH:3]=[CH2:4].[CH2:5]=[CH2:6]"
                ">>[CH2:1]1[CH:2]=[CH:3][CH2:4][CH2:5][CH2:6]1"
            ),
            context=ReactionContext.pericyclic,
            mechanism_type="Diels-Alder",
            metadata={"benchmark": True, "note": "core benchmark Diels-Alder"},
        ),
        DraftReactionInput(
            reaction_id="draft_aldol_acetone_self",
            reaction_smiles="CC(=O)C.CC(=O)C>>CC(=O)CC(O)C",
            context=ReactionContext.ionic,
            mechanism_type="Aldol",
            metadata={
                "benchmark": True,
                "note": "core benchmark aldol; no atom mapping; product may be simplified",
            },
        ),
        DraftReactionInput(
            reaction_id="draft_radical_bromination_methane",
            reaction_smiles="C.BrBr>>CBr.[H]Br",
            context=ReactionContext.radical,
            mechanism_type="radical_bromination",
            metadata={
                "benchmark": True,
                "note": (
                    "core benchmark radical bromination; methane has no detected "
                    "functional groups — most assignments will be spectator"
                ),
            },
        ),
    ]


def create_extended_draft_inputs() -> list[DraftReactionInput]:
    """Return starter inputs for extended benchmark examples.

    Covers additional SN2, E2, Aldol, Diels-Alder, and radical bromination
    variants.  Some product SMILES are simplified; see each metadata note.
    """
    return [
        DraftReactionInput(
            reaction_id="draft_sn2_methyl_iodide_cyanide",
            reaction_smiles="CI.[C-]#N>>CC#N.[I-]",
            context=ReactionContext.ionic,
            mechanism_type="SN2",
            metadata={"note": "SN2 with cyanide nucleophile"},
        ),
        DraftReactionInput(
            reaction_id="draft_sn2_benzyl_bromide_methoxide",
            reaction_smiles="c1ccccc1CBr.[CH3O-]>>c1ccccc1COC.[Br-]",
            context=ReactionContext.ionic,
            mechanism_type="SN2",
            metadata={"note": "SN2 benzyl bromide with methoxide"},
        ),
        DraftReactionInput(
            reaction_id="draft_e2_secondary_alkyl_bromide",
            reaction_smiles="CC(Br)C.[OH-]>>CC=C.[Br-].[OH2]",
            context=ReactionContext.ionic,
            mechanism_type="E2",
            metadata={
                "note": "E2 secondary alkyl bromide; draft input; product representation may be simplified"
            },
        ),
        DraftReactionInput(
            reaction_id="draft_e2_tertbutyl_bromide",
            reaction_smiles="CC(C)(C)Br.[OH-]>>CC(C)=C.[Br-].[OH2]",
            context=ReactionContext.ionic,
            mechanism_type="E2",
            metadata={
                "note": "E2 tert-butyl bromide; Zaitsev product; draft input; product representation may be simplified"
            },
        ),
        DraftReactionInput(
            reaction_id="draft_aldol_acetaldehyde_self",
            reaction_smiles="CC=O.CC=O>>CC(O)CC=O",
            context=ReactionContext.ionic,
            mechanism_type="Aldol",
            metadata={"note": "aldol acetaldehyde self-condensation; product may be simplified"},
        ),
        DraftReactionInput(
            reaction_id="draft_aldol_acetaldehyde_benzaldehyde",
            reaction_smiles="CC=O.c1ccccc1C=O>>CC(O)Cc1ccccc1",
            context=ReactionContext.ionic,
            mechanism_type="Aldol",
            metadata={
                "note": "cross-aldol acetaldehyde + benzaldehyde; draft input; product representation may be simplified"
            },
        ),
        DraftReactionInput(
            reaction_id="draft_diels_alder_cpd_ethylene",
            reaction_smiles="C1C=CC=C1.C=C>>C1CC2CC1CC2",
            context=ReactionContext.pericyclic,
            mechanism_type="Diels-Alder",
            metadata={
                "note": "cyclopentadiene + ethylene → norbornene; draft input; product representation may be simplified"
            },
        ),
        DraftReactionInput(
            reaction_id="draft_diels_alder_furan_acrylonitrile",
            reaction_smiles="C1=COC=C1.C=CC#N>>N#CC1CC2OC2C1",
            context=ReactionContext.pericyclic,
            mechanism_type="Diels-Alder",
            metadata={
                "note": "furan + acrylonitrile; draft input; product representation may be simplified"
            },
        ),
        DraftReactionInput(
            reaction_id="draft_radical_bromination_ethane",
            reaction_smiles="CC.BrBr>>CCBr.[H]Br",
            context=ReactionContext.radical,
            mechanism_type="radical_bromination",
            metadata={"note": "radical bromination of ethane"},
        ),
        DraftReactionInput(
            reaction_id="draft_radical_bromination_toluene_benzylic",
            reaction_smiles="c1ccccc1C.BrBr>>c1ccccc1CBr.[H]Br",
            context=ReactionContext.radical,
            mechanism_type="radical_bromination",
            metadata={"note": "benzylic bromination of toluene; benzylic_site should be detected"},
        ),
    ]
