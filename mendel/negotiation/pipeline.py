"""One-call pipeline wrappers around the negotiator."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mendel.identifier import identify_functional_groups
from mendel.negotiation.orchestrator import RuleBasedNegotiator
from mendel.negotiation.types import NegotiationResult, NegotiatorConfig
from mendel.parser import ParsedReaction, parse_reaction_smiles
from mendel.predictor import RolePrediction, predict_roles_for_reaction
from mendel.types import FunctionalGroup, ReactionContext


def negotiate_predictions(
    parsed_reaction: ParsedReaction,
    groups: list[FunctionalGroup],
    predictions: list[RolePrediction],
    config: NegotiatorConfig | None = None,
) -> NegotiationResult:
    """Convenience wrapper around RuleBasedNegotiator.negotiate."""
    return RuleBasedNegotiator(config).negotiate(parsed_reaction, groups, predictions)


def run_full_rule_pipeline(
    reaction_smiles: str,
    context: ReactionContext | str = ReactionContext.unknown,
    config: NegotiatorConfig | None = None,
) -> NegotiationResult:
    """One-call public pipeline: parser → identifier → predictor → negotiator.

    This is the first true one-call public interface for MENDEL.  It chains
    all implemented phases and returns a fully negotiated NegotiationResult.

    Args:
        reaction_smiles: Full reaction SMILES string (reactants>>products).
        context: Mechanistic category; accepts ReactionContext enum or plain string.
        config: Optional negotiator configuration.

    Returns:
        NegotiationResult with final coordinated assignments, mechanism hint,
        reaction center atoms, and warnings.
    """
    if isinstance(context, str):
        try:
            context = ReactionContext(context)
        except ValueError:
            context = ReactionContext.unknown

    parsed = parse_reaction_smiles(reaction_smiles, context=context)
    groups = identify_functional_groups(parsed)
    report = predict_roles_for_reaction(parsed, groups)
    return negotiate_predictions(parsed, groups, report.predictions, config)


def run_pipeline_with_mlp(
    reaction_smiles: str,
    mlp_checkpoint: str | Path,
    context: ReactionContext | str = ReactionContext.unknown,
    config: NegotiatorConfig | None = None,
    device: str = "cpu",
) -> NegotiationResult:
    """One-call pipeline using the Phase 7 MLP predictor instead of rule-based.

    parser → identifier → MLP predictor → negotiator

    Args:
        reaction_smiles: Full reaction SMILES (reactants>>products).
        mlp_checkpoint: Path to a checkpoint saved by MLPRolePredictor.save.
        context: Mechanistic category string or ReactionContext enum.
        config: Optional negotiator configuration.
        device: Torch device string ('cpu', 'cuda', 'mps').

    Returns:
        NegotiationResult identical in shape to run_full_rule_pipeline output.
    """
    from mendel.mlp import MLPRolePredictor

    if isinstance(context, str):
        try:
            context = ReactionContext(context)
        except ValueError:
            context = ReactionContext.unknown

    parsed = parse_reaction_smiles(reaction_smiles, context=context)
    groups = identify_functional_groups(parsed)
    mlp = MLPRolePredictor.load(Path(mlp_checkpoint), device=device)
    mlp_preds = mlp.predict_from_reaction(parsed, groups)

    group_type_by_id = {g.group_id: g.group_type for g in groups}
    predictions = [
        RolePrediction(
            group_id=p.group_id,
            group_type=group_type_by_id[p.group_id],
            predicted_role=p.predicted_role,
            confidence=p.confidence,
            reason="mlp_prediction",
            metadata={"prediction_source": "mlp", "predictor_name": "phase7_mlp"},
        )
        for p in mlp_preds
    ]
    return negotiate_predictions(parsed, groups, predictions, config)


def summarize_negotiation_result(result: NegotiationResult) -> dict[str, Any]:
    """Convenience wrapper for RuleBasedNegotiator.summarize_result."""
    return RuleBasedNegotiator().summarize_result(result)


def get_final_role_counts(result: NegotiationResult) -> dict[str, int]:
    """Return count of each final role across all assignments."""
    counts: dict[str, int] = {}
    for a in result.assignments:
        key = a.final_role.value
        counts[key] = counts.get(key, 0) + 1
    return counts


def get_reaction_center_group_ids(result: NegotiationResult) -> list[str]:
    """Return group_ids for assignments marked as reaction center."""
    return [a.group_id for a in result.assignments if a.is_reaction_center]
