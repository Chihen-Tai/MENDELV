"""MENDEL - Molecular Entity Negotiation for Dynamic Energy Landscapes.

Functional-group-level reaction role prediction framework.
Each functional group is treated as an agent that predicts its own role.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from mendel.constants import DEFAULT_VERSION

__version__: str = DEFAULT_VERSION

_EXPORT_MODULES: dict[str, str] = {
    # Phase 0 types
    "AtomRef": "mendel.types",
    "FunctionalGroup": "mendel.types",
    "FunctionalGroupType": "mendel.types",
    "ReactionContext": "mendel.types",
    "ReactionRecord": "mendel.types",
    "Role": "mendel.types",
    "RoleAssignment": "mendel.types",
    # Phase 1 parser
    "parse_reaction_record": "mendel.parser",
    "parse_reaction_smiles": "mendel.parser",
    "validate_reaction_smiles": "mendel.parser",
    # Phase 2 identifier
    "get_group_summary": "mendel.identifier",
    "has_group_type": "mendel.identifier",
    "identify_functional_groups": "mendel.identifier",
    "identify_functional_groups_in_mol": "mendel.identifier",
    # Phase 3 descriptor
    "GroupDescriptor": "mendel.descriptor",
    "build_descriptors": "mendel.descriptor",
    "build_group_descriptor": "mendel.descriptor",
    "descriptor_matrix": "mendel.descriptor",
    "get_feature_names": "mendel.descriptor",
    "validate_descriptor_schema": "mendel.descriptor",
    # Phase 4 labels
    "LabelValidationError": "mendel.labels",
    "LabeledGroupRole": "mendel.labels",
    "LabeledReaction": "mendel.labels",
    "labels_to_training_rows": "mendel.labels",
    "load_labeled_reactions": "mendel.labels",
    "save_labeled_reactions": "mendel.labels",
    "summarize_labeled_dataset": "mendel.labels",
    "validate_labeled_dataset": "mendel.labels",
    "validate_labeled_reaction": "mendel.labels",
    # Phase 5 predictor
    "PredictionReport": "mendel.predictor",
    "RolePrediction": "mendel.predictor",
    "RuleBasedPredictorConfig": "mendel.predictor",
    "RuleBasedRolePredictor": "mendel.predictor",
    "compare_predictions_to_labels": "mendel.predictor",
    "get_feature_value": "mendel.predictor",
    "predict_roles": "mendel.predictor",
    "predict_roles_for_reaction": "mendel.predictor",
    "summarize_predictions": "mendel.predictor",
    # Phase 6 negotiator
    "NegotiatedRoleAssignment": "mendel.negotiator",
    "NegotiationResult": "mendel.negotiator",
    "NegotiationWarning": "mendel.negotiator",
    "NegotiatorConfig": "mendel.negotiator",
    "RuleBasedNegotiator": "mendel.negotiator",
    "get_final_role_counts": "mendel.negotiator",
    "get_reaction_center_group_ids": "mendel.negotiator",
    "negotiate_predictions": "mendel.negotiator",
    "run_full_rule_pipeline": "mendel.negotiator",
    "run_pipeline_with_mlp": "mendel.negotiator",
    "summarize_negotiation_result": "mendel.negotiator",
    # Phase 8 benchmark (no torch required for these exports)
    "BenchmarkReport": "mendel.benchmark",
    "GroupBenchmarkRecord": "mendel.benchmark",
    "ReactionBenchmarkRecord": "mendel.benchmark",
    "compare_benchmark_reports": "mendel.benchmark",
    "evaluate_negotiated_rule_based": "mendel.benchmark",
    "evaluate_rule_based_predictor": "mendel.benchmark",
    # Phase 8.5 dataset quality (no torch required)
    "DatasetQualityIssue": "mendel.dataset_quality",
    "DatasetQualityReport": "mendel.dataset_quality",
    "build_dataset_quality_report": "mendel.dataset_quality",
    "canonicalize_mechanism_type": "mendel.dataset_quality",
    "normalize_labeled_dataset": "mendel.dataset_quality",
    "normalize_labeled_reaction": "mendel.dataset_quality",
}

__all__ = ["__version__", *_EXPORT_MODULES]


def __getattr__(name: str) -> Any:
    """Load public Phase 0-6 exports on demand.

    This keeps `import mendel` lightweight and avoids importing optional Phase 7
    dependencies such as PyTorch.
    """
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module 'mendel' has no attribute {name!r}")

    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
