"""MENDEL — Molecular Entity Negotiation for Dynamic Energy Landscapes.

Functional-group-level reaction role prediction framework.
Each functional group is treated as an agent that predicts its own role.
"""

from mendel.constants import DEFAULT_VERSION
from mendel.predictor import (
    PredictionReport,
    RolePrediction,
    RuleBasedPredictorConfig,
    RuleBasedRolePredictor,
    compare_predictions_to_labels,
    get_feature_value,
    predict_roles,
    predict_roles_for_reaction,
    summarize_predictions,
)
from mendel.labels import (
    LabelValidationError,
    LabeledGroupRole,
    LabeledReaction,
    labels_to_training_rows,
    load_labeled_reactions,
    save_labeled_reactions,
    summarize_labeled_dataset,
    validate_labeled_dataset,
    validate_labeled_reaction,
)
from mendel.descriptor import (
    GroupDescriptor,
    build_descriptors,
    build_group_descriptor,
    descriptor_matrix,
    get_feature_names,
    validate_descriptor_schema,
)
from mendel.identifier import (
    get_group_summary,
    has_group_type,
    identify_functional_groups,
    identify_functional_groups_in_mol,
)
from mendel.parser import parse_reaction_record, parse_reaction_smiles, validate_reaction_smiles
from mendel.types import (
    AtomRef,
    FunctionalGroup,
    FunctionalGroupType,
    ReactionContext,
    ReactionRecord,
    Role,
    RoleAssignment,
)

__version__: str = DEFAULT_VERSION

__all__ = [
    "__version__",
    # Phase 0 types
    "AtomRef",
    "FunctionalGroup",
    "FunctionalGroupType",
    "ReactionContext",
    "ReactionRecord",
    "Role",
    "RoleAssignment",
    # Phase 1 parser
    "parse_reaction_record",
    "parse_reaction_smiles",
    "validate_reaction_smiles",
    # Phase 2 identifier
    "get_group_summary",
    "has_group_type",
    "identify_functional_groups",
    "identify_functional_groups_in_mol",
    # Phase 5 predictor
    "PredictionReport",
    "RolePrediction",
    "RuleBasedPredictorConfig",
    "RuleBasedRolePredictor",
    "compare_predictions_to_labels",
    "get_feature_value",
    "predict_roles",
    "predict_roles_for_reaction",
    "summarize_predictions",
    # Phase 4 labels
    "LabelValidationError",
    "LabeledGroupRole",
    "LabeledReaction",
    "labels_to_training_rows",
    "load_labeled_reactions",
    "save_labeled_reactions",
    "summarize_labeled_dataset",
    "validate_labeled_dataset",
    "validate_labeled_reaction",
    # Phase 3 descriptor
    "GroupDescriptor",
    "build_descriptors",
    "build_group_descriptor",
    "descriptor_matrix",
    "get_feature_names",
    "validate_descriptor_schema",
]
