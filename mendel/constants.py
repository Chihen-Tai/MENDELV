"""Project-wide constants derived from the core type enums.

Import these instead of hardcoding enum members in downstream code.
"""

from mendel.types import FunctionalGroupType, ReactionContext, Role

DEFAULT_VERSION: str = "0.1.0"

SUPPORTED_ROLES: frozenset[Role] = frozenset(Role)

SUPPORTED_CONTEXTS: frozenset[ReactionContext] = frozenset(ReactionContext)

SUPPORTED_FUNCTIONAL_GROUPS: frozenset[FunctionalGroupType] = frozenset(FunctionalGroupType)
