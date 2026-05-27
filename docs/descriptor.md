# Module 2: Descriptor Builder

`mendel/descriptor.py` — Phase 3

Converts each `FunctionalGroup` into a deterministic 55-dimensional feature vector.
Descriptors feed the Phase 5 rule-based predictor and, later, the optional Phase 7 MLP.

---

## Five Descriptor Categories

### A. Identity (21 features)

One-hot over all 17 `FunctionalGroupType` values, plus four atom-count scalars:
`atom_count`, `heteroatom_count`, `aromatic_atom_fraction`, `ring_atom_fraction`.

### B. Electronic (10 features)

Formal charge, RDKit Gasteiger partial charges (mean/min/max), Pauling electronegativity
(mean/max), `has_electronegative_atom` (EN > 2.9), `has_pi_bond`, `is_aromatic_group`.

### C. Local Environment (9 features)

Topological (graph-distance) features and neighborhood flags:
`neighbor_heteroatom_count`, `distance_to_nearest_heteroatom`,
`distance_to_nearest_leaving_group_atom`, `neighbor_ewg_count`,
`env_alpha_carbon`, `env_benzylic_site`, `is_allylic_like`, `in_reactant`, `in_product`.
Distances use `Chem.GetDistanceMatrix`; sentinel 99.0 when no target atom exists.

### D. Mechanistic Heuristic Scores (5 features)

**These are chemistry priors only — not role predictions.** No group is labelled nucleophile or
electrophile here; these scores are inputs to the Phase 4 rule-based predictor.

| Feature | High (≥ 0.65) | Low (≤ 0.15) |
|---------|--------------|-------------|
| `nucleophilicity_score` | amine, alpha_carbon | nitrile, nitro, halide |
| `electrophilicity_score` | carbonyl, ester, halide | alcohol, ether, amine |
| `leaving_group_score` | I (0.95) Br (0.85) Cl (0.75) F (0.50) | most groups (0.05) |
| `acidity_score` | carboxylic_acid (0.90), phenol (0.65) | ether (0.05), aromatic (0.08) |
| `radical_stability_score` | benzylic_site (0.80), aromatic (0.50) | halide (0.10), nitro (0.10) |

Negative formal charge adds a linear boost to `nucleophilicity_score`.

### E. Reaction Context (10 features)

Four one-hot context flags (`context_ionic`, `context_radical`, `context_pericyclic`,
`context_unknown`) plus six optional condition flags read from `ParsedReaction.metadata`:
`condition_acidic/basic/neutral/thermal/photochemical`, `solvent_polarity_score`.
All condition flags default to 0.0 when absent.

---

## Public API

```python
from mendel.descriptor import (
    GroupDescriptor,          # dataclass: group_id, group_type, feature_names, values, metadata
    get_feature_names,        # () → list[str]  — 55 names, deterministic
    build_group_descriptor,   # (ParsedReaction, FunctionalGroup) → GroupDescriptor
    build_descriptors,        # (ParsedReaction, list[FunctionalGroup]) → list[GroupDescriptor]
    descriptor_matrix,        # (list[GroupDescriptor]) → (names, matrix n×55)
    summarize_descriptors,    # (list[GroupDescriptor]) → dict
    validate_descriptor_schema,  # (list[GroupDescriptor]) → bool
)
```

Constants: `ELECTRONEGATIVITY`, `EWG_GROUPS`, `LEAVING_GROUP_ATOMS`,
`FEATURE_SCHEMA_VERSION = "phase3_v1"`.

---

## Known Limitations

- Heuristic scores are fixed priors, not trained on reaction outcomes.
- Charge adjustments are linear; solvent, temperature, and concentration effects are absent.
- Gasteiger charges fail gracefully to 0.0 for exotic/charged species.
- `[OH-]` may not be detected as a group by Phase 2 (no C–O bond); its descriptor would be absent.

---

## Freeze Status

Descriptor building is part of the Phase 0-6 pre-training freeze. The next use of these
features is Phase 7 MLP role predictor training, which is paused/future.

Do not run `scripts/train_mlp.py` or `tests/test_mlp.py` as part of Phase 0-6 validation.
