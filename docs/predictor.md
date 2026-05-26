# Module 3 v0.1: Rule-Based Role Predictor

`mendel/predictor.py` — Phase 5

Assigns exactly one MENDEL role to each functional-group agent using deterministic
threshold rules on Phase 3 descriptor scores.  No MLP, no training, no negotiation.

---

## Purpose

Phase 5 gives each functional-group agent its first role prediction.  The predictor is
intentionally simple and interpretable: every decision traces back to a named heuristic
score and a threshold.  This baseline establishes a measurable accuracy floor before the
MLP is introduced in a later phase.

---

## Input and Output

**Input per group:**
- `GroupDescriptor` — 55-dimensional feature vector from Phase 3
- `ReactionContext` — ionic / radical / pericyclic / unknown

**Output per group:**
- `RolePrediction` — one of the five MENDEL roles, confidence in [0, 1], human-readable reason, and the five mechanistic scores

---

## Role Taxonomy

| Role | Meaning |
|------|---------|
| `reactive_nucleophile` | Electron donor; attacks an electrophile |
| `reactive_electrophile` | Electron acceptor; attacked by a nucleophile |
| `reactive_radical` | Radical centre; homolytic bond breaking or formation |
| `leaving_group` | Departs with the bonding electron pair |
| `spectator` | Does not participate in the bond-forming/breaking step |

Each group receives exactly one role.

---

## Rule Priority Order

Rules are evaluated in this fixed order.  The first rule that fires wins.

```
1. Radical context
   └─ benzylic_site type              → reactive_radical  (group-type override)
   └─ radical_stability_score >= 0.55 → reactive_radical
   └─ halide with leaving_group_score >= 0.50 → leaving_group
   └─ otherwise                       → spectator

2. Leaving group (all non-radical contexts)
   └─ leaving_group_score >= 0.50     → leaving_group

3. Pericyclic context
   └─ has_pi_bond and nuc >= elec     → reactive_nucleophile
   └─ has_pi_bond and elec > nuc      → reactive_electrophile
   └─ no pi_bond                      → spectator

4. Ionic / unknown context
   └─ alpha_carbon and acidity_score >= 0.45  → reactive_nucleophile
   └─ negative formal_charge and nuc >= 0.40  → reactive_nucleophile
   └─ both thresholds exceeded                → stronger score wins
   └─ nucleophilicity_score >= 0.55           → reactive_nucleophile
   └─ electrophilicity_score >= 0.55          → reactive_electrophile

5. Spectator fallback
   └─ no score exceeded any threshold → spectator
```

---

## Why Heuristic Scores

Phase 3 descriptor scores (`nucleophilicity_score`, `electrophilicity_score`,
`leaving_group_score`, `acidity_score`, `radical_stability_score`) are chemistry priors
computed from group type, formal charge, and substituent effects.  They encode known
chemical tendencies without referencing any reaction outcomes, keeping every prediction
fully interpretable.

Default thresholds (configurable via `RuleBasedPredictorConfig`):

| Parameter | Default |
|-----------|---------|
| `nucleophile_threshold` | 0.55 |
| `electrophile_threshold` | 0.55 |
| `leaving_group_threshold` | 0.50 |
| `acidity_threshold` | 0.45 |
| `radical_threshold` | 0.55 |
| `spectator_confidence` | 0.50 |

---

## Public API

```python
from mendel.predictor import (
    RuleBasedPredictorConfig,       # dataclass: threshold configuration
    RuleBasedRolePredictor,         # class: predict_group / predict / predict_from_reaction
    RolePrediction,                 # dataclass: group_id, predicted_role, confidence, reason, scores
    PredictionReport,               # dataclass: reaction_smiles, context, predictions
    get_feature_value,              # (descriptor, name, default=0.0) → float
    predict_roles,                  # (descriptors, context, config) → list[RolePrediction]
    predict_roles_for_reaction,     # (parsed_reaction, groups, config) → PredictionReport
    summarize_predictions,          # (predictions) → dict
    compare_predictions_to_labels,  # (predictions, labeled_reaction) → dict
)
```

---

## Known Limitations

**One role per group** — Each `FunctionalGroup` receives exactly one role.  The carbon of an
alkyl halide is simultaneously electrophilic and carries the leaving group; Phase 5 resolves
this by applying the leaving-group rule first.  Finer group granularity belongs to Phase 2.

**Flat taxonomy for pericyclic reactions** — Diene and dienophile partners are assigned
`reactive_nucleophile` or `reactive_electrophile` by score comparison.  A dedicated
`pericyclic_partner` role is out of scope for Phase 5.

**No negotiation** — Multiple groups in the same reaction may receive the same reactive role,
which is chemically inconsistent.  Conflict resolution belongs to Phase 6.

**No MLP** — Scores are fixed chemistry priors, not learned from reaction outcomes.

**No transition state or barrier prediction** — Phase 5 assigns roles only.  Activation
energies, barriers, and 3D geometry belong to the MLIP integration phase.

**Radical chain not modelled** — The Br• radical source in radical bromination has no SMARTS
match in Phase 2 and therefore no descriptor or role.

---

## What Phase 6 Should Implement Next

`mendel/negotiation.py` — rule-based conflict resolution:

- Input: `list[RolePrediction]` for all groups in one reaction.
- Output: `list[RolePrediction]` with conflicts resolved.
- Key rules: at most one nucleophile and one electrophile per step; `leaving_group` takes
  priority over `reactive_electrophile` for halides; aromatic groups are spectator unless
  explicitly reactive.
- Target: raise benchmark accuracy from the Phase 5 baseline toward ≥ 80 %.
