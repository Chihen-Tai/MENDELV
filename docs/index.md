# MENDEL Documentation

**Molecular Entity Negotiation for Dynamic Energy Landscapes**

Functional-group-level reaction role prediction for organic chemistry.

---

## Core Abstraction

**Functional group = agent.**

Each functional group in a molecule is treated as an independent decision unit that:

1. Observes its local chemical environment
2. Predicts its own role in the reaction (nucleophile, electrophile, radical, leaving group, or spectator)
3. Negotiates with neighbouring groups to produce a coherent, conflict-free assignment

This replaces atom-level or molecule-level role prediction with group-level reasoning — the natural unit of organic chemistry.

---

## Pipeline

```
reaction SMILES + context
         │
         ▼
  functional group detection
         │
         ▼
  per-group role prediction
         │
         ▼
  negotiation layer
         │
         ▼
  reaction center identification
         │
         ▼
  (optional) MLIP energy / 3D visualization
```

---

## Module 3 v0.1: Predictor — [docs/predictor.md](predictor.md)

`mendel/predictor.py` implements the first role predictor: a deterministic rule-based
baseline that assigns one of the five MENDEL roles to each functional-group agent.
Rules fire in priority order: radical context → leaving group → pericyclic context →
ionic nucleophile/electrophile → spectator fallback.  Thresholds are configurable via
`RuleBasedPredictorConfig`.  No MLP, no negotiation.

---

## Module 4: Labels — [docs/labels.md](labels.md)

`mendel/labels.py` defines `LabeledGroupRole` and `LabeledReaction` dataclasses with
load/save/validate/summarise utilities and a training-row flattener.
Ground-truth labels for the five benchmark reactions live in `data/reactions.json`.
Schema: `LabelValidationError` guards splits (`train`/`val`/`test`), duplicate group IDs,
and empty `atom_indices`.

---

## Module 2: Descriptor — [docs/descriptor.md](descriptor.md)

`mendel/descriptor.py` converts each `FunctionalGroup` into a deterministic 55-dimensional
feature vector (identity, electronic, local environment, mechanistic heuristic scores,
reaction context). Heuristic scores are chemistry priors — not role predictions.
Schema version: `phase3_v1`.

---

## Module 1: Identifier — [docs/identifier.md](identifier.md)

`mendel/identifier.py` identifies functional groups in parsed reaction molecules using
RDKit SMARTS matching. Returns `FunctionalGroup` objects with per-atom references.
Detection runs in three passes: aromatic rings, primary SMARTS groups (priority-ordered),
then contextual groups (`alpha_carbon`, `benzylic_site`).

---

## Module 0: Parser — [docs/parser.md](parser.md)

`mendel/parser.py` converts a reaction SMILES string (`reactants>>products`) into
`ParsedReaction` objects with per-molecule charge, radical, and atom-mapping info.
Functional group identification is **not** performed here.

---

## Phase 0 — Project Contracts

Phase 0 defines the data contracts used by all later phases:

- `mendel.types` — enums (`ReactionContext`, `Role`, `FunctionalGroupType`) and dataclasses (`AtomRef`, `FunctionalGroup`, `RoleAssignment`, `ReactionRecord`)
- `mendel.constants` — derived constant sets
- `data/reactions.example.json` — schema examples for SN2, Diels-Alder, and radical bromination

No chemistry parsing, SMARTS matching, or ML is included in Phase 0.

---

## Phases Overview

| Phase | Description |
|-------|-------------|
| 0 | Project scaffold and data contracts |
| 1 | Functional group identifier (RDKit + SMARTS) |
| 2 | Group descriptor builder |
| 3 | Group agent role predictor — v0.1 rule-based, v1.0 MLP |
| 4 | Labeled data schema + example labels |
| 5 | Rule-based role predictor — Module 3 v0.1 (current) |
| 6 | Negotiation / conflict resolution |
| 7 | MLP role predictor |
| 8 | MLIP wrapper + benchmarking + visualization |
