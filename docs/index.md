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

## Current Freeze Target

The current freeze target is **Phase 0-6**:

- Phase 0: project contracts
- Phase 1: reaction SMILES parser
- Phase 2: functional group identifier
- Phase 3: descriptor builder
- Phase 4: labeled data schema
- Phase 5: rule-based role predictor
- Phase 6: negotiation layer

Before Phase 7 can begin, **Phase 6.5 (dataset curation)** must be completed:
Phase 6.5 generates draft labels from the Phase 0–6 pipeline for manual review.
Only after enough curated labels exist in `data/reactions.json` should Phase 7 begin.

The next phase after curation is **Phase 7 MLP role predictor training**. It is paused
pending sufficient curated data. Phase 0–6 imports and validation must not require PyTorch.

## Pre-training Freeze Checklist

- `import mendel` works without `torch`.
- `from mendel.negotiator import run_full_rule_pipeline` works without `torch`.
- Phase 0-6 tests pass.
- `run_full_rule_pipeline(smiles, context)` works for the smoke cases.
- No training scripts are run.

---

## Phase 6.5: Dataset Curation / Draft Labels — [docs/curation.md](curation.md)

`mendel/curation.py` runs the Phase 0–6 pipeline over new reaction SMILES and exports
draft `LabeledReaction` records for manual review.  All draft records are marked
`confidence="draft"` and `needs_manual_review=true`.  A chemist must correct roles,
change split from `"draft"` to `"train"`/`"val"`/`"test"`, and set
`needs_manual_review=false` before records are added to `data/reactions.json`.

The CLI is `scripts/draft_labels.py` with `--core`, `--extended`, and `--input` options.
Phase 7 training must not begin until curated labels are ready.

---

## Phase 7: MLP Role Predictor — [docs/mlp.md](mlp.md)

`mendel/mlp.py` implements the first learned role predictor: a small PyTorch
MLP trained on Phase 3 descriptor vectors to classify each functional-group
agent into one of the five MENDEL roles.  Loss is CrossEntropyLoss on raw
logits; softmax is applied only at inference time.  The MLP is experimental
and supplements — but does not replace — the rule-based predictor and Phase 6
negotiation.  `train_from_labeled_json(path)` is the one-call training
entry point.

This is **Phase 7** and is paused/future for the current freeze. It trains role
classification only. MLIP, energy/force prediction, and Transition1x are out of scope.

---

## Phase 6: Negotiation Layer — [docs/negotiator.md](negotiator.md)

`mendel/negotiator.py` coordinates raw Phase 5 predictions into a globally consistent
reaction-level interpretation.  It infers a mechanism hint, resolves role conflicts
(aldol disambiguation, Diels-Alder diene/dienophile split, SN2/E2 leaving-group
confirmation, radical center promotion), identifies reaction center atoms, and emits
structured warnings for missing chemistry or v0.1 limitations.

`run_full_rule_pipeline(smiles, context)` is the first true one-call public pipeline,
chaining parser → identifier → predictor → negotiator internally.

---

## Phase 5: Rule-Based Role Predictor — [docs/predictor.md](predictor.md)

`mendel/predictor.py` implements the first role predictor: a deterministic rule-based
baseline that assigns one of the five MENDEL roles to each functional-group agent.
Rules fire in priority order: radical context → leaving group → pericyclic context →
ionic nucleophile/electrophile → spectator fallback.  Thresholds are configurable via
`RuleBasedPredictorConfig`.  No MLP, no negotiation.

---

## Phase 4: Labeled Data Schema — [docs/labels.md](labels.md)

`mendel/labels.py` defines `LabeledGroupRole` and `LabeledReaction` dataclasses with
load/save/validate/summarise utilities and a training-row flattener.
Ground-truth labels for the five benchmark reactions live in `data/reactions.json`.
Schema: `LabelValidationError` guards splits (`train`/`val`/`test`), duplicate group IDs,
and empty `atom_indices`.

---

## Phase 3: Descriptor Builder — [docs/descriptor.md](descriptor.md)

`mendel/descriptor.py` converts each `FunctionalGroup` into a deterministic 55-dimensional
feature vector (identity, electronic, local environment, mechanistic heuristic scores,
reaction context). Heuristic scores are chemistry priors — not role predictions.
Schema version: `phase3_v1`.

---

## Phase 2: Identifier — [docs/identifier.md](identifier.md)

`mendel/identifier.py` identifies functional groups in parsed reaction molecules using
RDKit SMARTS matching. Returns `FunctionalGroup` objects with per-atom references.
Detection runs in three passes: aromatic rings, primary SMARTS groups (priority-ordered),
then contextual groups (`alpha_carbon`, `benzylic_site`).

---

## Phase 1: Parser — [docs/parser.md](parser.md)

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
| 1 | Reaction SMILES parser |
| 2 | Functional group identifier (RDKit + SMARTS) |
| 3 | Group descriptor builder |
| 4 | Labeled data schema + example labels |
| 5 | Rule-based role predictor |
| 6 | Negotiation / conflict resolution |
| 6.5 | Dataset curation / draft label generation |
| 7 | MLP role predictor (paused — needs curated data) |
| 8 | MLIP wrapper + benchmarking + visualization |
