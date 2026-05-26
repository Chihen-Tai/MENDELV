# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

Use the dedicated conda env — never base or Homebrew python3:

```bash
conda create -n mendel python=3.12   # first time only
conda activate mendel
pip install -e ".[dev]"
```

## Commands

```bash
pytest                                                           # all tests
pytest tests/test_predictor.py::test_acceptance_snippet          # single test
ruff check mendel/ tests/                                        # lint
ruff format mendel/ tests/                                       # format
mypy mendel/                                                     # type check
```

## Architecture

MENDEL treats each functional group in a molecule as an independent agent that predicts its own reaction role. The pipeline is implemented through Phase 5:

```
reaction SMILES + context
    → parser.py        ParsedReaction  (reactant/product molecules, charge, atom-maps)
    → identifier.py    list[FunctionalGroup]  (RDKit SMARTS, 3-pass detection)
    → descriptor.py    list[GroupDescriptor]  (55-dim feature vector per group)
    → predictor.py     PredictionReport  (rule-based role assignment)
    → [negotiation.py] conflict resolution  — Phase 6, not yet implemented
    → [mlip.py]        MACE energy/forces   — Phase 8, not yet implemented
```

**Core abstraction: functional group = agent.** Every `FunctionalGroup` has a stable `group_id` (format `mol{N}_{type}_{M}`, e.g. `mol0_halide_0`) and predicts one of five mutually exclusive `Role` values.

### Implemented modules

| Module | File | Purpose |
|--------|------|---------|
| Types / contracts | `mendel/types.py` | `AtomRef`, `FunctionalGroup`, `RoleAssignment`, `ReactionRecord`, `Role`, `ReactionContext`, `FunctionalGroupType` |
| Constants | `mendel/constants.py` | `SUPPORTED_ROLES`, `SUPPORTED_CONTEXTS`, `SUPPORTED_FUNCTIONAL_GROUPS` as `frozenset`s |
| Parser | `mendel/parser.py` | Splits `reactants>>products` SMILES into `ParsedReaction`; no SMARTS here |
| Identifier | `mendel/identifier.py` | 3-pass SMARTS detection → `list[FunctionalGroup]` |
| Descriptor | `mendel/descriptor.py` | 55-dim `GroupDescriptor` per group; schema version `phase3_v1` |
| Labels | `mendel/labels.py` | `LabeledReaction` / `LabeledGroupRole`; load/save/validate labeled JSON datasets |
| Predictor | `mendel/predictor.py` | `RuleBasedRolePredictor` — threshold rules on descriptor scores |

### Role taxonomy (five mutually exclusive roles)

`reactive_nucleophile`, `reactive_electrophile`, `reactive_radical`, `leaving_group`, `spectator`

### Descriptor schema — `mendel/descriptor.py`

55 features in fixed order (schema version `phase3_v1`):

| Category | Count | Key features |
|----------|-------|--------------|
| A. Identity | 21 | One-hot over 17 `FunctionalGroupType` + atom/heteroatom counts |
| B. Electronic | 10 | Gasteiger charges, electronegativity, `has_pi_bond`, formal charge |
| C. Local environment | 9 | Neighbor heteroatom count, distances, `env_alpha_carbon`, `in_reactant` |
| D. Mechanistic heuristic scores | 5 | `nucleophilicity_score`, `electrophilicity_score`, `leaving_group_score`, `acidity_score`, `radical_stability_score` |
| E. Reaction context | 10 | `context_ionic/radical/pericyclic/unknown`, condition flags |

The mechanistic scores are chemistry priors, not role predictions — the predictor reads them as inputs.

### Predictor rule priority — `mendel/predictor.py`

1. Radical context (benzylic type override → high `radical_stability_score` → halide LG → spectator)
2. Leaving group (`leaving_group_score >= 0.50`)
3. Pericyclic context (pi-bond groups → nucleophile/electrophile by score comparison; flat taxonomy)
4. Ionic context (alpha carbon acidity → negative charge boost → score thresholds)
5. Spectator fallback

### SMARTS matching priority — `mendel/identifier.py`

Three passes:
1. **Aromatic rings** via `mol.GetRingInfo()`
2. **Primary SMARTS** (specific before general): `carboxylic_acid` → `ester` → `amide` → `carbonyl` → `phenol` → `alcohol` → `ether` → `nitro` → `nitrile` → `halide` → `amine` → `alkene` → `alkyne`
3. **Contextual** (second pass): `alpha_carbon`, `benzylic_site`

### Data files

| File | Contents |
|------|----------|
| `data/reactions.json` | 8 labeled reactions (5 benchmark + 3 extended) with `group_roles`, `reaction_center_atoms` |
| `data/reactions.minimal.json` | 2 reactions for fast unit tests |
| `data/label_schema.example.json` | Field reference for the JSON format |
| `groups/<name>.md` | Per-group SMARTS, allowed roles, negotiation rules (copy `TEMPLATE.md` to add a group) |

### Benchmark — `BENCHMARK.md`

Five reactions define the ≥ 80 % role-accuracy target: SN2, E2, Diels-Alder, Aldol, Radical bromination. Phase 5 rule-based predictor is the accuracy baseline; Phase 6 conflict resolution raises it.

### Phase-gated dependencies

| Phase | Status | New deps |
|-------|--------|----------|
| 0–5 | Implemented | `rdkit`, stdlib only |
| 6 — negotiation | Not started | none |
| 7 — MLP predictor | Not started | `torch` |
| 8 — MLIP/viz | Not started | `mace-torch`, `ase`, `py3Dmol`, `matplotlib` |
