# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

Use the dedicated conda env — never base or Homebrew python3:

```bash
conda create -n mendel python=3.12   # first time only
conda activate mendel
pip install -e ".[dev]"
pip install -e ".[ml]"               # Phase 7 only — installs torch
pip install -e ".[mlip]"             # Phase 9 only — installs mace-torch, ase
```

## Commands

```bash
pytest                                                           # all tests
pytest tests/test_predictor.py::test_acceptance_snippet          # single test
ruff check mendel/ tests/                                        # lint
ruff format mendel/ tests/                                       # format
mypy mendel/                                                     # type check
```

### Phase 0–6 freeze validation (canonical command)

```bash
PYTHONDONTWRITEBYTECODE=1 pytest -q -p no:cacheprovider \
  tests/test_phase0_scaffold.py tests/test_parser.py \
  tests/test_identifier.py tests/test_descriptor.py \
  tests/test_labels.py tests/test_predictor.py tests/test_negotiator.py
```

**Do not run `tests/test_mlp.py` or `scripts/train_mlp.py` unless intentionally starting Phase 7 work.**

**Do not run `tests/test_mlip.py`, `tests/test_mlip_env_scripts.py`, `tests/test_mlip_geometry_sanity.py`, or `tests/test_mlip_reference_benchmark.py` without Phase 9 deps installed.**

## Architecture

MENDEL treats each functional group in a molecule as an independent agent that predicts its own reaction role. The pipeline is:

```
reaction SMILES + context
    → parser.py          ParsedReaction  (reactant/product molecules, charge, atom-maps)
    → identifier.py      list[FunctionalGroup]  (RDKit SMARTS, 3-pass detection)
    → descriptor.py      list[GroupDescriptor]  (55-dim feature vector per group)
    → predictor.py       PredictionReport  (rule-based role assignment)        ← Phase 5
    → negotiator.py      NegotiationResult  (global consistency, mechanism hint) ← Phase 6
    → [mlp.py]           MLPRolePrediction  (learned role predictor)            ← Phase 7, paused
    → [center_head.py]   atom-level center predictions                         ← Phase 8.10
    → [mlip.py]          MACE energy/forces                                     ← Phase 9
    → [reference_data.py / md17.py / qo2mol.py]  reference conformer data      ← Phase 10
```

**Entry points:**
```python
# Rule-based pipeline (Phases 1–6) — no PyTorch required
from mendel import run_full_rule_pipeline
result = run_full_rule_pipeline("CCBr.[OH-]>>CCO.[Br-]", context="ionic")

# MLP training (Phase 7) — requires pip install -e ".[ml]"
from mendel.mlp import train_from_labeled_json, TrainingConfig   # NOT from mendel
predictor, history, report = train_from_labeled_json("data/reactions.json")

# CLI training (Phase 7 only)
# python scripts/train_mlp.py --data data/reactions.json --epochs 100
```

`import mendel` does **not** import PyTorch, matplotlib, ASE, or MACE. All optional deps are loaded lazily inside the modules that need them.

**Core abstraction: functional group = agent.** Every `FunctionalGroup` has a stable `group_id` (format `mol{N}_{type}_{M}`, e.g. `mol0_halide_0`) and predicts one of five mutually exclusive `Role` values.

### Phase status

| Phase | Status | Dependencies |
|-------|--------|--------------|
| 0–6 | Implemented | `rdkit`, stdlib only |
| 6.5 — Dataset curation / label drafting | In progress | `rdkit`, stdlib only |
| 7 — MLP role predictor training | Paused (needs curated data) | `torch>=2.0` |
| 8 — Benchmark, center head, dataset ops | Implemented | `rdkit`, stdlib only |
| 9 — MLIP single-point backend | Partial | `mace-torch`, `ase` |
| 10 — Reference energy/force data (MD17, QO2Mol) | Partial | stdlib only (no MLIP) |

### Implemented modules

| Module | File | Phase | Purpose |
|--------|------|-------|---------|
| Types / contracts | `mendel/types.py` | — | `AtomRef`, `FunctionalGroup`, `RoleAssignment`, `ReactionRecord`, `Role`, `ReactionContext`, `FunctionalGroupType` |
| Constants | `mendel/constants.py` | — | `SUPPORTED_ROLES`, `SUPPORTED_CONTEXTS`, `SUPPORTED_FUNCTIONAL_GROUPS` as `frozenset`s |
| Parser | `mendel/parser.py` | 1 | Splits `reactants>>products` SMILES into `ParsedReaction`; no SMARTS here |
| Identifier | `mendel/identifier.py` | 2 | 3-pass SMARTS detection → `list[FunctionalGroup]` |
| Descriptor | `mendel/descriptor.py` | 3 | 55-dim `GroupDescriptor` per group; schema version `phase3_v1` |
| Labels | `mendel/labels.py` | 4 | `LabeledReaction` / `LabeledGroupRole`; load/save/validate labeled JSON datasets |
| Predictor | `mendel/predictor.py` | 5 | `RuleBasedRolePredictor` — threshold rules on descriptor scores |
| Negotiator | `mendel/negotiator.py` | 6 | `RuleBasedNegotiator` — global consistency, mechanism hints, reaction center inference |
| Curation | `mendel/curation.py` | 6.5 | Draft `LabeledReaction` records from rule pipeline for manual review |
| MLP predictor | `mendel/mlp.py` | 7 | `RoleMLP` + `MLPRolePredictor` — learned descriptor→role classifier; paused |
| Benchmark | `mendel/benchmark.py` | 8 | Label-conditioned role and reaction-center accuracy evaluators |
| Dataset quality | `mendel/dataset_quality.py` | 8.5 | Dataset normalization and diagnostics before training |
| Promotion | `mendel/promotion.py` | 8.6 | Conservative promotion of human-reviewed auto-candidate labels |
| Diagnostics | `mendel/diagnostics.py` | 8.8 | Disagreement/failure analysis from saved benchmark reports |
| Center head | `mendel/center_head.py` | 8.10 | Binary atom-level reaction-center classifier from group features |
| Center validation | `mendel/center_validation.py` | 8.11 | Leakage-resistant center label audit and deterministic train/val splits |
| Center cleanup | `mendel/center_cleanup.py` | 8.12 | Conservative center label correction |
| Center expansion review | `mendel/center_expansion_review.py` | 8.13 | Review queue for center-expansion reactions |
| Mapping center | `mendel/mapping_center.py` | 8.14 | Atom-map-based reaction-center label audit and suggestion |
| MLIP backend | `mendel/mlip.py` | 9 | Optional MACE/ASE single-point energy and forces; lazy import |
| Local force analysis | `mendel/local_force_analysis.py` | 9 | Functional-group-local force error analysis against reference MLIP |
| Reference data | `mendel/reference_data.py` | 10 | `ReferenceStructureRecord` JSON schema and benchmark math |
| MD17 adapter | `mendel/md17.py` | 10.2 | Converts local MD17/rMD17 NPZ arrays to `ReferenceStructureRecord` |
| QO2Mol ingestion | `mendel/qo2mol.py` | 10 | Local QO2Mol sample → reference energy/force records (no download) |
| QO2Mol manager | `mendel/qo2mol_manager.py` | 10 | Manages QO2Mol ingestion jobs |
| Plotting | `mendel/plotting.py` | — | Matplotlib report plots; matplotlib imported lazily inside functions |

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

### MLP predictor — `mendel/mlp.py`

Key constants: `ROLE_TO_INDEX` (Role→int, fixed order), `INDEX_TO_ROLE` (reverse), `DEFAULT_MODEL_VERSION = "phase7_mlp_v1"`.

Architecture: `Linear(55, hidden_dim) → ReLU → Dropout → Linear(hidden_dim, 5)` returning raw logits. Softmax applied only at inference time. Loss is `CrossEntropyLoss` on raw logits — never apply softmax before the loss.

Key types: `TrainingExample` (group_id, features, role), `TrainingConfig` (hidden_dim=32, epochs=100, lr=1e-3, seed=42, early_stopping_patience=15), `TrainingHistory`, `MLPRolePrediction` (predicted_role, confidence, probabilities).

Checkpoint format (`.pt`): `{state_dict, input_dim, hidden_dim, output_dim, dropout, feature_names, model_version}` — safe for `weights_only=True` load.

**Scope boundary**: Phase 7 trains role classification only. Does NOT train MLIP, MACE, or any energy/force model.

### Negotiator — `mendel/negotiator.py`

Key types: `NegotiationResult`, `NegotiatedRoleAssignment` (carries `raw_role`, `final_role`, `subrole`, `is_reaction_center`), `NegotiationWarning`, `NegotiatorConfig`.

`negotiate()` dispatches to a mechanism-specific helper based on `infer_mechanism_hint()`:

| `mechanism_hint` | Trigger condition |
|---|---|
| `sn2_or_e2_like` | ionic + halide or leaving_group predicted |
| `aldol_like` | ionic + carbonyl + alpha_carbon |
| `diels_alder_like` | pericyclic context |
| `radical_bromination_like` | radical context |
| `ionic_addition_like` | ionic + nucleophile + electrophile, no halide |
| `unknown` | no rule matched; raw roles preserved |

Each helper mutates `assign_by_id` in place (the dict of `NegotiatedRoleAssignment`); input `groups` and `predictions` are never mutated.

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

| File / Pattern | Contents |
|----------------|----------|
| `data/reactions.json` | Core labeled reactions with `group_roles`, `reaction_center_atoms` |
| `data/reactions.minimal.json` | 2 reactions for fast unit tests |
| `data/reactions.*.json` | Pipeline-stage outputs: `center_cleaned`, `center_balanced`, `center_expanded`, `center_validated`, etc. |
| `data/reactions.draft.*.json` | Draft (needs human review) label sets at various stages |
| `data/draft_inputs.*.json` | Input SMILES batches for the curation pipeline |
| `data/label_schema.example.json` | Field reference for the labeled reaction JSON format |
| `data/auto_labeling_policy.md` | Policy doc for conservative auto-labeling decisions |
| `data/groups/<name>.md` | Per-group SMARTS, allowed roles, negotiation rules (copy `TEMPLATE.md` to add a group) |
| `reports/*.json` | Saved benchmark report outputs |

### Benchmark — `BENCHMARK.md`

Five reactions define the ≥ 80 % role-accuracy target: SN2, E2, Diels-Alder, Aldol, Radical bromination. Phase 5 rule-based predictor is the accuracy baseline; Phase 6 conflict resolution raises it.
