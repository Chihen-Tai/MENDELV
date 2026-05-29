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

## Phase 10.5: Energy/Force Figures — [docs/figures.md](figures.md)

Phase 10.5 generates fixed-conformer energy/force comparison figures from the
rMD17 ethanol reference benchmark. It plots raw and mean-shifted DFT-vs-MACE-OFF
energy parity, force RMSE by element, local force RMSE by group or pseudo-group,
and atom-level force error distributions.

This is benchmark visualization and MENDELV-local error decomposition. It does
not train MLIP, run DFT, or evaluate reaction paths or barriers.

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

## Phase 8: Benchmark Evaluator — [docs/benchmark.md](benchmark.md)

`mendel/benchmark.py` compares the rule-based local predictor, the negotiated
rule-based pipeline, and optional MLP checkpoints against curated
`LabeledReaction.group_roles`. Phase 8 validates whether the MLP improves over
rules and whether negotiation improves reaction-level consistency.

This phase is evaluation only. It does not train, does not introduce MLIP/MACE,
and does not measure energy, forces, transition states, or barriers.

## Phase 8.5: Dataset Quality and MLP Readiness — [docs/dataset_quality.md](dataset_quality.md)

`mendel/dataset_quality.py` normalizes mechanism labels, reports label quality
issues, and surfaces MLP training-data risks such as low role counts and draft
labels. It exists because Phase 8 showed `rule_based_negotiated` remains the
best current pipeline while the MLP is data-limited.

This phase improves data readiness only. It does not change role taxonomy,
functional-group taxonomy, model architecture, or the rule-based default.

## Phase 8.8: MLP Diagnostics — [docs/diagnostics.md](diagnostics.md)

`mendel/diagnostics.py` analyzes saved benchmark reports to explain the gap
between promoted MLP role accuracy and reaction-center F1. It reports
rule/MLP disagreements, high-confidence MLP errors, calibration bins,
spectator/reactive confusion, and reaction-center failure patterns.

This phase is analysis only. It does not train, does not add MLIP/MACE, and
does not replace `rule_based_negotiated` as the conservative default.

## Phase 8.10: Atom-Level Reaction-Center Head — [docs/center_head.md](center_head.md)

`mendel/center_head.py` adds an experimental atom binary classifier for
reaction-center recovery. It predicts whether each reactant atom belongs to the
reaction center using existing group membership, role predictions, confidence,
and mechanism context.

This phase targets the remaining gap between high promoted-MLP role accuracy
and lower MLP-guided reaction-center F1. It is not MLIP training and does not
predict energy, geometry, or barriers.

## Phase 8.11: Leakage-Resistant Center Validation — [docs/center_validation.md](center_validation.md)

`mendel/center_validation.py` audits `reaction_center_atoms` labels and creates
template-aware, mechanism-aware, source-aware, or reaction-ID-prefix splits to
test whether the atom-center head generalizes beyond generated templates.

This phase validates center supervision before any Phase 9 MLIP work. It does
not train MLIP and does not run energy, force, transition-state, IRC, NEB, MD,
or barrier prediction.

## Phase 8.13: Manual Center Expansion — [docs/center_expansion.md](center_expansion.md)

`mendel/center_expansion_review.py` conservatively promotes center-focused
candidate reactions and reruns mechanism-balanced strict validation on the
expanded cleaned dataset.

This phase broadens reaction-center supervision across mechanisms. It is still
not MLIP training and does not predict energy or force.

## Phase 8.14: Mapping-Aware Center Labels — [docs/mapping_center.md](mapping_center.md)

`mendel/mapping_center.py` audits `reaction_center_atoms` against atom-mapped
bond changes when mapped reaction SMILES are available. Phase 8.14 also adds a
`val_test_balanced_template` split strategy so both validation and test can
carry broader mechanism coverage.

This phase improves label reliability and validation balance only. It does not
train MLIP and does not predict energy or force.

## Phase 9: Optional Pretrained MLIP Backend — [docs/mlip.md](mlip.md)

`mendel/mlip.py` provides an optional single-point energy/force backend using
ASE and pretrained MACE when the `mlip` extra is installed. MENDELV supplies
reaction-center atoms, and the MLIP result is summarized over those atoms.

This phase does not train MLIP, does not fine-tune MACE, does not use
Transition1x, and does not run DFT, NEB, IRC, MD, transition-state search, or
barrier prediction.

## Phase 10: Reference Energy/Force Benchmark — [docs/reference_benchmark.md](reference_benchmark.md)

`mendel/reference_data.py` defines open reference energy/force records and
benchmark metrics. `mendel/qo2mol.py` provides a local QO2Mol sample adapter for
JSON, JSONL, and simple NPZ samples. The benchmark compares pretrained MACE-OFF
single-point predictions with reference molecular conformer energies/forces.

This phase does not download the full dataset by default, commit raw data, train
MLIP, run DFT, or claim reaction-path or barrier accuracy.

## Phase 10.1: QO2Mol Sample Manager — [docs/qo2mol.md](qo2mol.md)

`mendel/qo2mol_manager.py` and `scripts/qo2mol_sample_manager.py` inspect,
register, sample, and summarize local QO2Mol data. The manager prepares capped
MENDELV reference JSON files without committing raw QO2Mol data or requiring
ASE/MACE.

This phase is local data preparation only. It does not download the full
dataset by default, train MLIP, run DFT, or benchmark reaction barriers.

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
| 8 | Benchmark evaluator |
| 8.5 | Dataset normalization and MLP readiness diagnostics |
| 8.8 | MLP diagnostics and reaction-center failure analysis |
