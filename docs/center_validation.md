# Phase 8.11: Leakage-Resistant Center Validation

Phase 8.11 checks whether the Phase 8.10 atom-level reaction-center head is
robust or partly benefiting from generated-template leakage.

Functional group = agent remains the project abstraction. This phase validates
the downstream atom-center supervision that may later guide more expensive
analysis. It does not train MLIP, does not use MACE or Transition1x, and does
not predict energies or forces.

## Why Stricter Validation Is Needed

The non-strict Phase 8.10 benchmark was strong:

- `atom_center_head` reaction-center F1: 0.9054
- `rule_based_negotiated` reaction-center F1: 0.8929
- `new_mlp_aware_negotiated` reaction-center F1: 0.7973

That result is promising, but the promoted reactions include deterministic
template-generated examples. If related template instances appear across
train/val/test, the atom head can learn template-specific center patterns rather
than robust atom-level chemistry.

## Validation Strategy

`mendel/center_validation.py` provides:

- template-aware split assignment
- mechanism-aware, source-aware, and reaction-ID-prefix split strategies
- center-label audits
- leakage validation reports

The default strategy is `template`, which keeps all reactions with the same
template key in the same split.

## Center-Label Audit

The audit checks:

- invalid `reaction_center_atoms`
- duplicate center atoms
- controls with non-empty centers
- reactive mechanisms with empty centers
- unusually broad centers
- centers outside detected functional groups
- spectator-only reactions with centers
- mechanism-specific suspicious labels for SN2/E2, Diels-Alder, carbonyl
  addition, and radical bromination

Empty truth is treated as valid for control/no-reaction mechanisms. Empty truth
for reactive mechanisms is reported as a label issue.

## Commands

Create a leakage-resistant split and audit report:

```bash
python scripts/validate_center_labels.py \
  --data data/reactions.proposed_with_auto_promoted.normalized.json \
  --strategy template \
  --output-data data/reactions.center_validated.template_split.json \
  --report reports/center_validation_report.json
```

Retrain the atom head on the strict split:

```bash
python scripts/retrain_center_head_strict_split.py \
  --data data/reactions.center_validated.template_split.json \
  --role-checkpoint models/role_mlp_promoted.pt \
  --output models/atom_center_head_template_split.pt \
  --report reports/atom_center_training_template_split_report.json \
  --epochs 80 \
  --hidden-dim 32 \
  --batch-size 32 \
  --learning-rate 1e-3 \
  --threshold 0.5 \
  --use-class-weights
```

Benchmark train/val/test splits separately:

```bash
python scripts/benchmark_center_head_strict_split.py \
  --data data/reactions.center_validated.template_split.json \
  --center-checkpoint models/atom_center_head_template_split.pt \
  --role-checkpoint models/role_mlp_promoted.pt \
  --threshold 0.5 \
  --device cpu \
  --output reports/atom_center_benchmark_template_split_report.json \
  --comparison-output reports/center_head_template_split_comparison.json
```

## Interpretation

If strict-split test F1 remains strong, the atom-center head is better evidence
for robust reaction-center recovery and Phase 9 can be considered. If strict
test F1 collapses, treat the Phase 8.10 result as likely leakage-sensitive and
prioritize center-label cleanup, data expansion, mapping-aware labels, or a
mechanism-balanced split before MLIP work.

## Phase 8.12 Update

Phase 8.12 adds conservative center-label cleanup and a
`mechanism_balanced_template` split strategy. The cleanup pass standardizes
empty centers for controls, removes Diels-Alder substituent center atoms when
reacting alkene atoms remain, completes obvious SN2/E2 halide centers, and
keeps atom-center checkpoints separate from earlier Phase 8.10/8.11 runs.

Use the mechanism-balanced strategy when validating cleaned labels:

```bash
python scripts/validate_center_labels.py \
  --data data/reactions.center_cleaned.json \
  --strategy mechanism_balanced_template \
  --output-data data/reactions.center_cleaned.mechanism_balanced_split.json \
  --report reports/center_validation_cleaned_report.json
```

The atom center head remains experimental unless cleaned strict test coverage
is broad and reaction-center F1 remains strong.

## Phase 8.13 Update

Phase 8.13 promotes conservative center-expansion candidates and reruns the
mechanism-balanced strict split on the expanded cleaned dataset. The validation
targets are:

- test reactions >= 15
- test mechanisms >= 5
- strict test reaction-center F1 > 0.80

The workflow is:

```bash
python scripts/promote_center_expansion.py
python scripts/run_center_expansion_validation.py --device cpu
```

Passing this stricter expanded validation is stronger evidence for using the
atom-center head in later phases, but it still does not make MENDELV an MLIP
system.

## Phase 8.14 Update

Phase 8.14 adds mapping-aware center audit utilities and a
`val_test_balanced_template` split strategy. The mapping audit uses atom-map
numbers, when available, to infer bond-change centers and compare them with
`reaction_center_atoms`. If reaction SMILES are unmapped, the audit reports that
mapping is unavailable and continues with balanced validation.

The new split strategy keeps each template/leakage group in one split while
trying to make both validation and test broad:

- validation reactions >= 15 when possible
- test reactions >= 15 when possible
- validation mechanisms >= 5 when possible
- test mechanisms >= 5 when possible
- positive center atom counts reported by split

Run the Phase 8.14 workflow:

```bash
python scripts/audit_mapping_centers.py \
  --data data/reactions.center_expanded.cleaned.json \
  --output reports/mapping_center_audit_report.json \
  --apply-high-confidence \
  --apply-suggestions-output data/reactions.center_expanded.mapping_audited.json

python scripts/run_balanced_center_validation.py \
  --data data/reactions.center_expanded.cleaned.json \
  --use-mapping-suggestions \
  --device cpu \
  --epochs 80 \
  --threshold 0.5 \
  --output-prefix center_balanced
```

Stable val/test F1 is required before treating the atom-center head as credible
guidance for later MLIP-related analysis.
