# Phase 8.8: MLP Diagnostics

Phase 8.8 explains why the promoted-data MLP improved group-level role accuracy
while still trailing `rule_based_negotiated` on reaction-center F1.

Functional group = agent. The diagnostic unit is still the functional-group
agent decision, plus the reaction-level center inferred after negotiation.

## Current Phase 8.7 Results

| Predictor | Role accuracy | Reaction-center F1 |
|-----------|--------------:|-------------------:|
| `rule_based_local` | 0.7218 | 0.0000 |
| `rule_based_negotiated` | 0.8195 | 0.8929 |
| `old_mlp_negotiated` | 0.4511 | 0.4678 |
| `new_mlp_negotiated` | 0.9173 | 0.7481 |

The promoted MLP is the best experimental role-prediction mode, but
`rule_based_negotiated` remains the conservative default because its
reaction-center F1 is higher.

## Role Accuracy vs Reaction-Center F1

Role accuracy asks whether each labeled functional group got the correct role.
Reaction-center F1 asks whether the reaction-level atom set was inferred
correctly. A model can improve local role labels but still produce worse
reaction-center atoms if negotiation, atom mapping, group granularity, or
center extraction does not use the improved role information correctly.

## Diagnostic Categories

- rule correct / MLP wrong
- MLP correct / rule wrong
- both wrong
- high-confidence MLP errors
- role-correct center-wrong failures
- spectator/reactive confusion

## Command

```bash
python scripts/diagnose_mlp.py
```

The default command reads Phase 8.7 benchmark reports from
`reports/benchmark_promoted_full/` and writes
`reports/mlp_diagnostics_report.json`.

## Next-Step Logic

If MLP role accuracy is high but reaction-center F1 remains lower, improve
negotiation or reaction-center extraction before moving to MLIP work. Likely
next steps include MLP-aware negotiation, confidence calibration, additional
control/spectator data, more radical examples, or an atom-level
reaction-center head.

## Phase 8.9 Update

Phase 8.9 introduces MLP-aware negotiation as the first fix for high role
accuracy but weaker reaction-center F1. Run:

```bash
python scripts/benchmark_promoted_mlp.py \
  --data data/reactions.proposed_with_auto_promoted.normalized.json \
  --new-mlp-checkpoint models/role_mlp_promoted.pt \
  --old-mlp-checkpoint models/role_mlp.pt \
  --device cpu \
  --output-dir reports/benchmark_mlp_aware_full \
  --include-mlp-aware-negotiation
```

Then compare diagnostics:

```bash
python scripts/diagnose_mlp.py \
  --rule-report reports/benchmark_mlp_aware_full/rule_based_negotiated.json \
  --new-mlp-report reports/benchmark_mlp_aware_full/new_mlp_aware_negotiated.json \
  --old-mlp-report reports/benchmark_mlp_aware_full/new_mlp_negotiated.json \
  --output reports/mlp_aware_diagnostics_report.json
```

This phase does not train MLIP, does not use MACE or Transition1x, and does
not run energy, force, transition-state, IRC, NEB, MD, or barrier prediction.
