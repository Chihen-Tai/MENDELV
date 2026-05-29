# Phase 8.12: Center-Label Cleanup

Phase 8.12 cleans conservative reaction-center label issues before using the
atom-center head as evidence for later work.

Functional group = agent remains the project model. This phase only improves
`reaction_center_atoms` labels and validation splits. It does not train MLIP,
does not use MACE or Transition1x, and does not predict energy or force.

## Why Cleanup Is Needed

Phase 8.11 found strict-split test F1 remained useful, but center-label audit
still reported error and warning issues. Those issues can make reaction-center
F1 measure label noise rather than model quality.

Common issue types:

- `control_has_center_atoms`: controls should have an empty center.
- `spectator_only_with_center`: all groups are spectators but a center exists.
- `diels_alder_substituent_center_atoms`: EWG substituent atoms are included
  instead of only reacting alkene atoms.
- `center_atom_outside_detected_groups`: labels include atoms outside current
  detected functional-group agents.
- `reactive_empty_center`: a reactive mechanism has no center labels.
- `sn2_e2_missing_halide_center`: an SN2/E2 center misses the leaving group.

## Empty-Center Policy

- Controls and no-reaction examples should use `reaction_center_atoms = []`.
- Reactive mechanisms should usually have non-empty center labels.
- Empty truth should be separated from model failure: an empty true center in a
  reactive mechanism is a label issue, not a successful empty prediction.

## Commands

Cleanup only:

```bash
python scripts/cleanup_center_labels.py \
  --input data/reactions.proposed_with_auto_promoted.normalized.json \
  --output data/reactions.center_cleaned.json \
  --report reports/center_cleanup_report.json \
  --conservative
```

Full cleaned strict validation workflow:

```bash
python scripts/run_center_cleanup_validation.py
```

Manual equivalent:

```bash
python scripts/validate_center_labels.py \
  --data data/reactions.center_cleaned.json \
  --strategy mechanism_balanced_template \
  --output-data data/reactions.center_cleaned.mechanism_balanced_split.json \
  --report reports/center_validation_cleaned_report.json

python scripts/retrain_center_head_strict_split.py \
  --data data/reactions.center_cleaned.mechanism_balanced_split.json \
  --output models/atom_center_head_cleaned_strict.pt \
  --report reports/atom_center_training_cleaned_strict_report.json

python scripts/benchmark_center_head_strict_split.py \
  --data data/reactions.center_cleaned.mechanism_balanced_split.json \
  --center-checkpoint models/atom_center_head_cleaned_strict.pt \
  --output reports/atom_center_benchmark_cleaned_strict_report.json \
  --comparison-output reports/center_head_cleaned_strict_comparison.json
```

If cleaned strict validation drops, report the drop and inspect whether it came
from label-target changes, stricter splitting, remaining label noise, or limited
mechanism coverage.
