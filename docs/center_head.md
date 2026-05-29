# Phase 8.10: Atom-Level Reaction-Center Head

Phase 8.10 adds an experimental binary classifier for atom-level reaction-center
recovery:

```text
atom index -> center / non-center
```

Functional group = agent remains the project abstraction. The atom head is a
small downstream classifier that uses existing group membership, role
predictions, role confidence, atom properties, and mechanism context to decide
which reactant atoms should be in `reaction_center_atoms`.

## Why This Exists

Phase 8.7 showed that the promoted MLP role predictor has strong group-level
role accuracy. Phase 8.9 showed that MLP-aware negotiation improves
reaction-center F1, but still trails `rule_based_negotiated`.

That gap means role prediction is no longer the main bottleneck. The remaining
problem is atom-level center extraction: a functional-group agent can have the
right role while the predicted atom set is incomplete, too broad, or mismatched
to the labels.

## Scope

This is not MLIP training. It does not use 3D geometry, graph neural networks,
energy prediction, or external reaction datasets. The current head is a small
PyTorch MLP over explainable atom features:

- atomic number and electronegativity
- group membership count
- group role indicators
- role confidence summaries
- functional-group type indicators
- mechanism-type indicators
- simple local atom properties such as degree, charge, aromaticity, and ring flag

Torch is imported only inside training and prediction functions, so normal
MENDELV imports remain lightweight.

## Commands

Train the experimental atom head:

```bash
python scripts/train_center_head.py \
  --data data/reactions.proposed_with_auto_promoted.normalized.json \
  --role-checkpoint models/role_mlp_promoted.pt \
  --output models/atom_center_head.pt \
  --report reports/atom_center_training_report.json \
  --epochs 80 \
  --hidden-dim 32 \
  --batch-size 32 \
  --learning-rate 1e-3 \
  --threshold 0.5 \
  --use-class-weights
```

Benchmark the saved head:

```bash
python scripts/benchmark_center_head.py \
  --data data/reactions.proposed_with_auto_promoted.normalized.json \
  --center-checkpoint models/atom_center_head.pt \
  --role-checkpoint models/role_mlp_promoted.pt \
  --threshold 0.5 \
  --device cpu \
  --output reports/atom_center_benchmark_report.json \
  --comparison-output reports/center_head_comparison.json
```

The benchmark compares the atom head with the known Phase 8.9 references:

- `rule_based_negotiated` reaction-center F1: 0.8929
- `new_mlp_aware_negotiated` reaction-center F1: 0.7973

## Limitations

- Center labels may be noisy or incomplete.
- Control/no-reaction examples with empty truth are legitimate all-negative
  examples, but they create metric tension.
- The head uses current atom labels and group membership only.
- There is no 3D geometry and no graph neural network yet.
- Atom-map coverage and current functional-group granularity still constrain the
  best possible score.

Future work could make this mapping-aware or replace the feature MLP with a
small graph model once center labels are cleaner.
