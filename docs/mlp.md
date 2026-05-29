# Module 5 v0.1: MLP Role Predictor

`mendel/mlp.py` — Phase 7

Trains a small PyTorch MLP to classify each functional-group agent's reaction
role from its Phase 3 descriptor vector.  This is the first *learned* role
predictor in MENDEL.

---

## Purpose

Phase 7 introduces a data-driven alternative to the rule-based predictor
(Phase 5).  The MLP learns which descriptor features predict each of the five
MENDEL roles from the manually labeled dataset in `data/reactions.json`.

The rule-based predictor remains the default until the MLP is shown to
outperform it on the benchmark.  Phase 6 negotiation still provides global
consistency — the MLP predicts local (per-group) roles only.

---

## What This Phase Trains

**Role classification only:**

```
GroupDescriptor (55 floats)  →  softmax over 5 roles
```

This phase does **not** train:
- An MLIP (machine-learned interatomic potential)
- Energy or force prediction
- Transition state search or barrier prediction
- Any model using the Transition1x DFT dataset

---

## Input

A `GroupDescriptor` from Phase 3 — a 55-dimensional float vector (schema
version `phase3_v1`) encoding identity, electronic properties, local
environment, mechanistic heuristic scores, and reaction context.

---

## Output

A `MLPRolePrediction` containing:
- `predicted_role`: one of the five MENDEL roles
- `confidence`: scalar in [0, 1] — the softmax probability of the predicted class
- `probabilities`: full 5-class probability distribution

---

## Architecture

```
Input (55) → Linear(55, hidden_dim) → ReLU → Dropout(p) → Linear(hidden_dim, 5) → logits
```

Default: `hidden_dim=32`, `dropout=0.10`.  Logits are returned raw; softmax
is applied externally during inference.

---

## Loss Function

**CrossEntropyLoss** on raw logits.  Softmax is NOT applied before the loss.

```python
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)  # optional weights
loss = criterion(logits, y)  # logits: (batch, 5); y: (batch,) long int
```

Optional inverse-frequency class weighting is available via
`TrainingConfig(use_class_weights=True)`.

---

## Training

```python
from mendel.mlp import TrainingConfig, build_training_examples, train_mlp_role_predictor
from mendel.labels import load_labeled_reactions

reactions = load_labeled_reactions("data/reactions.json")
examples = build_training_examples(reactions, strict_group_matching=False)

config = TrainingConfig(hidden_dim=32, epochs=100, learning_rate=1e-3, seed=42)
predictor, history = train_mlp_role_predictor(examples, config=config)
```

Or via the CLI:

```bash
python scripts/train_mlp.py \
  --data data/reactions.json \
  --output models/role_mlp.pt \
  --report reports/mlp_training_report.json \
  --epochs 100 --hidden-dim 32
```

---

## Checkpoint Save / Load

```python
predictor.save("models/role_mlp.pt")
loaded = MLPRolePredictor.load("models/role_mlp.pt")
```

Checkpoint contains: `state_dict`, `input_dim`, `hidden_dim`, `output_dim`,
`dropout`, `feature_names`, `model_version`.

---

## Public API

```python
from mendel.mlp import (
    ROLE_TO_INDEX,           # dict[Role, int] — fixed class index mapping
    INDEX_TO_ROLE,           # dict[int, Role] — reverse mapping
    DEFAULT_MODEL_VERSION,   # str — "phase7_mlp_v1"
    TrainingExample,         # dataclass: reaction_id, group_id, features, role
    TrainingDatasetSummary,  # dataclass: counts and distributions
    TrainingConfig,          # dataclass: all hyperparameters
    TrainingHistory,         # dataclass: per-epoch loss and accuracy curves
    MLPRolePrediction,       # dataclass: predicted_role, confidence, probabilities
    RoleMLP,                 # nn.Module: Linear → ReLU → Dropout → Linear
    MLPRolePredictor,        # wrapper: predict_descriptor / save / load
    set_random_seed,         # (seed) → None
    build_training_examples, # (labeled_reactions, strict) → list[TrainingExample]
    summarize_training_examples,  # (examples) → TrainingDatasetSummary
    training_examples_to_tensors, # (examples) → (X, y, group_ids)
    compute_class_weights,   # (examples) → Tensor[5]
    train_mlp_role_predictor,     # (examples, config) → (MLPRolePredictor, TrainingHistory)
    evaluate_mlp_predictor,  # (predictor, examples) → dict
    train_from_labeled_json, # (path, config, strict) → (predictor, history, report)
    save_training_report,    # (report, path) → None
)
```

---

## Known Limitations

**Small dataset** — The labeled dataset in `data/reactions.json` contains only
26 reactions (48 labeled groups).  The MLP is still data-limited and
should not be used in production until more labeled data is available.

**Experimental status** — The MLP supplements but does not replace the
rule-based predictor (Phase 5).  It becomes the default only when its
benchmark accuracy exceeds the rule-based baseline (≥ 80 % on the five
benchmark reactions in `BENCHMARK.md`).

**No negotiation** — `MLPRolePredictor` assigns local per-group roles.
Global consistency (e.g., ensuring at most one nucleophile per step) still
requires Phase 6 `RuleBasedNegotiator`.

**No energy or force prediction** — This phase trains role classification
only.  MLIP, MACE, activation energies, barriers, and transition states are
out of scope.

**No Transition1x** — The model is trained on the small hand-labeled MENDEL
dataset, not on any DFT trajectory dataset.

---

## Phase 8/8.5 Follow-up

Phase 8 benchmarks rule-based, negotiated, and optional MLP checkpoint predictors.
Phase 8.5 normalizes dataset labels and reports training-readiness diagnostics.
Neither phase adds MLIP, MACE, Transition1x, energy, force, transition-state, or
barrier prediction.

---

## Phase 8.7: Promoted-Data Retraining

Phase 8.7 retrains the same small MLP role classifier on the promoted curated
dataset from Phase 8.6:

```bash
python scripts/train_promoted_mlp.py
python scripts/benchmark_promoted_mlp.py
```

The pre-promotion baseline on `data/reactions.normalized.json` was:

| Predictor | Overall role accuracy |
|-----------|----------------------:|
| `mlp_local` | 0.4167 |
| `mlp_negotiated` | 0.4167 |

Success criteria:

- Minimum: the new promoted-data MLP beats the old MLP checkpoint.
- Better: the new MLP beats `rule_based_local`.
- Strongest: the new MLP beats `rule_based_negotiated`.

`rule_based_negotiated` remains the default unless the benchmark clearly
supports replacing it. This remains role-predictor training only:
functional-group descriptor to role label. It is not MLIP training and does not
use MACE, Transition1x, energy, force, transition-state, or barrier data.
