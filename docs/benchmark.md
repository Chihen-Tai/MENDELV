# Phase 8: Benchmark Evaluator

`mendel/benchmark.py` compares MENDELV role predictors against curated
`LabeledReaction.group_roles`. It measures the current functional-group agents and
the negotiation layer; it does not train or change the model.

## Purpose

Phase 8 answers two questions:

1. Does the MLP local role predictor improve over the rule-based baseline?
2. Does negotiation improve reaction-level consistency over local predictions alone?

The rule-based predictor remains the default until benchmark evidence shows that the
MLP is better.

## Benchmark Levels

**Group-level role accuracy** compares each labeled `group_id` against the predictor's
role for that same functional-group agent.

**Confusion matrix** counts true role versus predicted role across the five existing
roles.

**Per-role accuracy** shows which chemical roles are reliable or weak.

**Per-mechanism accuracy** groups results by `mechanism_type`, such as SN2, E2,
Diels-Alder, aldol, radical bromination, and controls.

**Reaction-center precision/recall/F1** compares negotiated reaction-center atom IDs
against labeled `reaction_center_atoms` when those labels exist.

## Commands

Rule-based local predictor versus negotiated rule-based predictor:

```bash
python scripts/benchmark.py --data data/reactions.json --rule-based --negotiated
```

Optional MLP checkpoint benchmark. When a checkpoint is provided, the CLI evaluates
both `mlp_local` and `mlp_negotiated`:

```bash
python scripts/benchmark.py --data data/reactions.json --mlp-checkpoint models/role_mlp.pt --device cpu
```

The MLP command only evaluates an existing checkpoint. It does not train.

## Promoted Dataset Benchmark

Phase 8.7 retrains the experimental MLP role predictor on promoted curated
labels, then benchmarks the new checkpoint against rule-based baselines and the
old MLP checkpoint:

```bash
python scripts/train_promoted_mlp.py
python scripts/benchmark_promoted_mlp.py
```

The promoted benchmark writes separate reports for:

- `rule_based_local`
- `rule_based_negotiated`
- `new_mlp_local`
- `new_mlp_negotiated`
- `old_mlp_local` and `old_mlp_negotiated`, when the old checkpoint exists

Improvement over the old checkpoint is useful evidence that promoted labels
helped. It is not enough by itself to replace `rule_based_negotiated`; the MLP
must clearly beat the negotiated rule-based pipeline before becoming the
default.

Phase 8.9 adds an experimental MLP-aware negotiation benchmark:

```bash
python scripts/benchmark_promoted_mlp.py \
  --data data/reactions.proposed_with_auto_promoted.normalized.json \
  --new-mlp-checkpoint models/role_mlp_promoted.pt \
  --old-mlp-checkpoint models/role_mlp.pt \
  --device cpu \
  --output-dir reports/benchmark_mlp_aware_full \
  --include-mlp-aware-negotiation
```

This writes `new_mlp_aware_negotiated.json` and includes
`new_mlp_aware_negotiated` in `comparison.json`.

## Python API

```python
from mendel.labels import load_labeled_reactions
from mendel.benchmark import (
    evaluate_rule_based_predictor,
    evaluate_negotiated_rule_based,
    compare_benchmark_reports,
)

reactions = load_labeled_reactions("data/reactions.json")
rule_report = evaluate_rule_based_predictor(reactions)
negotiated_report = evaluate_negotiated_rule_based(reactions)
comparison = compare_benchmark_reports([rule_report, negotiated_report])
```

MLP checkpoint evaluation is intentionally imported through the benchmark module only
when called:

```python
from mendel.benchmark import evaluate_mlp_checkpoint, evaluate_negotiated_mlp_checkpoint

local_report = evaluate_mlp_checkpoint(reactions, "models/role_mlp.pt", device="cpu")
negotiated_report = evaluate_negotiated_mlp_checkpoint(
    reactions, "models/role_mlp.pt", device="cpu"
)
```

## Limitations

- The dataset is small, so all metrics have high variance.
- Evaluation is label-conditioned: only curated `group_roles` are scored.
- Unlabeled detected groups are ignored; they are not false positives.
- Labeled `group_id` values that are not detected by the current identifier count as
  incorrect missing groups.
- The current role taxonomy is flat and remains unchanged.
- Reaction-center metrics depend on atom-map coverage and current group granularity.
- This benchmark does not measure energy, force, potential energy surfaces,
  transition states, or reaction barriers.

## Status

The MLP is experimental until the benchmark shows improvement over the rule-based
predictor. Functional group = agent remains the benchmark unit.
