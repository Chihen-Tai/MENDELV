# Phase 8.13: Manual Center Expansion

Phase 8.13 promotes conservative center-focused expansion candidates and reruns
mechanism-balanced strict validation.

Functional group = agent remains the core abstraction. This phase expands and
validates reaction-center supervision only. It does not train MLIP and does not
predict energy or force.

## Why This Phase Exists

Phase 8.12 generated `data/draft_inputs.center_expansion.json`, but those
candidate reactions were not yet promoted into the cleaned center dataset. The
strict split also had unstable val/test behavior and limited mechanism coverage.

Phase 8.13 addresses that by:

- promoting safe center-expansion candidates with manually reviewed-style labels
- applying mechanism-specific center policies
- merging promoted records into `data/reactions.center_cleaned.json`
- rerunning center cleanup
- assigning a mechanism-balanced leakage-resistant split
- retraining and benchmarking the atom-center head on the expanded dataset

## Conservative Promotion Policy

Promotion is mechanism-specific:

- SN2/E2: halide is the leaving group and center includes represented C-X atoms.
- Diels-Alder: center includes reacting alkene atoms only.
- Carbonyl addition: carbonyl is electrophile and center includes carbonyl atoms.
- Controls: all detected groups are spectators and center is empty.
- Radical bromination: benzylic site is the radical center when present.
- Nitroalkane deprotonation: alpha carbon is the flat-taxonomy nitronate center.
- Ambiguous aldol/cross-aldol examples are skipped unless donor and acceptor are
  unambiguous.

## Commands

Promote center-expansion candidates:

```bash
python scripts/promote_center_expansion.py \
  --input data/reactions.draft.center_expansion.json \
  --fallback-input data/draft_inputs.center_expansion.json \
  --base data/reactions.center_cleaned.json \
  --output data/reactions.center_expansion.promoted.json \
  --merged-output data/reactions.center_expanded.cleaned_input.json \
  --report reports/center_expansion_promotion_report.json
```

Run the full expanded validation workflow:

```bash
python scripts/run_center_expansion_validation.py \
  --device cpu \
  --epochs 80 \
  --threshold 0.5
```

## Validation Goals

Minimum target:

- promoted center-expansion reactions > 0
- error-severity center-label issues after cleanup = 0
- test split has at least 15 reactions
- test split has at least 5 mechanisms
- strict test reaction-center F1 > 0.75

The atom-center head remains experimental if val/test behavior is unstable or
mechanism coverage is still narrow.
