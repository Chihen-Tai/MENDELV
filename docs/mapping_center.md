# Phase 8.14: Mapping-Aware Center Labels

Phase 8.14 adds atom-mapping and bond-change utilities for auditing
`reaction_center_atoms` labels. The goal is to improve center-label reliability
and held-out validation balance before any Phase 9 work.

This is not MLIP. It does not use MACE or Transition1x and does not predict
energy, force, transition states, IRC paths, NEB paths, MD trajectories, or
barriers.

## What Atom Mapping Provides

Atom-mapped reaction SMILES attach stable map numbers to corresponding atoms
across reactants and products. When mapped data are available, MENDELV can
compare reactant and product bonds:

- bonds present only in products imply bond formation
- bonds present only in reactants imply bond breaking
- changed bond orders imply a center change

Reactant atom-map numbers involved in those changes are a mapping-derived
reaction-center suggestion.

## Utilities

`mendel/mapping_center.py` provides:

- `has_atom_mapping(...)`
- `extract_mapped_atom_pairs(...)`
- `extract_bond_changes(...)`
- `infer_center_atoms_from_mapping(...)`
- `audit_labeled_centers_against_mapping(...)`
- `apply_mapping_center_suggestions(...)`

High-confidence suggestions can be applied conservatively, but unmapped
reactions are reported rather than treated as failures.

## Commands

Audit the expanded cleaned dataset:

```bash
python scripts/audit_mapping_centers.py \
  --data data/reactions.center_expanded.cleaned.json \
  --output reports/mapping_center_audit_report.json
```

Audit and write a dataset with high-confidence suggestions applied:

```bash
python scripts/audit_mapping_centers.py \
  --data data/reactions.center_expanded.cleaned.json \
  --output reports/mapping_center_audit_report.json \
  --apply-high-confidence \
  --apply-suggestions-output data/reactions.center_expanded.mapping_audited.json
```

Run the full balanced validation workflow:

```bash
python scripts/run_balanced_center_validation.py \
  --data data/reactions.center_expanded.cleaned.json \
  --use-mapping-suggestions \
  --device cpu \
  --epochs 80 \
  --threshold 0.5 \
  --output-prefix center_balanced
```

## Limitations

- Unmapped reaction SMILES cannot use bond-change inference.
- Atom mapping quality matters; incorrect maps produce incorrect suggestions.
- Current labels still use simple atom-map-number lists.
- This is an audit and validation layer, not a graph neural network.
- The atom-center head remains experimental until balanced val/test validation
  is stable.
