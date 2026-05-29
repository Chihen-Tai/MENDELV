# Phase 10.4: Local Force Error Analysis

Phase 10.4 localizes pretrained MLIP force error from whole-molecule metrics to
atoms and functional-group agents.

Functional group = agent remains the organizing idea: when functional-group
assignments are available, MENDELV can ask whether MACE-OFF force errors
concentrate around particular local chemical agents.

## Metrics

- Global force RMSE: one molecule-level force error summary over all atoms and
  Cartesian components.
- Per-element force RMSE: force error grouped by element, such as C, H, or O.
- Functional-group-local force RMSE: force error grouped by detected MENDELV
  functional groups from SMILES.
- Pseudo-group force RMSE: non-chemical fallback groups such as
  `whole_molecule`, `heavy_atoms`, `hydrogens`, and `element_C`.

## rMD17 Missing SMILES

Many rMD17 NPZ files provide coordinates, atomic numbers, energies, and forces,
but not SMILES. Without SMILES, MENDELV cannot reliably identify true chemical
functional groups.

In that case, local analysis still computes atom-level and per-element force
errors. Add `--use-pseudo-groups` to get useful local groupings without
claiming chemical functional-group semantics.

Pseudo-groups are marked with:

```json
{
  "pseudo_group": true,
  "not_chemical_functional_group": true
}
```

## Command

```bash
python scripts/analyze_functional_group_force_errors.py \
  --reference data/reference/rmd17_ethanol_sample_converted.reference.json \
  --predictions reports/mlip_rmd17_ethanol_converted_predictions.json \
  --benchmark reports/mlip_rmd17_ethanol_converted_benchmark.json \
  --output reports/functional_group_force_analysis_ethanol.json \
  --use-pseudo-groups
```

Use `--require-groups` only when the reference records contain SMILES and true
functional-group detection is expected.

## Using Local Force Analysis For Figures

Phase 10.5 uses the local force analysis report as the input for MENDELV-local
error plots:

```bash
python scripts/plot_energy_force_comparison.py \
  --reference data/reference/rmd17_ethanol_sample_converted.reference.json \
  --predictions reports/mlip_rmd17_ethanol_converted_predictions.json \
  --benchmark reports/mlip_rmd17_ethanol_converted_benchmark.json \
  --local-analysis reports/functional_group_force_analysis_ethanol.json \
  --output-dir reports/figures \
  --report reports/energy_force_plot_report.json
```

The figure script plots DFT reference vs pure MACE-OFF predictions, then uses
MENDELV's local analysis report to rank force RMSE by functional-group type or
pseudo-group type.

## Scope

This is energy/force quality analysis on fixed molecule conformers. It does not
train MLIP, run DFT, run NEB/IRC/MD, search transition states, or predict
reaction barriers.
