# Phase 10.5: Energy/Force Comparison Figures

Phase 10.5 turns the rMD17 ethanol MACE-OFF benchmark into publication-style
figures for fixed molecule conformer energy/force analysis.

The comparison is:

- DFT reference energies and forces from rMD17
- pure pretrained MACE-OFF single-point predictions
- MENDELV-guided local force-error decomposition

MENDELV currently organizes and decomposes errors. It does not correct
MACE-OFF predictions.

## Figures

### Raw Energy Parity

`energy_parity_raw.png` plots DFT reference energy against raw MACE-OFF
predicted energy. This shows absolute offset between the two energy scales.

### Mean-shifted Energy Parity

`energy_parity_mean_shifted.png` subtracts:

```text
offset = mean(predicted_energy - reference_energy)
```

from predicted energies before plotting. This is more useful for conformer
relative-energy behavior when absolute references have a constant offset.

### Energy RMSE Bar

`energy_rmse_bar.png` compares raw energy RMSE with mean-shifted energy RMSE.

### Force RMSE By Element

`force_rmse_by_element.png` shows global force RMSE and per-element force RMSE.
For the current rMD17 ethanol sample, oxygen-local error is highest.

### Local Force RMSE By Group

`local_force_rmse_by_group.png` ranks functional-group-local force RMSE when
true functional groups are available. rMD17 ethanol currently lacks SMILES, so
Phase 10.5 uses pseudo-groups such as `whole_molecule`, `heavy_atoms`,
`hydrogens`, and `element_O`.

Pseudo-groups are diagnostics, not chemical functional groups.

### Force Error Distribution

`force_error_distribution.png` plots the atom-level force error norm
distribution.

## Command

Install the optional plotting dependency if matplotlib is not already present:

```bash
pip install -e ".[plot]"
```

```bash
python scripts/plot_energy_force_comparison.py \
  --reference data/reference/rmd17_ethanol_sample_converted.reference.json \
  --predictions reports/mlip_rmd17_ethanol_converted_predictions.json \
  --benchmark reports/mlip_rmd17_ethanol_converted_benchmark.json \
  --local-analysis reports/functional_group_force_analysis_ethanol.json \
  --output-dir reports/figures \
  --report reports/energy_force_plot_report.json \
  --top-n 10
```

## Scope

This is energy/force prediction quality analysis on fixed molecule conformers.
It is not reaction-path, barrier, transition-state, IRC, or NEB analysis. It
does not train MLIP, fine-tune MACE, or run DFT.
