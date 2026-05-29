# Phase 10.3: MD17/rMD17 MACE-OFF Benchmark

Phase 10.3 adds a practical real-dataset path for molecule-conformer
energy/force benchmarking. MD17/rMD17 NPZ files are used because they are a
common open format with coordinates, energies, and forces. Phase 10.3b makes
unit conversion explicit and reuses the MACE calculator across structures.

This is not a MENDELV reaction benchmark. It does not evaluate reaction paths,
transition states, barriers, IRC, NEB, or MD trajectories.

## Scope

Included:

- local MD17/rMD17-style NPZ conversion to MENDELV reference JSON
- optional explicit URL fetch helper with size checks
- pretrained MACE-OFF single-point predictions on fixed conformer geometries
- explicit rMD17/MD17 unit conversion
- raw energy MAE/RMSE
- mean-shifted energy MAE/RMSE
- force MAE/RMSE
- per-element force RMSE
- reused calculator initialization for multi-structure benchmarks

Not included:

- MLIP training or fine-tuning
- quantum chemistry calculations
- reaction-path workflows
- barrier prediction
- committing raw datasets

## Obtain Data

Download an MD17 or rMD17 NPZ manually from an official or trusted open source,
then keep it under an ignored local path such as:

```text
data/external/md17/
```

The helper script does not download anything without an explicit URL and
`--allow-download`:

```bash
python scripts/fetch_md17_sample.py
python scripts/fetch_md17_sample.py \
  --url <URL> \
  --allow-download \
  --max-size-mb 500
```

Verify the source license before redistribution. Do not commit raw NPZ files.

## Convert A Local NPZ

```bash
python scripts/prepare_md17_sample.py \
  --input /path/to/md17_molecule.npz \
  --output data/reference/md17_sample.reference.json \
  --report reports/md17_sample_report.json \
  --max-records 100
```

By default MENDELV assumes MD17/rMD17 energies are `kcal/mol` and forces are
`kcal/mol/Angstrom`, then converts them to `eV` and `eV/Angstrom` before
benchmarking against MACE-OFF:

```text
1 kcal/mol = 0.0433641153087705 eV
1 kcal/mol/Angstrom = 0.0433641153087705 eV/Angstrom
```

Override only when you have verified dataset documentation:

```bash
python scripts/prepare_md17_sample.py \
  --input /path/to/md17_molecule.npz \
  --energy-unit eV \
  --force-unit eV/Angstrom \
  --no-convert-units
```

If no input is provided, `prepare_md17_sample.py` generates a tiny synthetic NPZ
only for plumbing tests. Synthetic records are marked with
`synthetic_test_data=true` and `not_scientific_reference=true`.

## Run MACE-OFF Benchmark

```bash
python scripts/run_md17_mace_benchmark.py \
  --input /path/to/real_md17_or_rmd17.npz \
  --output-reference data/reference/md17_real_sample.reference.json \
  --sample-report reports/md17_real_sample_report.json \
  --predictions-output reports/mlip_md17_real_predictions.json \
  --benchmark-output reports/mlip_md17_real_benchmark.json \
  --max-records 100 \
  --backend mace \
  --model-name mace-off-small \
  --device cpu \
  --continue-on-error
```

## Energy Metrics

Raw energy errors compare predicted and reference absolute energies directly.
Pretrained MLIPs and reference quantum chemistry datasets can have different
constant offsets, so raw energy MAE/RMSE may be dominated by that offset.

Mean-shifted energy errors subtract:

```text
offset = mean(predicted_energy - reference_energy)
```

from predicted energies before computing MAE/RMSE. This centered metric is
often more informative for conformer ranking and relative-energy behavior.

## Force Metrics

Force MAE/RMSE compares all Cartesian force components on the same fixed
geometries after reference force conversion to `eV/Angstrom`. For this kind of
conformer benchmark, converted force metrics are usually more directly
comparable than raw absolute energies.

## Local Force Analysis

Phase 10.4 adds atom-local and group-local force error analysis. For rMD17 files
without SMILES, use pseudo-groups:

```bash
python scripts/analyze_functional_group_force_errors.py \
  --reference data/reference/rmd17_ethanol_sample_converted.reference.json \
  --predictions reports/mlip_rmd17_ethanol_converted_predictions.json \
  --benchmark reports/mlip_rmd17_ethanol_converted_benchmark.json \
  --output reports/functional_group_force_analysis_ethanol.json \
  --use-pseudo-groups
```

Pseudo-groups are useful diagnostics, but they are not true chemical functional
groups. See [docs/local_force_analysis.md](local_force_analysis.md).

## Energy/Force Figures

Phase 10.5 generates benchmark figures from the converted rMD17 ethanol
reference JSON, MACE-OFF predictions, global benchmark report, and local force
analysis report:

```bash
python scripts/plot_energy_force_comparison.py \
  --reference data/reference/rmd17_ethanol_sample_converted.reference.json \
  --predictions reports/mlip_rmd17_ethanol_converted_predictions.json \
  --benchmark reports/mlip_rmd17_ethanol_converted_benchmark.json \
  --local-analysis reports/functional_group_force_analysis_ethanol.json \
  --output-dir reports/figures \
  --report reports/energy_force_plot_report.json
```

The figures compare raw and mean-shifted energy parity, energy RMSE, force RMSE
by element, local force RMSE by group, and atom-level force error distribution.
See [docs/figures.md](figures.md).

## Limitations

- MD17/rMD17 is molecule-conformer data, not reaction data.
- No MENDELV reaction-center validation is obtained from MD17 alone.
- No MLIP training is performed.
- No quantum chemistry is run.
- No reaction barriers are predicted.
- Unit assumptions should still be verified against the exact source file
  documentation.
