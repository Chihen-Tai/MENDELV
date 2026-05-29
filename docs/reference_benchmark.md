# Phase 10: QO2Mol Reference Energy/Force Benchmark Scaffold

Phase 10 adds an open reference-data scaffold for benchmarking pretrained MLIP
single-point energies and forces. The first target dataset is QO2Mol.

This is molecule conformer benchmarking, not reaction-path benchmarking. It
does not train MLIP, does not run DFT, does not use Transition1x, and does not
run NEB, IRC, MD, transition-state search, or barrier prediction.

## QO2Mol Metadata

- dataset_name: `QO2Mol`
- citation: `An Open Quantum Chemistry Property Database of 120 Kilo Molecules with 20 Million Conformers`
- source_url: `https://github.com/saiscn/QO2Mol/`
- reference_method: `B3LYP/def2-SVP`
- data_type: `molecular conformer energy/force`
- license_note: `license requires manual verification from source repository`

QO2Mol is appropriate for molecule-level energy/force benchmarking, not
reaction barrier or reaction-path validation.

## Reference JSON Format

```json
{
  "records": [
    {
      "structure_id": "sample_0",
      "molecule_id": "mol_0",
      "dataset_name": "QO2Mol",
      "smiles": "CCO",
      "xyz": [["C", 0.0, 0.0, 0.0]],
      "reference_energy": -1.0,
      "reference_energy_unit": "eV",
      "reference_forces": [[0.0, 0.0, 0.0]],
      "reference_force_unit": "eV/Angstrom",
      "reference_method": "B3LYP/def2-SVP"
    }
  ]
}
```

## Commands

Create a tiny synthetic reference file for software tests:

```bash
python scripts/create_tiny_reference_example.py \
  --output data/reference/tiny_reference_example.json
```

Inspect a local QO2Mol path:

```bash
python scripts/ingest_qo2mol.py \
  --input /path/to/qo2mol/sample \
  --inspect-only
```

Ingest a capped local sample:

```bash
python scripts/ingest_qo2mol.py \
  --input /path/to/qo2mol/sample \
  --output data/reference/qo2mol_sample.reference.json \
  --report reports/qo2mol_ingestion_report.json \
  --max-records 100
```

Run the optional MLIP benchmark when ASE and MACE are installed:

```bash
python scripts/run_mlip_reference_benchmark.py \
  --reference data/reference/qo2mol_sample.reference.json \
  --backend mace \
  --model-name mace-off-small \
  --device cpu \
  --predictions-output reports/mlip_qo2mol_predictions.json \
  --benchmark-output reports/mlip_qo2mol_benchmark.json \
  --continue-on-error
```

## Phase 10.3b Unit Handling

For MD17/rMD17-style NPZ files, MENDELV now handles reference units explicitly.
The default assumption is:

- energy: `kcal/mol`
- forces: `kcal/mol/Angstrom`

These are converted to MACE-compatible units before benchmarking:

- energy: `eV`
- forces: `eV/Angstrom`

The benchmark JSON records raw and mean-shifted energy MAE/RMSE. Mean-shifted
energy metrics subtract `mean(predicted - reference)` and are often more useful
than raw absolute energies when comparing pretrained MLIPs to external quantum
chemistry datasets. Force MAE/RMSE after unit conversion is the main
interpretable force metric.

See [docs/md17_benchmark.md](md17_benchmark.md) for the real MD17/rMD17
workflow.

## Phase 10.4 Local Force Analysis

After a reference benchmark is complete, MENDELV can localize force errors to
atoms, elements, functional groups, or pseudo-groups:

```bash
python scripts/analyze_functional_group_force_errors.py \
  --reference data/reference/rmd17_ethanol_sample_converted.reference.json \
  --predictions reports/mlip_rmd17_ethanol_converted_predictions.json \
  --benchmark reports/mlip_rmd17_ethanol_converted_benchmark.json \
  --use-pseudo-groups
```

See [docs/local_force_analysis.md](local_force_analysis.md).

## Phase 10.5 Benchmark Figures

The same reference, prediction, benchmark, and local-analysis JSON files can be
turned into comparison figures:

```bash
python scripts/plot_energy_force_comparison.py \
  --reference data/reference/rmd17_ethanol_sample_converted.reference.json \
  --predictions reports/mlip_rmd17_ethanol_converted_predictions.json \
  --benchmark reports/mlip_rmd17_ethanol_converted_benchmark.json \
  --local-analysis reports/functional_group_force_analysis_ethanol.json \
  --output-dir reports/figures \
  --report reports/energy_force_plot_report.json
```

The plots show raw and mean-shifted energy parity, energy RMSE, force RMSE by
element, local force RMSE by group, and atom-level force error distribution.
See [docs/figures.md](figures.md).

## Supported Local Sample Formats

The initial adapter supports generic local samples in JSON, JSONL, and simple
NPZ forms. Actual QO2Mol repository files may require extending the adapter.
Unsupported formats fail clearly rather than triggering a full dataset load.

## Phase 10.1 Sample Manager

Phase 10.1 adds `scripts/qo2mol_sample_manager.py` for local source inspection,
registry management, capped sampling, and reference JSON summaries:

```bash
python scripts/qo2mol_sample_manager.py inspect --input /path/to/QO2Mol
python scripts/qo2mol_sample_manager.py sample --input /path/to/QO2Mol/sample.json
python scripts/qo2mol_sample_manager.py summarize \
  --reference data/reference/qo2mol_sample.reference.json
```

The manager does not download the full dataset by default. Keep raw QO2Mol data
under ignored local paths and verify the license before redistribution.

## Limitations

- No raw QO2Mol data is downloaded or committed.
- No MLIP training or fine-tuning is performed.
- No DFT is run.
- No barrier, TS, IRC, NEB, or MD workflow is included.
- QO2Mol geometries are molecular conformers, not reaction paths.
- MENDELV reaction centers may not be available for generic conformer records.
