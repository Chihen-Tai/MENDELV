# Open Energy/Force Dataset Options

MENDELV Phase 10 uses open molecule-conformer datasets only for pretrained
MLIP single-point energy/force benchmarking. These datasets are not reaction
paths and should not be used to claim transition states, barriers, IRC, NEB, or
MD behavior.

## QO2Mol

- Scope: large organic molecule conformer energy/force dataset.
- Reference level: reported as B3LYP/def2-SVP in MENDELV metadata.
- Strength: broad organic chemistry coverage.
- Current MENDELV status: ingestion scaffold exists for local JSON/JSONL/NPZ
  samples, but the official repository metadata did not expose an obvious small
  sample.
- Best use: once a verified local sample is available, convert it with
  `scripts/qo2mol_sample_manager.py`.

## MD17 / rMD17

- Scope: molecular dynamics conformers for small molecules with energies and
  forces.
- Strength: practical NPZ files are common and easy to adapt for smoke tests.
- Current MENDELV status: Phase 10.2 adds an MD17/rMD17-style NPZ adapter.
- Phase 10.3 status: real-NPZ benchmark helper adds raw and mean-shifted
  energy metrics plus force metrics for MACE-OFF smoke benchmarks.
- Expected NPZ keys: `z`/`Z` atomic numbers, `R` coordinates, `E` energies, and
  optionally `F` forces.
- Best use: initial MACE-OFF benchmark smoke tests on a local small NPZ file.

```bash
python scripts/prepare_md17_sample.py \
  --input /path/to/md17_molecule.npz \
  --output data/reference/md17_sample.reference.json \
  --report reports/md17_sample_report.json \
  --max-records 100

python scripts/run_mlip_reference_benchmark.py \
  --reference data/reference/md17_sample.reference.json \
  --backend mace \
  --model-name mace-off-small \
  --device cpu \
  --predictions-output reports/mlip_md17_predictions.json \
  --benchmark-output reports/mlip_md17_benchmark.json \
  --continue-on-error
```

For the real MD17/rMD17 convenience workflow, see
[docs/md17_benchmark.md](md17_benchmark.md).

If no dataset is available, `prepare_md17_sample.py` creates a tiny synthetic
NPZ and converts it. That file is for plumbing tests only and is not a
scientific reference.

## ANI-1x

- Scope: broad organic conformer data with energies and forces.
- Strength: useful for organic ML potential benchmarking.
- MENDELV status: not yet implemented.
- Caveat: dataset layout and license should be verified before adding an
  adapter or committing derived samples.

## QM7-X

- Scope: quantum chemistry conformers with properties for small organic
  molecules.
- Strength: useful for molecule-level property and conformer benchmarking.
- MENDELV status: not yet implemented.
- Caveat: adapter and unit normalization would need review before benchmarking.

## Safety Policy

- Do not download huge datasets by default.
- Do not commit raw external data.
- Verify dataset licenses before redistribution.
- Do not train or fine-tune MLIP models in these benchmark scripts.
- Do not run DFT, NEB, IRC, MD, transition-state search, or barrier prediction.
