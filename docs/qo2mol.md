# Phase 10.1: QO2Mol Sample Manager

Phase 10.1 prepares small local QO2Mol samples for MENDELV reference
energy/force benchmarks. It does not download the full dataset by default and
does not commit raw QO2Mol data.

QO2Mol is molecular conformer data. It is not reaction-path data and should not
be used to claim reaction barriers, transition states, IRC paths, NEB paths, or
MD trajectories.

## Data Policy

- Keep raw external data under ignored paths such as `data/raw/`,
  `data/external/`, or `data/qo2mol/`.
- Do not commit raw QO2Mol archives or converted bulk files.
- Verify the dataset license before redistributing any sample.
- Commit only small curated MENDELV reference JSON files when appropriate.

## Manual Acquisition

QO2Mol source:

```text
https://github.com/saiscn/QO2Mol/
```

Use the repository instructions to obtain data manually. MENDELV does not
download the full dataset by default.

The safe downloader stub prints source and license guidance:

```bash
python scripts/download_qo2mol_stub.py
```

## Safe Sample Fetcher

Phase 10.2a adds a guarded fetcher for small official QO2Mol samples. It does
not download the full dataset by default.

Dry run:

```bash
python scripts/fetch_qo2mol_sample.py --dry-run
```

Fetch an explicitly provided small official sample URL only after reviewing the
source and size:

```bash
python scripts/fetch_qo2mol_sample.py \
  --url <URL> \
  --allow-download \
  --max-size-mb 200 \
  --output-dir data/external/qo2mol_sample \
  --report reports/qo2mol_fetch_report.json
```

Inspect after download:

```bash
python scripts/qo2mol_sample_manager.py inspect \
  --input data/external/qo2mol_sample \
  --registry reports/qo2mol_sources.json
```

Warnings:

- Do not commit raw QO2Mol data.
- Verify the QO2Mol license before redistribution.
- Use only the official QO2Mol repository, official URLs linked from that
  repository, or a user-provided path/URL.

## Inspect A Local Source

```bash
python scripts/qo2mol_sample_manager.py inspect \
  --input /path/to/QO2Mol \
  --registry reports/qo2mol_sources.json
```

The registry records source path, detected format, file count, size estimate,
and detected fields when they can be sampled cheaply.

## Sample Records

```bash
python scripts/qo2mol_sample_manager.py sample \
  --input /path/to/QO2Mol/sample.json \
  --output data/reference/qo2mol_sample.reference.json \
  --report reports/qo2mol_sample_report.json \
  --max-records 100 \
  --strategy random
```

Supported strategies:

- `first_n`
- `random`
- `element_filtered`
- `small_molecule_first`

Filter elements:

```bash
python scripts/qo2mol_sample_manager.py sample \
  --input /path/to/QO2Mol/sample.json \
  --elements C,H,O,N,F,Cl,Br,I \
  --strategy element_filtered
```

## Summarize A Reference JSON

```bash
python scripts/qo2mol_sample_manager.py summarize \
  --reference data/reference/qo2mol_sample.reference.json \
  --output reports/qo2mol_reference_summary.json
```

The summary includes element distribution, atom-count distribution, energy
range, force-norm range, and missing-field counts.

## Limitations

- The actual QO2Mol file layout may require adapter extension after inspection.
- Directory sources are inspected but not bulk-loaded.
- This phase does not train MLIP.
- This phase does not run DFT.
- This phase does not predict reaction barriers.
