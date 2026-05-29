# Phase 9: Optional Pretrained MLIP Backend

Phase 9 connects MENDELV reaction-center reasoning to an optional pretrained
MLIP calculator for single-point energy and force evaluation.

Functional group = agent remains the MENDELV abstraction. MENDELV supplies
roles and reaction-center atoms; the optional MLIP backend supplies approximate
single-point energy and forces for a generated or loaded structure.

## Scope

Included:

- optional ASE / MACE dependency extra
- RDKit 3D conformer generation
- RDKit molecule to ASE `Atoms` conversion
- XYZ loading through ASE
- pretrained MACE calculator creation when installed
- single-point energy and force calculation
- force summaries on MENDELV reaction-center atoms

Not included:

- MLIP training or fine-tuning
- Transition1x
- DFT comparison
- NEB
- IRC
- MD
- transition-state search
- barrier prediction
- PES exploration

## Installation

Normal MENDELV installation does not require MLIP dependencies:

```bash
pip install -e ".[dev]"
```

Install the optional MLIP backend only when needed:

```bash
pip install -e ".[mlip]"
```

`import mendel` and `import mendel.mlip` remain safe without ASE, MACE, or
torch. Optional imports raise clear errors only when MLIP functions are called.

## Commands

Single-point calculation from molecule SMILES:

```bash
python scripts/mlip_singlepoint.py \
  --smiles "CC(=O)C" \
  --backend mace \
  --model-name mace-off-small \
  --device cpu \
  --output reports/mlip_acetone.json
```

MENDELV-guided single-point calculation from reaction SMILES:

```bash
python scripts/mlip_singlepoint.py \
  --reaction-smiles "CBr.[OH-]>>CO.[Br-]" \
  --context ionic \
  --reaction-center-from-mendelv \
  --backend mace \
  --model-name mace-off-small \
  --device cpu \
  --output reports/mlip_sn2.json
```

## Warnings

- MLIP energy and force values are not DFT references.
- RDKit conformers are not reaction paths.
- Disconnected reactant complexes may have arbitrary relative geometry.
- Phase 9 does not compute barriers.
- Phase 9 does not replace `rule_based_negotiated` as the conservative default.

## Phase 9.2 Geometry Sanity Checks

MLIP single-point results now include a geometry sanity report in
`metadata.geometry_sanity`. The check records pairwise distance bounds,
force-norm bounds, disconnected-fragment charge risk, and a status of `pass`,
`warning`, `fail`, or `unknown`.

Neutral molecule SMILES mode is usually safer because RDKit embeds one
connected molecule. Reaction SMILES mode is still experimental: disconnected
charged reactants such as `CBr.[OH-]>>CO.[Br-]` can be embedded with arbitrary
relative geometry, which can produce huge MACE-OFF force norms. In that case
MENDELV keeps the successful MLIP calculation but marks the geometry as
suspicious and warns that a physically meaningful 3D complex or XYZ should be
provided for reliable force analysis.

Useful strict-mode command:

```bash
python scripts/mlip_singlepoint.py \
  --reaction-smiles "CBr.[OH-]>>CO.[Br-]" \
  --context ionic \
  --reaction-center-from-mendelv \
  --backend mace \
  --model-name mace-off-small \
  --device cpu \
  --fail-on-geometry-sanity \
  --output reports/mlip_sn2.json
```

These checks are heuristic guardrails. They do not turn a reaction SMILES
conformer into a reaction path, and they do not run NEB, IRC, MD, transition
state search, DFT, or barrier prediction.

## Phase 10 Reference Benchmarks

The same optional single-point backend is used by the Phase 10 reference
benchmark scaffold. See [docs/reference_benchmark.md](reference_benchmark.md)
for QO2Mol sample ingestion and MACE-OFF energy/force comparison commands.
