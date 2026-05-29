# MENDEL

**Molecular Entity Negotiation for Dynamic Energy Landscapes** — a fully local, functional-group-level reaction role prediction framework for organic chemistry.

Each functional group in a molecule is treated as an **agent** that observes its local chemical environment, predicts its own reaction role, then negotiates with neighbouring groups to produce a coherent, conflict-free assignment.

The same functional-group-as-agent lens applies to MLIP evaluation: MENDEL decomposes MLIP force errors by functional group type, revealing that reactive sites (e.g. the alcohol C–O bond in ethanol) carry ~2× the global RMSE — a fact that a single global number hides entirely.

---

## Quick Demo

```python
# Rule-based pipeline (no PyTorch required)
from mendel.negotiator import run_full_rule_pipeline

result = run_full_rule_pipeline("CBr.[OH-]>>CO.[Br-]", context="ionic")
print(result.mechanism_hint)    # sn2_or_e2_like
for ra in result.role_assignments:
    print(ra.group_id, ra.final_role)

# MLP pipeline (requires pip install -e ".[ml]")
from mendel.negotiator import run_pipeline_with_mlp

result = run_pipeline_with_mlp(
    "CBr.[OH-]>>CO.[Br-]",
    "models/role_mlp.pt",
    context="ionic",
)
```

`import mendel` does not import PyTorch, torchani, ASE, or MACE. All optional Phase 7–10 APIs must be imported directly from their submodule (`mendel.mlp`, `mendel.mlip`).

---

## Pipeline

```
reaction SMILES + context
         │
         ▼
  functional group detection       (identifier.py — RDKit SMARTS, 3-pass)
         │
         ▼
  per-group descriptor building    (descriptor.py — 65-dim, schema phase6_6_v1)
         │
         ▼
  per-group role prediction        (predictor.py — rule-based baseline)
         │
         ▼
  negotiation / conflict resolution (negotiator.py — mechanism hints, reaction center)
         │
         ▼
  [optional] MLP role predictor    (mlp.py — learned, Phase 7)
         │
         ▼
  [optional] MLIP energy / forces  (mlip.py — MACE-OFF / ANI-2x, Phase 9)
         │
         ▼
  [optional] MENDEL force decomposition  (local_force_analysis.py — per-group RMSE, Phase 10)
```

### Roles

Five mutually exclusive roles per functional group per reaction step:

| Role | Description |
|------|-------------|
| `reactive_nucleophile` | donates electrons |
| `reactive_electrophile` | accepts electrons |
| `reactive_radical` | radical center |
| `leaving_group` | departs with electron pair |
| `spectator` | uninvolved in this step |

---

## Benchmark Results

> **A compact empirical report.** *Question:* does the functional-group-as-agent
> decomposition, combined with a negotiation layer, predict reaction roles better
> than either component alone?

### Setup

- **Dataset.** 166 labeled reactions spanning 14 mechanism classes.
- **Split.** Mechanism-stratified, reaction-level train/val split — no group from a
  validation reaction is seen during training.
- **Task.** Assign each functional group one of five mutually exclusive roles
  (`reactive_nucleophile`, `reactive_electrophile`, `reactive_radical`,
  `leaving_group`, `spectator`); a group is additionally flagged if it sits on the
  reaction center.
- **Models compared.**
  1. **Rule + Negotiator** — hand-written threshold rules + global consistency layer.
  2. **MLP only** — per-group 65-dim descriptor → learned role classifier, *no* negotiation.
  3. **MLP + Negotiator** — per-group MLP predictions reconciled by the negotiation layer.
- **Metric.** Role accuracy over labeled groups; reaction-center precision / recall / F1.

![Role-prediction performance overview](reports/figures/role_performance_overview.png)

### Results

**Overall.** The agent + negotiation stack reaches **96.1%** role accuracy, ahead of
the learned predictor alone (90.7%) and the rule baseline (70.4%).

| Model | Overall | SN2/E2 | Aldol | Cross-aldol | Diels-Alder | Michael |
|-------|---------|--------|-------|-------------|-------------|---------|
| Rule + Negotiator | 70.4% | 100% | 83.0% | 95.2% | 81.9% | 25.0% |
| MLP only | 90.7% | 100% | 63.8% | 81.0% | 90.4% | 85.4% |
| **MLP + Negotiator** | **96.1%** | 100% | **78.7%** | 85.7% | 97.6% | **100%** |

**Per-role.** Negotiation lifts the hardest roles — `spectator` (43.0% → 93.9%) and
`reactive_electrophile` (80.5% → 100%) — without disturbing the roles that are already
saturated (`radical`, `leaving_group` at 100%).

| Role | Rule + Negotiator | MLP only | MLP + Negotiator |
|------|-------------------|----------|------------------|
| reactive_nucleophile | 94.5% | 98.9% | 94.5% |
| reactive_electrophile | 80.5% | 92.7% | 100% |
| reactive_radical | 100% | 100% | 100% |
| leaving_group | 100% | 100% | 100% |
| spectator | 43.0% | 82.4% | 93.9% |

**Reaction center.** The MLP alone produces *no* center predictions (F1 = 0) — reaction
center is an emergent property of negotiation between agents, not of independent
per-group classification. Adding the negotiation layer recovers it at F1 = 87.8%.

| Model | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| Rule + Negotiator | 82.5% | 90.4% | 91.6% |
| MLP only | 0.0% | 0.0% | 0.0% |
| **MLP + Negotiator** | **93.6%** | **84.5%** | **87.8%** |

### Discussion

1. **Negotiation supplies what independent agents cannot see.** Each functional group
   predicts its own role from its local descriptor, but reaction center, mechanism
   consistency, and the donor/acceptor distinction are *relational* facts. The MLP's
   0% center F1 makes this concrete: structure emerges only once agents are reconciled
   globally. 12 of 14 mechanisms reach 100% under MLP + Negotiator.

2. **Aldol is descriptor-limited, not data-limited.** Aldol (78.7%) and cross-aldol
   (85.7%) are the only classes below ceiling, and aldol is the *only* mechanism where
   the MLP (63.8%) underperforms the rule baseline (83.0%) — the learner cannot find a
   clean signal. The 65-dim descriptor cannot separate a donor carbonyl (→ spectator,
   it only activates the α-carbon) from an acceptor carbonyl (→ electrophile). An
   experiment trail adding aldol training examples confirms this: every data-side
   "fix" regressed *both* aldol and the cross-aldol it shares the ambiguity with.

   ![Aldol diagnosis: data-side fixes regress both classes](reports/aldol_diagnosis.png)

   The path past this ceiling is descriptor enrichment (e.g. an α-H acidity / partner
   electrophilicity contrast between the two carbonyls), not more aldol examples.

### Extrapolation (leave-one-mechanism-out)

How well does the learned predictor generalize to a mechanism it has *never seen*?
We retrain the MLP 14 times, each time holding out one entire mechanism class, and
measure role accuracy on the held-out reactions.

![Leave-one-mechanism-out extrapolation](reports/figures/lomo_extrapolation.png)

Mean held-out accuracy is **74.2%**, versus **94.3%** in-distribution — a ~20-point
extrapolation gap that is strongly mechanism-dependent:

- **Transfers well (≥ 90%):** `carbonyl_addition`, `e2`, `ester_control`,
  `nitrile_control`, `sn2`, `control`. These reuse role cues (a leaving group, a lone
  carbonyl electrophile) that recur across the training mechanisms.
- **Collapses (≤ 50%):** `diels_alder` (37%), `benzylic_radical_bromination` (43%),
  `nitroalkane_deprotonation` (47%), `hetero_diels_alder` (50%). These hinge on a cue
  unique to the held-out class and absent from training — e.g. with every Diels-Alder
  reaction removed, the model never learns the diene/dienophile π-role and defaults to
  spectator.

**Takeaway.** MENDEL interpolates strongly within its 14 known mechanisms but does not
yet extrapolate to genuinely novel ones; the negotiation layer degrades here too,
since it dispatches on a mechanism hint an unseen reaction will not match. Closing the
gap needs broader mechanism coverage and/or mechanism-agnostic role features, not just
more reactions per known class.

### Trained checkpoint

`models/role_mlp.pt` — architecture: `Linear(65, 64) → ReLU → Dropout → Linear(64, 5)`, trained on 166 reactions with mechanism-stratified reaction-level val split, class-weighted cross-entropy loss, early stopping on val loss.

---

## Install

```bash
git clone <repo-url>
cd mendel
conda create -n mendel python=3.12
conda activate mendel
pip install -e ".[dev]"          # Phases 0–6 + tests (rdkit only)
pip install -e ".[ml]"           # Phase 7 — installs torch
pip install -e ".[mlip]"         # Phase 9 — MACE-OFF (ASE + mace-torch)
pip install -e ".[ani2x]"        # Phase 9 — ANI-2x (torchani ≥ 2.2 + ASE)
pip install -e ".[mlip-all]"     # Phase 9 — MACE-OFF + ANI-2x together
```

> MACE on Apple Silicon requires `--device cpu` (MPS does not support float64).

---

## Current Status

| Phase | Description | Status |
|-------|-------------|--------|
| 0 | Project scaffold and data contracts | ✓ |
| 1 | Reaction SMILES parser | ✓ |
| 2 | Functional group identifier (RDKit + SMARTS) | ✓ |
| 3 | Group descriptor builder (55-dim → 65-dim in 6.6) | ✓ |
| 4 | Labeled data schema | ✓ |
| 5 | Rule-based role predictor | ✓ |
| 6 | Negotiation layer | ✓ |
| 6.5 | Dataset curation / draft label generation | ✓ |
| 6.6 | Descriptor upgrade — inter-molecular partner context (55→65 dim) | ✓ |
| 7 | MLP role predictor — 166 reactions, mechanism-stratified training | ✓ |
| 8 | Benchmark evaluator, center head, dataset ops | ✓ |
| 9 | Optional MLIP backend (MACE-OFF, ANI-2x) — design boundary confirmed | ✓ |
| 10 | rMD17/QO2Mol benchmark + MENDEL force decomposition | ✓ |

---

## Training

```bash
# Add Michael addition examples (12 reactions)
conda run -n mendel python scripts/add_michael_examples.py

# Add aldol examples (6 reactions with corrected label convention)
conda run -n mendel python scripts/add_aldol_examples.py

# Train MLP
conda run -n mendel python scripts/train_mlp.py \
  --data data/reactions.center_balanced.cleaned.json \
  --output models/role_mlp.pt \
  --epochs 150 --hidden-dim 64 --use-class-weights

# Run benchmark
conda run -n mendel python scripts/benchmark.py \
  --data data/reactions.center_balanced.cleaned.json \
  --mlp models/role_mlp.pt
```

---

## Validation

### Phase 0–6 (no PyTorch required)

```bash
PYTHONDONTWRITEBYTECODE=1 pytest -q -p no:cacheprovider \
  tests/test_phase0_scaffold.py tests/test_parser.py \
  tests/test_identifier.py tests/test_descriptor.py \
  tests/test_labels.py tests/test_predictor.py tests/test_negotiator.py
```

### Phase 7 — MLP (requires `pip install -e ".[ml]"`)

```bash
pytest tests/test_mlp.py tests/test_mlp_aware_negotiation.py -q
```

### Phase 9 — MLIP backend (no live MACE/torchani required for unit tests)

```bash
pytest tests/test_mlip.py -q
```

---

## Key Commands

```bash
# run all tests
pytest

# lint / format / type check
ruff check mendel/ tests/
ruff format mendel/ tests/
mypy mendel/

# generate draft labels from rule-based pipeline
python scripts/draft_labels.py --core --output data/reactions.draft.core.json

# train MLP (smoke test, no curated data needed)
python scripts/train_mlp.py \
  --data data/reactions.minimal.json \
  --output models/role_mlp_minimal.pt \
  --epochs 3 --hidden-dim 16 --allow-draft-labels

# MLIP single-point (Phase 9)
python scripts/mlip_singlepoint.py \
  --smiles "CC(=O)C" --backend mace --model-name mace-off-small --device cpu \
  --output reports/mlip_acetone.json

# MENDEL-guided MLIP (uses negotiated reaction center for force summary)
python scripts/mlip_singlepoint.py \
  --reaction-smiles "CBr.[OH-]>>CO.[Br-]" --context ionic \
  --reaction-center-from-mendel --device cpu \
  --output reports/mlip_sn2.json
```

---

## MLIP Comparison: Pure MLIP vs MENDEL + MLIP

Phase 10 runs MACE-OFF-small and ANI-2x against rMD17 ethanol (100 conformers, revPBE-D3 DFT) and decomposes force errors using MENDEL's functional-group agents.

### Global results (pure MLIP)

| Metric | MACE-OFF-small | ANI-2x |
|--------|---------------|--------|
| Force MAE (eV/Å) | 0.374 | 0.258 |
| Force RMSE (eV/Å) | 0.443 | 0.305 |
| Energy MAE (eV) | 11.35 | 7.23 |
| Per-element RMSE: C | 0.479 | 0.341 |
| Per-element RMSE: H | 0.418 | 0.293 |
| Per-element RMSE: O | 0.513 | 0.308 |

ANI-2x outperforms MACE-OFF-small by ~31–36% on all metrics. It was trained on CCSD(T)/CBS organic-molecule data, which is closer to rMD17's revPBE-D3 organic conformer distribution.

### MENDEL force decomposition

| Functional group | MACE RMSE (eV/Å) | ANI-2x RMSE (eV/Å) |
|-----------------|-----------------|------------------|
| alcohol C–O (reactive, MENDEL-identified) | **0.954** | **0.601** |
| hydroxyl H (O–H) | 0.771 | 0.482 |
| alpha C–H (reactive side) | 0.758 | 0.550 |
| methyl C–H (spectator) | 0.683 | 0.486 |
| methyl C (spectator) | 0.587 | 0.512 |

**Key finding**: both models concentrate their force error on the MENDEL-identified reactive functional group. The reactive-site RMSE is ~2× the global figure. A single global number hides this entirely.

MENDEL currently acts as a **diagnostic tool** — it reveals where each MLIP struggles chemically. The MLIP force values themselves are unchanged; MENDEL provides the chemical lens to interpret them.

### Reproduce

```bash
# install MACE + ANI-2x
pip install -e ".[mlip-all]"

# run MACE-OFF benchmark (--device cpu required on Apple Silicon)
python scripts/run_mlip_reference_benchmark.py \
  --reference data/reference/rmd17_ethanol_sample_converted.reference.json \
  --backend mace --model-name mace-off-small --device cpu

# run ANI-2x benchmark
python scripts/run_mlip_reference_benchmark.py \
  --reference data/reference/rmd17_ethanol_sample_converted.reference.json \
  --backend ani2x --device cpu

# generate comparison figures
python scripts/compare_mace_ani2x.py
# → reports/figures/mace_vs_ani2x_ethanol.png

python scripts/compare_pure_vs_mendel_mlip.py
# → reports/figures/pure_vs_mendel_mlip.png
```

See [docs/mlip_comparison.md](docs/mlip_comparison.md) for the full analysis.

---

## Design Principles

- **Functional group = agent** — the natural unit of organic chemistry decision-making
- **Interpretable** — every prediction is chemically explainable
- **Modular** — each phase is independently testable and swappable
- **Fully local** — no API calls, no external services
- **Honest diagnostics** — MENDEL shows where models struggle; it does not silently paper over errors
