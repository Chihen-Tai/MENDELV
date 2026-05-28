# MENDEL

**Molecular Entity Negotiation for Dynamic Energy Landscapes** — a fully local, functional-group-level reaction role prediction framework for organic chemistry.

Each functional group in a molecule is treated as an **agent** that observes its local chemical environment, predicts its own reaction role, then negotiates with neighbouring groups to produce a coherent, conflict-free assignment.

The same functional-group-as-agent lens applies to MLIP evaluation: MENDEL decomposes MLIP force errors by functional group type, revealing that reactive sites (e.g. the alcohol C–O bond in ethanol) carry ~2× the global RMSE — a fact that a single global number hides entirely.

---

## Quick Demo

```python
from mendel.negotiator import run_full_rule_pipeline

result = run_full_rule_pipeline(
    "CBr.[OH-]>>CO.[Br-]",
    context="ionic",
)

print(result.mechanism_hint)    # sn2_or_e2_like
for ra in result.role_assignments:
    print(ra.group_id, ra.final_role)
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
  per-group descriptor building    (descriptor.py — 55-dim feature vector)
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
| 3 | Group descriptor builder (55-dim) | ✓ |
| 4 | Labeled data schema | ✓ |
| 5 | Rule-based role predictor | ✓ |
| 6 | Negotiation layer | ✓ |
| 6.5 | Dataset curation / draft label generation | ✓ |
| 7 | MLP role predictor training | ✓ (needs more curated data) |
| 8 | Benchmark evaluator + diagnostics | ✓ |
| 9 | Optional MLIP backend (MACE-OFF, ANI-2x) | ✓ |
| 10 | rMD17/QO2Mol benchmark + MENDEL force decomposition | ✓ |

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

# generate MACE vs ANI-2x comparison figure
python scripts/compare_mace_ani2x.py
# → reports/figures/mace_vs_ani2x_ethanol.png

# generate pure-MLIP vs MENDEL decomposition figure
python scripts/compare_pure_vs_mendel_mlip.py
# → reports/figures/pure_vs_mendel_mlip.png
```

See [docs/mlip_comparison.md](docs/mlip_comparison.md) for the full analysis and next steps toward making functional-group agents actually improve prediction accuracy (reactive-site weighted fine-tuning).

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
pytest tests/test_mlp.py -q
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

# benchmark rule-based + negotiated pipeline
python scripts/benchmark.py --data data/reactions.json --rule-based --negotiated

# MLIP single-point (Phase 9)
python scripts/mlip_singlepoint.py \
  --smiles "CC(=O)C" --backend mace --model-name mace-off-small --device cpu \
  --output reports/mlip_acetone.json

# MENDEL-guided MLIP (uses negotiated reaction center for force summary)
python scripts/mlip_singlepoint.py \
  --reaction-smiles "CBr.[OH-]>>CO.[Br-]" --context ionic \
  --reaction-center-from-mendelv --device cpu \
  --output reports/mlip_sn2.json
```

---

## Design Principles

- **Functional group = agent** — the natural unit of organic chemistry decision-making
- **Interpretable** — every prediction is chemically explainable
- **Modular** — each phase is independently testable and swappable
- **Fully local** — no API calls, no external services
- **Honest diagnostics** — MENDEL shows where models struggle; it does not silently paper over errors
