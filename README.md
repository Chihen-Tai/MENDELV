# MENDEL

**Molecular Entity Negotiation for Dynamic Energy Landscapes** — a fully local, functional-group-level reaction role prediction framework for organic chemistry.

---

## Core Idea

Each functional group in a molecule is treated as an **agent** that predicts its own role in a reaction, then negotiates with neighbouring groups to produce a coherent, conflict-free assignment.

```
reaction SMILES + context
         │
         ▼
  functional group detection
         │
         ▼
  per-group role prediction (agent)
         │
         ▼
  negotiation layer
         │
         ▼
  reaction center identification
         │
         ▼
  (optional) MLIP energy / 3D visualization
```

Roles (mutually exclusive per group per step):

| Role | Description |
|------|-------------|
| `reactive_nucleophile` | donates electrons |
| `reactive_electrophile` | accepts electrons |
| `reactive_radical` | radical center |
| `leaving_group` | departs with electron pair |
| `spectator` | uninvolved in this step |

---

## Phase 0 — Project Scaffold (current)

Phase 0 sets up the project structure and defines the core data contracts used by all later phases.

**What is implemented:**
- `mendel/types.py` — enums and dataclasses (no chemistry, no ML)
- `mendel/constants.py` — derived constant sets
- `data/reactions.example.json` — schema examples (SN2, Diels-Alder, radical bromination)
- Full test suite for the scaffold

**What is NOT yet implemented:** reaction SMILES parsing, SMARTS matching, functional group detection, descriptors, role prediction, negotiation, MLIP, or visualization.

---

## Install

```bash
git clone <repo-url>
cd mendel
python -m pip install -e ".[dev]"
```

Requires Python ≥ 3.10. No heavy chemistry or ML dependencies in Phase 0.

---

## Run tests

```bash
pytest -q
```

---

## Quick demo (Phase 0)

```python
import mendel
from mendel.types import ReactionContext, Role, ReactionRecord

record = ReactionRecord(
    reaction_id="sn2_demo",
    reaction_smiles="CBr.[OH-]>>CO.[Br-]",
    context=ReactionContext.ionic,
)

print(mendel.__version__)               # 0.1.0
print(record.reaction_id)               # sn2_demo
print(Role.reactive_nucleophile.value)  # reactive_nucleophile
```

---

## Repository Structure

```
mendel/
├── pyproject.toml          ← build and dev config
├── README.md               ← this file
├── LICENSE                 ← MIT
├── .gitignore
├── mendel/
│   ├── __init__.py         ← package entry point, version
│   ├── types.py            ← core enums and dataclasses
│   └── constants.py        ← derived constant sets
├── data/
│   └── reactions.example.json
├── docs/
│   └── index.md
├── tests/
│   └── test_phase0_scaffold.py
├── DESIGN.md               ← full architecture spec
├── BENCHMARK.md            ← benchmark reactions
├── TEMPLATE.md             ← template for new functional groups
└── groups/                 ← per-group SMARTS specifications
```

---

## Design Principles

- **Functional group = agent** — the natural unit of chemical decision-making
- **Interpretable** — every prediction is chemically explainable
- **Modular** — each phase is independently swappable
- **Zero cost** — no API calls, fully local

---

## Phase Roadmap

| Phase | Goal |
|-------|------|
| 0 | Project scaffold and data contracts ✓ |
| 1 | Functional group identifier (RDKit + SMARTS) |
| 2 | Group descriptor builder |
| 3 | Group agent role predictor (MLP) |
| 4 | Negotiation layer + MLIP wrapper |
| 5 | Benchmarking and visualization |
