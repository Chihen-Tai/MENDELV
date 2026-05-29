# Atom Center Head Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an experimental atom-level reaction-center classifier for Phase 8.10.

**Architecture:** Keep the new model isolated in `mendel/center_head.py`, with separate training and benchmark CLIs. The head consumes labeled reactions and optional promoted-role MLP predictions as features, but it does not retrain role MLPs or touch MLIP code.

**Tech Stack:** Python dataclasses, RDKit parsing already present in MENDELV, optional PyTorch loaded only inside training/prediction functions, pytest.

---

### Task 1: Center Head Tests

**Files:**
- Create: `tests/test_center_head.py`

- [ ] Write failing tests for dataclass serialization, example construction, all-negative controls, summaries, aggregation, metrics, torch-gated training/prediction smoke tests, and CLI help.
- [ ] Run `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_center_head.py` and confirm RED failures for missing module/CLIs.

### Task 2: Center Head Module

**Files:**
- Create: `mendel/center_head.py`

- [ ] Implement required dataclasses with `to_dict`.
- [ ] Implement atom example building from reactant molecules, labeled `reaction_center_atoms`, group membership, and optional role predictions.
- [ ] Implement compact deterministic atom features with role, group type, mechanism, and local atom properties.
- [ ] Implement summary, aggregation, atom/reaction metrics, JSON save helpers.
- [ ] Implement optional PyTorch binary MLP training/loading/prediction without requiring torch at import time.
- [ ] Run the center-head tests and fix failures.

### Task 3: Phase 8.10 CLIs

**Files:**
- Create: `scripts/train_center_head.py`
- Create: `scripts/benchmark_center_head.py`

- [ ] Add training CLI that builds role features from `models/role_mlp_promoted.pt` when available, falls back to labels, refuses no-positive datasets, saves checkpoint/report.
- [ ] Add benchmark CLI that loads the center checkpoint, predicts atom centers, writes benchmark and comparison JSON, and prints whether it improves over Phase 8.9 references.
- [ ] Run CLI help tests and a manual smoke run.

### Task 4: Docs

**Files:**
- Create: `docs/center_head.md`
- Modify: `docs/index.md`
- Modify: `README.md`

- [ ] Document that Phase 8.10 is an atom binary classifier, not MLIP or energy/force modeling.
- [ ] Add commands and limitations.

### Task 5: Verification

- [ ] Run lint on new/changed files.
- [ ] Run acceptance tests without torch scope.
- [ ] Run torch acceptance tests in the current venv if torch is installed.
- [ ] Run manual train and benchmark commands, report honest metrics.
