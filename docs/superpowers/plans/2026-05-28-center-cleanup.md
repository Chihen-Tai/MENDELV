# Center Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Clean conservative reaction-center label issues, add mechanism-balanced strict splitting, and scaffold center-label-focused candidate expansion.

**Architecture:** Put cleanup rules in `mendel/center_cleanup.py`, keep split logic in `mendel/center_validation.py`, and expose scripts that write new Phase 8.12 artifacts without overwriting canonical datasets or checkpoints.

**Tech Stack:** Python dataclasses, existing MENDELV label/parser/identifier utilities, existing center-head training/benchmark CLIs, pytest.

---

### Task 1: RED Tests

**Files:**
- Create: `tests/test_center_cleanup.py`
- Create: `tests/test_center_expansion.py`
- Create: `tests/test_mechanism_balanced_split.py`

- [ ] Add tests for conservative corrections, expansion candidate metadata, and mechanism-balanced template split behavior.
- [ ] Run tests and confirm missing modules/scripts fail.

### Task 2: Cleanup Module and CLI

**Files:**
- Create: `mendel/center_cleanup.py`
- Create: `scripts/cleanup_center_labels.py`

- [ ] Implement correction dataclasses, conservative cleanup rules, report saving, and dataset saving.
- [ ] CLI audits before/after cleanup and writes cleaned dataset/report unless dry run.

### Task 3: Mechanism-Balanced Split

**Files:**
- Modify: `mendel/center_validation.py`

- [ ] Add `mechanism_balanced_template` strategy that keeps templates intact and tries to place multiple mechanisms in val/test.
- [ ] Report warnings when desired test coverage is impossible.

### Task 4: Expansion Scaffolding

**Files:**
- Create: `scripts/generate_center_expansion_candidates.py`
- Create: `scripts/run_center_expansion_pipeline.py`

- [ ] Generate local textbook-style candidate inputs with center-label focus metadata.
- [ ] Add pipeline wrapper that writes expected expansion artifacts and report without external data.

### Task 5: Convenience Runner and Docs

**Files:**
- Create: `scripts/run_center_cleanup_validation.py`
- Create: `docs/center_cleanup.md`
- Modify: `docs/center_validation.md`
- Modify: `README.md`

- [ ] Chain cleanup, validation, strict retraining, and strict benchmark.
- [ ] Document policy and commands.
- [ ] Run lint, acceptance tests, and manual Phase 8.12 core commands.
