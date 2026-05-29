# Center Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add leakage-resistant split validation and center-label auditing for Phase 8.11.

**Architecture:** Keep split/audit logic in a new `mendel/center_validation.py` module. Reuse `mendel.center_head` for atom example summaries and benchmarking, and add split-aware benchmark helpers without changing the existing API.

**Tech Stack:** Python dataclasses, existing MENDELV label/parser/identifier utilities, optional PyTorch only through existing center-head training/prediction functions, pytest.

---

### Task 1: RED Tests

**Files:**
- Create: `tests/test_center_validation.py`
- Modify: `tests/test_center_head.py`

- [ ] Add tests for deterministic template keys, split grouping, non-mutating split assignment, center-label audit issue detection, report generation, CLI smoke, and split-aware center-head metrics.
- [ ] Run the new tests and confirm they fail because `mendel.center_validation` and strict split CLIs are missing.

### Task 2: Center Validation Module

**Files:**
- Create: `mendel/center_validation.py`

- [ ] Implement `CenterLabelIssue`, `CenterSplitRecord`, and `LeakageValidationReport`.
- [ ] Implement leakage key inference and deterministic split assignment.
- [ ] Implement center-label audit checks for invalid atoms, controls, empty reactive centers, duplicates, broad centers, functional-group membership mismatches, and mechanism-specific suspicious centers.
- [ ] Implement report generation, JSON saving, and recommendations.

### Task 3: Split-Aware Center Head Metrics

**Files:**
- Modify: `mendel/center_head.py`

- [ ] Add `benchmark_atom_center_head_by_split(...)` that returns `train`, `val`, `test`, and `overall` report dictionaries.
- [ ] Preserve the existing `benchmark_atom_center_head(...)` behavior.

### Task 4: CLIs

**Files:**
- Create: `scripts/validate_center_labels.py`
- Create: `scripts/retrain_center_head_strict_split.py`
- Create: `scripts/benchmark_center_head_strict_split.py`

- [ ] Add validate CLI for split assignment and label audit.
- [ ] Add strict retrain CLI that writes `models/atom_center_head_template_split.pt` and never overwrites the non-strict checkpoint.
- [ ] Add strict benchmark CLI that reports train/val/test/overall metrics and comparison JSON.

### Task 5: Docs and Verification

**Files:**
- Create: `docs/center_validation.md`
- Modify: `docs/index.md`
- Modify: `README.md`

- [ ] Document leakage risk, empty-truth policy, commands, and success/failure interpretation.
- [ ] Run lint, acceptance tests, manual validate/retrain/benchmark commands, and report results honestly.
