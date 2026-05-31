# MENDEL developer tasks — mechanical equivalents of the README quality flow.
.PHONY: help install install-ml lint format-check typecheck \
        test-core test-ml test-mlip-light coverage quality

PYTEST ?= pytest
RUFF   ?= ruff
MYPY   ?= mypy

# Phase 9 MLIP test files that require Phase 9 deps to even import.
MLIP_IGNORES := --ignore=tests/test_mlip_env_scripts.py \
                --ignore=tests/test_mlip_geometry_sanity.py \
                --ignore=tests/test_mlip_reference_benchmark.py

help:
	@grep -E '^[a-zA-Z0-9_-]+:.*?# ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?# "}{printf "  %-18s %s\n",$$1,$$2}'

install:           # editable install with dev tooling
	pip install -e ".[dev]"

install-ml:        # dev + ML (torch) tooling
	pip install -e ".[dev,ml]"

lint:              # ruff lint
	$(RUFF) check mendel/ tests/

format-check:      # ruff format check (no writes)
	$(RUFF) format --check mendel/ tests/

typecheck:         # mypy
	$(MYPY) mendel/

# core: phases 0–8 with no torch/ASE; ml/mlip tests self-skip via importorskip.
test-core:         # rule-based + non-ML suite
	PYTHONDONTWRITEBYTECODE=1 $(PYTEST) -q -p no:cacheprovider \
	  --ignore=tests/test_mlip.py $(MLIP_IGNORES)

# ml: requires .[ml]; torch-marked tests run.
test-ml:           # MLP / torch suite
	PYTHONDONTWRITEBYTECODE=1 $(PYTEST) -q -p no:cacheprovider \
	  --ignore=tests/test_mlip.py $(MLIP_IGNORES)

# mlip-light: unit-level MLIP tests; live MACE/torchani parts self-skip via importorskip.
test-mlip-light:   # MLIP unit tests (no live backend needed)
	PYTHONDONTWRITEBYTECODE=1 $(PYTEST) -q -p no:cacheprovider tests/test_mlip.py

coverage:          # coverage over the non-MLIP suite
	PYTHONDONTWRITEBYTECODE=1 $(PYTEST) -q -p no:cacheprovider \
	  --cov=mendel --cov-report=term-missing --cov-report=xml \
	  --ignore=tests/test_mlip.py $(MLIP_IGNORES)

quality: lint format-check typecheck test-core  # full local gate
