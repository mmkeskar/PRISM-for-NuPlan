# PRISM Lab Makefile
# ─────────────────────────────────────────────────────────────────────────────
# Usage:
#   source lab.env          # set paths (once per shell session)
#   make setup              # one-time: create conda env and install packages
#   make check              # verify environment is ready
#   make cache-mini         # build scenario cache from mini dataset (~minutes)
#   make hyperparams-mini   # compute hyperparams from mini (~minutes)
#   make train-mini         # smoke-test training on mini dataset
#   make cache              # build full scenario cache (hours, run once)
#   make hyperparams        # compute hyperparams from full dataset
#   make train              # run full PRISM training
#   make test               # run unit tests
#
# Override any variable on the command line:
#   make cache NUPLAN_DATA_ROOT=/my/path CACHE_PATH=/my/cache
# ─────────────────────────────────────────────────────────────────────────────

# ── Load lab.env if present (machine-local paths) ────────────────────────────
-include lab.env

# ── Configuration (override via lab.env or CLI) ───────────────────────────────
NUPLAN_DATA_ROOT  ?= /data/nuplan/dataset
NUPLAN_MAP_ROOT   ?= $(NUPLAN_DATA_ROOT)/maps
NUPLAN_MINI_ROOT  ?= $(NUPLAN_DATA_ROOT)/nuplan-v1.1-mini
NUPLAN_DEVKIT_PATH ?= /data/nuplan/nuplan-devkit
CACHE_PATH        ?= /data/prism_cache
MINI_CACHE_PATH   ?= /data/prism_mini_cache
CONDA_ENV         ?= prism
TORCH_CUDA_VERSION ?= cu121

# Use the conda env's python if activated; otherwise fall back to system python
PYTHON            := python

.PHONY: all setup check-env check cache cache-mini hyperparams hyperparams-mini \
        train train-mini test clean help

# ─────────────────────────────────────────────────────────────────────────────
# Default target
# ─────────────────────────────────────────────────────────────────────────────

all: help

help:
	@echo ""
	@echo "PRISM Lab Makefile"
	@echo "────────────────────────────────────────────────────"
	@echo "  make check-env          Validate lab.env paths (no conda needed)"
	@echo "  make setup              One-time environment setup"
	@echo "  make check              Verify environment is ready"
	@echo "  make cache-mini         Build cache from mini dataset"
	@echo "  make hyperparams-mini   Compute hyperparams (mini, ~50 rollouts)"
	@echo "  make train-mini         Smoke-test training on mini dataset"
	@echo "  make cache              Build cache from full train dataset"
	@echo "  make hyperparams        Compute hyperparams (full, 200 rollouts)"
	@echo "  make train              Full PRISM training run"
	@echo "  make test               Run unit tests"
	@echo "  make clean              Remove __pycache__ and .pyc files"
	@echo ""
	@echo "Set paths in lab.env or pass on CLI."
	@echo ""

# ─────────────────────────────────────────────────────────────────────────────
# lab.env validation (no conda needed — run this first)
# ─────────────────────────────────────────────────────────────────────────────

check-env:
	@bash scripts/check_env.sh

# ─────────────────────────────────────────────────────────────────────────────
# One-time setup
# ─────────────────────────────────────────────────────────────────────────────

setup: environment.yml
	@echo "── Creating/updating conda env '$(CONDA_ENV)' ──"
	conda env create -f environment.yml --name $(CONDA_ENV) 2>/dev/null || \
	    conda env update -f environment.yml --name $(CONDA_ENV) --prune
	@echo ""
	@echo "── Installing PyTorch (CUDA $(TORCH_CUDA_VERSION)) ──"
	conda run -n $(CONDA_ENV) pip install torch \
	    --index-url https://download.pytorch.org/whl/$(TORCH_CUDA_VERSION)
	@echo ""
	@echo "── Installing nuplan-devkit dependencies ──"
	conda run -n $(CONDA_ENV) pip install -r $(NUPLAN_DEVKIT_PATH)/requirements.txt
	@echo ""
	@echo "── Installing nuplan-devkit ──"
	conda run -n $(CONDA_ENV) pip install -e $(NUPLAN_DEVKIT_PATH)
	@echo ""
	@echo "── Installing CaRL (carl_nuplan) ──"
	conda run -n $(CONDA_ENV) pip install -e ../nuPlan
	@echo ""
	@echo "── Installing PRISM package ──"
	conda run -n $(CONDA_ENV) pip install -e .
	@echo ""
	@echo "Setup complete. Activate with: conda activate $(CONDA_ENV)"
	@echo "Then run: make check"

# ─────────────────────────────────────────────────────────────────────────────
# Verification
# ─────────────────────────────────────────────────────────────────────────────

check:
	@bash scripts/lab_check.sh

# ─────────────────────────────────────────────────────────────────────────────
# Scenario cache — mini (quick smoke test, ~500 scenarios, minutes)
# ─────────────────────────────────────────────────────────────────────────────

cache-mini:
	@echo "── Building mini scenario cache ──"
	@echo "   data : $(NUPLAN_MINI_ROOT)"
	@echo "   maps : $(NUPLAN_MAP_ROOT)"
	@echo "   out  : $(MINI_CACHE_PATH)"
	$(PYTHON) scripts/build_cache.py \
	    --data_root   $(NUPLAN_MINI_ROOT) \
	    --map_root    $(NUPLAN_MAP_ROOT) \
	    --cache_path  $(MINI_CACHE_PATH) \
	    --max_scenarios 500 \
	    --workers 4

# ─────────────────────────────────────────────────────────────────────────────
# Scenario cache — full training set (hours, run once)
# ─────────────────────────────────────────────────────────────────────────────

cache:
	@echo "── Building full scenario cache (this takes a while) ──"
	@echo "   data : $(NUPLAN_DATA_ROOT)/nuplan-v1.1/splits/train"
	@echo "   maps : $(NUPLAN_MAP_ROOT)"
	@echo "   out  : $(CACHE_PATH)"
	$(PYTHON) scripts/build_cache.py \
	    --data_root   $(NUPLAN_DATA_ROOT)/nuplan-v1.1/splits/train \
	    --map_root    $(NUPLAN_MAP_ROOT) \
	    --cache_path  $(CACHE_PATH) \
	    --workers 8

# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameters — mini (fast approximation, 20 rollouts)
# ─────────────────────────────────────────────────────────────────────────────

hyperparams-mini:
	@echo "── Computing hyperparams from mini dataset ──"
	$(PYTHON) compute_hyperparams.py \
	    --output_path      hyperparams.json \
	    --n_warmup_rollouts 20 \
	    --nuplan_data_root  $(NUPLAN_MINI_ROOT)

# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameters — full (200 IDM rollouts as per CLAUDE.md)
# ─────────────────────────────────────────────────────────────────────────────

hyperparams:
	@echo "── Computing hyperparams from full dataset ──"
	$(PYTHON) compute_hyperparams.py \
	    --output_path      hyperparams.json \
	    --n_warmup_rollouts 200 \
	    --nuplan_data_root  $(NUPLAN_DATA_ROOT)

# ─────────────────────────────────────────────────────────────────────────────
# Training — mini smoke test
# ─────────────────────────────────────────────────────────────────────────────

train-mini:
	@echo "── Smoke-test training on mini dataset ──"
	$(PYTHON) scripts/train.py \
	    --config     configs/prism_default.yaml \
	    --hyperparams hyperparams.json \
	    --nuplan_data_root $(NUPLAN_MINI_ROOT) \
	    --cache_path  $(MINI_CACHE_PATH) \
	    --output_dir  runs/mini_test \
	    --n_policies  2

# ─────────────────────────────────────────────────────────────────────────────
# Training — full run
# ─────────────────────────────────────────────────────────────────────────────

train:
	@echo "── Full PRISM training ──"
	$(PYTHON) scripts/train.py \
	    --config      configs/prism_default.yaml \
	    --hyperparams hyperparams.json \
	    --nuplan_data_root $(NUPLAN_DATA_ROOT) \
	    --cache_path  $(CACHE_PATH) \
	    --output_dir  runs/prism_run_001

# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

test:
	$(PYTHON) -m pytest tests/ -v

# ─────────────────────────────────────────────────────────────────────────────
# Housekeeping
# ─────────────────────────────────────────────────────────────────────────────

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
