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
        check-hyperparams train train-mini train-instability-mini train-dpmorl-only-mini \
        train-dpmorl-only-mini-parallel test clean help

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
	@echo "  make hyperparams-mini   Compute hyperparams (mini, 200 rollouts)"
	@echo "  make check-hyperparams  Validate hyperparams.json (PASS/FAIL per value)"
	@echo "  make train-mini         Smoke-test training on mini dataset"
	@echo "  make train-instability-mini  Instability-analysis experiment (mini dataset)"
	@echo "  make train-dpmorl-only-mini  DPMORL-only experiment, safety off (mini dataset)"
	@echo "  make train-dpmorl-only-mini-parallel  Same, K policies as concurrent processes"
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
	@echo "   (filtering hydra-core — already pinned in environment.yml at a working version)"
	grep -vE "^(hydra-core|opencv-python|numpy)" $(NUPLAN_DEVKIT_PATH)/requirements.txt > /tmp/nuplan_req_filtered.txt
	conda run -n $(CONDA_ENV) pip install -r /tmp/nuplan_req_filtered.txt --prefer-binary
	@echo ""
	@echo "── Installing CaRL nuplan dependencies ──"
	grep -vE "^(hydra-core|opencv-python|numpy)" ../nuPlan/requirements_nuplan.txt > /tmp/carl_nuplan_req_filtered.txt
	conda run -n $(CONDA_ENV) pip install -r /tmp/carl_nuplan_req_filtered.txt --prefer-binary
	@echo ""
	@echo "── Installing CaRL dependencies ──"
	conda run -n $(CONDA_ENV) pip install -r ../nuPlan/requirements_carl.txt --prefer-binary
	@echo ""
	@echo "── Installing nuplan-devkit ──"
	conda run -n $(CONDA_ENV) pip install -e $(NUPLAN_DEVKIT_PATH) --prefer-binary
	@echo ""
	@echo "── Installing CaRL (carl_nuplan) ──"
	conda run -n $(CONDA_ENV) pip install -e ../nuPlan --prefer-binary
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
# Hyperparameters — mini (200 rollouts; fast enough, better tail statistics)
# ─────────────────────────────────────────────────────────────────────────────

hyperparams-mini:
	@echo "── Computing hyperparams from mini dataset ──"
	$(PYTHON) compute_hyperparams.py \
	    --output_path      hyperparams.json \
	    --n_warmup_rollouts 200 \
	    --nuplan_data_root  $(NUPLAN_MINI_ROOT)

# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameter validation
# ─────────────────────────────────────────────────────────────────────────────

check-hyperparams:
	@bash scripts/check_hyperparams.sh

# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameters — full (200 rollouts as per CLAUDE.md)
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
# Training — instability-analysis experiment (mini smoke test)
# ─────────────────────────────────────────────────────────────────────────────

# n_policies is intentionally NOT passed as --n_policies here (unlike
# train-mini's --n_policies 2, which caps K for a quick pipeline smoke test):
# this is a real diagnostic run, not a smoke test, so K=4 from
# prism_instability_experiment.yaml is the source of truth.
train-instability-mini:
	@echo "── Instability-analysis experiment (indicator-only TTC cost, mini dataset, K=4) ──"
	$(PYTHON) scripts/train.py \
	    --config     configs/prism_instability_experiment.yaml \
	    --hyperparams hyperparams.json \
	    --nuplan_data_root $(NUPLAN_MINI_ROOT) \
	    --cache_path  $(MINI_CACHE_PATH) \
	    --output_dir  runs/instability_ttc

# ─────────────────────────────────────────────────────────────────────────────
# Training — DPMORL-only experiment (safety term off, mini smoke test)
# ─────────────────────────────────────────────────────────────────────────────

train-dpmorl-only-mini:
	@echo "── DPMORL-only experiment (beta=0, cf_coef=0, mini dataset, K=2) ──"
	$(PYTHON) scripts/train.py \
	    --config     configs/prism_dpmorl_only.yaml \
	    --hyperparams hyperparams.json \
	    --nuplan_data_root $(NUPLAN_MINI_ROOT) \
	    --cache_path  $(MINI_CACHE_PATH) \
	    --output_dir  runs/dpmorl_only

# ─────────────────────────────────────────────────────────────────────────────
# Training — DPMORL-only experiment, K policies as concurrent processes.
# Real wall-clock parallelism (separate OS processes/CPU cores), not matrix
# batching -- see --policy_ids's help text and CHANGES.md for why. Runs
# Stage 1 once (saves utility functions), then launches one process per
# policy_id in the background, each restricted to its own index, and waits
# for all of them. Per-policy stdout/stderr goes to its own log file since
# concurrent processes writing to one terminal would interleave.
# ─────────────────────────────────────────────────────────────────────────────

# Auto-detected from the config so this can't silently drift out of sync;
# override on the CLI if you want fewer than the config's full K, e.g.
# `make train-dpmorl-only-mini-parallel N_POLICIES=1`.
N_POLICIES ?= $(shell $(PYTHON) -c "import yaml; print(yaml.safe_load(open('configs/prism_dpmorl_only.yaml'))['n_policies'])")
DPMORL_ONLY_OUT := runs/dpmorl_only
DPMORL_ONLY_EXP := prism_dpmorl_only_001

# Optional instability-analysis ablation passthrough (see CHANGES.md), e.g.:
#   make train-dpmorl-only-mini-parallel UTILITY_ABLATION=linear_projection LOG_REWARD_COMPONENTS=1
UTILITY_ABLATION ?=
LOG_REWARD_COMPONENTS ?=
ABLATION_FLAGS := $(if $(UTILITY_ABLATION),--utility_ablation $(UTILITY_ABLATION)) $(if $(LOG_REWARD_COMPONENTS),--log_reward_components)

train-dpmorl-only-mini-parallel:
	@echo "── DPMORL-only experiment: Stage 1, then $(N_POLICIES) polic$(if $(filter 1,$(N_POLICIES)),y,ies) as concurrent processes ──"
	$(PYTHON) scripts/train.py \
	    --config     configs/prism_dpmorl_only.yaml \
	    --hyperparams hyperparams.json \
	    --nuplan_data_root $(NUPLAN_MINI_ROOT) \
	    --cache_path  $(MINI_CACHE_PATH) \
	    --output_dir  $(DPMORL_ONLY_OUT) \
	    $(ABLATION_FLAGS) \
	    --stage1_only
	@echo "── Stage 1 done. Launching policy_ids 0..$$(($(N_POLICIES)-1)) concurrently -- logs in $(DPMORL_ONLY_OUT)/policy_*.log ──"
	@for k in $$(seq 0 $$(($(N_POLICIES)-1))); do \
	    $(PYTHON) scripts/train.py \
	        --config     configs/prism_dpmorl_only.yaml \
	        --hyperparams hyperparams.json \
	        --nuplan_data_root $(NUPLAN_MINI_ROOT) \
	        --cache_path  $(MINI_CACHE_PATH) \
	        --output_dir  $(DPMORL_ONLY_OUT) \
	        --skip_stage1 \
	        --utility_fn_dir $(DPMORL_ONLY_OUT)/$(DPMORL_ONLY_EXP)/stage1 \
	        $(ABLATION_FLAGS) \
	        --policy_ids $$k > $(DPMORL_ONLY_OUT)/policy_$$k.log 2>&1 & \
	done; \
	wait; \
	echo "── All $(N_POLICIES) polic$(if $(filter 1,$(N_POLICIES)),y,ies) finished. Metrics in $(DPMORL_ONLY_OUT)/$(DPMORL_ONLY_EXP)/policy_*/  ──"

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
