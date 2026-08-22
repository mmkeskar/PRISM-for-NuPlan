# PRISM — Training Guide

Pareto-optimal Risk-constrained drIving Style Manifold.  
Two backbone options: **CaRL PPO** (BEV raster) and **Alpamayo** (camera + VLM).

---

## Lab Computer Quick Start

Requires conda, the nuPlan mini dataset on disk, and nuplan-devkit source cloned.
All setup and training commands are driven by `make`.

### Step 1 — Clone and enter the repo

```bash
git clone <repo-url>
cd PRISM/PRISM
```

### Step 2 — Verify `lab.env`

`lab.env` holds all machine-local paths. Confirm they match your machine, then run the automated check:

```bash
cat lab.env          # inspect values
source lab.env
make check-env       # validates paths and file types, no conda required
```

| Variable | Description |
|---|---|
| `NUPLAN_DATA_ROOT` | Root of the nuPlan dataset (`maps/`, `nuplan-v1.1/` live here) |
| `NUPLAN_MAP_ROOT` | Path to the nuPlan maps folder |
| `NUPLAN_MINI_ROOT` | Path to the mini split directory containing `.db` files |
| `NUPLAN_DEVKIT_PATH` | Path to cloned nuplan-devkit source |
| `MINI_CACHE_PATH` | Output path for mini scenario cache |
| `CONDA_ENV` | Conda environment name (default: `prism`) |
| `TORCH_CUDA_VERSION` | CUDA version for PyTorch wheel (e.g. `cu124`) |
| `NUPLAN_SENSOR_ROOT` | Path to sensor blob directory (camera JPEGs) — Alpamayo only; harmless if unset for CaRL |

Check your CUDA version: `nvidia-smi | grep "CUDA Version"`

### Step 3 — Create the conda environment

```bash
source lab.env
make setup
```

`make setup` creates the `prism` conda environment from `environment.yml` (Python 3.10,
all non-torch deps), then installs PyTorch with the correct CUDA wheel, nuplan-devkit
from source, and the PRISM package.

### Step 4 — Activate and verify

```bash
conda activate prism
make check
```

`make check` runs `scripts/lab_check.sh` to confirm imports, data paths, and GPU
visibility are all healthy before you build any cache.

Add `source /path/to/PRISM/PRISM/lab.env` to `~/.bashrc` so paths are always set.

### Step 5 — Build the mini scenario cache

```bash
make cache-mini
```

Builds the cache from `$NUPLAN_MINI_ROOT` into `$MINI_CACHE_PATH` (500 scenarios,
takes a few minutes). Update `cache_path` in your training config to match.

### Step 6 — Compute hyperparameters

```bash
make hyperparams-mini   # 200 expert log rollouts on mini set (~minutes)
make check-hyperparams  # validate with colour-coded PASS/WARN/FAIL output
```

`hyperparams-mini` replays 200 nuPlan expert trajectories to calibrate reward
scaling, safety cost indicator weights, and z_t normalisation statistics. All
values must pass `check-hyperparams` before training. Use `make hyperparams`
(full dataset) before a real training run if mini statistics look marginal.
Note: the CVaR safety objective is an unconstrained penalty (`beta * CVaR`,
fixed `beta` in the training config) -- there is no threshold calibrated
from expert rollouts (see `CHANGES.md`).

### Step 7 — Smoke-test training

```bash
make train-mini
```

Runs a short CaRL PPO training on the mini cache with K=2 policies. Confirm loss
curves look sane before committing to a full run.

---

## Full training workflow

Once the smoke test passes, run on the full dataset:

```bash
make cache          # build full scenario cache (~hours, run once)
make hyperparams    # compute hyperparams from full dataset (200 rollouts)
make train          # full PRISM training run
```

Or with explicit `python` commands — see the Manual section below.

---

## Makefile targets reference

| Target | Description |
|---|---|
| `make setup` | Create conda env and install all packages |
| `make check-env` | Validate `lab.env` paths and file types (no conda needed) |
| `make check` | Verify environment, imports, data paths, GPU |
| `make cache-mini` | Build cache from mini dataset (500 scenarios) |
| `make cache` | Build cache from full training split |
| `make hyperparams-mini` | Compute hyperparams from mini (200 expert rollouts) |
| `make check-hyperparams` | Validate hyperparams.json — PASS/WARN/FAIL per value |
| `make hyperparams` | Compute hyperparams from full dataset (200 expert rollouts) |
| `make train-mini` | Smoke-test training on mini cache (K=2) |
| `make train` | Full PRISM training run |
| `make test` | Run unit tests |
| `make clean` | Remove `__pycache__` and `.pyc` files |

Any variable can be overridden on the command line:
```bash
make cache NUPLAN_DATA_ROOT=/my/path CACHE_PATH=/my/cache
```

---

## Training — CaRL PPO backbone

Uses CaRL's BEV semantic segmentation observation space.

```bash
make train-mini   # smoke test: mini cache, K=2 policies
make train        # full run: full cache, K=5 policies, output → runs/prism_run_001
```

Or with explicit arguments:

```bash
python scripts/train.py \
    --config configs/prism_default.yaml \
    --hyperparams hyperparams.json \
    --nuplan_data_root $NUPLAN_DATA_ROOT \
    --n_policies 5 \
    --total_timesteps 10000000 \
    --output_dir runs/carl_run_001
```

No backbone phase flags needed. A single training run covers both Stage 1
(utility function initialisation) and Stage 2 (per-policy PPO with an
unconstrained CVaR penalty) for all K=5 policies sequentially.

---

## Training — Alpamayo backbone

Alpamayo uses raw camera images (CAM_F0, CAM_L0, CAM_R0) and a 10B VLM
backbone. Training is split into two phases to protect the pretrained
representations.

### Phase A — frozen backbone

Only the stochastic action head and Q-Former critic train. Run this first.

```bash
python scripts/train.py \
    --config configs/prism_alpamayo.yaml \
    --hyperparams hyperparams.json \
    --nuplan_data_root $NUPLAN_DATA_ROOT \
    --backbone-phase a \
    --total_timesteps 5000000 \
    --output_dir runs/alpamayo_phase_a
```

The backbone (`nvidia/Alpamayo-R1-10B`) is downloaded from HuggingFace on first
run. To use a local copy, set `backbone_model_name` in
`configs/prism_alpamayo.yaml` to the local path.

### Phase B — LoRA fine-tuning

Requires a completed Phase A output directory. LoRA adapters are added to the
VLM (rank 16) and action expert (rank 32). The CVaR buffer resets so it
reflects only Phase B experience.

```bash
python scripts/train.py \
    --config configs/prism_alpamayo.yaml \
    --hyperparams hyperparams.json \
    --nuplan_data_root $NUPLAN_DATA_ROOT \
    --backbone-phase b \
    --phase-a-dir runs/alpamayo_phase_a \
    --total_timesteps 2000000 \
    --output_dir runs/alpamayo_phase_b
```

---

## Switching backbones at the command line

The `--model-backend` flag overrides the `model_backend` field in the config
without editing any file:

```bash
# Force CaRL PPO regardless of what the config says
python scripts/train.py --config configs/prism_alpamayo.yaml \
    --model-backend carl_ppo ...

# Force Alpamayo
python scripts/train.py --config configs/prism_default.yaml \
    --model-backend alpamayo ...
```

> **Note:** backend-specific config sections (`alpamayo_obs`, `alpamayo.*`)
> must still be present in the YAML for the chosen backend. Using
> `configs/prism_alpamayo.yaml` as the base config is safest.

---

## Evaluation

Evaluate a trained run on the nuPlan Val14 benchmark:

```bash
python scripts/evaluate.py \
    --policy_dir runs/alpamayo_phase_b \
    --config configs/prism_alpamayo.yaml \
    --hyperparams hyperparams.json \
    --n_episodes 200 \
    --output_dir runs/alpamayo_phase_b/eval
```

---

## Stage 1 utilities — advanced options

Run Stage 1 only and save utility functions for reuse:
```bash
python scripts/train.py --config configs/prism_default.yaml \
    --hyperparams hyperparams.json \
    --stage1_only \
    --output_dir runs/stage1_only
```

Skip Stage 1 and load pre-saved utility functions:
```bash
python scripts/train.py --config configs/prism_default.yaml \
    --hyperparams hyperparams.json \
    --skip_stage1 \
    --utility_fn_dir runs/stage1_only/stage1 \
    --output_dir runs/carl_run_002
```

---

## CLI reference

| Argument | Default | Description |
|---|---|---|
| `--config` | `configs/prism_default.yaml` | YAML config file |
| `--hyperparams` | `hyperparams.json` | Pre-computed hyperparameter file |
| `--nuplan_data_root` | — | Path to nuPlan dataset root |
| `--n_policies` | from config | Number of policies K |
| `--total_timesteps` | from config | Total env steps across all policies |
| `--output_dir` | from config | Directory for checkpoints and summaries |
| `--seed` | from config | Random seed |
| `--model-backend` | from config | `carl_ppo` or `alpamayo` — overrides config |
| `--backbone-phase` | — | `a` (frozen) or `b` (LoRA) — Alpamayo only |
| `--phase-a-dir` | — | Phase A output dir — required for `--backbone-phase b` |
| `--stage1_only` | false | Run Stage 1 and exit |
| `--skip_stage1` | false | Skip Stage 1, load from `--utility_fn_dir` |
| `--utility_fn_dir` | — | Directory containing saved utility function checkpoints |

---

## Output structure

```
runs/
└── alpamayo_phase_a/
    ├── stage1/
    │   ├── utility_fn_0.pth … utility_fn_4.pth
    └── policy_{k}/
        ├── policy_{k}_model_final.pth
        ├── policy_{k}_utility_final.pth
        ├── policy_{k}_lagrangian_final.json
        └── training_summary.json
```

Phase B loads `policy_{k}_model_final.pth` from the Phase A directory via
`--phase-a-dir`. The Lagrangian state is always reset fresh for Phase B.
