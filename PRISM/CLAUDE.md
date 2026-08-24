# PRISM — Claude Code Project Brief

## What This Project Is

PRISM (Pareto-optimal Risk-constrained drIving Style Manifold) is a
research codebase implementing a safe, personalised autonomous driving
planner. It learns K=5 policies forming a CVaR-safe Distributional
Pareto Front of driving styles (comfort-seeking, progress-oriented,
disciplined, cautious, balanced) on the nuPlan benchmark.

This is a PhD research project targeting IEEE IV / IROS. The paper
draft is in `docs/prism_paper.tex`. Every implementation decision must
be consistent with the formulation in that paper.

See `CHANGES.md` for a dated log of substantive changes to the codebase
(what changed, why, and any implementation decisions made along the way) and
`handoff.md` for session-by-session debugging notes.

---

## Repository Structure

```
prism/
├── CLAUDE.md                   ← this file (read first, always)
├── hyperparams.json            ← cached hyperparameters (auto-generated)
├── compute_hyperparams.py      ← run once before any training
│                                  uses collect_expert_rollouts() — nuPlan log
│                                  replay, NOT simulated IDM (avoids cache
│                                  chicken-and-egg; expert logs are equivalent
│                                  for calibration purposes)
├── handoff.md                  ← session handoff notes (updated each session)
├── docs/
│   ├── prism_paper.tex         ← full paper draft
│   └── references.bib
├── prism/
│   ├── env/
│   │   ├── nuplan_env.py       ← nuPlan gym wrapper
│   │   ├── rewards.py          ← all four style reward functions
│   │   ├── safety_cost.py      ← two-tier safety cost c_t
│   │   └── regime_detector.py  ← free-flow / car-following / congested
│   ├── morl/
│   │   ├── utility_functions.py ← non-decreasing neural networks (Stage 1)
│   │   ├── dpmorl_trainer.py   ← Stage 2 policy optimisation
│   │   └── cvar_penalty.py     ← state-augmented cost, VaR/nu, dense signal (g_nu)
│   ├── curriculum/
│   │   └── alpha_schedule.py   ← alpha(n) curriculum (no epsilon — see CVaR section)
│   └── utils/
│       ├── hyperparams.py      ← load hyperparams.json
│       └── zt_normaliser.py    ← z_t running normalisation with EMA
├── scripts/
│   ├── train.py                ← main training entry point
│   ├── evaluate.py             ← evaluation on nuPlan Val14
│   ├── visualise_pareto.py     ← plot the distributional Pareto front
│   ├── check_hyperparams.sh    ← thin bash wrapper: file existence only
│   └── check_hyperparams.py    ← Python validation: PASS/WARN/FAIL per value
├── configs/
│   └── prism_default.yaml      ← all training hyperparameters
└── tests/
    └── test_rewards.py         ← unit tests for reward functions
```

---

## External Repositories and Packages

### Base RL Planner — CaRL
- **Repo**: https://github.com/autonomousvision/CaRL
- **What we use**: observation space (BEV semantic segmentation), nuPlan
  environment wrappers, PPO training infrastructure, scenario filtering.
- **What we do NOT use**: CaRL's reward function (route completion + penalties).
  We replace this entirely with our style reward vector.
- **Key files to reference**: `carl/environment/`, `carl/training/ppo_trainer.py`
- **Do NOT copy CaRL's reward code** into `prism/env/rewards.py`.

### DPMORL — Distributional Pareto MORL
- **Repo**: https://github.com/zpschang/DPMORL
- **Paper**: NeurIPS 2023 — Cai et al., "Distributional Pareto-Optimal
  Multi-Objective Reinforcement Learning"
- **What we use**:
  - Non-decreasing neural network architecture for utility functions
    (`utility_functions.py`)
  - Diversity-based objective for generating K diverse utility functions
  - State augmentation with cumulative returns z_t (Algorithm 1)
  - Scalar reward transformation R_t = gamma^{-t}[f(z_{t+1}) - f(z_t)]
- **Key files**: `dpmorl/utility_function.py`, `dpmorl/algorithm.py`
- **What we add on top**: CVaR Lagrangian constraint (not in DPMORL)

### nuPlan
- **Repo**: https://github.com/motional/nuplan-devkit
- **What we use**: closed-loop simulation, scenario database, expert log
  replay for hyperparameter calibration, Val14 evaluation protocol
- **Install**: follow nuplan-devkit README for dataset setup
- **Key classes**:
  - `nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder.NuPlanScenarioBuilder`
  - `nuplan.planning.scenario_builder.scenario_filter.ScenarioFilter`
  - `nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario`
  - nuPlan metrics: `nuplan.planning.metrics`
- **NuPlanScenarioBuilder constructor** — must pass these kwargs (match `build_cache.py`):
  ```python
  NuPlanScenarioBuilder(
      data_root=..., map_root=...,
      sensor_root=nuplan_data_root,  # required even if unused
      db_files=None, map_version="nuplan-maps-v1.0",
      include_cameras=False, max_workers=1, verbose=False,
  )
  ```
- **angular_velocity caveat**: `DynamicCarState.angular_velocity` returns 0.0
  for all ego states in the nuPlan **mini** dataset. Detect with
  `np.max(np.abs(ang_vel)) < 1e-5` and fall back to finite-diff of heading.
- **Env var name**: `NUPLAN_MAPS_ROOT` (with S) not `NUPLAN_MAP_ROOT` — check
  lab.env and Makefile carefully; mismatch causes silent wrong-path errors.

### Stable-Baselines3
- **Package**: `stable-baselines3`
- **What we use**: PPO implementation as the inner RL algorithm
- **Important**: we use SB3's PPO but with a custom reward that includes
  the Lagrangian penalty. The PPO algorithm itself is unmodified.

---

## Core Mathematical Formulation

Read `docs/prism_paper.tex` for full derivations. Key equations:

### Style Reward Vector (4-dimensional, all in (0,1])

**Comfort** (Eq. comfort):
```
r_comfort = exp(-(j_lon^2 + j_lat^2) / sigma_j_sq)
```

**Progress** (Eq. progress):
```
r_progress = r_speed * r_accel * (0.5 + 0.5 * r_lane)

r_speed = exp(-max(0, v_des - v_ego) / (beta * v_des))
r_accel = 1 - exp(-|a_ego| / gamma_a)
r_lane  = lane_index / (N_lanes - 1)   [0.5 if N_lanes == 1]
```

**Desired speed regime detection** (check in this order):
1. Congested:     v_lane_avg < 0.5 * v_limit  →  v_des = v_lane_avg
2. Car following: lead vehicle in ego lane within horizon  →  v_des = v_lead
3. Free flow:     otherwise  →  v_des = v_limit

**Lateral discipline** (Eq. lateral):
```
r_lateral = r_dev * r_heading

r_dev     = 0.3 + 0.7 * exp(-(d_lat / sigma_d)^2)   [sigma_d = 0.2m]
r_heading = exp(-(|delta_psi| / phi)^2)
```

**Spacing** (Eq. spacing):
```
r_spacing = 0.2 + 0.8 * (1 - exp(-(max(0, TTC) / tau)^2))

TTC = d_lead / (v_ego - v_lead)   when v_ego > v_lead
TTC = inf                          otherwise or no lead vehicle
```

### Safety Cost (two-tier)

```
c_t = c_outcome + sum_j(c_lead_j)
```

**Outcome weights** (from hyperparams.json):
- VRU collision:        100
- Wrong direction:      100
- Vehicle collision:     80
- Red light:             80
- Stop sign:             70
- Drivable area:         65
- Object collision:      40

**Indicator weights** (computed by compute_hyperparams.py):
```
w_j = sum_{i in O_j} W_i / (|O_j| * T_{j->i})
```

**Episode-level cap** for persistent indicators (TTC, THW, speed):
```
sum_t(c_lead_j) <= mean(W_i for i in O_j)
```

### CVaR Safety Penalty — state-augmented cost + dual critic (v2)

**PRISM does not constrain CVaR against a threshold, and does not use
REINFORCE.** (v1 tried an unconstrained REINFORCE-style penalty; code
review found it violated the envelope theorem and had no importance-ratio
correction across PPO's epochs — see `CHANGES.md` for the full account.)
v2 extends DPMORL's own state-augmentation trick — already used for the
reward side below — to the safety cost, and trains a **separate cost
critic** through the same PPO machinery as the reward critic:

```
e_t = cumulative discounted safety cost               # mirrors z_t exactly
e_{t+1} = e_t + gamma^t * c_t                          # e_0 = 0 each episode
nu = quantile(episode_costs, alpha)                    # VaR threshold, updated
                                                        # once per training update
                                                        # from a rolling episode buffer
g_nu(e) = tau * softplus((e - nu) / tau)               # FIXED smooth hinge at nu
                                                        # (not learned, not per-policy)
c~_t = gamma^{-t} * [g_nu(e_{t+1}) - g_nu(e_t)]         # dense per-timestep cost signal
```

`c~_t` telescopes exactly to `g_nu(C^pi) - g_nu(0)` for the same reason
`R_t` below telescopes to `f(z_T) - f(z_0)` — the `gamma^{-t}` prefactor
cancels GAE's `gamma^l` discounting. It is dense (nonzero whenever raw cost
fires) because `g_nu` is a smooth softplus hinge, not a hard `(e-nu)^+`.
**Do not use the naive `(e_{t+1}-nu)^+ - (e_t-nu)^+` difference** — it is
sparse (exactly zero whenever both `e_t` and `e_{t+1}` are below `nu`),
which is the density problem `g_nu` exists to avoid.

`c~_t` feeds a cost critic `V^C(s, e_t)` via GAE, exactly parallel to how
the reward critic is trained on `R_t`. Reward and cost combine into a
single PPO update:

```
A_total = normalize(A_reward) - beta * normalize(A_cost)   # both streams
ppo_loss = clip(A_total, ratio)                             # ONE ratio, ONE clip
total_loss = ppo_loss + vf_coef*v_loss + cf_coef*cost_critic_loss - ent_coef*ent_loss
```

Reward and cost share the same importance ratio, clip, and epochs — there
is no separate REINFORCE term, so the v1 off-policy-drift and
envelope-theorem bugs cannot recur by construction. `beta` and `tau` are
fixed hyperparameters (`configs/*.yaml`), not learned. `beta` must be
tuned per backbone (CaRL vs. Alpamayo) even though advantage normalisation
makes it more portable than v1's raw-log-prob-scaled weight was.

Alpamayo's actor also conditions on `e_t`/`nu` via the same instruction-text
mechanism used for `z_t` (`prism/models/alpamayo/instruction.py`,
`AlpamayoAdapter.set_nu()`), tokenised and passed to the backbone as
`input_ids`/`attention_mask` — this was previously built but never actually
reaching the model; see `prism/models/alpamayo/adapter.py`'s `_run_backbone()`
for the one remaining risk (Qwen-VL-family models may require image
placeholder tokens interleaved via a processor, not independent kwargs —
verify on the lab machine before trusting it in the RL loop).

See `prism/morl/cvar_penalty.py` (`g_nu`, `dense_cost_signal`, `update_var`)
and `prism/morl/dpmorl_trainer.py` (dual GAE, combined loss) for the
implementation, and `CHANGES.md` for the full derivation and the Muni et
al. (arXiv:2602.03778) formula-verification notes that motivated the `g_nu`
construction over a literal transcription of that paper.

### State Augmentation (DPMORL)

```
s_tilde_t = concat(s_t, z_t_normalised)
z_{t+1} = z_t + gamma^t * r_t           # cumulative style returns
z_norm_i = (z_i - mu_i) / sigma_i       # normalised with EMA stats
```

### Scalar Reward (DPMORL Algorithm 1)

```
R_t = gamma^{-t} * [f_theta_k(z_{t+1}) - f_theta_k(z_t)]
```

---

## Hyperparameter File Protocol

**CRITICAL**: `hyperparams.json` must exist before any training.

### To compute (run once):
```bash
python compute_hyperparams.py \
    --output_path hyperparams.json \
    --n_warmup_rollouts 200 \
    --nuplan_data_root /path/to/nuplan/nuplan-v1.1-mini \
    --gamma 0.99
```

Or via Makefile (preferred):
```bash
source lab.env && conda activate prism
make hyperparams-mini    # mini dataset, 200 rollouts (~minutes)
make check-hyperparams   # validate values with PASS/WARN/FAIL output
```

### In training script — guard pattern:
```python
import subprocess, sys
result = subprocess.run(
    ["python", "compute_hyperparams.py",
     "--check_only", "--output_path", "hyperparams.json"],
    capture_output=True)
if result.returncode != 0:
    print("Hyperparams not found. Run compute_hyperparams.py first.")
    sys.exit(1)
```

### Loading in code:
```python
from prism.utils.hyperparams import load_hyperparams
hp = load_hyperparams("hyperparams.json")
sigma_j_sq = hp["reward_scaling"]["sigma_j_sq"]
lead_times = hp["lead_times"]         # nested dict
```

Note: `hyperparams.json` no longer has an `epsilon_curve` key. There is no
CVaR threshold — `beta` (the fixed penalty weight) lives in
`configs/*.yaml`, not in the calibrated hyperparameter file. Expert
rollouts (`collect_expert_rollouts`) are still used for reward scaling,
indicator lead times, and z_t normalisation.

---

## Training Entry Point

```bash
python scripts/train.py \
    --config configs/prism_default.yaml \
    --hyperparams hyperparams.json \
    --nuplan_data_root /path/to/nuplan \
    --n_policies 5 \
    --total_timesteps 10000000 \
    --n_rollouts_per_iter 512 \
    --gamma 0.99 \
    --n_curriculum_iters 5000 \
    --output_dir runs/prism_run_001
```

Training runs K=5 policies sequentially (or in parallel if multi-GPU).
Each policy is trained independently with its own cost critic and rolling
episode-cost buffer (see Key Design Decision #4 below); `beta` and `tau`
are shared, fixed hyperparameters, not per-policy state.

---

## Key Design Decisions (Do Not Change Without Discussion)

1. **No CaRL weight initialisation**: policies train from scratch.
   Rationale: CaRL's scalar reward would bias the style objectives.

2. **Expert log replay, not random rollouts**: hyperparams computed from
   nuPlan expert trajectories (`collect_expert_rollouts`), not random
   policy or simulated IDM. Rationale: random policy produces degenerate
   trajectories; expert replay avoids the cache chicken-and-egg problem
   (no built cache needed) while providing realistic kinematic statistics.

3. **Episode-level cap for persistent indicators**: TTC, THW, speed
   violation costs are capped per episode. Rationale: prevents
   cumulative indicator cost from exceeding outcome event costs.

4. **Per-policy episode-cost buffer and cost critic, shared beta/tau**: each
   of K=5 policies maintains its own rolling `EpisodeCostBuffer`, VaR (`nu`)
   estimate, and cost critic `V^C(s, e_t)` weights (different style
   preferences produce different safety cost distributions), but `beta` and
   `tau` are single fixed hyperparameters shared across all K policies --
   neither is learned or per-policy. (This replaces the old per-policy
   Lagrange multiplier `lambda_k`.)

5. **Trajectory-level CVaR, not timestep-level**: CVaR is computed
   over C^i = sum(gamma^t * c_t) per rollout. Rationale: captures
   cumulative trajectory danger, not momentary spikes.

6. **Regime check order: congested first**: prevents a slow lead
   vehicle from triggering car-following when the broader lane
   is congested. Always check congestion before car-following.

7. **N=1 lane edge case**: r_lane = 0.5 for single-lane roads.
   Rationale: lane choice is neutral when there is no choice.

---

## Files to Modify When Implementing Each Component

| Component | File |
|---|---|
| Style rewards | `prism/env/rewards.py` |
| Safety cost | `prism/env/safety_cost.py` |
| Regime detection | `prism/env/regime_detector.py` |
| nuPlan gym env | `prism/env/nuplan_env.py` |
| Utility functions | `prism/morl/utility_functions.py` |
| DPMORL Stage 2 | `prism/morl/dpmorl_trainer.py` |
| CVaR penalty | `prism/morl/cvar_penalty.py` |
| Alpha curriculum | `prism/curriculum/alpha_schedule.py` |
| z_t normalisation | `prism/utils/zt_normaliser.py` |
| Hyperparam loading | `prism/utils/hyperparams.py` |
| Hyperparam computation | `compute_hyperparams.py` (root level) |
| Hyperparam validation | `scripts/check_hyperparams.sh` + `scripts/check_hyperparams.py` |
| Main training loop | `scripts/train.py` |
| Evaluation | `scripts/evaluate.py` |

---

## Common Tasks for Claude Code

### Add a new reward component
1. Implement in `prism/env/rewards.py` following the existing pattern
2. Ensure output is in (0,1]
3. Add corresponding scaling param to `compute_hyperparams.py`
4. Add unit test in `tests/test_rewards.py`
5. Update paper equation in `docs/prism_paper.tex`

### Modify the safety cost
1. Edit `prism/env/safety_cost.py`
2. Update `INDICATOR_OUTCOME_MAP` or `OUTCOME_WEIGHTS` in
   `compute_hyperparams.py` if adding new signals
3. Delete `hyperparams.json` and rerun `make hyperparams-mini`
4. Run `make check-hyperparams` to validate the new values
5. Update Table I in `docs/prism_paper.tex`

### Change the alpha curriculum
1. Edit `prism/curriculum/alpha_schedule.py`
2. Update alpha_start/alpha_end in `configs/prism_default.yaml`
3. The epsilon curve is always read from `hyperparams.json` --
   no need to rerun hyperparams if only changing alpha_start/end

### Add a new policy (change K)
1. Change `n_policies` in `configs/prism_default.yaml`
2. Stage 1 will automatically generate K utility functions
3. Stage 2 loops over K -- no other changes needed

---

## Important Conventions

- **nuPlan runs at 10 Hz**: all time thresholds in paper (1.5s TTC,
  2.0s THW) correspond to 15 and 20 timesteps respectively.
- **Rewards are always in (0,1]**: never return 0 or negative from
  style reward functions. Use floor values (delta_d=0.3, delta_s=0.2).
- **Safety cost is always >= 0**: never return negative safety cost.
- **gamma = 0.99** everywhere: in rewards, z_t accumulation, and
  CVaR computation. Changing this requires rerunning hyperparams.
- **EMA beta = 0.01** for z_t normalisation update during training (unrelated
  to the CVaR penalty weight, also named `beta`, in `configs/*.yaml`).
- **Blind spot geometry**: defined as ±45° to ±135° from ego heading,
  within 10m longitudinal and 4m lateral of ego centre.
- **Three different `beta`s -- do not confuse them**: `reward_scaling.beta`
  (speed-shortfall scaling for `r_progress`, from `hyperparams.json`),
  `ZtNormaliser`'s EMA `beta=0.01` (z_t running-average rate), and the CVaR
  penalty weight `beta` in `configs/*.yaml` (fixed, e.g. `1.0`) are
  unrelated hyperparameters that happen to share a name.
- **angular_velocity not in mini DB**: `DynamicCarState.angular_velocity`
  returns 0.0 for all ego states in the nuPlan mini dataset. Fall back to
  `np.gradient(savgol_filter(heading_array, 7, 2), dt)`.
- **Jerk computation**: do NOT triple-difference position. Use
  `rear_axle_acceleration_2d` from `DynamicCarState`, apply
  `scipy.signal.savgol_filter(window=7, polyorder=2)`, then one
  `np.gradient` call. Triple-differencing amplifies noise by ~1000×.

---

## Dependencies

```
nuplan-devkit          # nuPlan simulation and data
stable-baselines3      # PPO implementation
torch                  # neural networks (GPU via conda/pip CUDA wheel)
numpy
scipy
matplotlib             # Pareto front visualisation
wandb                  # experiment tracking (optional)
gym==0.26.2            # CRITICAL: must be exactly 0.26.2
                       # newer gym breaks the step() return signature
                       # (returns 5-tuple; CaRL/nuPlan wrappers expect 4-tuple)
```

### Setup pipeline (run in order, once per machine)

```bash
# 1. Set machine-local paths (edit lab.env first)
source lab.env

# 2. Create conda env, install all dependencies
make setup              # installs PyTorch, nuplan-devkit, CaRL, PRISM

# 3. Verify environment
make check              # runs scripts/lab_check.sh

# 4. Build scenario cache (required for training; takes minutes for mini)
make cache-mini         # mini dataset (~500 scenarios, fast)
make cache              # full training set (hours, run once)

# 5. Compute hyperparameters (must run before any training)
make hyperparams-mini   # 200 rollouts from mini dataset (~minutes)
make check-hyperparams  # validate with PASS/WARN/FAIL output

# 6. Smoke-test training
make train-mini         # K=2 policies, mini dataset

# 7. Full training
make train              # K=5 policies, full dataset
```

`make setup` handles: conda env creation, PyTorch CUDA wheel, nuplan-devkit,
CaRL dependencies, and `pip install -e .` for the PRISM package itself. See
`Makefile` and `environment.yml` for pinned versions.

---

## Do Not Do

- Do not modify `hyperparams.json` by hand
- Do not add safety signals to the style reward functions
- Do not add style signals to the safety cost
- Do not use CaRL's reward function
- Do not hardcode scaling parameters -- always read from hyperparams.json
- Do not compute the VaR/CVaR *statistic* at timestep level -- `nu` and the
  logged `CVaR` diagnostic are always estimated from trajectory-level
  (episode) cumulative costs (`update_var`, `compute_empirical_cvar` in
  `prism/morl/cvar_penalty.py`). This is separate from the dense
  per-timestep *signal* `c~_t` fed to the cost critic, which is a reward-
  shaping construction that telescopes to a trajectory-level quantity, not
  itself a per-timestep CVaR estimate.
- Do not reintroduce a Lagrange multiplier / dual update for the safety
  constraint, or a threshold (`epsilon`/`d`) calibrated from the IDM/expert
  baseline. See `CHANGES.md` for why the constrained formulation was
  replaced (v1) and then why v1's own REINFORCE penalty was replaced in turn
  (v2).
- Do not reintroduce a REINFORCE-style term (raw `log_prob` multiplied by a
  cost weight) for the safety objective. Cost must flow through a cost
  critic + GAE + PPO's existing clipped-ratio objective, combined with the
  reward advantage before clipping (`A_total = A_reward - beta*A_cost`) --
  see the CVaR Safety Penalty section above for why (envelope-theorem
  violation and missing importance-ratio correction in v1).
- Do not estimate CVaR/VaR via a Gaussian closed form
  (`mu_C + phi(Phi^-1(alpha))/(1-alpha) * sigma_C`). Always use the
  empirical estimators in `prism/morl/cvar_penalty.py` -- the two-tier
  safety cost distribution is right-skewed.
- Do not use the sparse hinge difference `(e_{t+1}-nu)^+ - (e_t-nu)^+` for
  the dense cost signal -- it is exactly zero whenever both `e_t` and
  `e_{t+1}` are below `nu`. Always use `g_nu`'s smooth softplus hinge
  (`dense_cost_signal` in `prism/morl/cvar_penalty.py`).
- Do not add inline comments after values in `lab.env` — GNU Make parses
  `VAR=value   # comment` as the value including trailing spaces and the `#`.
  Put all comments on their own line.

---

## Karpathy Agentic Engineering Principles

This project is built under the agentic engineering discipline as defined by Andrej Karpathy (Sequoia Ascent 2026). These are not suggestions — they are the operating philosophy for how this codebase should be written and extended.

### The core distinction

Vibe coding raises the floor — anyone can ship something. Agentic engineering raises the ceiling — professionals ship fast _without sacrificing quality_. This project operates at the ceiling. That means:

- Never blindly accept generated code
- Design specs first, then let agents implement
- Inspect every diff, write tests, create evaluation loops
- Preserve correctness, security, taste, and maintainability throughout

### Software 3.0 thinking

The context window is the primary lever. Before writing any code, ask: what is the right context to give the agent so it can do this correctly? The spec in this `CLAUDE.md` is the Software 3.0 program. The implementation is what emerges from it.

### Macro actions only

Do not work line by line. Work in macro actions:

- "Implement this feature end to end"
- "Refactor this subsystem"
- "Write tests, run them, fix failures"
- "Compare approaches and propose a plan"

If you find yourself making small edits to individual lines, stop and reframe at a higher level.

### The MenuGen mistake — avoid it here

Karpathy's agent matched Stripe emails to Google account emails. Plausible code, broken product logic. The equivalent risk in this project: using session names as IDs instead of UUIDs, or matching users by anything other than `user_id`. Always use persistent, stable identifiers. Never cross-correlate by human-readable fields.

### Verifiability first

Build things that can be verified. Every feature should have a clear pass/fail signal:

- CLI commands should have predictable output that can be asserted
- API calls should return typed responses validated with Zod
- Memory injection should be testable: given X memories, system prompt contains Y
- Session consolidation should be inspectable before saving

### What agents handle vs what humans decide

Agents handle: API syntax, boilerplate, routine CRUD, filling in typed implementations, running and fixing tests.

Humans (you) decide: system boundaries, identity and auth logic, security rules, what makes it into the consolidation, which memories matter, the shape of the data model.

Do not let agents make decisions about: user ID strategy, payment logic, permission scoping, what gets persisted to EverMind.

### Agent-native by design

This product is for agentic engineers. Build it to be used by agents too:

- All CLI commands should be scriptable and produce machine-readable output when passed `--json`
- The `CLAUDE.md` is the agent-native interface to this codebase — keep it current
- Avoid UI-only flows: anything a human can do in the webapp, an agent should be able to do via CLI or API
- Write docs for agents first: precise, copy-pasteable, no "click here" instructions

### Taste matters

The agentic engineer is still in charge of aesthetics, judgment, and taste. Generated code can be bloated, copy-pasted, awkwardly abstracted, and brittle. It is your job to reject that. Specifically:

- No unnecessary abstraction layers — if two things do the same job, pick one
- No copy-pasted blocks — extract to a shared utility
- No magic strings — everything named, typed, and in `constants/`
- No silent failures — every async operation has explicit error handling

### You cannot outsource understanding

The quote that guides this project: _"You can outsource your thinking, but you can't outsource your understanding."_ Know why every architectural decision was made. If you cannot explain it, do not commit it.
