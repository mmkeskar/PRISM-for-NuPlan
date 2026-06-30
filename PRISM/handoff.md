# PRISM Session Handoff

**Date:** 2026-06-30

**Session type:** Development / debugging

**Topic:** Debugged and fixed the full training pipeline from `make cache-mini` through `make train-mini`, resolving a cascade of environment, library, and logic bugs to reach a running training loop with an active Lagrangian.

---

## Current Project State

**Current milestone:** Get `make hyperparams-mini && make train-mini` running cleanly end-to-end on the lab machine.

- `make setup` — complete and verified
- `make check` — passing for all modules including `prism.env.nuplan_env`
- `make cache-mini` — **complete**: 500/500 scenarios cached to `/data/prism_mini_cache`
- `make hyperparams-mini` — complete with 200 rollouts; values mostly in range (phi=0.42 still a WARN, deferred — see Open Questions)
- `make check-hyperparams` — passing (phi warns, does not fail)
- `make train-mini` — **blocked**: K=2 policies, 10000 updates each. Pipeline runs without errors. CVaR and lambda are non-zero (safety cost path confirmed working). However, lambda diverges without bound (reached 1182 at update 500) and CVaR does not fall — the Lagrangian is not converging. See Open Questions.

### Architecture summary

PRISM trains K=5 policies forming a CVaR-safe Distributional Pareto Front of driving styles on nuPlan. The stack is:

- **PPO backbone:** CaRL (`../nuPlan/carl_nuplan/`) — provides BEV observation space, nuPlan gym wrapper, scenario cache format
- **MORL layer:** DPMORL (`prism/morl/`) — utility functions, Stage 1/2 trainer, CVaR Lagrangian
- **Environment:** `prism/env/` — PRISMEnv wraps CaRL's EnvironmentWrapper, adds z_t state augmentation, Lagrangian penalty, 4D style reward vector, two-tier safety cost
- **Hyperparams:** `hyperparams.json` (auto-generated, never edit by hand) — reward scaling, epsilon_curve, indicator weights, z_t normalisation stats

Key paths:
- Training entry point: `scripts/train.py`
- Config: `configs/prism_default.yaml`
- Hyperparams computation: `compute_hyperparams.py` (root level)
- Lab paths: `lab.env` (gitignored, real paths for lab machine)
- Makefile: all workflow targets

---

## Code Context

### Files modified this session

#### `lab.env`
Three additions to fix runtime environment issues on the lab machine:

- **`NUPLAN_MAPS_ROOT`** (new): CaRL's `gym_scenario.py` reads `os.getenv("NUPLAN_MAPS_ROOT")` (with trailing S), not `NUPLAN_MAP_ROOT`. Both are now exported and kept in sync. Missing this caused `TypeError: stat: path should be string … not NoneType` on every scenario load.
- **`LD_PRELOAD`** (new): Lab machine's system `libstdc++.so.6` lacks `CXXABI_1.3.15`, which conda's ICU/sqlite3 requires. Preloading the conda env's libstdc++ resolves it: `export LD_PRELOAD=/home/mi3/miniconda3/envs/prism/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}`.

**Critical gotcha — inline comments in lab.env corrupt Make variables:** The Makefile uses `-include lab.env`, which parses the file as Makefile syntax in addition to bash sourcing it manually. GNU Make preserves whitespace before `#` in variable assignments (bash does not). Any line of the form `export VAR=value   # comment` will set the Make variable to `value   ` (with trailing spaces) even though bash sees a clean value. This caused `os.path.isdir("…/maps   ")` to fail silently while `source lab.env` + `cat -A` looked clean. **All assignment lines in lab.env must have comments on their own preceding line, never inline after a value.**

#### `environment.yml`
Added `gym==0.26.2` to the pip section. CaRL's `ppo_model.py` imports the legacy `gym` package (not `gymnasium`). This was missing from the environment and not caught by `make check` because the failing import only triggers when the PPO agent is actually built inside the training loop, not at module import time.

#### `prism/env/nuplan_env.py`
Fixed a silent bug where the safety constraint was never active:

- `PRISMEnv.step()` was calling `info.pop("safety_cost", 0.0)` to extract the cost internally, which removed the key from the dict before returning it. The trainer then called `info.get("safety_cost", 0.0)` and always got 0.0. Fix: after popping internally, immediately re-expose: `info["safety_cost"] = c_t`.
- Removed the stale `info["episode_cost"] = float(c_t)` line at episode end, which was incorrectly exposing only the final-timestep cost as if it were the episode total. The trainer correctly accumulates per-step costs itself via `compute_episode_cost(episode_step_costs, gamma=...)`.

### Key reference files (unmodified)
- `scripts/build_cache.py` — authoritative example for `NuPlanScenarioBuilder` and `ScenarioFilter` call signatures
- `prism/morl/dpmorl_trainer.py` — training loop; logs every 100 updates; lagrangian saved as separate `.json`, not inside `.pth`
- `prism/morl/cvar_lagrangian.py` — CVaR estimator; episode costs accessed via `state_dict()["costs"]`
- `../nuPlan/carl_nuplan/planning/gym/cache/gym_scenario.py` — reads `NUPLAN_MAPS_ROOT` (with S) at module level as a constant

---

## Decisions & Reasoning

### 1. Expert log replay instead of live IDM simulation
*(carried forward from previous session — still the active decision)*

**Decision:** `collect_expert_rollouts` replays nuPlan's stored ground-truth ego trajectories rather than running the IDM planner in closed-loop simulation.

**Reason:** `compute_hyperparams.py` must run before the scenario cache exists. Live IDM simulation requires `SimulationRunner` + Hydra — bypassed in `build_cache.py`. Expert trajectories are equivalent for calibration purposes.

*Alternatives rejected:* Live IDM (requires Hydra, not cache-free), `PRISMEnv` wrapper (requires cache to exist first).

### 2. Savitzky-Golay smoothing before differentiation
*(carried forward — smoothing is applied, phi is still 0.42, unresolved)*

**Decision:** Apply `scipy.signal.savgol_filter(window=7, polyorder=2)` to acceleration and heading arrays before computing jerk and heading rate.

**Reason:** `rear_axle_acceleration_2d` is already a finite difference of velocity; differencing again for jerk amplifies noise by 100×. `angular_velocity` is absent from the mini DB, so heading rate requires an additional finite diff. Smoothing brought sigma_j_sq from ~4183 to ~5 (passes). phi went from 0.495 → 0.42 (still WARN, deferred).

### 3. No inline comments after assignment values in lab.env

**Decision:** All comments in `lab.env` must be on their own line above the assignment, never trailing after the value.

**Reason:** The Makefile's `-include lab.env` parses the file as GNU Make syntax. Make preserves whitespace before `#` in variable assignments, so `VAR=value   # comment` sets Make's variable to `value   ` with trailing spaces. Bash silently truncates at `#` so manual `source lab.env` diagnostics looked clean while `make train-mini` used the space-tainted value. This caused `os.path.isdir` to fail on a path that visually appeared correct in all log output (the trailing `\r` or spaces were disguised by terminal rendering).

*Alternatives rejected:* Using `$(strip ...)` in the Makefile — fragile, doesn't address the root cause for future edits.

### 4. Re-expose `safety_cost` in info after popping

**Decision:** `PRISMEnv.step()` pops `"safety_cost"` from info to consume it internally, then immediately re-adds it so the trainer can read it.

**Reason:** The pop was originally intended to keep the info dict clean, but the trainer needs per-step `c_t` to accumulate `episode_step_costs` for the discounted CVaR computation. Without re-exposing it, the trainer always got 0.0, the CVaR buffer stayed empty, and lambda never moved. Re-exposing is cleaner than changing the trainer to pull costs from a different source.

*Alternatives rejected:* Change `pop` to `get` everywhere — `pop` is still correct for the internal consumption; the re-add makes the intent explicit.

---

## Next Immediate Actions

- [ ] First attempt: lower `eta_lambda` from 0.01 → 0.0001 in `configs/prism_default.yaml`, then `rm -rf runs/mini_test && make train-mini`. Watch update 100–300 for CVaR trend.
- [ ] If CVaR still flat/rising at update 300 with eta_lambda=0.0001: add `lambda_max=50.0` cap to `CVaRLagrangian` in `prism/morl/cvar_lagrangian.py` and re-run.
- [ ] In parallel: add a one-off diagnostic log of mean per-step `c_t` per episode to compare against what `compute_hyperparams.py` observed — this tests the epsilon underestimation hypothesis.
- [ ] After Lagrangian convergence is confirmed on mini: scale up to `make train` (K=5, full dataset).

---

## Open Questions

### phi = 0.42 rad/s (above expected range of 0.01–0.3)

**Question:** The lateral heading-rate scale parameter phi is 0.42 after Savitzky-Golay smoothing. The check script warns (not FAILs) at this value. Is this a calibration artifact of the mini dataset, or is 0.42 genuinely representative?

**Status:** Confirmed value after smoothing. Deferred — the rest of the pipeline is now running, so this is a lower priority. phi=0.42 means sharp-turn heading rates (common in urban intersection scenarios in the mini split) sit at the 1-sigma point of `r_heading`, compressing the reward less than a calibrated value would. Training will still work; style discrimination for lateral discipline may be slightly reduced.

**Approaches:**
- (a) Widen the check-script upper bound to 0.5 and document the urban mini dataset bias — quick fix, allows `make check-hyperparams` to fully pass
- (b) Increase SG filter window (e.g., window=11 or 15) for heading — may reduce phi further at the cost of over-smoothing genuine heading changes
- (c) Use full training split (not mini) for hyperparams computation — more diverse scenarios should bring phi into range, but requires full cache first

**Recommendation:** Option (a) as an immediate fix; revisit with option (c) before the final training run for the paper.

### gamma_a = 0.19 m/s² (below expected range of 0.3–3.0, WARN only)

**Question:** Is this a calibration artifact of the mini dataset's smooth scenarios, or a genuine measurement?

**Status:** Unchanged from previous session. WARN, not FAIL. ZtNormaliser EMA (beta=0.01) adapts during training, so this is not blocking.

**Approaches:**
- (a) Accept — self-corrects during training
- (b) Increase n_rollouts to 500 for more diverse acceleration distribution
- (c) Inspect r_progress distributions in tensorboard after training starts

### z_mu[1] (progress dimension) ~10× lower than other dimensions

**Question:** z_mu for progress (≈28) is far below comfort/lateral/spacing (≈290–400).

**Status:** Root cause known: `_EmptyMapProxy` forces v_des = 13.89 m/s (50 km/h) for all steps; urban mini scenarios have v_ego << v_des, collapsing r_speed ≈ 0. ZtNormaliser EMA will self-correct within a few hundred training episodes.

**Recommendation:** Accept for now. Note in paper appendix that z_t normalisation bootstraps from 50 km/h default and adapts online.

### Lagrangian divergence during train-mini — BLOCKING

**Question:** CVaR is stuck at 200–300 and lambda is growing linearly without bound. The safety constraint is not converging. What is causing this and how do we fix it?

**Status:** Confirmed divergence at update 500. Lambda reached 1182 growing ~24/10 updates (~2.4/update). CVaR did not fall — it fluctuated 191–298 with no downward trend, actually increasing slightly after update 200. The policy is not learning to reduce safety violations despite the constraint penalty growing to dominate the reward.

**Observed data (selected):**

| update | CVaR | lambda |
|--------|------|--------|
| 40 | 222.5 | 112 |
| 100 | 221.1 | 230 |
| 200 | 199.7 | 436 |
| 300 | 256.7 | 673 |
| 400 | 270.4 | 935 |
| 500 | 242.6 | 1182 |

**Root cause analysis:**

The gap between epsilon (~7) and CVaR (~250) is ~35×. The math: with ~150-step episodes and gamma=0.99, a constant per-step c_t of ~3.2 produces a discounted episode cost of ~250. Epsilon=6.987 was calibrated from expert rollouts where safety violations were rare. A random untrained policy violates safety every step. The Lagrangian is designed for policies near the constraint boundary, not 35× over it. Lambda grows linearly as `eta_lambda * (CVaR - epsilon) ≈ 0.01 * 245 ≈ 2.45/update`, reaching 24,500 by update 10,000 with no mechanism to stop it.

Additionally, at lambda=1182 with c_t≈3.2/step, the Lagrangian penalty dominates effective_reward by ~3800× relative to the DPMORL scalar R_t (which is sub-1.0 scale). PPO is receiving a nearly pure safety signal with negligible style gradient — yet CVaR is not falling. This suggests the PPO update is not successfully propagating the safety penalty into better behavior, possibly because the gradient magnitude is so large it destabilizes learning.

**Suspected secondary cause — epsilon underestimated:** During `compute_hyperparams.py`, safety costs were computed using `_EmptyMapProxy` (no map data) and `_DetectionProxy` (expert tracked objects). In the live training environment, CaRL's `EnvironmentWrapper` provides real map and detection data, potentially generating higher indicator costs (TTC, THW) every step than were present in the expert calibration. This would make epsilon systematically too low for the training distribution.

**Approaches (ranked by recommendation):**

- **(a) Lower `eta_lambda` drastically** — change from 0.01 → 0.0001 in `configs/prism_default.yaml`. This slows lambda growth by 100×, giving the policy time to learn before the penalty becomes numerically overwhelming. Tradeoff: safety constraint activates much more slowly; may need more total training updates.

- **(b) Clip lambda to a maximum value** — add `lambda_max` parameter (e.g., 10.0 or 50.0) to `CVaRLagrangian` and `cvar_lagrangian.py`. This prevents divergence while still applying the constraint. Tradeoff: a hard cap is not theoretically principled for a Lagrangian; the cap value requires tuning.

- **(c) Warm-start with lambda=0** — train for N_warmup updates (e.g., 500–1000) with `lambda_k` held at 0, then reset lambda to 0 and begin the curriculum. This lets the policy learn basic driving before the constraint is introduced. The `--phase` flag already exists in `train.py` for phase A/B separation; could extend this. Tradeoff: adds a training phase, requires implementation.

- **(d) Investigate epsilon underestimation** — log per-step c_t means during a training rollout and compare to what `compute_hyperparams.py` saw. If training c_t is systematically higher than calibration c_t (due to real vs. proxy map/detection data), re-calibrate epsilon by running `collect_expert_rollouts` with real CaRL env data (requires cache to exist first). Tradeoff: complex; requires rebuilding hyperparams after cache is built.

- **(e) Scale down safety cost weights** — divide all `OUTCOME_WEIGHTS` in `safety_cost.py` by a constant (e.g., 10) to bring episode costs to ~25 instead of ~250, closer to epsilon. Tradeoff: changes the meaning of the safety cost signal and invalidates the current hyperparams.json; requires rerunning `compute_hyperparams.py`.

**Recommendation:** Try (a) first — one-line config change, no code changes, fast to iterate. If CVaR still doesn't fall by update 500 with eta_lambda=0.0001, move to (b) combined with (a) to cap lambda at 10–50 while learning proceeds. Investigate (d) in parallel to understand if epsilon is the root mismatch.

---

## Constraints & Gotchas

### lab.env inline comments corrupt Make variable values
GNU Make's `-include lab.env` preserves trailing whitespace before `#` in assignments. `export VAR=value   # comment` → Make sets VAR to `value   `. This is invisible to `source lab.env` + bash inspection, and the trailing spaces survive into Python's `os.environ`, causing `os.path.isdir` failures. All comments must be on their own line above the assignment.

### CaRL reads `NUPLAN_MAPS_ROOT` (with S), not `NUPLAN_MAP_ROOT`
`carl_nuplan/planning/gym/cache/gym_scenario.py` line 21: `NUPLAN_MAPS_ROOT = os.getenv("NUPLAN_MAPS_ROOT")`. This is evaluated at module import time. If unset, every scenario load raises `TypeError: stat: path should be string … not NoneType`.

### System libstdc++ on lab machine lacks CXXABI_1.3.15
Conda's ICU library (`libicui18n.so.78`), pulled in by conda's sqlite3, requires `CXXABI_1.3.15`. The lab machine's system `/lib/x86_64-linux-gnu/libstdc++.so.6` only has up to CXXABI_1.3.14. Fix: `LD_PRELOAD` the conda env's `libstdc++.so.6` in `lab.env`. Previously, `opencv-python<5.0` was pinned in `environment.yml` for the same underlying reason (5.x also requires CXXABI_1.3.15).

### `gym==0.26.2` required (legacy gym, not gymnasium)
CaRL's `ppo_model.py` does `import gym` — the old OpenAI gym, not the `gymnasium` fork. Must be pinned to `gym==0.26.2` (last stable release before the fork). Now in `environment.yml`. Not caught by `make check` because the import only triggers when `_build_agent()` runs inside `train.py`, not at module load time.

### `safety_cost` is popped from info in PRISMEnv.step()
`PRISMEnv.step()` (`prism/env/nuplan_env.py:442`) pops `"safety_cost"` from the info dict, then re-adds it. The trainer reads `info.get("safety_cost", 0.0)` per step to accumulate episode costs for CVaR. If the re-add is ever removed, the CVaR buffer silently fills with zeros and lambda never moves — no error is raised.

### Lagrangian state is in a separate `.json`, not the `.pth` checkpoint
`policy_{k}_model_{tag}.pth` contains only the agent weights. `policy_{k}_lagrangian_{tag}.json` contains the CVaR buffer (`costs` key) and lambda history. To inspect the cost buffer: `cat policy_0_lagrangian_000000499.json | python -c "import json,sys; d=json.load(sys.stdin); print(d['costs'][:10])"`.

### Train-mini logs only every 100 updates
`dpmorl_trainer.py:218`: `if (update + 1) % 100 == 0`. With 512 steps/update and slow nuPlan stepping, the first log line appears after many minutes. No output ≠ hung. Verify with `nvidia-smi` or `top -p $(pgrep -f train.py)`.

### nuPlan map utils RuntimeWarning: invalid value encountered in cast
`nuplan/common/maps/nuplan_map/utils.py:413` — NaN→int cast on optional map attributes in the mini dataset. Fires intermittently during cache build and scenario loading. Gracefully degrades (drops bad map elements). Safe to ignore; not a PRISM bug.

### NuPlanScenarioBuilder requires `sensor_root`
Constructor requires `sensor_root` as a positional/keyword arg even if sensors are unused. Pass `sensor_root=nuplan_data_root` as a dummy.

### ScenarioFilter kwargs must exactly match build_cache.py
See `scripts/build_cache.py` lines 95–110 for the working call signature. Extra or missing kwargs cause TypeError.

### angular_velocity not stored in nuPlan mini DB
`DynamicCarState.angular_velocity` returns 0.0 for all ego states. Detect with `np.max(np.abs(ang_vel)) < 1e-5`; fall back to finite-differenced heading with Savitzky-Golay smoothing.

### epsilon_curve is monotone INCREASING (not decreasing)
CVaR_alpha increases with alpha. The check script verifies increasing order. This was previously backwards and caused a false FAIL.

### compute_hyperparams and build_cache are independent
Neither is a prerequisite for the other. Both feed into `train.py` and can run in parallel or in either order.

---

## References

- nuplan-devkit 1.2.2: https://github.com/motional/nuplan-devkit
- CaRL (installed at `../nuPlan/`): https://github.com/autonomousvision/CaRL
- DPMORL (NeurIPS 2023): https://github.com/zpschang/DPMORL
- Paper draft: `docs/prism_paper_v2.tex`
- Lab machine paths: `lab.env` (gitignored)
