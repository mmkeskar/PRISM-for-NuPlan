# PRISM Session Handoff

**Date:** 2026-07-08

**Session type:** Development / debugging

**Topic:** Comprehensive reward pipeline audit and fix: resolved 13 code-vs-paper discrepancies across the safety cost, reward functions, epsilon calibration, training loop, and Lagrangian control. All fixes implemented but not yet tested or pushed to GitHub.

---

## Current Project State

**Current milestone:** Get `make hyperparams-mini && make train-mini` running cleanly end-to-end with a converging Lagrangian.

- `make setup` — complete and verified
- `make check` — passing for all modules including `prism.env.nuplan_env`
- `make cache-mini` — **complete**: 500/500 scenarios cached to `/data/prism_mini_cache`
- `make hyperparams-mini` — previously run with 200 rollouts; **must be re-run** after code changes (delete `hyperparams.json` first — calibration logic has changed substantially)
- `make check-hyperparams` — will need re-run; phi=0.42 was a WARN previously, see Open Questions
- `make train-mini` — **not yet re-run** after this session's fixes; previous run had lambda divergence (reached 1182 at update 500). Root causes now addressed in code. Needs retesting.

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

#### `prism/env/safety_cost.py` — complete rewrite (×2 across session)

Fixes for bugs 2, 4, 5, 7, 8, 9 from the audit, plus directional blind spot and combined must-stop indicator:

**Indicator severity (Bug 2):** `_cap_indicator(name)` now takes `severity: float`. Per-step cost = `w_j * severity`. Graded severities: `s_ttc = (thresh-TTC)/thresh`, `s_thw = (thresh-THW)/thresh`, `s_spd = (v_ego-v_limit)/v_limit`. Previously all indicators fired at flat weight regardless of how far inside the threshold.

**Blind spot directionality (Bug 4 + refinement):** `_blind_spot_sides_occupied()` (replaces `_blind_spot_occupied()`) returns `(left_occ, right_occ)` using the signed lateral offset `d_lat_signed = -sin_h*dx + cos_h*dy`. In `compute()`, the indicator only fires when ego is *converging* with the occupied side: `left_occ and v_lat_ego > lc_thresh` OR `right_occ and v_lat_ego < -lc_thresh`. Moving away from an occupied blind spot does not fire.

**Speed violation (Bug 5):** Removed `1 + delta_road` tolerance from speed indicator. Exceeding the legal limit is a violation regardless of road type.

**Stop sign violation — state machine (Bug 7, redesigned):** Old approach was point-in-polygon at time of crossing (unreliable at 10 Hz for thin stripe polygons, vulnerable to speed-blasting). New approach is a CARLA Leaderboard 2.0-style state machine inside `_check_stop_sign_violation()` (instance method, not static):
- **Encounter:** sign centroid enters approach window `0 < d_lon < approach_m` (default 5m) — start tracking
- **Track:** record whether `v_ego < stop_threshold_ms (0.5 m/s)` at any point while sign is ahead
- **Verdict:** when ego passes the sign (`d_lon ≤ 0`), if no stop was recorded → violation, clean up state
- Stopping after the line does not satisfy the check (approach window tracking ends at the crossing event)
- State machine lives in `self._stop_sign_state: Dict[str, Dict]`, cleared on `reset()`
- Stop sign state (`_stop_sign_tokens_fired` set) is no longer tracked externally in `PRISMRewardBuilder`

**Collision weights (Bug 8):** Outcome block now distinguishes `had_vru_collision` (weight 100), `had_vehicle_collision` (weight 80), `had_object_collision` (weight 40) via `elif` chain. Previously all collisions used a single weight.

**Red light indicator → must_stop_ahead (Bug 9 + stop sign combined):** `_red_light_ahead()` renamed to `_must_stop_ahead()`. Now checks both red traffic light connectors (from `map_cache.lane_connectors` + `TrafficLightStatusType.RED`) and stop signs (from `map_cache.stop_lines` with `StopLineType.STOP_SIGN`). Both use `_estimate_time_to_arrival()` with the same `arrival_horizon_s` threshold. `_MAX_LOOKAHEAD_M = 80.0` proximity shortcut skips polygon distance computation for distant targets. `SafetyCostComponents.red_light_ahead_active` renamed to `must_stop_ahead_active`.

**compute() signature change:** Removed `had_stop_sign_violation` parameter — violation is now detected entirely inside `SafetyCostBuilder` via `_check_stop_sign_violation()`.

#### `prism/env/nuplan_env.py`

**SG smoothing for live training (Bug 3):** `PRISMRewardBuilder` now has `_accel_lon_buf` and `_accel_lat_buf` deques (maxlen=7). At each step, acceleration is appended, Savitzky-Golay filter (window up to 7, polyorder=2) applied to the buffer, and jerk computed from the smoothed values. Consistent with `compute_hyperparams.py` calibration. Buffer cleared on `reset()`. Falls back to raw acceleration when buffer has fewer than 5 samples.

**Collision classification (Bug 8):** After `calculate_non_stationary_collisions`, collided tokens are classified by `tracked_object_type` into `had_vru_collision` (`PEDESTRIAN`, `BICYCLE`), `had_vehicle_collision` (`VEHICLE`), or `had_object_collision` (other). Previously all collisions set `had_collision=True` and the caller couldn't distinguish types.

**terminate_on_failure propagation (Bug B):** `make_prism_env()` now passes `terminate_on_collision=terminate_on_failure` and `terminate_on_off_road=terminate_on_failure` to `PRISMRewardBuilder`. Previously `PRISMRewardBuilder` hardcoded `terminate_on_collision=True, terminate_on_off_road=True` regardless of the `terminate_on_failure` flag passed to the env.

**v_lat_ego extraction:** `rear_axle_velocity_2d` projected into ego frame (`-sin_h*vx + cos_h*vy`) and passed to `SafetyCostBuilder.compute()` for the directional blind spot check.

**Removed external stop sign state:** `_stop_sign_tokens_fired` set and `SafetyCostBuilder._stop_sign_violated()` call removed from `PRISMRewardBuilder`. Stop sign violation is now internal to `SafetyCostBuilder`.

**Updated `compute()` call:** `had_stop_sign_violation` argument removed (no longer accepted). Collision args split into `had_vru_collision`, `had_vehicle_collision`, `had_object_collision`.

#### `prism/env/rewards.py`

**sigma_d key bug:** `compute_style_rewards()` was reading `hp.get("floor_values", {}).get("delta_d", 0.2)` as the `sigma_d` argument to `compute_lateral_discipline()`. The key is wrong — `sigma_d` is a reward scaling parameter stored at `reward_scaling["sigma_d"]`, not `floor_values["delta_d"]`. Fixed to `scaling.get("sigma_d", 0.2)`.

#### `prism/morl/cvar_lagrangian.py`

**Lambda cap (Bug C):** Added `lambda_max: float = float("inf")` parameter to `__init__`. `update_lambda()` now applies `min(self._lambda_max, max(0.0, ...))`. Included in `state_dict()` / `from_state_dict()` for checkpoint consistency.

#### `prism/morl/dpmorl_trainer.py`

**Lambda warmup (Bug C):** `_lambda_warmup = cfg.get("lambda_warmup_updates", 0)`. During the first `lambda_warmup_updates` PPO updates, `set_lambda(0.0)` is called and `update_lambda()` is skipped. Episode costs are still accumulated in the CVaR buffer during warmup, so the Lagrangian starts from a realistic distribution when it activates. Improved logging reports `[warmup]` tag and uses `summary` lists instead of undefined `cvar_hat`/`new_lambda` variables from outside the warmup branch.

**Lambda max passed through:** `CVaRLagrangian` now constructed with `lambda_max=cfg.get("lambda_max", float("inf"))`.

#### `compute_hyperparams.py`

**Full safety signal in epsilon calibration (Bug A):** `_extract_episode()` previously only computed TTC and THW costs inline (two channels out of five). Now constructs `SafetyCostBuilder(bootstrap_hp)` per episode and calls `compute()` at each step with all five indicator channels active. Bootstrap indicator weights (computed from fallback lead times) are populated in `bootstrap_hp["indicator_weights"]` / `bootstrap_hp["indicator_caps"]` before passing to `SafetyCostBuilder`. All outcome flags passed as `False` (expert data doesn't crash). `v_lat_ego=float(v_lat_raw[i])` passed for the directional blind spot check. This is the primary fix for the epsilon underestimation bug.

**Discounted z_t normalisation (Bug 6):** `compute_zt_normalisation()` previously used `ep["style_r"].sum(axis=0)` (undiscounted sum). Now uses `(gamma**t * r).sum(axis=0)` — discounted cumulative sum matching how `PRISMEnv.step()` accumulates `z_{t+1} = z_t + gamma^t * r_t`. Added `gamma: float = 0.99` parameter; call in `main()` passes `gamma=args.gamma`.

**`red_light_arrival_horizon_s` added to output:** Safety thresholds in the assembled `hyperparams` dict now include `"red_light_arrival_horizon_s": 4.0`.

#### `configs/prism_default.yaml`

Added under CVaR Lagrangian block:
```yaml
lambda_warmup_updates: 500      # hold lambda=0 for first N updates
lambda_max: 50.0                # cap on Lagrange multiplier
```

### Key reference files (unmodified)
- `scripts/build_cache.py` — authoritative example for `NuPlanScenarioBuilder` and `ScenarioFilter` call signatures
- `../nuPlan/carl_nuplan/planning/gym/environment/reward_builder/default_reward_builder.py` — CaRL's `_calculate_red_light()` (in-polygon check + expert infraction allowlist) and `_calculate_stop_sign()` (stub, not implemented)
- `CARLA/team_code/reward/criteria/run_stop_sign.py` — CARLA Leaderboard 2.0 stop sign state machine (inspiration for our positive-check approach)

---

## Decisions & Reasoning

### 1. Expert log replay instead of live IDM simulation
*(carried forward — still the active decision)*

**Decision:** `collect_expert_rollouts` replays nuPlan's stored ground-truth ego trajectories rather than running the IDM planner in closed-loop simulation.

**Reason:** `compute_hyperparams.py` must run before the scenario cache exists. Live IDM simulation requires `SimulationRunner` + Hydra. Expert trajectories are equivalent for calibration purposes.

*Alternatives rejected:* Live IDM (requires Hydra, not cache-free), `PRISMEnv` wrapper (requires cache to exist first).

### 2. Savitzky-Golay smoothing before differentiation
*(carried forward — smoothing is applied in both calibration and live training)*

**Decision:** Apply `scipy.signal.savgol_filter(window=7, polyorder=2)` to acceleration and heading arrays before computing jerk and heading rate.

**Reason:** `rear_axle_acceleration_2d` is already a finite difference of velocity; differencing again for jerk amplifies noise by 100×. `angular_velocity` is absent from the mini DB, so heading rate requires an additional finite diff. Smoothing is applied both in `compute_hyperparams.py` (full episode batch) and in `PRISMRewardBuilder` (rolling deque of 7 steps) so calibration and training see the same jerk distribution.

### 3. No inline comments after assignment values in lab.env
*(carried forward)*

**Decision:** All comments in `lab.env` must be on their own line above the assignment, never trailing after the value.

**Reason:** The Makefile's `-include lab.env` parses the file as GNU Make syntax. Make preserves whitespace before `#` in assignments, so `VAR=value   # comment` sets Make's variable to `value   `. This is invisible to `source lab.env` + bash inspection. All comments must be on their own line.

### 4. Re-expose `safety_cost` in info after popping
*(carried forward)*

**Decision:** `PRISMEnv.step()` pops `"safety_cost"` from info internally, then immediately re-adds it so the trainer can accumulate per-step costs.

### 5. Lambda control: warmup + cap, eta_lambda unchanged

**Decision:** Three independent levers in config: `lambda_warmup_updates: 500`, `lambda_max: 50.0`, `eta_lambda: 0.01` (unchanged). All are config-driven and independently tunable.

**Reason:** The 35× gap between epsilon (~7) and CVaR (~250) had multiple root causes. Fixing epsilon (Bug A) is the primary fix, but the training loop needs protection while the policy is still random at episode start. Warmup gives the policy 500 updates of style-reward-only learning before the constraint activates. The cap prevents numerical PPO collapse if epsilon is still underestimated after rerunning hyperparams. `eta_lambda` is left at 0.01 until we see the new epsilon values — lowering it is a secondary lever if convergence is still slow.

*Alternatives rejected:*
- Lower `eta_lambda` only — delays divergence but doesn't address root cause if epsilon is still wrong
- Constraint relaxation / epsilon scaling — adds a tuning parameter without principled grounding; redundant once the alpha curriculum co-schedules epsilon correctly
- Entropy bonus (`ent_coef`) — helps exploration but doesn't prevent value function collapse at high lambda

### 6. Stop sign violation: positive-check state machine over point-in-polygon

**Decision:** CARLA Leaderboard 2.0-style state machine tracking minimum approach speed before the crossing event, rather than a point-in-polygon check at the stop line.

**Reason:** Point-in-polygon is unreliable at 10 Hz. At 14 m/s, ego moves 1.4m per step — thin stop line polygons can be skipped entirely. The positive check is robust to frame rate and exact line geometry: it asks "did the ego slow to near-zero before passing the sign?" rather than "did the ego's centre intersect the polygon while moving fast?"

**On approach window (5m):** The approach window is a detection parameter, not a reaction-time parameter. The indicator signal (`_must_stop_ahead`) handles shaping; the violation checker just audits the outcome. A 5m window is just wide enough to catch the ego entering the deceleration zone before the sign but tight enough not to credit a stop for unrelated reasons.

*Alternatives rejected:*
- Point-in-polygon with 0.5m buffer — still misses fast crossings; fires incorrectly for stopped vehicles abutting the line from a previous step
- 25m window — too large; credits a stop 25m before the sign as compliance

### 7. Blind spot: directional gate only (converging, not any lane change)

**Decision:** Blind spot indicator fires only when ego is moving *toward* the occupied side (`left_occ and v_lat_ego > lc_thresh` OR `right_occ and v_lat_ego < -lc_thresh`). Moving away from an occupied blind spot does not fire.

**Reason:** Penalizing a correct avoidance maneuver (moving away from an occupied blind spot) produces false gradient signal. The agent is doing the right thing. Directional gating gives a precise cost (converging = bad, diverging = neutral) and is strictly better for learning than the symmetric version.

### 8. Stop sign + red light combined in one indicator channel (`red_light`)

**Decision:** `_must_stop_ahead` checks both red traffic lights and stop signs under a single indicator key `"red_light"` in `hyperparams.json`. Both controls require a full stop; the temporal arrival check is identical for both.

**Reason:** Keeping them separate would require a new `"stop_sign_ahead"` key in `INDICATOR_OUTCOME_MAP`, separate calibration, and separate cap tracking — all for a signal that is functionally identical (arrive-within-horizon → must stop). The outcome weights differ (80 vs 70) but the calibrated indicator weights from fallback lead times are similar enough that one channel serves both.

*Alternative considered:* Separate `"stop_sign_ahead"` indicator channel — more precise but adds complexity; can be split later if ablation studies show it matters.

---

## Next Immediate Actions

- [ ] **Delete `hyperparams.json` and rerun:** `rm hyperparams.json && make hyperparams-mini` — calibration logic now includes all 5 indicator channels via `SafetyCostBuilder.compute()` rather than inline TTC/THW only. This is the primary fix for the Lagrangian divergence.
- [ ] **Check epsilon:** `make check-hyperparams` — verify `epsilon@0.95` is substantially higher than the old ~7. Expected range after fix: 50–300+ depending on how often speed/blind_spot fire on expert data. If epsilon is still < 20, investigate indicator_weights in `hyperparams.json` and whether the bootstrap SafetyCostBuilder is firing.
- [ ] **Rerun training:** `rm -rf runs/mini_test && make train-mini` — watch update 500 (end of warmup). Lambda should be 0.0 at update 500 and CVaR should be non-trivially changing. After update 500, lambda should grow slowly and either plateau (constraint satisfied) or grow to `lambda_max=50.0` (constraint violated but capped).
- [ ] **Validate stop sign state machine:** in a short diagnostic run, confirm `safety_components.stop_sign` fires at least occasionally (nuPlan mini has stop signs). If zero fires across a full episode run, check `map_cache.stop_lines` is populated in the live env (it may not be if CaRL builds the cache without `load_stop_lines=True`).
- [ ] **Push to GitHub** once training shows convergent lambda behavior on mini.
- [ ] After Lagrangian convergence is confirmed on mini: scale up to `make train` (K=5, full dataset).

---

## Open Questions

### phi = 0.42 rad/s (above expected range of 0.01–0.3)

**Question:** The lateral heading-rate scale parameter phi is 0.42 after Savitzky-Golay smoothing. Is this a calibration artifact of the mini dataset, or is 0.42 genuinely representative?

**Status:** Confirmed value after smoothing. Deferred — the rest of the pipeline is now running. phi=0.42 means sharp-turn heading rates (common in urban intersection scenarios in the mini split) sit at the 1-sigma point of `r_heading`, compressing the reward less than a calibrated value would. Training will still work; style discrimination for lateral discipline may be slightly reduced.

**Approaches:**
- (a) Widen the check-script upper bound to 0.5 and document the urban mini dataset bias — quick fix
- (b) Increase SG filter window (e.g., 11 or 15) for heading — may reduce phi further at cost of over-smoothing genuine heading changes
- (c) Use full training split for hyperparams computation — more diverse scenarios should bring phi into range, but requires full cache first

**Recommendation:** Option (a) as an immediate fix; revisit with option (c) before the final training run for the paper.

### gamma_a = 0.19 m/s² (below expected range of 0.3–3.0, WARN only)

**Question:** Is this a calibration artifact of the mini dataset's smooth scenarios, or a genuine measurement?

**Status:** Unchanged. WARN, not FAIL. ZtNormaliser EMA (beta=0.01) adapts during training.

**Approaches:**
- (a) Accept — self-corrects during training
- (b) Increase n_rollouts to 500 for more diverse acceleration distribution
- (c) Inspect r_progress distributions in tensorboard after training starts

### z_mu[1] (progress dimension) ~10× lower than other dimensions

**Question:** z_mu for progress (≈28) is far below comfort/lateral/spacing (≈290–400).

**Status:** Root cause known: `_EmptyMapProxy` forces v_des = 13.89 m/s for all calibration steps; urban mini scenarios have v_ego << v_des, collapsing r_speed ≈ 0. Now partially improved because Bug 6 (discounted z_t) is fixed — the undiscounted sum was also inflating the other three dimensions relative to progress. ZtNormaliser EMA will self-correct within a few hundred training episodes.

**Recommendation:** Accept for now. Note in paper appendix.

### Lagrangian divergence — partially addressed, not yet retested

**Question:** Will the Lagrangian converge after the epsilon calibration fix, lambda warmup, and lambda cap?

**Status:** Previous run: lambda reached 1182 at update 500 with CVaR stuck at 200–300 and epsilon ~7 (35× gap). Root causes addressed this session:
- Bug A fix: epsilon should now reflect all 5 indicator channels in expert data — expected to be much higher
- Lambda warmup (500 updates): policy learns basic driving before constraint activates
- Lambda cap (50.0): prevents numerical PPO collapse if epsilon is still underestimated

**What to watch in the next run:**
- Update 500 (end of warmup): lambda should be 0.0, CVaR should be within ~2-3× of new epsilon
- Updates 500–1500: lambda should grow slowly and stabilise or plateau at 50.0
- If lambda still diverges to cap immediately: epsilon is still too low — investigate `indicator_weights` in `hyperparams.json` or whether indicators are firing on expert data

---

## Constraints & Gotchas

### lab.env inline comments corrupt Make variable values
GNU Make's `-include lab.env` preserves trailing whitespace before `#` in assignments. `export VAR=value   # comment` → Make sets VAR to `value   `. This is invisible to `source lab.env` + bash inspection, and the trailing spaces survive into Python's `os.environ`, causing `os.path.isdir` failures. All comments must be on their own line above the assignment.

### CaRL reads `NUPLAN_MAPS_ROOT` (with S), not `NUPLAN_MAP_ROOT`
`carl_nuplan/planning/gym/cache/gym_scenario.py` line 21: `NUPLAN_MAPS_ROOT = os.getenv("NUPLAN_MAPS_ROOT")`. Evaluated at module import time. If unset, every scenario load raises `TypeError: stat: path should be string … not NoneType`.

### System libstdc++ on lab machine lacks CXXABI_1.3.15
Conda's ICU library requires `CXXABI_1.3.15`. Fix: `LD_PRELOAD` the conda env's `libstdc++.so.6` in `lab.env`.

### `gym==0.26.2` required (legacy gym, not gymnasium)
CaRL's `ppo_model.py` does `import gym`. Must be pinned to `gym==0.26.2`. Not caught by `make check` because the import only triggers when `_build_agent()` runs inside `train.py`.

### `safety_cost` is popped from info in PRISMEnv.step()
`PRISMEnv.step()` pops `"safety_cost"` from the info dict, then re-adds it. If the re-add is ever removed, the CVaR buffer silently fills with zeros and lambda never moves — no error is raised.

### Lagrangian state is in a separate `.json`, not the `.pth` checkpoint
`policy_{k}_model_{tag}.pth` contains only agent weights. `policy_{k}_lagrangian_{tag}.json` contains the CVaR buffer (`costs` key) and lambda. To inspect: `cat policy_0_lagrangian_000000499.json | python -c "import json,sys; d=json.load(sys.stdin); print(d['costs'][:10])"`.

### Train-mini logs only every 100 updates
`dpmorl_trainer.py`: `if (update + 1) % 100 == 0`. No output ≠ hung. Verify with `nvidia-smi` or `top -p $(pgrep -f train.py)`.

### nuPlan map utils RuntimeWarning: invalid value encountered in cast
`nuplan/common/maps/nuplan_map/utils.py:413` — NaN→int cast on optional map attributes in the mini dataset. Gracefully degrades. Safe to ignore.

### NuPlanScenarioBuilder requires `sensor_root`
Pass `sensor_root=nuplan_data_root` as a dummy even if sensors are unused.

### ScenarioFilter kwargs must exactly match build_cache.py
See `scripts/build_cache.py` lines 95–110 for the working call signature.

### angular_velocity not stored in nuPlan mini DB
`DynamicCarState.angular_velocity` returns 0.0 for all ego states. Detect with `np.max(np.abs(ang_vel)) < 1e-5`; fall back to finite-differenced heading with Savitzky-Golay smoothing.

### epsilon_curve is monotone INCREASING (not decreasing)
CVaR_alpha increases with alpha. The check script verifies increasing order.

### compute_hyperparams and build_cache are independent
Neither is a prerequisite for the other. Both feed into `train.py` and can run in parallel or in either order.

### `map_cache.stop_lines` may not be populated in the live CaRL env
CaRL's `environment_cache_manager` may not load stop lines unless the cache was built with `load_stop_lines=True`. If `getattr(map_cache, "stop_lines", {})` returns an empty dict in training, the stop sign violation checker and must_stop_ahead indicator will silently return False for stop signs. The TTC/THW/speed/blind_spot indicators are unaffected. Verify by logging `len(map_cache.stop_lines)` in one episode; if zero throughout, add `load_stop_lines=True` to the environment cache builder call.

### SafetyCostBuilder.compute() signature changed
`had_stop_sign_violation` parameter removed. Any caller (including test code) using the old signature will break with an unexpected keyword argument. Stop sign detection is now fully internal.

### `SafetyCostComponents.red_light_ahead_active` renamed to `must_stop_ahead_active`
Any code that previously logged or tested `comp.red_light_ahead_active` will need updating.

---

## References

- nuplan-devkit 1.2.2: https://github.com/motional/nuplan-devkit
- CaRL (installed at `../nuPlan/`): https://github.com/autonomousvision/CaRL
- DPMORL (NeurIPS 2023): https://github.com/zpschang/DPMORL
- CARLA RunStopSign (Leaderboard 2.0 pattern): `CARLA/team_code/reward/criteria/run_stop_sign.py`
- Paper draft: `docs/prism_paper_v2.tex`
- Lab machine paths: `lab.env` (gitignored)
