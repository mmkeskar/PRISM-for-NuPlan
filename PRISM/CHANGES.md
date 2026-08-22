# PRISM Change Log

Running record of substantive changes to the PRISM codebase, kept alongside
`CLAUDE.md` (architecture/spec) and `handoff.md` (session-by-session
debugging notes). Add a new dated entry per change; do not rewrite history.

---

## 2026-08-22 — CVaR refactor: constrained Lagrangian → unconstrained penalty with empirical CVaR + return capping

### Motivation

`handoff.md` (2026-06-30) documented a blocking failure in `make train-mini`:
the CVaR-Lagrangian dual variable `lambda_k` grew unbounded (1182 by update
500, projected ~24,500 by update 10,000) while CVaR never fell (fluctuated
191–298, no downward trend). Root cause: `epsilon` (the CVaR threshold,
calibrated from expert/IDM rollouts) was ~35x smaller than an untrained
policy's actual CVaR, so the constraint was infeasible from initialization
and the primal-dual feedback loop never found a foothold.

This entry replaces the constrained Lagrangian formulation with a fixed,
unconstrained penalty on empirical CVaR, using Rockafellar-Uryasev return
capping so the full rollout batch (not just the worst `(1-alpha)` tail)
contributes gradient signal.

**Note on the "old formulation" premise:** the request that prompted this
refactor described the prior CVaR estimator as a Gaussian closed form
(`mu_C + phi(Phi^-1(alpha))/(1-alpha) * sigma_C`). The actual code in
`prism/morl/cvar_lagrangian.py::estimate_cvar` was already an empirical
estimator (sort episode costs, average the top `(1-alpha)` tail) — no
Gaussian assumption was present in this codebase. What *was* present, and
*is* the actual subject of this refactor, is the Lagrangian dual-ascent
machinery (`lambda_k`, `eta_lambda`) and the `epsilon_curve` threshold
computed from IDM/expert-log rollouts in `compute_hyperparams.py` — both of
which map directly onto the blocking failure in `handoff.md` and are
removed below.

### New formulation

```
actor_loss = reward_loss + beta * CVaR_alpha(C^pi)
```

- No Lagrange multiplier, no dual update, no threshold.
- `beta` is a fixed hyperparameter (`configs/*.yaml`, default `1.0`).
- `alpha` still follows the existing curriculum
  (`alpha_start` → `alpha_end` over `n_curriculum_iters`).
- `CVaR_alpha` is estimated empirically and via Rockafellar-Uryasev return
  capping (`prism/morl/cvar_penalty.py`), never via a closed-form formula.

### Files changed

| File | Change |
|---|---|
| `prism/morl/cvar_lagrangian.py` | **Deleted.** Superseded by `cvar_penalty.py`. |
| `prism/morl/cvar_penalty.py` | **New.** `compute_empirical_cvar()`, `compute_cvar_with_return_capping()` (Rockafellar-Uryasev), `EpisodeCostBuffer` (rolling episode-cost window, no dual variable), `compute_episode_cost()` (unchanged, carried over). |
| `prism/curriculum/alpha_schedule.py` | `AlphaSchedule.get(n)` now returns `alpha` only (was `(alpha, epsilon)`). Constructor no longer takes `hp`. `compute_epsilon()` / epsilon lookup removed. |
| `prism/morl/dpmorl_trainer.py` | Removed `lambda`/dual-update logic. Added per-step episode-id tracking so each *local* rollout episode's return-capped cost can be broadcast to its own timesteps. Actor loss is now `reward_loss + beta * cost_penalty`, with the cost term computed via REINFORCE (`log_prob * capped_cost`, no PPO clipping — see Design Notes). New logging fields: `alpha, beta, CVaR, mu_c, sigma_c, reward_loss, cost_penalty, actor_loss` (removed `lambda, epsilon`). Checkpoint files renamed `policy_{k}_lagrangian_{tag}.json` → `policy_{k}_costbuffer_{tag}.json`. |
| `prism/env/nuplan_env.py` | `PRISMEnv` no longer takes `lambda_k` / has `set_lambda()`. `step()` returns the raw DPMORL reward `R_t` (previously `R_t - lambda_k * c_t`). `c_t` is still exposed via `info["safety_cost"]` for the trainer to use in the CVaR penalty. `make_prism_env()` factory signature updated to match. |
| `scripts/train.py` | Removed dead `CVaRLagrangian` import. `_build_env()` no longer takes `lambda_k`. `run_stage2()` logging/`training_summary.json` updated to the new summary schema (`cvar_history`, `mu_c_history`, `sigma_c_history`, `reward_loss_history`, `cost_penalty_history`, `actor_loss_history` — no `lambda_history`). |
| `scripts/evaluate.py` | Removed `lambda_k=0.0` arg. CVaR@0.95 is now reported as a diagnostic only (`compute_empirical_cvar`, plus `mu_c`/`sigma_c`) — no `epsilon`/"safe" comparison, since there is no threshold. |
| `scripts/explore_simulator.py` | Removed `lambda_k=0.0` arg and the `epsilon_curve` placeholder key from `_PLACEHOLDER_HP`. |
| `compute_hyperparams.py` | Removed `compute_cvar_epsilon_curve()` and the `epsilon_curve` key from the output JSON and CLI summary. Removed now-unused `ALPHA_VALUES` constant. Expert rollouts (`collect_expert_rollouts`) are unchanged and still feed reward scaling, lead times, and z_t normalisation. |
| `prism/utils/hyperparams.py` | `epsilon_curve` removed from `_validate()`'s required keys. `get_epsilon()` removed. |
| `scripts/check_hyperparams.py` | Removed the "CVaR epsilon curve" validation section and `epsilon_curve` from `REQUIRED`. |
| `configs/prism_default.yaml`, `configs/prism_alpamayo.yaml` | Removed `eta_lambda`. Added `beta: 1.0` (fixed CVaR penalty weight). Kept `cvar_buffer_size` (now sizes `EpisodeCostBuffer`, not a Lagrangian buffer). |
| `tests/test_rewards.py` | `default_hp` fixture no longer includes `epsilon_curve`. `TestAlphaSchedule` updated for the new `AlphaSchedule` signature/return type. `TestCVaR` rewritten against `cvar_penalty.py`: empirical CVaR tests carried over, added tests for `compute_cvar_with_return_capping` (Rockafellar-Uryasev equivalence to direct empirical CVaR, all-episodes weighting, empty-batch), and `EpisodeCostBuffer` (rolling window, checkpoint round-trip). Removed the old Lagrangian-update tests (no `lambda` to test). |
| `CLAUDE.md` | Rewrote the CVaR math section, hyperparameter-loading example, files-to-modify table, design-decision #4 (per-policy buffer/shared beta, not per-policy lambda), and "Do Not Do" list (dropped the shared-lambda rule; added rules against reintroducing a Lagrangian/threshold or a Gaussian CVaR formula). Flagged the `beta` name collision across three unrelated hyperparameters (progress-reward speed scaling, z_t EMA rate, CVaR penalty weight). |
| `README.md` | Updated the hyperparameter-computation and CaRL-training-summary prose to drop "CVaR epsilon curve" / "CVaR Lagrangian" language. |

### Design notes (decisions not fully specified by the request)

1. **Rolling episode-cost buffer (`EpisodeCostBuffer`).** The request's
   pseudocode assumes a batch of N≈100 i.i.d. episodes. In practice
   `steps_per_update=512` with `max_episode_steps=150` yields only ~3
   completed episodes per PPO update — far too few for a stable tail
   quantile at `alpha=0.95`. `VaR_alpha` (and the diagnostic `CVaR`, `mu_c`,
   `sigma_c` logged every 100 updates) are therefore estimated from a
   rolling window of the last `cvar_buffer_size` episodes (default 500,
   same knob the old `CVaRLagrangian` used). The return-capped costs
   actually used in the policy-gradient loss are still computed per update,
   local to that update's batch, using `var_alpha` from the stable buffer
   as the cap threshold — this keeps the gradient signal timely while the
   quantile estimate itself is variance-reduced.

2. **Partial trailing episodes get zero cost-gradient weight.** If an
   episode is still in progress when `steps_per_update` steps have been
   collected, its outcome (and therefore its cost) is unknown, so its
   timesteps receive `capped_cost = 0` for that update. The episode's cost
   is recorded once it completes in a later update. This mirrors how the
   pre-existing trainer already dropped/deferred partial-episode
   bookkeeping across update boundaries.

3. **Cost term uses plain REINFORCE, not the clipped PPO surrogate.** The
   request specifies `log pi(a|s,w) * capped_cost_i` directly. The reward
   term keeps PPO's clipped surrogate (unchanged, per the request). Over
   `update_epochs=4` epochs the behavior policy and current policy diverge,
   which is exactly what PPO's clipping guards against for the reward term
   — the cost term has no such guard. This is implemented as specified;
   flagged here as the first thing to inspect if the 500-update smoke test
   (checklist item below) shows unstable/oscillating `cost_penalty`.

4. **`actor_loss` in logs vs. the backward-pass loss.** Per the request,
   logged `actor_loss = reward_loss + cost_penalty`. The tensor actually
   passed to `.backward()` additionally includes the pre-existing PPO value
   loss and entropy bonus (`vf_coef * v_loss - ent_coef * ent_loss`), since
   the policy and value heads share one optimizer — this bookkeeping is
   unchanged from before the refactor and orthogonal to the CVaR penalty.

5. **numpy, not torch, for the CVaR statistics.** `compute_empirical_cvar`
   and `compute_cvar_with_return_capping` operate on numpy arrays (the
   request's pseudocode used `torch.Tensor`), matching how GAE
   advantages/returns are already computed in `dpmorl_trainer.py`. The
   capped-cost weights are non-differentiable constants from the policy's
   perspective (like a REINFORCE return-to-go), so no autograd graph is
   needed until they're multiplied by `log_prob` inside `_ppo_update`.

6. **Checkpoint filename renamed.** `policy_{k}_lagrangian_{tag}.json` →
   `policy_{k}_costbuffer_{tag}.json`. No code reads these files back for
   resume (only Phase B loads model *weights*, not this state), so the
   rename is safe, but tooling/scripts outside this repo that parsed the
   old filename pattern will need updating.

### Removed hyperparameters

`lambda_init`, `eta_lambda`, `epsilon_curve` (from `hyperparams.json`),
`AlphaSchedule`'s `hp` argument / epsilon lookup.

### Added hyperparameters

`beta` (float, default `1.0`) in `configs/prism_default.yaml` and
`configs/prism_alpamayo.yaml` — fixed CVaR penalty weight.

### Kept unchanged

Reward critic and its training, policy network architecture (FiLM
conditioning, backbone), preference-vector sampling, the two-tier safety
cost function itself (`prism/env/safety_cost.py`), the environment/nuPlan
interface (beyond dropping `lambda_k`), the alpha curriculum shape,
`cvar_buffer_size`'s role as a rolling-window knob.

### Outstanding — not yet done in this session

Per the request's implementation checklist (§8):

- [ ] Run a short training smoke test (~500 updates) to verify: CVaR is
      computed correctly, `actor_loss` has reasonable magnitude (reward and
      cost terms in a similar ballpark), no NaN/Inf, and cost trends down
      (even slowly) instead of diverging. **Not run** — no `hyperparams.json`
      or nuPlan dataset is present in this checkout; this needs to happen on
      the lab machine per `handoff.md`'s setup pipeline.
- [ ] Tune `beta` from the observed early-training magnitudes of
      `reward_loss` vs. `CVaR` once the smoke test above produces numbers
      (the request suggests starting from `beta=1.0` and adjusting so
      `beta * CVaR` is roughly the same order of magnitude as `reward_loss`).
- [x] Unit tests updated and passing (`tests/test_rewards.py`) — pure
      Python/numpy, no nuPlan/gym dependency, run with
      `python -m pytest tests/test_rewards.py -v`.
