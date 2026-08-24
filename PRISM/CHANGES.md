# PRISM Change Log

Running record of substantive changes to the PRISM codebase, kept alongside
`CLAUDE.md` (architecture/spec) and `handoff.md` (session-by-session
debugging notes). Add a new dated entry per change; do not rewrite history.

---

## 2026-08-23 — CVaR v2: state-augmented cost + dual critic (replaces v1's REINFORCE penalty)

### Motivation

Code review of the 2026-08-22 entry below (`cvar-penalty` branch) found real bugs in how
the unconstrained penalty was applied, not just in the formulation:

1. **Envelope-theorem violation.** The Rockafellar-Uryasev formula
   `capped_costs = var_alpha/n + excess/((1-alpha)*n)` is a valid decomposition of the CVaR
   *value* (`sum(capped_costs) == cvar`), but v1 reused it directly as a per-episode
   REINFORCE weight on `log_prob`. The correct policy gradient of CVaR (holding VaR fixed via
   the envelope theorem) only involves the excess term — the additive `var_alpha/n` constant
   has zero true gradient contribution, but as a REINFORCE weight it injected a uniform,
   non-discriminating pressure on every episode's log-prob regardless of that episode's actual
   cost.
2. **No importance-ratio correction across PPO epochs.** The reward term explicitly forms and
   clips `ratio = (log_prob - old_log_prob).exp()` for exactly this reason; the REINFORCE cost
   term used raw `log_prob` directly, becoming increasingly off-policy-biased across
   `update_epochs=4` passes over the same batch with no correction.
3. **Single `beta` uncalibratable across backbones.** Diagonal-Gaussian `log_prob` magnitude
   scales ~linearly with action dimensionality (2D for `carl_ppo` vs. 128D for `alpamayo`, a
   ~64x gap at the configured `init_log_std`), so a `beta` tuned on one backend was wrong for
   the other by construction.

v2 removes the REINFORCE mechanism entirely. See `PRISM/CLAUDE.md`'s "CVaR Safety Penalty"
section for the resulting math; this entry covers what changed and why.

### New formulation

State-augment with cumulative safety cost `e_t` (mirroring DPMORL's existing cumulative-reward
`z_t`), train a **separate cost critic** `V^C(s, e_t)` via GAE on a dense per-timestep cost
signal, and combine `A_total = A_reward - beta * A_cost` through PPO's *existing* clipped-ratio
objective — reward and cost now share one ratio, one clip, one set of epochs. No REINFORCE, no
raw `log_prob` weighting, so none of the three bugs above can recur by construction.

**Dense cost signal — reused, not imported.** The original ask cited Muni et al. 2026
(arXiv:2602.03778) for the telescoping trick. I fetched the actual paper (via WebFetch of the
arXiv HTML rendering) and read their real recursion: `z' = (r+z)/gamma` (note the *division* by
gamma every step — a compounding rescale, not PRISM's `e_{t+1} = e_t + gamma^t*c_t` additive
convention) and dense reward `r~ = z_- - (r+z)_-`. A literal transcription of the originally
proposed formula (`c~_t = (nu-e_t)^+ - (nu-e_{t+1})^+`) does not match this recursion. I verified
by direct summation that it telescopes to `min(C^pi, nu)` — constant once an episode exceeds
VaR, meaning the signal stops discriminating "just over budget" from "catastrophic," backwards
for a CVaR objective.

Instead of trying to force-fit Muni et al.'s differently-scaled recursion, v2 reuses PRISM's own
already-implemented, already-correct telescoping mechanism — `R_t = gamma^{-t} * [f(z_{t+1}) -
f(z_t)]` in `prism/env/nuplan_env.py`, which telescopes exactly because the `gamma^{-t}` prefactor
cancels GAE's `gamma^l` discounting — applied to a **fixed** smooth hinge instead of a learned
reward utility:

```
g_nu(e) = tau * softplus((e - nu) / tau)
c~_t = gamma^{-t} * [g_nu(e_{t+1}) - g_nu(e_t)]
```

This telescopes correctly by the identical proof, is dense everywhere (`g_nu' = sigmoid((e-nu)/tau)
> 0`), and needed zero new unverified math — verified numerically in
`tests/test_rewards.py::TestCVaR::test_dense_cost_signal_telescopes`.

**Pre-existing gap fixed as a prerequisite.** Alpamayo's actor never actually received `z_t`
before this change: `AlpamayoInstructionBuilder` builds an instruction string every step, but
`AlpamayoAdapter._run_backbone()` had `input_ids`/`attention_mask` commented out with a
`# TODO: tokenise instruction and pass to backbone` — the string was built and discarded, despite
the module docstring marking "Problem 3 — z_t injection" as done. `e_t` conditioning needed the
same pathway, so this was fixed once for both: `_load_backbone()` now loads
`AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)`, and `_run_backbone()`
tokenises the instruction and passes it through. **Flagged as the highest-uncertainty step in
this change**: Qwen-VL-family models (this backbone's language model) commonly require
`input_ids` to already contain interleaved image-placeholder tokens matched against
`pixel_values` via a processor/chat-template, not independent kwargs — this cannot be fully
verified without the real model loaded. Do one isolated forward-pass smoke test on the lab
machine before trusting it in the RL rollout loop.

### Files changed

| File | Change |
|---|---|
| `prism/env/nuplan_env.py` | Added `e_t` tracking (`self._et`), identical recursion/timing to `z_t`. Exposed as a new obs key `"cumulative_cost"` (raw, unnormalised — must stay in the same units as `nu`) and `info["cumulative_cost"]`. Confirmed via reading `carl_nuplan`'s `EnvironmentWrapper` that nothing validates obs dicts against their declared `gym.spaces.Dict`, so adding a new key needed no changes to CaRL's or Alpamayo's observation builders. |
| `prism/models/base.py` | `PolicyOutput` gained a `cost_value: torch.Tensor` field. No `forward()` signature change — `obs` may now contain `"cumulative_cost"`. |
| `prism/morl/cvar_penalty.py` | Removed `compute_cvar_with_return_capping` (v1 REINFORCE-era) and `EpisodeCostBuffer`'s v1 role as a REINFORCE buffer (kept unmodified — it's a plain statistics ring buffer, reused for the `nu` window). Added `g_nu()`, `dense_cost_signal()`, and `update_var()` (promoted from a private trainer method; `torch.quantile`-based, clamps `alpha <= 0.999`). `compute_empirical_cvar` kept as a pure logging diagnostic — CVaR is no longer in the loss. |
| `prism/morl/dpmorl_trainer.py` | Full rewrite of the update loop. `RolloutBuffer` gained `cost_values`, `dense_costs`. New `_compute_dense_costs()` reconstructs each local episode's `e_t` sequence post-hoc (once `nu` is known) and computes `c~_t` per step — including a still-in-progress trailing episode, since `dense_cost_signal` only needs the local `e_t`/`e_{t+1}` pair, not the episode's final outcome (unlike v1, which had to mask incomplete episodes). `_ppo_update()` now runs two parallel GAE passes (reward unchanged, cost new) and combines advantages *before* the single clipped-ratio loss. Added `cost_critic_loss` (MSE, weighted by new `cf_coef`). Logging extended per the new format (`var_nu`, `cost_critic_loss`, `total_loss`; dropped `actor_loss`). Added guards: `alpha` clamped to `<=0.999` before use, `nu` NaN/Inf caught and clamped to the last valid value, WARNING logged if all episode costs are zero for 10+ consecutive updates. |
| `prism/models/carl_ppo/adapter.py` | State augmentation applied on BOTH sides, per the spec's `s~_t = (s_t, z_t, e_t)`: (1) the actor's `features_aug` now concatenates `e_t` alongside `z_t` (`style_policy_head`'s input widened by 1) — the policy itself needs to see accumulated cost so it can act more conservatively approaching `nu`, not just have a critic judge it after the fact; (2) added `cost_value_head` (new MLP reading raw, unstyled `features` + `e_t` — no FiLM/`w_k`, matching the existing reward critic's own style-independence) for the cost critic. **Opportunistic fix**: `forward()` was calling `self.policy.get_value(obs)`, which internally recomputes `get_features(obs)` from scratch (confirmed in `ppo_model.py`) even though `features` was already computed one line above — replaced with a direct `value_head(cat([features, value_measurements]))` call, halving CaRL's per-step forward compute now that `features` has three consumers. |
| `prism/models/alpamayo/adapter.py` | Added `cost_critic` param (a second `QFormerCritic`, `style_dim=1`, reading the same already-detached `backbone_hidden_states` as the reward critic — no extra VLM forward pass). Added `set_nu()` (mirrors `set_w_k`'s once-per-update, not once-per-step, cadence). Wired the tokenizer (see above). **Found and fixed a pre-existing, unrelated bug** while touching `_run_backbone()`: `getattr(outputs, "trajectory", None) or getattr(outputs, "logits", None)` raises `RuntimeError: Boolean value of Tensor with more than one value is ambiguous` the moment `trajectory` is a real multi-element tensor — `or` on tensors doesn't do what it looks like it does. Fixed to explicit `is None` checks. Caught by the new smoke test (`or` never fired in the field because the pathway was never functionally reachable before this change). |
| `prism/models/alpamayo/instruction.py` | `AlpamayoInstructionBuilder.build()` gained `e_t`/`nu` params, extending the per-step instruction with a margin-to-`nu` sentence alongside the existing `z_t` sentence. |
| `scripts/train.py` | `_build_agent()`'s Alpamayo branch now also constructs the cost `QFormerCritic` from the new `cost_critic_*` config keys and passes it to `AlpamayoAdapter`. `run_stage2()`'s summary/logging updated to the new field names (`var_nu_history`, `cost_critic_loss_history`, `total_loss_history` — dropped `actor_loss_history`). |
| `configs/prism_default.yaml`, `configs/prism_alpamayo.yaml` | Added `tau` (hinge smoothness, default `20.0` — must be commensurate with raw episode-cost scale, ~100s, not `[0,1]`) and `cf_coef` (cost critic loss weight, default `0.5`, parallels `vf_coef`). `configs/prism_alpamayo.yaml` additionally gained a `cost_critic_*` block (own `style_dim: 1`). `beta`'s meaning changed (combined-advantage penalty weight, not a REINFORCE weight) — comments updated in both files; explicitly noted that a `beta` tuned on one backend should still be re-checked on the other despite advantage normalisation making it more portable than v1's raw-log-prob-scaled weight was. |
| `tests/test_rewards.py` | Removed the v1 return-capping tests (`test_return_capping_*`). Added tests for `update_var`, `g_nu` (non-decreasing, positive, requires `tau>0`), and `dense_cost_signal` — including a numerical telescoping-identity check against a synthetic right-skewed cost trajectory, the single most important correctness test in this change. `EpisodeCostBuffer`/`compute_episode_cost`/`compute_empirical_cvar` tests carried over unmodified. |
| `CLAUDE.md` | Rewrote the CVaR math section for the dual-critic/dense-signal formulation. Updated "Do Not Do" (added rules against reintroducing REINFORCE or the sparse hinge difference; clarified that "no timestep-level CVaR" refers to the VaR/CVaR *statistic*, not the new dense per-timestep *signal*, which is a different thing). Fixed three v1-era doc-drift spots a code-review pass found still describing the removed `lambda_k`/`epsilon_curve` design (repo structure listing, "own lambda_k" sentence). |

### Design notes (decisions not fully specified by the request)

1. **`e_t` via a new obs-dict key, not an explicit `forward()` parameter.** My first draft
   proposed threading `e_t` as an explicit new `forward(obs, actions, e_t)` argument, reasoning
   that both backbones declare a strict `gym.spaces.Dict` observation space. A Plan-agent review
   (reading `carl_nuplan`'s `EnvironmentWrapper` directly) found nothing anywhere validates obs
   dicts against that declared schema — `PRISMEnv` already exploits this today by repurposing
   `value_measurements` for `z_t`. Putting `e_t` in the obs dict instead means `PRISMEnv`
   auto-manages it exactly like `z_t` (reset to 0 at episode start, updated every step), and the
   trainer's tensor-stacking code (`b_obs = {k: torch.stack(...) for k in buffer.obs[0].keys()}`)
   is already fully generic over dict keys — zero new plumbing needed in the rollout loop. The
   explicit-param approach would have required threading `e_t` by hand through `info` into
   trainer-held state and back into three separate `forward()` call sites, recreating exactly the
   kind of fragile "pop from info, must remember to re-add" pattern `handoff.md` already
   documents as a past bug source for `safety_cost`.
2. **Advantage normalization: separately, then combined.** Neither the request nor prior code had
   a precedent for combining two advantage streams. Chose to normalize `A_reward` and `A_cost`
   independently (same per-minibatch mean/std normalization already applied to the single
   reward advantage) before combining into `A_total = A_r_norm - beta*A_c_norm`. This makes
   `beta` comparable in principle across backbones regardless of each one's raw advantage scale,
   directly targeting the v1 cross-backbone calibration bug — an alternative (normalize once,
   after combining) would leave `beta`'s effective strength dependent on the ratio of the two
   raw scales, reintroducing a version of the same problem. Also confirmed *per-minibatch*
   (not once per full rollout batch) during a later spec review that flagged the distinction
   explicitly — kept per-minibatch, matching the pre-existing reward-side convention.
3. **`reward_loss`/`cost_penalty` diagnostics are unclipped surrogates, not exact loss
   decomposition.** The actual backprop loss clips the *combined* advantage once
   (`max(-A_total*ratio, -A_total*ratio_clip)`), which is nonlinear, so it cannot be exactly
   split into separate reward/cost contributions after the fact. The logged `reward_loss` /
   `cost_penalty` fields report the unclipped surrogates (`-A_r*ratio` and `beta*A_c*ratio`
   respectively) purely as monitoring diagnostics for `beta` calibration (per the "if reward
   stalls while CVaR drops, beta may be too large" guidance) — they will not sum exactly to
   `total_loss`, and that's expected, not a bug.
4. **`nu` estimated from the rolling `EpisodeCostBuffer`, threaded to the Alpamayo actor via
   `set_nu()`.** `nu` is computed once per training update, after that update's rollout, from the
   buffer (not the ~3-episode local batch alone — same variance-reduction reasoning as v1).
   Since it's needed by the *next* update's rollout for the Alpamayo instruction text, `set_nu()`
   is called once per update immediately after computing `nu`, mirroring `set_w_k`'s cadence
   exactly. `dense_cost_signal` itself reads `nu` directly from the trainer, independent of
   `set_nu()` — that method exists purely for the text-conditioning side channel.
5. **Cost critic weights fully separate from the reward critic, for both backbones.** Matches
   `PRISMCriticBase`'s existing gradient-isolation contract (documented for the Alpamayo critic)
   and avoids the reward/safety objectives interfering through a shared trunk.
6. **State augmentation reaches the actor, not just the cost critic, on both backbones —
   via different mechanisms appropriate to each architecture.** The spec's `s~_t = (s_t, z_t,
   e_t)` means the POLICY needs `e_t`, not only a critic judging it after the fact. For CaRL,
   `e_t` is concatenated directly into the actor's `features_aug` (a simple tensor op, mirroring
   how `z_t` was already added there). For Alpamayo, there is no equivalent raw-vector input to
   concatenate into — the frozen VLM's only input surface is camera pixels and (now-functional,
   see above) text — so `e_t` reaches the actor via the same instruction-text mechanism `z_t`
   already uses (`AlpamayoInstructionBuilder`, tokenised and passed as `input_ids`). Caught in
   review: an earlier pass added `e_t` to the CaRL cost critic but forgot to also add it to the
   actor's `features_aug`, satisfying only half of the state-augmentation requirement for that
   backbone. Fixed before commit; both backbones now state-augment both the actor and the critic.

### Removed hyperparameters / config keys

None removed from v1's already-reduced set — v2 does not reintroduce anything v1 removed.

### Added hyperparameters / config keys

`tau` (dense cost hinge smoothness), `cf_coef` (cost critic loss weight), and (Alpamayo only)
`cost_critic_n_queries`, `cost_critic_query_dim`, `cost_critic_n_heads`, `cost_critic_style_dim`,
`cost_critic_value_hidden_dims`.

### Kept unchanged

Reward critic and its training, FiLM conditioning for `w_k`, preference-vector sampling, the
two-tier safety cost function itself, the alpha curriculum shape, `cvar_buffer_size`'s role,
PPO's core clipped-ratio mechanics for the reward side.

### Post-implementation verification pass (before commit)

A written spec-vs-implementation review against this entry's own design surfaced two more
corrections, applied before anything was pushed:

7. **Runtime sanity checks (A1, A2) added to `dpmorl_trainer.py`.** Section 11 of the review spec
   called for a per-episode runtime telescoping check and NaN/Inf detection on `c~_t`, `A_t^C`,
   and `total_loss` — neither existed; only `nu`'s NaN guard did, and it silently recovered rather
   than halting. Added: **A1**, a WARNING-only check after each *completed* local episode (the
   still-in-progress trailing episode is skipped, since its cost isn't final) verifying
   `|sum_t gamma^t c~_t - (g_nu(C) - g_nu(0))| / max(|g_nu(C)|, 1) < 1e-4` — a correctness monitor
   for the dense-signal computation itself, not a halt condition. **A2**, NaN/Inf checks on `c~_t`
   (in `_compute_dense_costs`), `A_t^C` (after the cost GAE pass), and `total_loss` (per
   minibatch, where a bad minibatch also skips its own `optimizer.step()` to avoid corrupting
   weights with a NaN gradient). A2 tracks a consecutive-bad-*update* streak (not consecutive
   bad minibatches, which conflates granularities) and raises `RuntimeError` after 3 consecutive
   updates contain any NaN/Inf occurrence — deliberately more tolerant than halting on the first
   occurrence, so a single transient issue logs loudly without necessarily ending a long run.
   Both were verified positively (not just "silent on clean data"): a targeted test monkeypatched
   `dense_cost_signal` to break the telescoping identity and confirmed A1 fires, and separately
   to always return NaN and confirmed A2 halts at exactly 3 consecutive updates, no earlier.
8. **CaRL actor was missing `e_t` in its state augmentation** — caught separately, see design
   note 6 above.

### Outstanding — not yet done in this session

- [ ] Real training smoke test (~500 updates) on the lab machine, extended from v1's checklist to
      also confirm `cost_critic_loss` decreases and `var_nu` trends downward. **Not run** — no
      `hyperparams.json` or nuPlan dataset in this checkout, same constraint as v1.
- [ ] The Alpamayo tokenizer/backbone forward-signature verification flagged above — do the
      isolated smoke test with the real model before the RL loop depends on it.
- [ ] `beta` (and now `tau`) tuning from real early-training magnitudes, per backbone.
- [x] Advantage normalization: confirmed as per-minibatch (matches the pre-existing reward-only
      PPO code's convention, now applied identically to both the reward and cost streams). Raised
      during spec review as a discrepancy against a stricter "normalize once per batch, before
      the PPO epochs" reading — explicitly decided to keep per-minibatch. No code change made.
- [x] Structural verification done in this session without the real dataset/model:
      `tests/test_rewards.py` (44/44 passing, including the new telescoping-identity test), a
      full synthetic rollout → dense-cost → dual-GAE → combined-loss → backward/optimizer-step
      pass against a fake CaRL policy + fake env (multiple updates, finite losses throughout,
      real gradients), a fake-backbone/fake-tokenizer pass confirming `input_ids`/
      `attention_mask` actually reach the Alpamayo backbone call and the cost critic/instruction
      extensions work end-to-end, and positive verification that A1/A2 fire correctly when
      deliberately triggered (not just silent on clean data).

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
