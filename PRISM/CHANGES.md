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

---

## 2026-08-25 — Instability-analysis experiment: indicator-only cost, ablation toggles, verbose logging (branch `instability-analysis`)

### Motivation

A real `make train-mini` run on the lab GPU server (8800 updates, CaRL backbone, full
CVaR v2 pipeline from the 2026-08-23 entry above) showed sustained, unrecovered
instability: `cost_critic_loss` exploding into the tens of thousands, `var_nu` growing
~30x and never settling, `sigma_c` staying flat while `var_nu` climbed (a heavy-tail
signature). Root-cause discussion (see conversation log, not reproduced here) converged
on two leading hypotheses, ranked by evidence strength:

1. **Cost formulation** — Tier-1 outcome costs (`prism/env/safety_cost.py`) are applied
   as single-timestep lumps of 65-100 (and can stack past 200, since `wrong_direction`/
   `red_light`/`off_road` are independent `if` blocks, not mutually exclusive). The dense
   cost signal `c~_t = gamma^{-t}[g_nu(e_{t+1}) - g_nu(e_t)]` depends on the PER-STEP
   INCREMENT, not the episode total — a hand-derived toy example reproduced the exact
   >99% loss-domination signature seen in the real logs when a single large increment
   hits late in an episode (`gamma^{-t}` up to ~4.5x at t=149, squared in the critic
   loss). Tier-2 indicator costs are structurally different: `_cap_indicator()` caps each
   indicator's PER-STEP contribution at its `ind_weight` (~2-7) regardless of remaining
   cap headroom, so even worst-case simultaneous multi-indicator firing bounds the
   per-step increment an order of magnitude below a single outcome lump.
2. **`nu` as a moving target** — `nu = quantile(episode_cost_buffer, alpha)` is
   recomputed every update from BOTH a climbing `alpha` curriculum (0.20 -> 0.95 over
   5000 iterations) AND a rolling buffer that mixes old/current-policy episodes —
   directly analogous to the DQN "moving target" problem (target networks / soft updates
   are the standard fix). Random CaRL initialization was also discussed but ranked below
   both of the above: it plausibly amplifies the SEVERITY of the earliest instability
   trigger, but by update 5000-8800 (thousands of gradient steps later) it is a weak
   direct explanation for behavior that persists that long.

This experiment isolates both hypotheses in one run rather than three, by comparing
against the existing 8800-update full-cost run as a baseline with exactly the changes
below and nothing else.

### Changes

**1. `outcome_costs_enabled` ablation switch** (`prism/env/safety_cost.py`,
   `prism/env/nuplan_env.py`, `scripts/train.py`) — new `SafetyCostBuilder.__init__`
   param, default `True` (unchanged baseline behavior). When `False`, Tier-1 outcome
   events no longer contribute to `c_outcome`; their boolean flags (`vru_collision`,
   `off_road`, etc.) are still set every step, so infraction telemetry stays complete
   even when a component is excluded from cost. Threaded through
   `make_prism_env(outcome_costs_enabled=...)` and read from `cfg["outcome_costs_enabled"]`
   in `scripts/train.py`'s `_build_env()`.

**2. `active_indicators` ablation switch** (same three files) — new
   `SafetyCostBuilder.__init__` param, a list of indicator names (`"ttc"`, `"thw"`,
   `"speed"`, `"blind_spot"`, `"red_light"`) or `None` (default, all active — unchanged
   baseline behavior). Same flag-always-set / contribution-gated pattern as #1, via a new
   `_indicator_enabled()` helper.

**3. `cost_scale` ablation switch** (`prism/env/nuplan_env.py`) — new `PRISMEnv.__init__`
   param, default `1.0`. Applied ONCE, in `PRISMEnv.step()`, as a uniform multiplier on
   the raw per-step safety cost `c_t` before it feeds `e_t` accumulation — deliberately
   NOT applied inside `SafetyCostBuilder` (episode caps stay expressed/enforced in their
   original calibrated units from `hyperparams.json`; only the final scalar is rescaled).
   `info["safety_cost_raw"]` is added (unscaled) alongside the existing (now scaled)
   `info["safety_cost"]`, for diagnostics.

**4. `fixed_alpha` ablation switch** (`prism/morl/dpmorl_trainer.py`) — new `cfg` key,
   default `None` (unchanged curriculum behavior via the existing `AlphaSchedule`). When
   set, `DPMORLTrainer.train()` uses this constant alpha for every update, bypassing
   `alpha_start`/`alpha_end`/`n_curriculum_iters` entirely. Chosen as `0.5` (median) for
   this experiment specifically because it isolates the curriculum-driven component of
   `nu`'s movement while also being a lower-variance quantile to estimate from a rolling
   buffer than 0.9+ would be. Note for later: alpha=0.5 is a debug-only choice — CVaR's
   value proposition is tail risk, so the eventual model still needs a fixed HIGH alpha
   (0.9-0.95, still no curriculum) tested once this diagnostic clarifies whether
   `nu`-non-stationarity is a real contributor.

**5. This run's specific combination** (`configs/prism_instability_experiment.yaml`, new
   file, and `make train-instability-mini`, new Makefile target) — copies
   `prism_default.yaml` with `outcome_costs_enabled: false`, `cost_scale: 0.5`,
   `active_indicators: [ttc]`, `fixed_alpha: 0.5`. TTC chosen as the single indicator
   both because it is the most standard risk-sensitive signal in the AV safety
   literature (easiest to cite/justify) and because "K policies, different driving
   styles, all respecting one TTC-based safety rule" is a clean, defensible fallback
   publishable result if the full 5-indicator + outcome-cost system doesn't converge
   before the ICRA deadline. `cost_scale=0.5` is a starting guess to bring indicator-only
   cost magnitude (~2-7/step, uncut) closer to reward's bounded `(0,1]`-ish per-step
   scale without eliminating the safety signal — not a precisely derived value; the
   `mu_c`/`z_T` ratio should be checked empirically in the first ~50 updates of the real
   run and this factor revisited if still grossly mismatched.

**6. Gradient-norm-by-group diagnostic** (`prism/models/base.py`: new
   `PRISMPolicyBase.cost_critic_parameters()`, default empty, overridden in both
   `CaRLPPOAdapter` and `AlpamayoAdapter`; `prism/morl/dpmorl_trainer.py`'s
   `_ppo_update()`) — after `total_loss.backward()` but BEFORE
   `clip_grad_norm_` (which rescales `.grad` in place), splits the combined loss's
   gradient norm into cost-critic vs. everything else, at zero extra backward-pass cost
   (pure `.grad` inspection). Directly tests hypothesis "the cost critic's gradients
   dominate the shared clip" — previously asserted from log correlation, not measured.
   `clip_grad_norm_`'s own return value (the pre-clip TOTAL norm) is also captured.
   Logged every update as `grad_norm_total_preclip` / `grad_norm_cost_critic` /
   `grad_norm_other`.

**7. Verbose per-update metrics + GPU utilization logging** (`prism/utils/metrics_logger.py`,
   `prism/utils/gpu_monitor.py`, both new; wired into `dpmorl_trainer.py`'s `__init__`/
   `train()`) — requested so a full run can be analyzed offline in pandas instead of
   depending on terminal scrollback:
   - `MetricsLogger` writes one JSON line per update (not just every 100th, unlike the
     existing INFO log) to `{output_dir}/policy_{id}_metrics.jsonl`: every existing
     diagnostic (`alpha`, `var_nu`, `cvar_hat`, `mu_c`, `sigma_c`, `reward_loss`,
     `cost_penalty`, `cost_critic_loss`, `total_loss`, NaN/streak state) plus the new
     grad-norm-by-group fields, per-update episode-cost mean/std (this update's ~3-4
     episodes, distinct from the buffer's rolling `mu_c`/`sigma_c` — a check on the
     small-sample-noise hypothesis), and a timing breakdown
     (`rollout_time_s`/`ppo_update_time_s`/`update_total_time_s`). The first line is a
     `record_type: "config"` record echoing all static hyperparameters and the 4 ablation
     toggles above, so the file is self-describing without also needing the run's YAML.
   - `GPUMonitor` samples `nvidia-smi` (utilization%, memory used/total, temperature,
     power draw) every `cfg["gpu_log_interval_s"]` seconds (default 2.0) in a background
     thread, for the explicit purpose of telling whether the training-speed bottleneck is
     GPU-bound (in which case a better GPU would help) or something else (data loading,
     simulation stepping, CPU-bound reward/cost computation). Full-resolution samples
     stream to `{output_dir}/policy_{id}_gpu.jsonl`; a windowed average
     (`average_window()`) is also embedded into each update's metrics line. Chose
     `nvidia-smi` subprocess parsing over `pynvml` deliberately — no extra Python
     dependency to install on the lab machine, `nvidia-smi` ships with any NVIDIA driver.
     Probes once on construction and disables itself (single WARNING, never raises) if
     `nvidia-smi` is unavailable — GPU logging must not be able to break training.
   - `DPMORLTrainer.train()` was split into `train()` (setup + `_gpu_monitor.start()` /
     `try: self._train_loop(...) finally: stop()+close()` / final checkpoint) and a new
     `_train_loop()` holding the per-update body — needed so the GPU monitor and metrics
     file close cleanly even when the existing A2 NaN-halt `RuntimeError` fires
     mid-training (verified: the halting update's metrics line is still written, since
     logging happens before the halt check, not after).

### Verification

- `python -m pytest tests/` — 44/44 passing (no regressions).
- `SafetyCostBuilder` toggle behavior (outcome flag-vs-cost split, indicator
  flag-vs-cost split) verified against hand-constructed fake ego-state/detection-cache
  objects — confirmed flags always set, cost contribution correctly gated.
- `cost_critic_parameters()` verified as a strict subset of `trainable_parameters()` for
  `CaRLPPOAdapter`, and verified to return an empty iterator (no crash) for
  `AlpamayoAdapter` when constructed with `cost_critic=None`.
- Full trainer smoke test (fake CaRL policy + fake env, CPU, no nuPlan/GPU dependency):
  `fixed_alpha=0.5` confirmed to pin `alpha` at exactly 0.5 across every update
  (curriculum bypassed); `metrics.jsonl` confirmed to contain one config record (all 4
  ablation toggles correctly echoed) plus one update record per training update, with
  `grad_norm_cost_critic` nonzero on at least one update (confirms the split is wired to
  real parameters, not silently always zero).
- `GPUMonitor` confirmed to disable gracefully (no exception) on this CPU-only dev
  machine (`nvidia-smi` unavailable) — `average_window()` returns `None` rather than
  raising. NOT verified against a real GPU / real `nvidia-smi` output — first real signal
  on that comes from the actual lab-machine run.
- A1 (telescoping WARNING) and A2 (NaN/Inf halt after 3 consecutive updates) sanity
  checks re-run against the `train()`/`_train_loop()` split — both still fire correctly;
  confirmed the halting update's metrics line is written before the `RuntimeError`
  propagates through the `try/finally`.
- **Not verifiable here, flagged for the lab machine**: real GPU utilization numbers
  (only a CPU dev box is available in this checkout); whether `cost_scale=0.5` actually
  brings cost and reward magnitudes into a comparable range (needs the real `mu_c` vs
  `z_T` numbers from the first ~50 updates); whether this combination is actually stable
  over a full run — that is the entire point of running it.

### Outstanding

- [ ] Run `make train-instability-mini` (or the full-dataset equivalent) on the lab
      server and review `runs/instability_ttc/policy_0_metrics.jsonl` +
      `policy_0_gpu.jsonl`.
- [ ] If stable: cost formulation was likely dominant — plan the next experiment (fixed
      HIGH alpha, e.g. 0.9-0.95, no curriculum) to confirm CVaR's tail-risk framing still
      trains stably before reintroducing outcome costs incrementally.
- [ ] If still visibly unstable (smaller magnitude but still spiky `cost_critic_loss` /
      oscillating `var_nu`): `nu`-non-stationarity is independently significant — consider
      a target-network-style soft update for `nu` (EMA) as the next change, per the DQN
      moving-target analogy in the motivation section above.
- [ ] Reward-only (`beta=0`, `cf_coef=0`) and cost-only (`reward_weight=0`, `beta=1`)
      ablations remain deferred, not part of this experiment — would need a new
      symmetric `reward_weight` config knob (not yet implemented).

### Addendum — K reduced 5 → 4

Cuts sequential wall-clock by 20% (K trainers run sequentially per
`scripts/train.py`'s docstring). `_get_preference_vectors()` gained an explicit
`n_policies == 4` case — the existing K=5 curated list minus "balanced" — rather than
falling through to the generic per-K formula, which for K=4 produces a less extreme
skew (~42/19/19/19 vs. the curated 55/15/15/15) and for K=3 specifically never
emphasizes one of the 4 reward dimensions at all (`k % reward_dim` never reaches it
within 3 iterations). K=4 was chosen over K=3 specifically to keep full coverage across
all 4 style dimensions — relevant here since instability could be style-dependent (e.g.
a progress-weighted policy driving more aggressively than a comfort-weighted one).

Also fixed a latent bug this surfaced: `make train-instability-mini`'s Makefile target
had copied `--n_policies 2` from `train-mini` (whose job is a fast pipeline smoke test,
not a substantive run) — since CLI `--n_policies` overrides the config file's
`n_policies` (`scripts/train.py`: `if args.n_policies: cfg["n_policies"] = args.n_policies`),
this would have silently run K=2 regardless of what `prism_instability_experiment.yaml`
specified. Removed the CLI override; `n_policies: 4` in the YAML is now the sole source
of truth for this target.

### Addendum — GPUMonitor sample interval 2.0s → 15.0s default (20.0s for this experiment)

First real (lab-machine) run reported the whole machine (used for both training and
interactive desktop work) becoming severely sluggish; the run was killed after a single
update. Suspected cause: `GPUMonitor` spawns a fresh `nvidia-smi` subprocess from a
background thread every `gpu_log_interval_s` — at 2.0s this is frequent enough to add
real fork/exec + GPU-driver-query overhead, plausibly contending with the training
process's own CUDA calls on some driver setups. Not conclusively proven (only one
update's data exists from that run), but low-risk and easy to fix: raised the code
default (`dpmorl_trainer.py`) from 2.0s to 15.0s and this experiment's config value to
20.0s — roughly 10x fewer subprocess spawns, still ample resolution over a run spanning
thousands of updates. If system-wide slowdown persists at the new interval, GPU
utilization logging should be suspected less and something else (e.g. driver-level
contention independent of poll frequency, or unrelated system load) investigated instead.

Incidentally, the single update logged before the kill (`gpu_util_pct: 5.25`,
`rollout_time_s: 7.32` vs `ppo_update_time_s: 0.82`) suggests rollout collection
(CPU-bound nuPlan simulation) dominates update wall-clock, not GPU compute — a
preliminary signal (one data point, not conclusive) that training speed may not be
GPU-bound at all.

---

## 2026-08-26 — DPMORL-only experiment (safety term off) + z_T / episode-length logging

### Motivation

785 updates into the indicator-only-TTC instability-analysis run (see previous entry),
`cvar_hat`/`cost_critic_loss`/`mu_c` showed a real dip (updates ~400-620) followed by a
rise back toward starting levels (updates ~700-785) — smaller in scale than the original
full-cost run's divergence, but not fully resolved. Checked the two diagnostics built for
exactly this purpose against the real data: `grad_norm_cost_critic` never exceeded ~23% of
the total pre-clip gradient norm (rules out the shared-grad-clip-dominance hypothesis as
the driver of THIS residual pattern), and `mean_episode_cost_this_update` (local, fresh
episodes, no buffer lag possible) tracked the rolling buffer's `mu_c` closely rather than
lagging it (rules out buffer staleness as the primary driver). Leading remaining
hypothesis: `beta` is a fixed penalty weight, not an adaptive Lagrangian-style dual
variable — nothing in the current design increases pressure when cost trends back up,
which is a well-documented failure mode (bounded oscillation around the constraint
boundary) for fixed-penalty constrained RL.

Before chasing that further, it's worth checking a completely orthogonal question that
instability-analysis so far has never isolated: does the DPMORL (personalization) half of
the system work cleanly on its own, independent of the cost/safety side entirely? Every
diagnosis so far has been on the cost side; if reward-only training is stable and produces
K distinguishable driving styles, that further narrows the instability specifically to the
`beta`/cost-critic interaction. If it doesn't, that's a more fundamental, unrelated
problem — and worth knowing now, since every fallback plan (see prior entries) depends on
personalization working at all.

### Changes

**1. DPMORL-only experiment config** (`configs/prism_dpmorl_only.yaml`, new, +
   `make train-dpmorl-only-mini`, new Makefile target) — `beta: 0.0`, `cf_coef: 0.0`, no
   other code path changes needed: both are already-existing config keys.
   `A_total = A_reward - 0*A_cost = A_reward` drops the cost advantage out of the PPO
   objective entirely, and `cf_coef=0` zeroes the cost critic's own gradient (via
   `0 * cost_critic_loss` in `total_loss`) so it doesn't even consume shared grad-clip
   budget, though the cost critic still runs a forward pass every step (harmless — same
   architecture, safety term switched off, not a different code path). Verified the cost
   critic's Adam moment estimates stay at zero throughout (zero grad every step -> zero
   parameter update), so it doesn't drift from initialization while inert. K=4, same
   curated preference vectors as the instability-analysis experiment.

**2. z_T and episode-length logging** (`prism/morl/dpmorl_trainer.py`) — the one gap that
   would have made the DPMORL-only experiment impossible to actually judge: `episode_zt`
   was already available per completed episode (used for the in-memory `summary` dict) but
   never reached the per-update metrics file. Added `local_episode_zts` /
   `local_episode_lengths` tracking (parallel to the existing `local_episode_costs`
   pattern) in `_train_loop()`'s rollout loop; new fields in `policy_{k}_metrics.jsonl`:
   `z_comfort` / `z_progress` / `z_lateral` / `z_spacing` (mean z_T per style dimension
   across episodes completed that update — `_STYLE_DIM_NAMES`, falls back to generic
   `z_dim{i}` naming if a future config's `reward_dim` isn't 4), `mean_episode_length` /
   `min_episode_length` / `max_episode_length` (cheap capability proxy — longer episodes
   suggest fewer early terminations), and `mean_reward_this_update` (mean raw `R_t` across
   the update's rollout, for judging reward-critic health). Judging the DPMORL-only
   experiment: compare the 4 policies' `z_*` trends against each other — each should trend
   higher on its OWN preferred dimension (per `_get_preference_vectors`) than the others.
   These fields are populated for every experiment going forward, not just this one — in
   particular they'll also retroactively help interpret the instability-analysis run's
   `mu_c` rise (is it coming with longer episodes / more genuine traffic exposure, or not).

### Verification

- `python -m pytest tests/` — 44/44 passing.
- Full trainer smoke test (fake CaRL policy + fake env, `beta=0`, `cf_coef=0`): confirmed
  `z_comfort`/`z_progress`/`z_lateral`/`z_spacing`/`mean_episode_length`/
  `min_episode_length`/`max_episode_length`/`mean_reward_this_update` all present in every
  update record.
- Re-ran the instability-analysis and A1/A2 sanity-check smoke tests from the prior two
  entries against a clean output directory — both still pass unchanged (no regression from
  the rollout-loop additions).
- **Not verifiable here, flagged for the lab machine**: whether the 4 DPMORL-only policies
  actually produce distinguishable styles, and whether reward-only training is itself
  stable over a long run — that's the entire point of running it.

### Outstanding

- [ ] Run `make train-dpmorl-only-mini` and compare the 4 policies' `z_*` trends.
- [ ] If DPMORL-only is stable and styles diverge as expected: strong evidence the
      instability lives specifically in the `beta`/cost-critic interaction — prioritize
      the adaptive-`beta` (dual-ascent) change over further cost-formulation ablations.
- [ ] If DPMORL-only is ALSO unstable or styles don't diverge: unrelated, more fundamental
      problem — re-scope investigation away from the CVaR/safety mechanism entirely.
- [ ] Adaptive `beta` (dual-ascent-style, reintroducing the original Lagrangian
      constrained-RL idea inside the already-correct v2 dual-critic architecture, not the
      buggy v1 REINFORCE mechanism) remains proposed, not implemented — deferred pending
      the DPMORL-only result above.

---

## 2026-08-26 — "Is it learning" logging fields + standalone analysis script

### Motivation

The 785-update read of the instability-analysis run's `metrics.jsonl` could only speak to
whether the cost/CVaR math was numerically stable, not whether the policy was actually
improving — no reward, style-vector, or episode-outcome signal was reaching the log at
all. Asked directly whether the training was "working" in the sense of decreasing loss/CVaR
and got an honest "no clean trend, and we can't even tell if driving is improving" answer.
Fixing that gap is also a prerequisite for the newly-agreed plan (DPMORL-only run first,
warm-start the safety run from it if styles diverge cleanly, investigate rewards from
scratch if they don't) — that plan can't be judged without exactly this data.

Also, `metrics.jsonl` files are growing into the multi-MB, thousands-of-lines range (the
instability-analysis run is at 6522+ updates). Manually re-deriving binned trends by hand
each time (as done for the two previous entries) doesn't scale — asked for a reusable
script instead.

### Changes

**1. New per-update fields** (`prism/morl/dpmorl_trainer.py`):
   - `v_loss`, `entropy` — the reward critic's own MSE loss and the policy's mean entropy,
     both already computed inside `_ppo_update()`'s minibatch loop but never returned or
     logged past that method. `v_loss` should trend down as the critic learns to predict
     returns; `entropy` should trend down as the policy commits to more confident actions
     (but collapsing to ~0 immediately would suggest premature convergence, not learning).
   - `z_comfort` / `z_progress` / `z_lateral` / `z_spacing` — carried over from the prior
     entry, unchanged.
   - `mean_reward_this_update` — carried over from the prior entry, unchanged.
   - `frac_collision` / `frac_off_road` / `frac_completed` — fraction of episodes completed
     THIS update that ended in a collision, went off-road, or reached scenario
     end/truncation. New `local_episode_outcomes` tracking in `_train_loop()`'s rollout
     loop, classified from `info["safety_components"]`'s boolean flags (already set
     unconditionally regardless of `outcome_costs_enabled`, per the first
     instability-analysis entry) at each episode's terminal step. Deliberately a SEPARATE
     signal from `mean_episode_length`: a stationary/timid policy can also produce long
     episodes without driving anywhere, so length alone can't distinguish "surviving
     because it drives well" from "surviving because it barely moves." Direct capability
     signal, and directly relevant to interpreting the episode-count/cost spike found in
     the prior entry's updates ~1300-3260 window.
   - `grad_norm_*` fields unchanged (already added in the first instability-analysis
     entry); no new gradient-diagnostic work here.

**2. `scripts/analyze_metrics.py`** (new, standalone, stdlib-only) — parses one or more
   `policy_*_metrics.jsonl` files (handles multiple config-record "runs" appended to the
   same file from a restarted process, and a truncated final line from copying a file
   still being written), and prints a concise report per file: config, wall-clock pace,
   NaN/stability check, a binned trend table (cost/CVaR side + the new learning-signal
   fields + episode outcomes + cost-critic gradient-norm fraction), z_T-by-dimension
   early-vs-late trend, and a handful of early-half-vs-late-half directional flags
   (explicitly framed as pointers to check against the binned table, not an automated
   verdict — a flat/noisy metric can trip a >5% threshold either direction without meaning
   anything). When multiple files are given (one per K policy), also prints a cross-policy
   z_T comparison table assuming the standard K=4 preference ordering from
   `_get_preference_vectors()`, to check whether each policy trends highest on its own
   preferred dimension. Intended workflow: run this on the lab machine or after copying
   the log over, share only the printed report, not the raw multi-MB file.

### Verification

- `python -m pytest tests/` — 44/44 passing.
- Trainer smoke test with a fake env that cycles collision/off_road/completed outcomes:
  confirmed `frac_collision`/`frac_off_road`/`frac_completed` sum to 1.0 each update and
  correctly reflect the injected outcome sequence; confirmed `v_loss`/`entropy` populated.
- `analyze_metrics.py` run against the real 6522-update instability-analysis log (older
  format, predates `z_comfort`/`frac_*`/`v_loss`/`entropy`): reproduced the exact binned
  `mu_c`/`cvar_hat`/`cost_critic_loss` trend derived by hand in the prior entry, printed
  `n/a` gracefully for every missing field rather than crashing.
- `analyze_metrics.py` run against fresh smoke-test output containing all new fields:
  confirmed the full binned table, z_T section, and multi-file cross-policy comparison
  code paths execute without error (too few updates in the smoke test itself for the
  early/late-half comparison to populate, expected and harmless at that scale).

### Outstanding

- [ ] Run `make train-dpmorl-only-mini` with this logging in place; use
      `scripts/analyze_metrics.py runs/dpmorl_only/policy_*_metrics.jsonl` to read the
      result instead of manual analysis.
- [ ] Everything else from the prior entry's Outstanding section still applies unchanged.
