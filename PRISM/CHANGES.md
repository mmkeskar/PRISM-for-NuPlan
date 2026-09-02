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

---

## 2026-08-26 — Unbounded log_std fix, K reduced to 2 for the next DPMORL-only run

### Motivation

First real DPMORL-only run (1605 updates, K=4, beta=0/cf_coef=0): `mean_reward_this_update`
stayed completely flat (0.031-0.034) the entire run -- no improvement at all -- while
`entropy` climbed monotonically from 1.98 to 3.74 and never turned around, and
`approx_kl` was enormous in the first 3 bins (11.3, 10.6, 23.8 -- healthy PPO updates are
normally ~0.001-0.05). Traced the mechanism: `StochasticActionHead.log_std`
(`prism/models/common/action_heads.py`) had no upper bound. The PPO surrogate loss's
gradient contribution to `log_std` is structurally close to zero (per-minibatch advantage
normalization + ratio~1 make it near-self-cancelling -- the same effect already documented
for `reward_loss`/`cost_penalty`, and now understood to extend to `ppo_loss` itself), while
the entropy bonus (`-ent_coef * entropy` in `total_loss`) provides a constant, unopposed
pull toward higher entropy. With nothing capping `log_std`, action noise could grow for the
entire run instead of shrinking as the policy commits to learned behavior -- and growing
noise plausibly explains the flat reward directly: the network's learned mean action gets
increasingly drowned out by its own randomness. This is architecture-level code, shared by
both experiments (not personalization- or safety-specific), so it's also a plausible
contributor to the burst of short/crash-like episodes seen in the safety-cost
(instability-analysis) run's updates ~1300-3260 -- not confirmed directly (that run predates
entropy logging and was never restarted), but a consistent, plausible mechanism.

### Changes

**1. `log_std` clamp** (`prism/models/common/action_heads.py`) -- new module constants
   `_LOG_STD_MIN = -5.0`, `_LOG_STD_MAX = 1.0` (std range ~[0.007, 2.72], chosen for actions
   normalised to [-1, 1] per the existing docstring; wide enough not to artificially
   constrain exploration, bounded enough that noise can't grow without limit). Applied via
   `.clamp()` on a computed copy inside `forward()`, not in-place on the stored
   `nn.Parameter` -- stays differentiable, doesn't fight the optimizer's own momentum
   state. The raw parameter can still drift outside the range under gradient descent (seen
   directly in a stress test: forced to 50.0, stayed at 50.0 after further training) but its
   EFFECT is always bounded, since every forward pass re-clamps before computing the
   distribution.

**2. `tests/test_action_heads.py`** (new) -- direct unit tests on `StochasticActionHead`:
   forcing `log_std` to 100 confirms entropy reflects the clamped ceiling, not the raw
   value; forcing it to -100 confirms `log_prob` stays finite (an uncapped floor would
   collapse std toward 0 and blow up log-probability for any off-mean action); a value
   already inside the range passes through unchanged; and a sanity check that the default
   `init_log_std=-0.5` falls strictly inside the clamp bounds.

**3. K reduced 4 → 2 for the next DPMORL-only run** (`configs/prism_dpmorl_only.yaml`,
   Makefile echo text updated to match) -- less training before getting a first signal on
   whether personalization works at all. Uses `_get_preference_vectors()`'s existing
   generic fallback for `n_policies != 5` (no new code needed): for K=2 this lands on
   policy 0 = comfort-preferring, policy 1 = progress-preferring -- two clearly contrasting,
   behaviorally distinct styles. If these two diverge cleanly once the log_std fix is in,
   scale back to `n_policies: 4` for full coverage before the warm-start/safety-combination
   step from the earlier plan.

### Verification

- `python -m pytest tests/` -- 48/48 passing (44 existing + 4 new).
- Stress-test smoke run: `log_std` force-set to 50.0 before training; confirmed training
  completes without crashing and the EFFECTIVE (clamped) log_std used in the action
  distribution is correctly bounded at 1.0 regardless of the raw parameter's value.
- `configs/prism_dpmorl_only.yaml` confirmed parses with `n_policies: 2`;
  `make -n train-dpmorl-only-mini` dry-run confirmed correct.

### Outstanding

- [ ] Run `make train-dpmorl-only-mini` (K=2, log_std fix in place) and check with
      `scripts/analyze_metrics.py`: does entropy now trend down (or at least plateau)
      instead of climbing unboundedly, and does reward show any improvement this time?
- [ ] If yes to both: scale back to K=4, then proceed to the warm-start/safety-combination
      step from the earlier plan.
- [ ] If entropy is fixed but reward is still flat: the log_std runaway wasn't the (only)
      cause -- investigate the reward signal/critic more directly.
- [ ] Consider whether to apply the same stress-test style check retroactively once the
      instability-analysis (safety-cost) run is restarted with current code, to see whether
      entropy behaves the same way there and whether it correlates with the
      updates-1300-3260 episode-termination burst as hypothesized above.

---

## 2026-08-26 (cont.) — ent_coef 0.01 → 0.0: clamp bounded the symptom, didn't fix the cause

### Motivation

The K=2 + log_std-clamp run above was allowed to continue past the first 263 updates
covered in the previous entry. At 2347 updates, `entropy` had climbed nearly every single
bin (2.07 -> 4.51), approaching the clamp's theoretical ceiling (~4.84 for this 2D action
space) rather than plateauing before reaching it -- the clamp bounded the worst case but
did not address why entropy keeps climbing. Meanwhile `v_loss` genuinely improved
(0.035 -> ~0.010-0.014) and episode outcomes genuinely improved (`frac_completed` 49% ->
82%, collision rate 34% -> 11%), but `mean_reward_this_update` stayed completely flat
(0.031-0.034) the entire run. Read together: real learning is visibly happening (critic,
survival/completion), but the actor's own reward score never moves, while its actions get
steadily noisier -- consistent with the entropy bonus (`-ent_coef * entropy` in the PPO
loss) being effectively unopposed the whole run, not just early on, per the mechanism in
the previous two entries (ppo_loss's gradient contribution to log_std is structurally weak
under per-minibatch advantage normalization).

### Change

`ent_coef: 0.01 -> 0.0` in `configs/prism_dpmorl_only.yaml`'s `stage2` block (config-only,
no code change -- `ent_coef` was already trainer-config-driven). Zero chosen deliberately
over a smaller-but-nonzero value: it's the cleanest test of the exact hypothesis above (if
entropy still climbs with the pull fully removed, something else entirely is driving it,
cleanly falsifying this theory; if it stabilizes, the theory is confirmed) -- a smaller
value would give a muddier signal either way. Also matches common practice: several
well-known PPO implementations default entropy coefficient to 0.0 specifically for
continuous-action control, relying on the Gaussian action distribution's own sampling
noise for exploration rather than an explicit bonus (unlike discrete-action settings, where
an entropy bonus is more commonly needed to prevent premature deterministic collapse).

### Verification

Config-only change; confirmed `configs/prism_dpmorl_only.yaml` parses with
`stage2.ent_coef == 0.0`. No code touched, so the existing 48/48 test suite is unaffected
(not re-run for this entry specifically since nothing it covers changed).

### Outstanding

- [ ] Restart `make train-dpmorl-only-mini` with `ent_coef=0.0`; watch whether entropy
      plateaus/decreases this time, and whether `mean_reward_this_update` finally shows
      sustained movement instead of sitting flat at ~0.033.
- [ ] If entropy stabilizes and reward improves: hypothesis confirmed -- consider whether
      some small nonzero floor is worth reintroducing later (e.g. if entropy collapses too
      fast / policy converges prematurely), but only after seeing this run's behavior.
- [ ] If entropy STILL climbs with ent_coef=0.0: the entropy bonus was not the (sole)
      driver -- look elsewhere (e.g. the log_prob/ratio computation's own gradient
      contribution to log_std, or the advantage-normalization mechanism itself).
- [ ] Everything else from the prior two entries' Outstanding sections still applies.

---

## 2026-08-27 — Root cause found: reward barely differentiated between policies at all

### Motivation

The 6560-update run with ent_coef=0.0 (previous entry) confirmed entropy growth had slowed
substantially (a real, partial win) but `mean_reward_this_update` was STILL pinned in the
same ~0.032-0.034 band as every prior run, regardless of entropy dynamics -- three
different configurations now landing in that same narrow range. That ruled entropy out as
the explanation for flat reward specifically. Rather than run a fourth expensive experiment
to gather more of the same evidence, read `prism/morl/utility_functions.py` (`f_k`, the
function `R_t = gamma^{-t} * (f_k(z_{t+1}) - f_k(z_t))` is computed from) directly, and found
a concrete, quantifiable root cause -- no GPU/nuPlan needed to find or verify it.

`UtilityFunction.forward()` first min-max normalises `z` using `_min_val`/`_max_val`, running
statistics that track the full range of `z` ever observed (updated on EVERY call, not just
during training). Since every episode starts at `z=0` and grows to roughly 50-80 per
dimension by the end, this range spans the full cumulative episode scale. A single
timestep's contribution to `z` (~0.1-1, one step's style reward) is therefore always a tiny
fraction of that range, at every point in every episode -- so the monotone neural network,
operating on this heavily-squashed input, barely reacts to what happened on any single step,
regardless of the step's actual quality.

`forward()` also adds `linear_term = lamda * z.mean(dim=-1)` -- per its own comment, "to keep
gradient non-zero," i.e. a small safety net for exactly this squashing scenario. But it uses
a **plain, unweighted mean** across all 4 style dimensions, not each policy's own preference
weights `w_k`. An offline check (`init_utility_functions_from_preferences`, simulating a
realistic 150-step episode, zero GPU/nuPlan dependency) confirmed both parts precisely:
`mean |R_t| = 0.030` -- matching the real training runs' ~0.032-0.034 almost exactly -- and
**99.4-99.5% of the total signal came from that unweighted linear term alone**, not the
neural pathway. A comfort-preferring and a progress-preferring policy got near-identical
mean `|R_t|` (0.03016 vs 0.03017). The "safety net" had quietly become nearly the entire
reward signal, and it doesn't know or care which policy it's training.

### Changes

**1. Preference-weighted linear term** (`prism/morl/utility_functions.py`) -- new
   `_pref_weights` buffer (registered alongside the existing `_min_val`/`_max_val`, so it's
   part of `state_dict()` and round-trips through checkpoint save/load automatically, no
   extra plumbing needed at the two call sites in `scripts/train.py` that load a saved
   utility function). Defaults to uniform (`ones(reward_dim)/reward_dim`, mathematically
   identical to the old `z.mean()` -- exact backward compatibility until set). New
   `set_preference(pref)` method. `forward()`'s `linear_term` changed from
   `lamda * z.mean(dim=-1)` to `lamda * (z * self._pref_weights).sum(dim=-1)` --  a convex
   combination either way (both existing preference vectors and the uniform default sum to
   1), so this is a scale-neutral swap, not a magnitude change: same contribution size,
   correctly weighted per-policy instead of generically averaged.
   `init_utility_functions_from_preferences()` now calls `uf.set_preference(pref)` for each
   policy (previously only used `pref` to bias `fc_in`'s weights).

**2. `max_weight` raised 0.1 → 1.0** (`UtilityFunction.__init__` default) -- the neural
   pathway's weights are clamped to `[0, max_weight]` after every optimizer step
   (`make_monotone()`) to preserve monotonicity (non-negative weights => output
   non-decreasing in each `z_i`); 0.1 was too small for the network to contribute a
   meaningful signal on top of the (now correctly preference-weighted, but still only a
   linear proxy) fallback term. Not threaded through `cfg`/YAML -- changed at the class
   default level, affecting all current call sites uniformly, matching how other defaults in
   this module were already handled before any config-level tuning was added.

**3. `tests/test_utility_functions.py`** (new, 7 tests) -- `set_preference()` behavior
   (default uniform, updates the buffer, rejects wrong shapes, survives a state_dict
   round-trip); the linear term itself is preference-weighted (isolated from the neural
   pathway); end-to-end, a comfort-preferring and progress-preferring policy value a boost
   to their own dimension more than the other's; monotonicity still holds after the
   `max_weight` change. Two of these initially failed for an instructive reason: `_min_val`/
   `_max_val` update on every forward call (train or eval), so a naive sequence of calls
   with different `z` shifts the normalization mid-comparison and isn't a fair
   before/after check -- fixed by pinning them to a realistic fixed range before comparing
   (the regime training settles into once enough episodes have shown the achievable range).
   Worth flagging as a secondary, smaller observation for later: this means the
   normalization is a genuine moving target for the ENTIRE run, not just early on, similar
   in spirit to `nu`'s moving-target issue on the cost side -- likely much less severe since
   the range plausibly saturates fairly quickly, but not verified, and not addressed here.

### Verification

- `python -m pytest tests/` -- 55/55 passing (48 previous + 7 new).
- Offline diagnostic (no GPU/nuPlan): after the fix, the linear term's share of the total
  signal dropped from >99% to ~56-61% (the neural pathway now contributes meaningfully).
  Direct, controlled check (fixed `z`, boost one dimension by exactly 1.0, fixed
  normalization range): the comfort-preferring policy valued a +1 comfort boost ~3.5x more
  than a +1 progress boost; the progress-preferring policy showed the mirror-opposite ratio.
  This is the differentiation that was completely absent before.
- Full trainer smoke test (fake CaRL policy + fake env) with the real
  `init_utility_functions_from_preferences()` path: completed without crashing, all
  diagnostic histories finite.

### Outstanding

- [ ] Restart `make train-dpmorl-only-mini` (K=2, log_std clamp, ent_coef=0.0, and now this
      fix) and check with `scripts/analyze_metrics.py`: does `mean_reward_this_update`
      finally show real, sustained movement, and do the two policies' `z_*` trends diverge
      more clearly than before?
- [ ] If reward improves and styles diverge: proceed to the warm-start/safety-combination
      step from the earlier plan (scale back to K=4 first).
- [ ] If reward is STILL flat even now: the remaining ~40-44% neural-pathway share may still
      not be enough, or the moving-normalization-range issue noted above may be more
      significant than assumed -- worth deliberately verifying rather than guessing further.
- [ ] The moving min/max normalization range (noted above) is unaddressed -- revisit if the
      above still doesn't resolve things.
- [ ] Everything else from the prior entries' Outstanding sections still applies.

---

## 2026-08-27 (cont.) — Two confirmed style-reward bugs, found by auditing against the paper

### Motivation

Before restarting training on the utility-function fix, audited `prism/env/rewards.py`
against `docs/prism_paper_v2.tex`'s formal spec (eq. progress, eq. lateral) rather than
running another experiment to find out empirically. Found two real, confirmed
implementation-vs-spec mismatches.

### Changes

**1. `r_lane` direction was inverted** (`prism/env/rewards.py::compute_progress`). The paper
   states explicitly: "Leftmost (fastest) lane scores 1; rightmost scores 0." The code
   computed `r_lane = lane_index / (n_lanes - 1)` with `lane_index=0` = leftmost (per its own
   comment) -- meaning `r_lane` was HIGHEST at the rightmost lane and LOWEST at the leftmost,
   the exact opposite of spec. Fixed to `r_lane = 1.0 - lane_index / (n_lanes - 1)`.
   `tests/test_rewards.py::test_single_lane_neutral_r_lane` was asserting the old (buggy)
   direction and has been corrected to match the paper.

**2. `sigma_d` read from the wrong hyperparameter key** (`prism/env/rewards.py::compute_style_rewards`).
   The paper defines two DISTINCT constants for lateral discipline: `sigma_d=0.2` (Gaussian
   width, from empirical lane-keeping data, cited) and `delta_d=0.3` (the additive floor,
   separately hardcoded in `compute_lateral_discipline`'s formula, preventing lane-change
   maneuvers from being over-penalized). `compute_hyperparams.py` correctly produces both
   under their own keys (`reward_scaling.sigma_d` and `floor_values.delta_d` respectively) --
   but `compute_style_rewards` was reading `hp["floor_values"]["delta_d"]` and passing it as
   `sigma_d`, silently using 0.3 instead of the intended, cited 0.2 in production. The test
   fixture's `floor_values.delta_d` happened to already be 0.2, masking the wrong-key read --
   `default_hp` now deliberately keeps `reward_scaling.sigma_d=0.2` and
   `floor_values.delta_d=0.3` different (matching real `compute_hyperparams.py` output), and
   a new regression test (`test_uses_reward_scaling_sigma_d_not_floor_delta_d`) checks
   `compute_style_rewards`'s output against a direct `compute_lateral_discipline` call using
   the correct key, so this can't silently regress again. Also fixed the same missing-key
   issue in `scripts/explore_simulator.py`'s placeholder hyperparams dict (would have raised
   `KeyError` on `scaling["sigma_d"]` once the direct-index read landed).

**Not changed, flagged as an intentional design property worth being aware of:** `r_accel`
(part of `r_progress`) uses `1 - exp(-|a_ego|/gamma_a)` -- rewards acceleration MAGNITUDE in
either direction (speeding up or braking), explicitly per the paper ("Magnitude captures
both assertive acceleration and decisive braking," cited to Surmann2025), not "is the car
moving." This means a policy cruising smoothly at exactly the desired speed (`a_ego≈0`)
scores near `r_accel`'s floor (0.01) on this sub-term specifically, in real tension with
`r_comfort`'s jerk-minimization goal. This is deliberate per the citation, not a bug -- but
worth keeping in mind when interpreting results, since it means `r_progress` can be
structurally small during otherwise-ideal driving, and the personalization axis between
"comfort" and "progress" preferences may partly reflect this designed-in tension rather than
being purely orthogonal.

### Verification

- `python -m pytest tests/` -- 56/56 passing (55 previous + 1 new regression test; 1 existing
  test corrected to match the fixed lane direction).
- Confirmed via direct code read (not simulation) against `compute_hyperparams.py`'s actual
  output structure (`reward_scaling.sigma_d=0.2`, `floor_values.delta_d=0.3` in the real,
  non-test-fixture bootstrap/output hyperparams) that the sigma_d bug is real in production,
  not just a test-fixture artifact.
- Grepped the full repo for other `floor_values`/`delta_d`/`lane_index` usages to confirm no
  other call site depends on the old (buggy) behavior.

### Outstanding

- [ ] Both fixes apply automatically to the next training run (no config change needed) --
      restart `make train-dpmorl-only-mini` with all fixes from today's entries in place.
- [ ] Everything else from today's earlier entries' Outstanding sections still applies.

---

## 2026-08-27 (cont.) — `--policy_ids` for running K policies as concurrent processes

### Motivation

Asked whether K policies could train "simultaneously via matrix math." They can't
meaningfully -- `rollout_time_s` (CPU-bound nuPlan simulation) dominates `ppo_update_time_s`
(the actual GPU/tensor math) by roughly an order of magnitude in every run analyzed so far,
so batching the network computation across policies would only shave a few percent off
wall-clock. The real lever is process-level parallelism: run each policy's full
rollout+update loop as a separate OS process, on a separate CPU core, concurrently. GPU
utilization has been consistently low (5-16%) in every run, so one GPU has ample headroom
for K lightweight models running concurrently -- the limiting resource is CPU cores, not GPU.

### Change

`scripts/train.py`: new `--policy_ids` CLI flag (comma-separated indices, e.g. `"0"` or
`"2,3"|`), threaded through to `run_stage2(policy_ids=...)`, which now skips any `k` not in
the given set (`None`, the default, trains all `n_policies` sequentially -- unchanged prior
behavior). Each policy already writes to its own `policy_{k}/` subdirectory, so concurrent
processes never touch the same output files -- no restructuring needed beyond the filter
itself. Combines with the existing (pre-existing, unmodified) `--stage1_only` /
`--skip_stage1 --utility_fn_dir` flags: run Stage 1 once to save utility functions, then
launch one process per policy pointed at that same saved output, e.g. for K=2:

```
python scripts/train.py --config configs/prism_dpmorl_only.yaml --stage1_only \
    --output_dir runs/dpmorl_only
python scripts/train.py --config configs/prism_dpmorl_only.yaml \
    --skip_stage1 --utility_fn_dir runs/dpmorl_only/prism_dpmorl_only_001/stage1 \
    --output_dir runs/dpmorl_only --policy_ids 0 &
python scripts/train.py --config configs/prism_dpmorl_only.yaml \
    --skip_stage1 --utility_fn_dir runs/dpmorl_only/prism_dpmorl_only_001/stage1 \
    --output_dir runs/dpmorl_only --policy_ids 1 &
wait
```

### Verification

- `python -m pytest tests/` -- 56/56 passing (no logic in the tested modules touched).
- `python scripts/train.py --help` -- caught and fixed a real bug introduced while writing
  this: a literal `%` in the new flag's help text broke argparse's help formatter (`%` is a
  format-string directive there), crashing `--help` entirely. Fixed by rewording; `--help`
  now runs cleanly and shows the new flag.
- Not run end-to-end on the lab machine (needs the actual nuPlan/CaRL environment) -- flagged
  for the user to try when launching the next DPMORL-only run.

### Outstanding

- [ ] Try the concurrent-process workflow above on the next DPMORL-only restart; watch CPU
      usage (e.g. `htop`) to see whether the machine has enough cores for a real ~2x speedup
      or whether concurrent simulation contends and the gain is smaller.
- [ ] Not addressed: the K policies still train fully independent backbones (each
      `_build_agent()` call constructs a fresh network, no backbone sharing across policies)
      -- unrelated to this change, noted here only because it came up while reading this code
      path; a shared-backbone design would be a much bigger change, out of scope for now.

`make train-dpmorl-only-mini-parallel` added (`Makefile`): wraps the `--policy_ids` workflow
above -- runs Stage 1 once, then launches `N_POLICIES` (auto-detected from
`configs/prism_dpmorl_only.yaml`'s `n_policies`, overridable on the CLI) concurrent
background processes via a shell `for`/`wait` loop, one per `policy_id`, each with its
stdout/stderr redirected to its own `runs/dpmorl_only/policy_{k}.log` (concurrent processes
writing to one terminal would otherwise interleave). Verified with `make -n` dry-runs at both
the auto-detected K=2 and an `N_POLICIES=1` override (including singular/plural wording in
the echo lines) -- not run for real (needs the lab machine's nuPlan/CaRL environment).

---

## 2026-08-29 — Three bundled fixes for late-training instability

### Motivation

The K=2, full-5000-update run (both fixes from the previous two entries in place) told a
different story than the 2000-update checkpoint had suggested: ALL FOUR z_T dimensions
regressed by the end, for BOTH policies, not just `z_progress`. `policy_0`'s `approx_kl` hit
54.4 in the final bin (4501-5000) -- an order of magnitude past anything seen in any prior
run -- with `clip_fraction` climbing steadily the entire second half (0.082 -> 0.365).
Cross-referencing the bins: entropy bottomed out and outcomes peaked around update 2500-3000
for `policy_0`, then both reversed in the back half, culminating in that late spike. Three
candidate mechanisms, bundled into one run rather than three separate ~10.6h runs given time
cost, though each is independently well-motivated (not competing guesses for the same
phenomenon):

### Changes

**1. Linear learning-rate decay** (`prism/morl/dpmorl_trainer.py`) -- `learning_rate` was
   fixed at its initial value for the entire run, including the final updates, where a
   still-full-sized step is least wanted. Standard PPO practice (original PPO paper, CleanRL,
   etc.) linearly decays LR to 0 over the run; this codebase had no such schedule. Implemented
   directly on `optimizer.param_groups` each update (`frac = 1 - update/n_updates`) rather
   than via `torch.optim.lr_scheduler`, matching how e.g. CleanRL does it -- simpler to reason
   about alongside the update-indexed loop already there. New `cfg["lr_decay"]` toggle
   (default `True`); `current_lr` now logged every update. A late, still-large LR step is a
   plausible direct contributor to the late approx_kl spike.

**2. `ent_coef` 0.0 -> 0.001** (`configs/prism_dpmorl_only.yaml`) -- 0.0 fixed the *runaway*
   entropy problem from two entries back, and entropy did decrease cleanly for ~2500 updates
   -- but with literally no floor, nothing stops entropy from collapsing too low either, and
   its lowest point lined up with roughly where the run started destabilizing. A fully
   deterministic, over-confident policy is more brittle: less able to recover if something
   else (e.g. still-high LR) perturbs it. 0.001 (10x smaller than the original problematic
   0.01) is a reasoned starting point for a small floor, explicitly not a precisely calibrated
   value -- may need further tuning based on what this run shows.

**3. Frozen normalization range** (`prism/morl/utility_functions.py`) -- the "still-unfixed"
   item flagged in the two prior entries. `_min_val`/`_max_val` previously updated on every
   single `forward()` call, forever, for the entire run: the same raw `z` value gets
   normalised differently depending on when during training it's evaluated, since the range
   it's divided by keeps growing. A moving target the network never gets to settle against,
   not just a warmup-period effect -- and one that could plausibly introduce an uneven jolt
   at any point (any new record max shifts it), including late in a run when the policy is
   otherwise stabilizing. New `normalization_warmup_calls` param (default 50,000 -- roughly
   165-330 episodes, enough to get a representative range without leaving it open for most of
   a run) and a `_calls_seen` buffer (survives checkpoint save/load); `_update_running_stats`
   now no-ops once the threshold is reached.

### Verification

- `python -m pytest tests/` -- 58/58 passing (56 previous + 2 new: normalization freezes at
  the exact threshold and survives a state_dict round-trip).
- Offline check (no GPU/nuPlan): forced calls past the warmup threshold, confirmed
  `_calls_seen` and `_max_val` both stop changing exactly at the configured limit.
- Trainer smoke test (fake CaRL policy + fake env): confirmed `current_lr` starts at the
  configured value and decays monotonically to near-zero by the final update.
- `scripts/analyze_metrics.py` updated to include `current_lr` in the directional-flags
  section, so all three fixes have some visibility even bundled into one run.

### Outstanding

- [ ] Restart with all three fixes in place (plus everything from the prior entries). Given
      they're bundled: if this run is healthy for the full 5000 updates, we won't know which
      fix(es) mattered without follow-up runs reverting one at a time -- acceptable given the
      time cost of isolating first, but worth remembering before overclaiming "X fixed it" in
      any writeup.
- [ ] If late-training instability recurs despite all three: reconsider the `r_progress`
      multiplicative-AND-structure question from two entries back -- it was set aside in favor
      of this broader fix, not ruled out.
- [ ] `ent_coef=0.001` and `normalization_warmup_calls=50000` are both reasoned starting
      points, not calibrated values -- revisit based on what this run shows.
- [ ] Everything else from prior entries' Outstanding sections still applies.

---

## 2026-08-29 (cont.) — `scripts/inspect_utility_functions.py`: is divergence in the weights, not just the behavior?

### Motivation

Asked to judge personalization directly from the two policies' utility-function WEIGHTS
rather than inferring it from z_T trajectories, which are confounded by training stability,
exploration noise, and episode randomness -- exactly the things the last several entries have
been fighting. Worth noting explicitly: in "preferences" mode (what every run so far has
used), the utility function's neural network weights are never updated during Stage 2 at all
-- `run_stage2()`'s optimizer is built only from `agent.trainable_parameters()`, and every
call to `utility_fn` from `PRISMEnv.step()` goes through `as_callable()`, which wraps it in
`torch.no_grad()`. So the utility functions are fixed at whatever
`init_utility_functions_from_preferences()` set them to at construction, and checking their
weights is checking whether that ONE-TIME construction actually took hold -- independent of
anything that happened during RL training.

### Change

New `scripts/inspect_utility_functions.py` -- loads saved `utility_fn_{k}.pth` checkpoints
(needs the `prism` package + torch, run from the repo root in the training conda env, unlike
the stdlib-only `analyze_metrics.py`) and reports: whether `_pref_weights` matches the
declared preference; `fc_in.weight`'s per-dimension mean magnitude (which dimension the first
layer emphasizes); and a direct sensitivity test -- hold `z` fixed, boost one dimension by
+1.0 at a time, check whether each checkpoint's OWN preferred dimension produces the largest
utility gain. Preferences are inferred from checkpoint position using the standard K<=4
convention (`scripts/train.py`'s `_get_preference_vectors`) or can be given explicitly via
`path:w1,w2,w3,w4`.

Two things caught while building and smoke-testing this against locally-constructed
checkpoints (not yet the user's real ones):
- The existing checkpoint the user has predates the normalization-freeze fix (this session's
  earlier entry) and won't have the `_calls_seen` buffer -- `load_state_dict` now uses
  `strict=False` so this doesn't crash, with a printed note when keys are missing.
- The sensitivity test's own sequential `fn()` calls were themselves shifting
  `_min_val`/`_max_val` between the base and boosted evaluations on a checkpoint with
  `_calls_seen` below the freeze threshold -- the exact moving-target problem this whole
  investigation started from, now self-inflicted inside the diagnostic tool meant to check
  for it. Fixed by pinning the normalization range (freezing `_calls_seen` to the warmup
  threshold, or falling back to a fixed `[0, 50]` range if the checkpoint's stats were still
  `inf`/`-inf`, i.e. never saw a real value) immediately after loading, before running the
  test -- confirmed via a before/after smoke-test comparison that this removes spurious
  negative "utility gain" values the unpinned version produced.

### Verification

- `python -m pytest tests/` -- 58/58 passing (no test-suite code touched).
- Smoke-tested against two locally-constructed (not the user's real) checkpoints built via
  `init_utility_functions_from_preferences`: `_pref_weights` correctly matched the declared
  preference for both; the sensitivity test correctly showed each checkpoint's own preferred
  dimension producing the largest gain, with the pinning fix removing spurious negative gains
  present in an earlier, unpinned version of the same test.
- Not yet run against the user's actual saved checkpoints from the completed 5000-update run
  -- next step.

### Outstanding

- [ ] Run against the real checkpoints:
      `python scripts/inspect_utility_functions.py runs/dpmorl_only/prism_dpmorl_only_001/stage1/utility_fn_0.pth runs/dpmorl_only/prism_dpmorl_only_001/stage1/utility_fn_1.pth`
- [ ] If both show their own preferred dimension winning: divergence exists at the utility-
      function level -- any remaining lack of z_T divergence in behavior is downstream (actor
      not exploiting the signal, or training instability), not because the utility functions
      themselves are the same.
- [ ] If not: a real problem at the utility-function level specifically, worth investigating
      before anything else (e.g. whether `max_weight`/preference-biasing at construction is
      actually producing enough separation, independent of training).

---

## 2026-08-31 — `r_progress` simplification (drop `r_accel`) + `r_lane` direction reverted (self-correction)

### Motivation

Real checkpoints confirmed via `inspect_utility_functions.py` that both K=2 utility functions
are correctly preference-differentiated (own-dimension-highest for both), so the earlier
entries' remaining open question -- policy_1 (progress-preferring) training much worse than
policy_0 despite no instability -- traces to `r_progress` itself, not the utility functions or
PPO mechanics. `reward_advantage_std` measured ~4x smaller for policy_1 than policy_0
throughout the run (0.26-0.30 vs 0.99-1.18), which explains both the flat/declining z_progress
trend and why the same fixed `ent_coef` failed to hold entropy down for policy_1 specifically
(a weaker reward-driven "confidence" signal loses the tug-of-war against the entropy bonus more
easily).

Root cause: `r_accel = 1 - exp(-|a_ego|/gamma_a)` rewards acceleration *magnitude*
unconditionally, with no notion of whether a speed correction is actually needed. This
structurally punishes ideal steady-state cruising at `v_des` -- exactly the behavior `r_speed`
already fully rewards -- and directly fights `r_comfort`'s jerk-minimization objective. Since
`r_progress = r_speed * r_accel * (...)` is a multiplicative AND, a policy driving well (at
`v_des`, low jerk) was structurally capped near `r_progress`'s floor by `r_accel`, leaving very
little signal for the utility function to differentiate on -- exactly the small,
low-variance reward-advantage measured above.

Separately, self-correction: the 2026-08-27 entry ("Two confirmed style-reward bugs") inverted
`r_lane` to `1.0 - lane_index/(n_lanes-1)`, based on an in-code comment claiming "leftmost
lane = 0, rightmost = N_lanes-1". That comment was itself wrong. Re-deriving `_lane_position()`
(`prism/env/nuplan_env.py`) from scratch: it sorts candidate lanes ascending by
`-sin(heading)*dx + cos(heading)*dy`, the projection onto the LEFT-perpendicular direction from
ego heading (verified with a worked example: heading=0/facing east, a lane to the north at
dy=+5 gives a positive value, and north is correctly "left" when facing east under the standard
right-handed, counterclockwise-positive heading convention -- the same `cos_h`/`sin_h`
decomposition is used identically in `safety_cost.py`'s blind-spot check, confirming this
convention is consistent across the codebase). Ascending sort means the most-negative
(rightmost) lane lands at index 0 and the most-positive (leftmost) lane lands at the highest
index -- i.e. `lane_index=0` is the RIGHTMOST/slowest lane and `lane_index=n_lanes-1` is the
LEFTMOST/fastest lane. This exactly matches the paper's stated convention (`docs/prism_paper_v2.tex`,
eq. progress: "lane_index is 0 for the rightmost (slowest) lane and N-1 for the leftmost
(fastest) lane"), meaning the 2026-08-27 inversion was wrong and is reverted here. Lesson: that
earlier fix trusted a misleading in-code comment instead of independently re-deriving the
geometry -- the comment has been removed rather than corrected, to avoid the same failure mode
recurring.

### Change

`prism/env/rewards.py`:
- `compute_progress()`: removed the `a_ego`/`gamma_a` parameters and the `r_accel` term
  entirely. New formula: `r_progress = r_speed * (0.5 + 0.5 * r_lane)`. Assertiveness (closing
  a speed gap quickly) doesn't need a separate term -- a policy slow to close a gap accumulates
  more low-`r_speed` timesteps, which the cumulative `z_t`/`R_t` machinery already penalizes
  over the course of an episode.
- `r_lane` reverted to `lane_index / (n_lanes - 1)` (no `1.0 -` inversion), with the misleading
  comment replaced by the correct derivation.
- `compute_style_rewards()`: dropped the now-unused `a_ego` parameter and the `gamma_a` lookup.

`prism/env/nuplan_env.py`: dropped `a_ego=a_lon` from the `compute_style_rewards()` call
(`a_lon` itself is still used for jerk computation, unrelated to this change).

`compute_hyperparams.py`: dropped `a_ego=float(a_lon[i])` from its own `compute_style_rewards()`
call in the bootstrap-scaling rollout loop (this call site would otherwise have raised
`TypeError` on the next `make hyperparams-mini`/`make hyperparams` run). `gamma_a` is still
computed and written to `hyperparams.json` for now (harmless, simply unread by `rewards.py`
going forward) -- not touched, out of scope for this change.

`CLAUDE.md`: updated the Progress equation section to match (formula, removed `r_accel` line,
added an explicit note on the `lane_index` convention and why it must be re-derived from
`_lane_position()`'s geometry, not assumed from a comment).

`tests/test_rewards.py`: updated `TestProgress` for the new `compute_progress()` signature;
`test_single_lane_neutral_r_lane` reverted to assert `r2_right < r1 < r2_left` (rightmost scores
lowest); `test_at_desired_speed_no_speed_penalty` updated to drop the `r_accel` term from its
expected value; `test_zero_acceleration_reduces_progress` removed (no longer applicable, since
acceleration no longer affects `r_progress` at all by design); added
`test_leftmost_lane_scores_higher_than_rightmost` as an explicit direction-convention regression
check. Updated the three `TestStyleRewardVector` call sites to drop `a_ego`.

### Verification

- `python -m pytest tests/` -- 58/58 passing.
- `python -c "import ast; ast.parse(open('compute_hyperparams.py').read())"` -- syntax OK.
- Not yet run against a real DPMORL-only training run -- next step, before treating the
  policy_1 issue as resolved or using these checkpoints for anything downstream (see
  Outstanding).

### Outstanding

- [ ] Re-run `make train-dpmorl-only-mini-parallel` (or equivalent) with this fix and confirm
      both K=2 policies now show healthy, comparable `reward_advantage_std` and improving z_T
      trends on their own preferred dimension -- the diagnosis above is well-evidenced but not
      yet empirically confirmed under the corrected formula.
- [ ] `docs/prism_paper_v2.tex`: Eq. (progress) and the "Acceleration component" subsection
      (Eq. 7, its `gamma_a` calibration text, its assertiveness citation) need a matching
      revision to drop the acceleration term -- not done as part of this change (code-only;
      paper prose left for the user to revise in their own voice). The lane-choice section
      (Eq. 8) already matches the code as reverted here and needs no change.
- [ ] Only use checkpoints from a run trained under this corrected formula as a warm-start basis
      for the planned indicator-costs (safety) experiment -- the existing saved checkpoints from
      the already-completed 5000-update run were trained under the old, buggy `r_progress`
      formula and are not good warm-start candidates for that reason.
- [ ] Everything else from prior entries' Outstanding sections still applies.

---

## 2026-09-01 — Git commit logging, ent_coef decay, and a redundant utility_fn call fix

### Motivation

Full re-audit of the DPMORL training pipeline (utility function, `PRISMEnv`'s z_t/e_t/R_t
plumbing, GAE/PPO update, entropy head) requested after several rounds of narrow, symptom-by-
symptom fixes started feeling slow. Three real, independently-fixable issues found; a fourth
(the utility function's neural pathway being ~99% dominated by its linear regularization term,
since Stage 1 `"preferences"` mode never gradient-trains the network -- see the 2026-08-27
"root cause found" entry) was re-confirmed but deliberately left as-is pending a decision on
whether to revisit deferring `"diversity"` mode Stage 1 training.

### Change 1: git commit hash in the metrics config record

Immediately motivated by a real mix-up this session: confirming which run reflected the
`r_progress`/`r_lane` fix (previous entry) required manually cross-referencing a `git pull`
terminal timestamp against the push time, because `MetricsLogger` appends rather than
overwrites (see `metrics_logger.py`), so a `policy_k_metrics.jsonl` left over from an un-cleared
output dir can silently mix runs from different code versions with no way to tell which is
which after the fact. `prism/morl/dpmorl_trainer.py` now captures `git rev-parse --short HEAD`
once at import time and logs it in every policy's `config` record; `scripts/analyze_metrics.py`
surfaces it in its printed config line.

### Change 2: `ent_coef` linear decay to 0 (mirrors the existing `learning_rate` decay)

`action_heads.py`'s own docstring already diagnoses the mechanism behind the entropy-runaway
family of bugs this branch has repeatedly hit: PPO's gradient on `log_std` is "structurally
close to zero" (per-minibatch advantage normalization + ratio~1 make it near-self-cancelling),
while `-ent_coef * entropy` is a constant, unopposed pull toward higher entropy. `learning_rate`
already decays linearly to 0 over a run (2026-08-29 bundled-fix entry), but `ent_coef` never
did -- it was read fresh from the static config dict every update, so even the reduced
`ent_coef=0.001` guaranteed the upward pull never actually disappeared, just slowed down. This
is consistent with the late-training entropy uptick seen in policy_0's most recent run (after
the `r_progress`/`r_lane` fix) despite `ent_coef` already having been reduced twice this
project. `DPMORLTrainer` now decays `ent_coef` linearly to 0 over `n_updates`, same
`frac = 1.0 - update/n_updates` pattern as the LR decay, controlled by a new `ent_coef_decay`
config key (default `True`). `_ppo_update()` now takes `ent_coef` as a parameter (this update's
decayed value) instead of reading `self.cfg` directly, since `self.cfg` only holds the initial
value. Logged per-update as `current_ent_coef`, and `ent_coef_decay` is logged in the config
record.

### Change 3: redundant utility-function forward pass per env step

`PRISMEnv.step()` computed `R_t = gamma^{-t} * [f(z_{t+1}) - f(z_t)]` via two separate calls to
the utility function every step -- `f_next = f(z_{t+1})` and `f_curr = f(z_t)`. But `f(z_t)` is
the exact same value already computed as `f_next` on the *previous* step (`self._zt` doesn't
change except via `step()`), never cached. Fixed: `PRISMEnv` now caches `self._f_zt` (computed
once at `reset()` for `f(z_0)`, then updated to `f_next` at the end of each `step()`) and reuses
it as the next step's `f_curr`, halving the utility function's per-step call count. Verified
offline that this doesn't change the telescoping identity `sum(gamma^t * R_t) == f(z_T) - f(z_0)`
once the normalization range is frozen (post-warmup) -- confirmed exact equality (diff=0) for
both the old and cached call patterns in a synthetic 20-step rollout; the two patterns only
disagree during the warmup phase itself, because the normalization range (and therefore `f`) is
a genuinely moving target then regardless of caching -- a pre-existing, already-accepted property
of the warmup design, not something this change introduces or worsens.

Side effect: `UtilityFunction.__init__`'s `normalization_warmup_calls` docstring assumed ~1
`forward()` call per env step ("~50k calls is roughly 165-330 episodes at ~150-300
calls/episode"), which was actually wrong before this fix (2 calls/step means the real freeze
point was ~82-165 episodes, half what the comment claimed) and is now accurate again. Comment
updated to note this explicitly.

### Change 4 (not made): utility function neural pathway still ~99% linear-term-dominated

Re-confirmed, not re-litigated: in `"preferences"` mode (used by every run so far),
`init_utility_functions_from_preferences()` constructs each `UtilityFunction` once and biases
`fc_in`'s weights toward the preference vector, but the network's parameters are **never**
included in Stage 2's optimizer (`run_stage2()` builds it only from
`agent.trainable_parameters()`), so the monotone-NN pathway never receives a gradient update --
its behavior is whatever the one-time construction produced, forever. Combined with the
already-documented desensitization (`utility_functions.py:136-148`: the neural pathway operates
on `z` normalized against the full episode-cumulative range, so a single step's change is
invisible to it), what's actually running is close to linear preference-weighted scalarization
with a monotone-NN wrapper, not DPMORL's actual non-linear, diversity-trained utility functions.
This is the same gap flagged in the 2026-08-27 entry and deferred by explicit user choice ("we
can do that later"); resurfaced here because it's likely the single largest remaining lever for
DPMORL personalization quality, and worth an explicit decision rather than continuing to defer
implicitly.

### Verification

- `python -m pytest tests/` -- 58/58 passing (no test-suite code touched by this entry).
- `python -c "import ast; ..."` syntax check on all three touched files.
- Offline synthetic-rollout script (see Change 3) confirming the caching fix preserves the R_t
  telescoping identity post-warmup.
- Not yet run against a real training run -- next step.

### Outstanding

- [ ] Run DPMORL-only again with `ent_coef_decay` active and confirm the late-training entropy
      upticks (seen in both the original entropy-runaway case and policy_0's most recent run)
      are gone or substantially reduced.
- [ ] Decide whether to revisit Stage 1 `"diversity"` mode (Change 4) now, given how long
      `"preferences"`-mode iteration has taken to get right -- needs the still-unfixed
      unbounded/unregularized diversity loss (2026-08-27 area) addressed first if so.
- [ ] Everything else from prior entries' Outstanding sections still applies.

---

## 2026-09-01 (cont.) — UtilityFunction output self-normalisation, ported from the real DPMORL repo

### Motivation

User has the actual DPMORL reference implementation checked out locally (`MI3/DPMORL/`, the code
behind the NeurIPS 2023 paper this project builds on) and asked to reconcile our implementation
against it, plus a review of the paper draft against both. Full findings below; this entry covers
only the one change made so far (K stays <= 5 for now, so the other findings are deferred).

### What the reference repo actually does (not what the paper's prose implies)

Read `main_generate_utility.py`, `utility_function_parameterized.py`,
`utility_function_programmed.py`, and `utility_env_wrapper.py`. Key findings:

- For `policy_idx < reward_dim + 1` (our exact K=4/5 regime with `reward_dim=4`),
  `main_policy.py`'s `get_utility()` uses hand-coded closed-form functions (plain mean, and
  "weight one dimension 4x then renormalise the sum") -- **not** gradient-trained diversity
  functions. Learned diversity training (`main_generate_utility.py`) is only used to generate
  *additional* utility functions beyond this "obvious" corner set. This means our `"preferences"`
  mode is methodologically consistent with the reference for K<=5 -- not a shortcut, contrary to
  what this session's earlier entries implied.
- The reference's utility function **self-normalises its output**: every `forward()` call
  additionally evaluates the same network at the normalised domain's corners (all-0s, all-1s) and
  rescales the batch's raw output via `(util - min_util) / (max_util - min_util)` before use. Our
  `UtilityFunction.forward()` only ever normalised the *input*, never the *output* -- this is very
  likely the actual mechanism behind two things found earlier this session: the linear
  regularisation term dominating >99% of the observed signal, and the (separate, currently unused)
  diversity-training loss growing unboundedly in an offline test. Neither implementation clamps
  the output layer's bias (`make_monotone()` only clamps weights, in both repos) -- the reference
  doesn't need to, because the corner-based rescaling makes a constant bias shift cancel out of the
  ratio almost entirely. Ours had no such correction, so bias (and general output scale) could
  drift freely.
- If real diversity training is ever built (deferred -- see prior entry's Change 4/Outstanding),
  the reference's actual loss is a **sequential/greedy soft-min (logsumexp) loss** against all
  previously-fixed functions, not our dormant `train_utility_functions_stage1()`'s joint pairwise
  sum-of-squares trained simultaneously -- soft-min naturally saturates once separation is
  achieved, which is likely why the reference doesn't hit the unbounded-loss problem we found.
  Also: the paper's Eq. 30 gradient-difference term (`J^grad`) is **not actually implemented** in
  the reference code -- it's hardcoded to zero (`dist_grad = torch.zeros_like(...)`, loss weight
  `0.0`) and approximated instead by an angle/"degree" term. Faithfully porting this later means
  porting what the code does, not what the paper's equation says.
- One place NOT ported: the reference's env wrapper drops the `gamma^t`/`gamma^{-t}` factors
  entirely from the z_t/utility-difference pipeline (`zt_next = self.zt + reward`, undiscounted;
  `new_reward = f(zt_next) - f(zt)`, no `gamma^{-t}`) -- `self.gamma` is set but never used in that
  file. Our implementation and the paper's Eq. 31-32 both include these factors, which is what
  makes the discounted sum telescope exactly to `f(z_T) - f(z_0)` (DPMORL's Theorem 2, the actual
  proof this construction relies on). Kept our version; the reference's appears to be a practical
  simplification that doesn't reflect the theorem as cleanly.

### Change: output self-normalisation ported into `UtilityFunction.forward()`

`prism/morl/utility_functions.py`: every `forward()` call now additionally batches through the
normalised domain's corners (`zeros(reward_dim)`, `ones(reward_dim)`) in the same forward pass,
and rescales the raw network output via `(raw_util - min_util) / (max_util - min_util + 1e-6)`
before combining with the linear term. Monotonicity (all weights clamped >= 0, including through
the triple-clamp activation) guarantees all-0s is the network's global minimum and all-1s its
global maximum over the normalised domain, so this rescaling is exact, not approximate.

Deliberately NOT ported: the reference's "keep_scale" option (forcing all dimensions to share one
normalisation window) -- our 4 style dimensions have genuinely different natural scales (e.g.
z_progress ~13-14 vs z_comfort ~70-80 in a real run), so sharing a window would compress the
already-smaller dimension further, working against the goal here. Also not ported: the reference's
`scale_back`/`*=2` post-processing that rescales the final output back to roughly the raw z scale
-- our downstream pipeline (PPO's per-minibatch advantage normalisation, `vf_coef`, learning rate)
was tuned around the utility function's existing small output scale; changing that scale
substantially would need its own re-tuning pass, and isn't needed for the core fix (bounding /
centering the network's contribution against its own achievable range, not matching the
reference's absolute output magnitude).

The linear term was also changed from an additive add-on (`net_out + lamda * linear_term`, using
raw un-normalised `z`) to a convex blend (`(1-lamda)*net_out + lamda*linear_term`, both terms now
using the same normalised `x` and both landing in ~[0,1]) -- previously the linear term's raw
magnitude (up to ~4, from `lamda=0.05` times a z-component sum up to ~80) swamped the network's
un-normalised output regardless of `lamda`'s intended weighting; now both terms are on a
comparable scale, so `lamda` actually controls their relative contribution as intended.

### Verification

- `python -m pytest tests/` -- 58/58 passing (including `TestMonotonicity` and
  `test_two_policies_value_their_own_dimension_more`, unmodified -- both still pass against the
  new forward() implementation).
- Offline diagnostic (isolating `net_out` alone by forcing `lamda=0`): a comfort-preferring
  policy's utility gain from a +0.7 boost to `z_comfort` (0.00358) is now ~3.4x its gain from the
  same boost to `z_progress` (0.00106) **from the neural pathway alone**, confirming it's no
  longer structurally desensitized -- previously this pathway's contribution was found to be
  negligible (<1% of the observed signal, all differentiation coming from the linear term).
- Offline telescoping-identity check (same script as the previous entry's Change 3 verification):
  `sum(gamma^t * R_t) == f(z_T) - f(z_0)` still holds exactly (diff=0.00e+00) post-warmup with the
  new forward() implementation.
- Not yet run against a real training run -- next step.

### Outstanding

- [ ] Run DPMORL-only again with this change (plus the ent_coef decay and call-caching fixes from
      the prior entry) and confirm both K=2 policies show meaningfully differentiated
      `reward_advantage_std` and cleanly improving z_T trends now that the neural pathway
      contributes real, non-desensitized signal.
- [ ] Paper reconciliation needed (not yet done, larger scope, discussed but not actioned this
      entry): Section IV-B/C + Algorithm 1 describe the abandoned Lagrangian CVaR mechanism
      (Eq. 33, 36), not the actual v2 dual-critic design; Algorithm 1 line 4 ("Initialize policy
      from CaRL checkpoint") contradicts CLAUDE.md's Design Decision #1 (train from scratch); the
      utility function's domain is described as `[0,1]^4` (Section IV-A) but `Z^pi` is an
      unbounded discounted sum (Eq. 14) -- needs precise language, not a design change. Also
      worth noting: the planned `MORL-Linear` ablation baseline (Section V-C) is close in
      mechanism to what `"preferences"`-mode PRISM has actually been running -- the K<=5
      finding above (reference repo also uses hand-coded functions at this K) mitigates this
      concern somewhat, but it's still worth being precise in the paper about what's actually
      being compared.
- [ ] Everything else from prior entries' Outstanding sections still applies.

---

## 2026-09-02 — `UtilityFunction` output scale collapsed after the normalisation fix; added `output_scale`

### Motivation

Real run (git_commit=a57ec50, i.e. the r_progress/r_lane fix + ent_coef decay + call-caching +
output self-normalisation all together, stopped at update ~3600/5000) showed a regression, not
the expected improvement: policy_0's entropy went flat (~1.7-1.8 the whole run, not decreasing)
and policy_1's entropy climbed from ~1.9 to ~2.9-3.0 -- worse than either prior run, including the
one *before* ent_coef decay was added. `z_T` trends confirmed this wasn't cosmetic: policy_1
declined on all four style dimensions, and the progress-divergence that had appeared in the
previous run (policy_1 beating policy_0 on `z_progress`) collapsed to a near-tie (11.53 vs 11.59).

Root cause, confirmed both from the real run's data and an offline simulated-trajectory
comparison (old formula vs. new formula, same network, same z trajectory): the previous entry's
output-normalisation fix correctly bounded `f(z)` to `~[0,1]`, but that also removed an
UNINTENTIONAL scale inflation the old, unbounded output had (an unconstrained output bias --
neither implementation clamps it, see previous entry -- could push raw output arbitrarily large).
Measured: per-step utility deltas are ~16-17x smaller after the fix (offline: 0.087 -> 0.005 mean
absolute delta on a realistic simulated trajectory; the real run's `reward_advantage_std` dropped
by a consistent ~15-20x for both policies). `ent_coef`, `vf_coef`, and `learning_rate` were all
tuned against the old, larger (if unintentional) scale. Since the run was stopped at ~72% of its
schedule, `ent_coef` had only decayed to ~28% of its initial value by then -- so even with decay
active, the entropy-bonus-to-reward-signal ratio got roughly 5x MORE entropy-dominant than before
(reward shrank ~17x, ent_coef only shrank ~3.6x by that point), which is exactly why entropy
failed to decrease instead of improving further.

### Change

`prism/morl/utility_functions.py`: added `output_scale: float = 16.0` to `UtilityFunction.__init__`,
applied as a final multiplier on `out` after the corner-normalisation + lamda blend (so it
preserves the now-correct relative shape/differentiation from the previous entry, and only
restores absolute magnitude). 16.0 is a reasoned value from the measured ~16-17x ratio, not a
precisely calibrated one -- may need further tuning, same as `ent_coef`'s own history in this
project.

### Verification

- `python -m pytest tests/` -- 58/58 passing.
- Offline: re-ran the same old-vs-new simulated-trajectory comparison from the previous entry with
  `output_scale=16.0` applied -- ratio is now ~1.01-1.09x (old and new are essentially the same
  scale), down from ~16-17x before this fix.
- Offline: telescoping identity `sum(gamma^t * R_t) == f(z_T) - f(z_0)` still holds exactly
  (diff=0.00e+00) with the scale factor applied (a constant multiplier on `f` doesn't affect
  telescoping, confirmed rather than assumed).
- Not yet run against a real training run -- next step.

### Outstanding

- [ ] Run DPMORL-only again with `output_scale=16.0` and confirm entropy decreases cleanly for
      both policies through the full 5000-update schedule this time (this is now the 3rd distinct
      mechanism targeting this class of problem -- log_std clamp, ent_coef floor/decay, and now
      reward-scale calibration -- each of which was individually well-justified and empirically
      verified offline, but the interaction between them wasn't fully anticipated until seeing
      real training data. Worth watching closely rather than assuming this closes it out.)
- [ ] If entropy is still not cleanly decreasing with output_scale=16.0, the next lever is
      `output_scale` itself (try a somewhat larger value) rather than another change to `ent_coef`
      -- the analysis above suggests scale mismatch, not an incorrectly-shaped decay schedule, was
      the actual problem.
- [ ] Everything else from prior entries' Outstanding sections still applies.

---

## 2026-09-02 (cont.) — Three new diagnostics for the K-policy-divergence investigation

### Motivation

output_scale=16.0 (previous entry) fixed policy_0 (cleanest entropy decrease seen yet) but left
policy_1 with a flat-then-plateauing entropy (~2.0-2.1) and a progress-divergence margin that
shrank to a near-tie (14.322 vs 14.217). An offline check of whether comfort/progress reward
SHAPES intrinsically differ in per-step variance (synthetic input distributions) did not confirm
that hypothesis (variance ratio ~0.97x, i.e. roughly equal) -- but it did reveal the assumed input
distributions were unrealistic (real z_progress/z_comfort ratios imply mean r_progress~0.16 vs
mean r_comfort~0.89, the opposite of what the synthetic sweep produced), meaning reasoning from
synthetic data isn't reliable here. User (via a separate consultation) proposed two decisive
tests instead of further synthetic guessing, plus asked for direct visibility into reward
component spread. All three built this entry; none required another full training run to be
useful (Test 1 and the ablation both work with data/infra already in place; the component
logging needs one new run, but is cheap to add to whatever runs next).

### Test 1: `scripts/check_gradient_alignment.py`

Computes nabla f_0(z) and nabla f_1(z) via autograd at REAL z points -- every update's mean z_T,
pulled straight from the two policies' existing metrics.jsonl files (no new run needed) -- and
reports the distribution of cosine similarity between the two gradients. Near 1.0 means the two
(different-by-construction) utility functions point in nearly the same direction across the
region actually reached during training, which alone would explain weak divergence regardless of
PPO/environment/reward-formula correctness. Reuses `inspect_utility_functions.load_checkpoint()`
and `analyze_metrics.load_runs()` rather than duplicating their parsing/loading logic. Verified
against synthetic checkpoints + a synthetic metrics.jsonl (mean cosine ~0.81 for two utility
functions built from clearly-different-but-not-orthogonal preference vectors, as expected).

### Test 2: `LinearProjectionUtility` + `--utility_ablation linear_projection`

`prism/morl/utility_functions.py`: new `LinearProjectionUtility(reward_dim, dim)` -- f(z) = z[dim]
exactly, bypassing ALL learned/preference-biased machinery (no monotone network, no
normalisation, no linear-term blend). Gradients are one-hot and orthogonal everywhere in R^4 by
construction -- no saturation, no shared-direction confound possible, unlike the real utility
functions Test 1 checks. `scripts/train.py --utility_ablation linear_projection` bypasses Stage 1
entirely (`run_stage1()` short-circuits before the preferences/diversity dispatch) and constructs
one projection per policy with matching one-hot `w_k` (FiLM conditioning stays consistent with
what the utility function actually rewards). Cleanly separates two hypotheses: if K policies
diverge under this, the DPMORL/PPO machinery works and the bottleneck is specifically Stage 1
utility-function construction; if they still don't diverge, the problem is downstream (reward
formulas, environment) and Stage 1 isn't the bottleneck.

Side effect worth knowing (documented in the class docstring): because `PRISMEnv` updates
`z_{t+1} = z_t + gamma^t * r_vec`, `f(z_{t+1}) - f(z_t) = gamma^t * r_vec[dim]_t` exactly when
`f(z) = z[dim]`, so `R_t = gamma^{-t} * (...) = r_vec[dim]_t` -- this ablation is simultaneously a
test of "does PPO on the RAW per-step reward component alone, with no DPMORL z_t/utility
wrapping at all, produce divergence." Requires `n_policies <= reward_dim` (asserted) -- no
sensible K+1-th projection. `inspect_utility_functions.py` won't produce a meaningful report
against checkpoints from this ablation (it assumes the real `UtilityFunction`'s buffers) -- not
needed anyway, since this ablation's behavior is fully known by construction.

### Reward-component logging + `scripts/analyze_reward_spread.py`

Refactored `compute_progress()`/`compute_lateral_discipline()` in `prism/env/rewards.py` to share
new `_progress_components()`/`_lateral_components()` helpers with a new
`compute_style_rewards_verbose()` (returns `(r_vec, components_dict)`, verified byte-identical
`r_vec` to `compute_style_rewards()`) -- one implementation of each formula, not two that could
drift apart. `PRISMRewardBuilder` gets a new `log_components` flag (off by default); when on,
`build_reward()` exposes `info["reward_components"]` (`r_speed`, `r_lane`, `r_dev`, `r_heading`,
`jerk_lon`, `jerk_lat`, `ttc` -- `ttc` is `None`, not `inf`, when there's no lead vehicle, since
`json.dumps` can't serialise `inf`/`nan`). `DPMORLTrainer` accumulates these per update and logs
`rc_<name>_mean`/`rc_<name>_std` in the metrics record, alongside the existing `z_stats`/
`outcome_stats` pattern. Wired through `make_prism_env()` and `_build_env()` to a new
`scripts/train.py --log_reward_components` flag / `cfg["log_reward_components"]` key.

`scripts/analyze_reward_spread.py` (stdlib-only, matches `analyze_metrics.py`'s portability)
reports each component's mean/std over a run and flags two saturation directions: pinned near its
ceiling with low std (little room left to differentiate -- the "reward is basically maxed out"
case), or pinned near its floor with low std (the more concerning case in practice -- usually
means the policy is persistently far from what the component wants, e.g. consistently well below
desired speed, not that the reward is well-tuned and topped out). Verified against synthetic
metrics.jsonl data for both saturation directions and a healthy-spread case; correctly produces
no flag for a merely-low-but-not-pinned value (mean=0.16, floor-room=0.16, above the 0.05
threshold).

### Verification

- `python -m pytest tests/` -- 58/58 passing (no test-suite code touched; the rewards.py refactor
  is behavior-preserving, confirmed by the existing suite still passing unmodified, plus a direct
  check that `compute_style_rewards_verbose()`'s `r_vec` is byte-identical to
  `compute_style_rewards()`'s for the same inputs).
- `run_stage1()` with the linear_projection ablation tested directly (offline, no nuPlan/CaRL
  needed): produces `f(z)=z[dim]` exactly for both policies, checkpoint save/load roundtrips
  correctly, and the `n_policies > reward_dim` case raises the expected `AssertionError`.
- `check_gradient_alignment.py` and `analyze_reward_spread.py` both run end-to-end against
  synthetic checkpoints/metrics.jsonl data and produce sensible, correctly-triggering output.
- `PRISMRewardBuilder(log_components=True)` constructs without error (offline, no simulation
  needed for the constructor itself).
- Not yet run against real data for Test 1 (needs the user's actual checkpoints + metrics.jsonl)
  or a real training run with `--log_reward_components` -- both are ready to use now.

### Outstanding

- [ ] Run Test 1 against the real K=2 checkpoints + metrics.jsonl from the most recent completed
      run (git_commit=3a6c771) -- decisive, no new training needed.
- [ ] Launch a `--utility_ablation linear_projection` run (K=2, same config otherwise) --
      decisive, cleanly separates "Stage 1 is the bottleneck" from "something downstream is."
- [ ] Launch (or combine with the above) a run with `--log_reward_components` and check
      `analyze_reward_spread.py`'s output, especially for `r_speed` given the real z_progress/
      z_comfort ratio already suggests it may be floor-saturated, not just low.
- [ ] Everything else from prior entries' Outstanding sections still applies.

---

## 2026-09-02 (cont.) — Test 1 result; `--skip_stage1` gap for the linear_projection ablation

### Test 1 result: root cause found, no training needed

Ran `check_gradient_alignment.py` against the real K=2 checkpoints + metrics.jsonl (git_commit=
3a6c771). Result: cosine similarity between nabla f_0(z) and nabla f_1(z) is essentially CONSTANT
across all 2000 real z points sampled -- mean=0.8069, std=0.0054 (min=0.7978, max=0.8175). That
tight a clustering, independent of where in the reachable region z falls, points away from
"saturation at specific points" and toward "the two functions are just not different enough,
globally, by construction."

Confirmed algebraically: the K=2 generic-fallback preference vectors from `_get_preference_vectors()`
are `[0.4231, 0.1923, 0.1923, 0.1923]` (policy 0) and `[0.1923, 0.4231, 0.1923, 0.1923]` (policy 1).
Their cosine similarity, computed directly: 0.8164 -- matching the measured ~0.807 almost exactly
(the small gap is the residual, now-meaningful neural pathway's own contribution, on top of the
still-dominant linear term). The two vectors are structurally similar because the fallback formula
(`base = 1/reward_dim` for every dimension, `+0.4*(1-1/reward_dim)` added to ONLY the policy's own
dimension, then renormalised) leaves the OTHER `reward_dim - 1` dimensions identical between any
two policies -- for K=2, policies 0 and 1 share IDENTICAL weight (0.1923) on both the lateral and
spacing dimensions, and differ only in which ONE dimension gets the boost. That structurally caps
how different two K=2 preference vectors from this formula can ever be, regardless of anything
about PPO, the environment, or reward formulas.

This is a clean, load-bearing finding: the divergence weakness traces to Stage 1's preference-
vector CONSTRUCTION for non-curated K (the generic fallback, used for K=2; the curated K=4/K=5
lists don't have this problem, since they're hand-picked, not formula-generated). Not yet fixed --
candidate fix is a fallback formula that also SUPPRESSES weight on other dimensions when boosting
one, not just adds to one while leaving the rest untouched (e.g. redistribute the boost's cost
across the other dimensions rather than only growing the numerator), making the K=2 case closer in
spirit to the curated lists' more differentiated vectors. Needs discussion before implementing --
this affects the ALREADY-COMPLETED K=2 runs' interpretation (their divergence weakness has a
concrete, non-buggy explanation now) and needs to be weighed against just moving to a curated K=4
run instead, where this specific formula isn't used at all.

### Bug found via a real run: `--skip_stage1 --utility_fn_dir` didn't handle the ablation

User's attempt to launch `--utility_ablation linear_projection` crashed on an unrelated
`--cache_path` issue (ran the raw command by hand, without the Makefile's `--cache_path`/
`--nuplan_data_root`/`--output_dir`, which resolve from `lab.env`; the config file's own
`cache_path` default, `/data/prism_cache`, doesn't exist on this machine -- `/data/prism_mini_cache`,
which the Makefile always supplies, does). That crash happened before reaching Stage 2, but it
surfaced a real, separate bug in the process: `run_stage1()`'s ablation branch (previous entry)
was fixed, but the OTHER path that constructs utility functions --
`main()`'s `elif args.skip_stage1 and args.utility_fn_dir:` branch, used by the Makefile's
`*-mini-parallel` targets' per-policy concurrent processes -- unconditionally tried to
`UtilityFunction().load_state_dict(...)` against the ablation's saved checkpoint, which only ever
contains `LinearProjectionUtility`'s single dummy buffer. Would have raised a missing-keys error
on the very next step of the same launch. Fixed: this branch now also checks
`cfg["utility_ablation"]` and reconstructs `LinearProjectionUtility` directly (no state to load --
it's fully determined by `reward_dim`/`dim` alone), matching `run_stage1()`'s own branch exactly
(same one-hot `preference_vectors`).

Also added `UTILITY_ABLATION`/`LOG_REWARD_COMPONENTS` optional Makefile variables to
`train-dpmorl-only-mini-parallel`, threaded into both the `--stage1_only` call and every per-policy
parallel call, so the ablation/logging flags don't need to be hand-assembled against `lab.env`'s
resolved paths ever again:
```
make train-dpmorl-only-mini-parallel UTILITY_ABLATION=linear_projection LOG_REWARD_COMPONENTS=1
```

### Verification

- `python -m pytest tests/` -- 58/58 passing.
- Offline: reconstructed the `--skip_stage1 --utility_fn_dir` ablation branch's logic directly
  (stage1_only save -> reconstruct, matching what the Makefile's two-phase launch actually does)
  and confirmed `f_k(z) = z[k]` still holds exactly after the round trip.
- `make -n train-dpmorl-only-mini-parallel UTILITY_ABLATION=linear_projection LOG_REWARD_COMPONENTS=1`
  (dry run) confirms both flags land correctly in both the stage1_only call and the per-policy
  parallel calls; the same dry run with neither variable set is byte-identical to the pre-existing
  target (no regression).

### Outstanding

- [ ] Decide on a fix for the K=2 fallback preference-vector formula (see Test 1 finding above)
      before relying on any further K=2 "preferences"-mode results -- or switch straight to a
      curated K=4 run, which doesn't hit this formula at all.
- [ ] Re-launch the linear_projection ablation via
      `make train-dpmorl-only-mini-parallel UTILITY_ABLATION=linear_projection LOG_REWARD_COMPONENTS=1`
      now that both the cache-path issue (user error, not a bug) and the `--skip_stage1` gap
      (real bug, now fixed) are resolved.
- [ ] Everything else from prior entries' Outstanding sections still applies.

---

## 2026-09-02 (cont.) — Fixed the K=2 preference-vector concentration (Test 1's root cause)

### Change

`scripts/train.py`'s `_get_preference_vectors()` generic fallback (used for any K not covered by
the curated K=4/K=5 lists, i.e. our actual K=2 setup) previously derived each policy's vector as
`base=1/reward_dim` with `+=0.4*(1-base)` boost to its own dimension, then renormalised. That
produces an own:other concentration ratio of only ~2.2 (e.g. 0.4231/0.1923 for K=2) -- much weaker
than the curated lists' ratio of 3.67 (0.55/0.15). Ratio, not the post-renormalisation values, is
what determines cosine similarity between two policies' vectors (renormalising doesn't change a
vector's direction), so this directly explains the previous entry's Test 1 finding: real K=2
utility-function gradients had cosine similarity ~0.81, matching the OLD formula's own vectors'
cosine similarity (0.8164) almost exactly.

Fixed to use the same 0.55/0.15 concentration the curated lists already use, generalised to any
`reward_dim`/K via `low=0.15`, `high=1-(reward_dim-1)*low` (asserts `high > low`, i.e. sensible up
to `reward_dim<=6` for `low=0.15`). For `reward_dim=4`, this reproduces the curated K=4 list
exactly, which made the separate `n_policies==4` special case in `_get_preference_vectors()` pure
duplicate logic -- removed it, folding into the (now-correct) generic formula. The `n_policies==5`
case is NOT redundant and was kept: the generic formula's `k % reward_dim` indexing would wrap
policy index 4 back to dimension 0, duplicating policy 0's vector, rather than producing the
intended "balanced" `[0.25,0.25,0.25,0.25]` 5th vector. Also fixed the mirrored (but, per
`run_stage1()`'s actual call pattern, unreachable in practice) fallback inside
`init_utility_functions_from_preferences()` in `prism/morl/utility_functions.py`, since its own
docstring already documents it as matching `_get_preference_vectors()` -- kept in sync rather than
leaving a second, inconsistent copy of the same formula.

New K=2 cosine similarity: 0.5676 (down from 0.8164) -- a real, not cosmetic, improvement. Not
zero: two non-negative, fixed-sum, non-sparse preference vectors can't be fully orthogonal by
construction; that would need one-hot vectors (`LinearProjectionUtility`'s ablation), a bigger
design departure (a policy would then care nothing at all about its non-preferred dimensions,
rather than mostly-but-not-only).

Pre-existing, still-unfixed limitation noted but not touched: for `n_policies=3`, dimension 3 is
never emphasized (`k % reward_dim` only reaches 0,1,2 for k=0,1,2) -- not relevant to the current
K=2 setup, left as a known gap for if/when K=3 is ever used.

### Verification

- `python -m pytest tests/` -- 58/58 passing.
- `_get_preference_vectors(2, 4)` -> `[[0.55,0.15,0.15,0.15], [0.15,0.55,0.15,0.15]]`; measured
  cosine similarity 0.5676 (matches hand calculation).
- `_get_preference_vectors(4, 4)` and `_get_preference_vectors(5, 4)` -- unchanged output vs.
  before this fix (K=4 now via the generic path instead of a separate special case; K=5 still via
  its own kept special case) -- confirmed no regression for the two K values every prior run this
  session actually used.
- `init_utility_functions_from_preferences(n_policies=2, reward_dim=4)`'s resulting
  `_pref_weights` match `_get_preference_vectors(2, 4)` exactly -- the two paths stay in sync as
  the docstring claims.

### Outstanding

- [ ] Re-run K=2 DPMORL-only with this fix (git_commit will disambiguate automatically now) and
      check whether the entropy/divergence issues from the last several entries improve --
      **calibrated expectation, not a guarantee**: this fixes the diagnosed Stage-1 cause (weak
      preference-vector separation), but doesn't touch `r_progress`'s own low mean/possible
      floor-saturation (still-open question from the reward-component-spread diagnostic, not yet
      run) or the entropy-decay-vs-reward-scale interaction from two entries back -- those could
      still limit policy_1 specifically even with better-separated utility functions. Don't
      over-read a clean or a messy result in isolation from those other still-open threads.
- [ ] Everything else from prior entries' Outstanding sections still applies.
