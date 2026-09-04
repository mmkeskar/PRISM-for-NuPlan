"""
Unit tests for PRISM style reward functions.

All tests are pure Python / NumPy — no nuPlan or gym dependencies required.
Run with:
    cd PRISM && python -m pytest tests/test_rewards.py -v
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from prism.env.rewards import (
    compute_comfort,
    compute_lateral_discipline,
    compute_progress,
    compute_spacing,
    compute_style_rewards,
    compute_ttc,
)
from prism.utils.zt_normaliser import ZtNormaliser
from prism.curriculum.alpha_schedule import AlphaSchedule, compute_alpha


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def default_hp():
    # sigma_d (reward_scaling) and delta_d (floor_values) deliberately kept
    # DIFFERENT (0.2 vs 0.3, matching real compute_hyperparams.py output) --
    # equal values previously masked a wrong-key bug where
    # compute_style_rewards read delta_d and used it as sigma_d. See
    # CHANGES.md and TestStyleRewardVector.test_uses_reward_scaling_sigma_d_not_floor_delta_d.
    return {
        "reward_scaling": {
            "sigma_j_sq": 1.0,
            "beta": 0.5,
            "phi": 0.3,
            "tau": 2.0,
            "sigma_d": 0.2,
        },
        "floor_values": {"delta_d": 0.3, "delta_s": 0.2},
        "alpha_curriculum": {"alpha_start": 0.20, "alpha_end": 0.95},
    }


# ─────────────────────────────────────────────────────────────────────────────
# compute_comfort
# ─────────────────────────────────────────────────────────────────────────────

class TestComfort:

    def test_zero_jerk_returns_one(self):
        r = compute_comfort(j_lon=0.0, j_lat=0.0, sigma_j_sq=1.0)
        assert r == pytest.approx(1.0, abs=1e-8)

    def test_positive_jerk_less_than_one(self):
        r = compute_comfort(j_lon=1.0, j_lat=0.0, sigma_j_sq=1.0)
        assert 0.0 < r < 1.0

    def test_output_in_zero_one_interval(self):
        for j_lon in np.linspace(-5, 5, 11):
            for j_lat in np.linspace(-5, 5, 11):
                r = compute_comfort(j_lon, j_lat, sigma_j_sq=1.0)
                assert 0.0 < r <= 1.0, f"Out of (0,1] for j_lon={j_lon}, j_lat={j_lat}"

    def test_large_jerk_near_zero_but_positive(self):
        # j=10 m/s³ is already severe (braking hard); gives exp(-200) which is
        # non-zero in float64 (underflow only occurs around j≈95 with sigma=1).
        r = compute_comfort(j_lon=10.0, j_lat=10.0, sigma_j_sq=1.0)
        assert r > 0.0

    def test_lat_lon_symmetric(self):
        r1 = compute_comfort(j_lon=2.0, j_lat=0.0, sigma_j_sq=1.0)
        r2 = compute_comfort(j_lon=0.0, j_lat=2.0, sigma_j_sq=1.0)
        assert r1 == pytest.approx(r2, abs=1e-8)

    def test_sigma_scaling(self):
        # Larger sigma_j_sq → less penalisation → higher reward for same jerk
        r_small = compute_comfort(1.0, 1.0, sigma_j_sq=0.5)
        r_large = compute_comfort(1.0, 1.0, sigma_j_sq=2.0)
        assert r_large > r_small


# ─────────────────────────────────────────────────────────────────────────────
# compute_progress
# ─────────────────────────────────────────────────────────────────────────────

class TestProgress:

    def test_output_in_zero_one_interval(self):
        for v_ego in [0, 5, 10, 15]:
            for v_des in [5, 10, 15]:
                r = compute_progress(v_ego=v_ego, v_des=v_des, beta=0.5)
                assert 0.0 < r <= 1.0, (
                    f"Out of (0,1] for v_ego={v_ego}, v_des={v_des}"
                )

    def test_at_desired_speed_no_speed_penalty(self):
        # When v_ego == v_des, shortfall = 0, r_speed = 1, so r_progress = 1
        # exactly (the delta_v floor only bites when r_speed < 1).
        r = compute_progress(10, 10, beta=0.5)
        assert r == pytest.approx(1.0, rel=1e-6)

    def test_exceeding_desired_speed_no_penalty(self):
        # shortfall is clamped at 0 -- going faster than v_des is not
        # penalised (r_progress is a one-sided shortfall penalty).
        r_at = compute_progress(10, 10, beta=0.5)
        r_over = compute_progress(15, 10, beta=0.5)
        assert r_over == pytest.approx(r_at, rel=1e-6)

    def test_large_shortfall_approaches_floor(self):
        # v_ego=0 against a large v_des with a tight beta drives r_speed -> 0,
        # so r_progress -> delta_v (0.1), the numerical floor.
        r = compute_progress(v_ego=0.0, v_des=20.0, beta=0.05)
        assert r == pytest.approx(0.1, abs=1e-3)

    def test_larger_beta_is_more_forgiving(self):
        # Same shortfall, larger beta (softer regime target) -> higher reward.
        r_tight = compute_progress(v_ego=5.0, v_des=10.0, beta=0.2)
        r_soft = compute_progress(v_ego=5.0, v_des=10.0, beta=1.0)
        assert r_soft > r_tight


# ─────────────────────────────────────────────────────────────────────────────
# compute_lateral_discipline
# ─────────────────────────────────────────────────────────────────────────────

class TestLateral:

    def test_perfect_alignment_max_reward(self):
        r = compute_lateral_discipline(d_lat=0.0, delta_psi=0.0, sigma_d=0.2, phi=0.3)
        expected = (0.3 + 0.7 * 1.0) * 1.0
        assert r == pytest.approx(expected, abs=1e-6)

    def test_output_in_zero_one_interval(self):
        for d_lat in [0, 0.1, 0.5, 1.0, 2.0]:
            for delta_psi in [0, 0.1, 0.5, 1.0, math.pi]:
                r = compute_lateral_discipline(d_lat, delta_psi, sigma_d=0.2, phi=0.3)
                assert 0.0 < r <= 1.0

    def test_floor_value(self):
        # r_dev has floor of 0.3 (when d_lat -> inf)
        r = compute_lateral_discipline(d_lat=1000.0, delta_psi=0.0, sigma_d=0.2, phi=0.3)
        assert r >= 0.3 - 1e-6  # r_dev >= 0.3, r_heading = 1.0

    def test_large_heading_error(self):
        r_small = compute_lateral_discipline(0.0, 0.1, 0.2, 0.3)
        r_large = compute_lateral_discipline(0.0, 1.5, 0.2, 0.3)
        assert r_large < r_small


# ─────────────────────────────────────────────────────────────────────────────
# compute_spacing / compute_ttc
# ─────────────────────────────────────────────────────────────────────────────

class TestSpacing:

    def test_no_lead_vehicle_max_spacing(self):
        ttc = compute_ttc(d_lead=50.0, v_ego=10.0, v_lead=0.0, has_lead=False)
        assert ttc == float("inf")
        r = compute_spacing(ttc, tau=2.0)
        assert r == pytest.approx(1.0, abs=1e-6)

    def test_closing_fast_low_spacing(self):
        ttc = compute_ttc(d_lead=10.0, v_ego=20.0, v_lead=0.0, has_lead=True)
        assert ttc == pytest.approx(0.5, rel=1e-6)
        r = compute_spacing(ttc, tau=2.0)
        assert r < 0.5  # low reward when TTC is very small

    def test_output_in_zero_one_interval(self):
        for ttc in [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, float("inf")]:
            r = compute_spacing(ttc, tau=2.0)
            assert 0.0 < r <= 1.0, f"Out of (0,1] for ttc={ttc}"

    def test_floor_value(self):
        r = compute_spacing(ttc=0.0, tau=2.0)
        assert r == pytest.approx(0.2, abs=1e-6)  # floor = 0.2

    def test_not_closing_returns_inf_ttc(self):
        ttc = compute_ttc(d_lead=20.0, v_ego=5.0, v_lead=10.0, has_lead=True)
        assert ttc == float("inf")  # not closing (v_lead > v_ego)


# ─────────────────────────────────────────────────────────────────────────────
# compute_style_rewards (vector)
# ─────────────────────────────────────────────────────────────────────────────

class TestStyleRewardVector:

    def test_returns_4d_array(self, default_hp):
        r = compute_style_rewards(
            j_lon=0.0, j_lat=0.0,
            v_ego=10.0, v_des=10.0,
            beta=default_hp["reward_scaling"]["beta"],
            d_lat=0.0, delta_psi=0.0,
            d_lead=50.0, v_lead=0.0, has_lead=False,
            hp=default_hp,
        )
        assert r.shape == (4,)
        assert r.dtype == np.float32

    def test_all_components_in_zero_one(self, default_hp):
        r = compute_style_rewards(
            j_lon=1.0, j_lat=0.5,
            v_ego=8.0, v_des=10.0,
            beta=default_hp["reward_scaling"]["beta"],
            d_lat=0.1, delta_psi=0.05,
            d_lead=20.0, v_lead=5.0, has_lead=True,
            hp=default_hp,
        )
        for i, val in enumerate(r):
            assert 0.0 < val <= 1.0, f"Component {i} = {val} out of (0,1]"

    def test_uses_reward_scaling_sigma_d_not_floor_delta_d(self, default_hp):
        """Regression test: compute_style_rewards's r_lateral must use
        hp["reward_scaling"]["sigma_d"] (0.2 in the fixture, matching real
        compute_hyperparams.py output) for the Gaussian width, NOT
        hp["floor_values"]["delta_d"] (0.3 in the fixture -- a different
        constant, the additive floor hardcoded inside
        compute_lateral_discipline itself). The fixture deliberately keeps
        these different so this test fails loudly if the wrong key is ever
        read again."""
        r_direct = compute_lateral_discipline(
            d_lat=0.15, delta_psi=0.0,
            sigma_d=default_hp["reward_scaling"]["sigma_d"],
            phi=default_hp["reward_scaling"]["phi"],
        )
        r_vec = compute_style_rewards(
            j_lon=0.0, j_lat=0.0,
            v_ego=10.0, v_des=10.0,
            beta=default_hp["reward_scaling"]["beta"],
            d_lat=0.15, delta_psi=0.0,
            d_lead=50.0, v_lead=0.0, has_lead=False,
            hp=default_hp,
        )
        assert r_vec[2] == pytest.approx(r_direct, rel=1e-6), (
            "r_lateral does not match the value computed with "
            "reward_scaling.sigma_d -- check which hp key compute_style_rewards reads"
        )


# ─────────────────────────────────────────────────────────────────────────────
# ZtNormaliser
# ─────────────────────────────────────────────────────────────────────────────

class TestZtNormaliser:

    def test_initial_normalisation_with_zero_input(self):
        norm = ZtNormaliser(reward_dim=4, beta=0.01)
        z = np.zeros(4)
        z_norm = norm.normalise(z)
        assert z_norm.shape == (4,)
        # With zero mean and unit var initialisation: normalised = 0
        assert np.allclose(z_norm, 0.0, atol=0.1)

    def test_update_shifts_mean(self):
        norm = ZtNormaliser(reward_dim=4, beta=0.1)
        for _ in range(100):
            norm.update(np.ones(4) * 5.0)
        # Mean should be near 5 now
        assert np.allclose(norm._mu, 5.0, atol=0.5)

    def test_normalise_output_clipped(self):
        norm = ZtNormaliser(reward_dim=4, beta=0.01)
        z = np.ones(4) * 1000.0  # extreme value
        z_norm = norm.normalise(z)
        assert np.all(z_norm <= 5.0)
        assert np.all(z_norm >= -5.0)

    def test_state_dict_roundtrip(self):
        norm = ZtNormaliser(reward_dim=4, beta=0.05)
        for _ in range(50):
            norm.update(np.random.randn(4))
        sd = norm.state_dict()
        norm2 = ZtNormaliser.from_state_dict(sd, reward_dim=4)
        assert np.allclose(norm._mu, norm2._mu)
        assert np.allclose(norm._var, norm2._var)


# ─────────────────────────────────────────────────────────────────────────────
# AlphaSchedule
# ─────────────────────────────────────────────────────────────────────────────

class TestAlphaSchedule:

    def test_starts_at_alpha_start(self):
        sched = AlphaSchedule(alpha_start=0.2, alpha_end=0.95, n_curriculum=1000)
        assert sched.get(0) == pytest.approx(0.2, abs=1e-6)

    def test_reaches_alpha_end(self):
        sched = AlphaSchedule(alpha_start=0.2, alpha_end=0.95, n_curriculum=1000)
        assert sched.get(1000) == pytest.approx(0.95, abs=1e-6)

    def test_monotone_increasing(self):
        sched = AlphaSchedule(alpha_start=0.2, alpha_end=0.95, n_curriculum=1000)
        alphas = [sched.get(n) for n in range(0, 1001, 100)]
        for a, b in zip(alphas, alphas[1:]):
            assert b >= a

    def test_clamped_after_curriculum(self):
        sched = AlphaSchedule(alpha_start=0.2, alpha_end=0.95, n_curriculum=1000)
        alpha_end = sched.get(1000)
        alpha_over = sched.get(5000)
        assert alpha_end == pytest.approx(alpha_over, abs=1e-6)

    def test_compute_alpha_formula(self):
        alpha = compute_alpha(n=500, alpha_start=0.2, alpha_end=0.9, n_curriculum=1000)
        expected = 0.2 + (0.9 - 0.2) * 0.5
        assert alpha == pytest.approx(expected, abs=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# CVaR estimation
# ─────────────────────────────────────────────────────────────────────────────

class TestCVaR:

    def test_empirical_cvar_basic(self):
        from prism.morl.cvar_penalty import compute_empirical_cvar
        costs = list(range(1, 101))  # 1..100
        # CVaR at alpha=0.95: top 5% = [96..100], mean = 98
        cvar = compute_empirical_cvar(costs, alpha=0.95)
        assert cvar == pytest.approx(98.0, abs=1.0)

    def test_empirical_cvar_at_alpha_zero_is_mean(self):
        # CVaR_0 = expected value over the FULL distribution = mean.
        # (alpha=0 means "top 100% tail", not just the single worst.)
        from prism.morl.cvar_penalty import compute_empirical_cvar
        costs = [1, 5, 3, 9, 2]
        cvar = compute_empirical_cvar(costs, alpha=0.0)
        assert cvar == pytest.approx(float(np.mean(costs)), rel=1e-6)

    def test_empirical_cvar_at_high_alpha_is_single_worst(self):
        # alpha=0.8, N=5: tail fraction = 1-0.8 = 0.2 → ceil(0.2*5) = 1 episode.
        # CVaR = mean of top-1 = max.
        from prism.morl.cvar_penalty import compute_empirical_cvar
        costs = [1, 5, 3, 9, 2]
        cvar = compute_empirical_cvar(costs, alpha=0.8)
        assert cvar == pytest.approx(float(max(costs)), rel=1e-6)

    def test_empirical_cvar_empty_batch_is_zero(self):
        from prism.morl.cvar_penalty import compute_empirical_cvar
        assert compute_empirical_cvar([], alpha=0.95) == 0.0

    def test_update_var_matches_quantile(self):
        from prism.morl.cvar_penalty import update_var
        costs = list(range(1, 101))  # 1..100
        # alpha=0.95 quantile of 1..100 (torch.quantile, linear interpolation)
        nu = update_var(costs, alpha=0.95)
        assert nu == pytest.approx(95.05, abs=0.5)

    def test_update_var_clamps_alpha_below_one(self):
        # alpha=1.0 must not raise or produce inf/nan -- clamped to 0.999 internally.
        from prism.morl.cvar_penalty import update_var
        costs = [1.0, 2.0, 3.0, 4.0, 5.0]
        nu = update_var(costs, alpha=1.0)
        assert np.isfinite(nu)

    def test_update_var_empty_batch_is_zero(self):
        from prism.morl.cvar_penalty import update_var
        assert update_var([], alpha=0.95) == 0.0

    def test_g_nu_is_nondecreasing_and_positive(self):
        from prism.morl.cvar_penalty import g_nu
        nu, tau = 100.0, 20.0
        es = np.linspace(0.0, 300.0, 50)
        vals = [g_nu(float(e), nu, tau) for e in es]
        assert all(v > 0.0 for v in vals)  # dense: never exactly zero
        assert all(b >= a for a, b in zip(vals, vals[1:]))  # non-decreasing

    def test_g_nu_requires_positive_tau(self):
        from prism.morl.cvar_penalty import g_nu
        with pytest.raises(ValueError):
            g_nu(1.0, nu=1.0, tau=0.0)

    def test_dense_cost_signal_telescopes(self):
        # sum_t gamma^t * c~_t must equal g_nu(e_T) - g_nu(e_0) exactly,
        # for ANY per-step cost trajectory (the core correctness property
        # that makes GAE on c~_t a valid proxy for the g_nu(C^pi) objective).
        from prism.morl.cvar_penalty import dense_cost_signal, g_nu

        gamma = 0.99
        nu, tau = 50.0, 15.0
        rng = np.random.default_rng(42)
        step_costs = rng.exponential(scale=3.0, size=37)  # right-skewed, like safety cost

        e = 0.0
        e_trace = [0.0]
        for t, c in enumerate(step_costs):
            e = e + (gamma ** t) * c
            e_trace.append(e)

        total = 0.0
        for t, c in enumerate(step_costs):
            c_tilde = dense_cost_signal(e_trace[t], e_trace[t + 1], nu, tau, t, gamma)
            total += (gamma ** t) * c_tilde

        expected = g_nu(e_trace[-1], nu, tau) - g_nu(e_trace[0], nu, tau)
        assert total == pytest.approx(expected, rel=1e-6, abs=1e-9)

    def test_dense_cost_signal_dense_below_var(self):
        # Below VaR, c~_t must be nonzero whenever raw cost fires -- unlike
        # the sparse hinge-difference alternative, which is exactly zero
        # whenever both e_t and e_{t+1} stay below nu.  Use a nu only
        # moderately above e_t (not astronomically far) so the density
        # is numerically meaningful, not just technically nonzero.
        from prism.morl.cvar_penalty import dense_cost_signal
        nu, tau, gamma = 50.0, 20.0, 0.99
        c_tilde = dense_cost_signal(e_t=10.0, e_tp1=13.0, nu=nu, tau=tau, t=5, gamma=gamma)
        assert abs(c_tilde) > 1e-4

    def test_episode_cost_buffer_rolling_window(self):
        from prism.morl.cvar_penalty import EpisodeCostBuffer
        buf = EpisodeCostBuffer(buffer_size=10)
        for c in range(20):
            buf.add_episode(float(c))
        # Only the most recent 10 episodes are retained
        assert len(buf) == 10
        assert buf.costs == [float(c) for c in range(10, 20)]

    def test_episode_cost_buffer_state_dict_roundtrip(self):
        from prism.morl.cvar_penalty import EpisodeCostBuffer
        buf = EpisodeCostBuffer(buffer_size=5)
        buf.add_episodes([1.0, 2.0, 3.0])
        restored = EpisodeCostBuffer.from_state_dict(buf.state_dict(), buffer_size=5)
        assert restored.costs == buf.costs

    def test_compute_episode_cost(self):
        from prism.morl.cvar_penalty import compute_episode_cost
        costs = [1.0, 1.0, 1.0]
        gamma = 0.99
        expected = 1.0 + 0.99 + 0.99**2
        assert compute_episode_cost(costs, gamma) == pytest.approx(expected, rel=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
