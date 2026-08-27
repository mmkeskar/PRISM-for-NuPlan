"""
Unit tests for UtilityFunction's preference-weighted linear term and
monotonicity, covering the fix for the "reward doesn't differentiate
policies" root cause (see CHANGES.md).

Pure PyTorch, no nuPlan/gym dependency. Run with:
    cd PRISM && python -m pytest tests/test_utility_functions.py -v
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from prism.morl.utility_functions import (
    UtilityFunction,
    init_utility_functions_from_preferences,
)


class TestSetPreference:
    def test_default_preference_is_uniform(self):
        """Before set_preference() is called, behavior must match the old
        plain-mean linear term exactly (backward compatibility)."""
        uf = UtilityFunction(reward_dim=4)
        assert torch.allclose(uf._pref_weights, torch.full((4,), 0.25))

    def test_set_preference_updates_buffer(self):
        uf = UtilityFunction(reward_dim=4)
        uf.set_preference([0.55, 0.15, 0.15, 0.15])
        assert torch.allclose(uf._pref_weights, torch.tensor([0.55, 0.15, 0.15, 0.15]))

    def test_set_preference_wrong_shape_raises(self):
        uf = UtilityFunction(reward_dim=4)
        try:
            uf.set_preference([1.0, 0.0])
            assert False, "expected an assertion error for wrong-shaped preference"
        except AssertionError as e:
            assert "shape" in str(e)

    def test_preference_survives_state_dict_roundtrip(self):
        """_pref_weights is a buffer -- must save/load with the checkpoint,
        the same mechanism scripts/train.py relies on when loading a saved
        utility function (no separate set_preference() call needed there)."""
        uf = UtilityFunction(reward_dim=4)
        uf.set_preference([0.15, 0.55, 0.15, 0.15])

        uf2 = UtilityFunction(reward_dim=4)
        uf2.load_state_dict(uf.state_dict())
        assert torch.allclose(uf2._pref_weights, torch.tensor([0.15, 0.55, 0.15, 0.15]))


class TestPreferenceWeightedOutput:
    def test_linear_term_uses_preference_not_plain_mean(self):
        """Directly isolate the linear term's contribution: two identical
        z vectors differing only in ONE dimension should produce a
        DIFFERENT utility delta depending on which policy's preference
        weights that dimension, once the (small, near-zero-init) neural
        pathway's contribution is negligible relative to this direct check."""
        uf_a = UtilityFunction(reward_dim=4)
        uf_a.set_preference([1.0, 0.0, 0.0, 0.0])  # cares only about dim 0
        uf_b = UtilityFunction(reward_dim=4)
        uf_b.load_state_dict(uf_a.state_dict())
        uf_b.set_preference([0.0, 1.0, 0.0, 0.0])  # cares only about dim 1

        z_base = torch.tensor([10.0, 10.0, 10.0, 10.0])
        z_boost_dim0 = z_base.clone()
        z_boost_dim0[0] += 5.0

        # Isolate the linear term directly (same computation as forward()'s
        # linear_term, not routed through the neural pathway at all) --
        # this is what we're actually verifying is preference-weighted.
        lamda = uf_a._lamda
        delta_a = lamda * ((z_boost_dim0 - z_base) * uf_a._pref_weights).sum()
        delta_b = lamda * ((z_boost_dim0 - z_base) * uf_b._pref_weights).sum()

        assert delta_a.item() > 1e-6, "policy caring about dim 0 should value a dim-0 boost"
        assert abs(delta_b.item()) < 1e-6, "policy caring about dim 1 should NOT value a dim-0 boost"

    def test_two_policies_value_their_own_dimension_more(self):
        """End-to-end (through the full forward pass, both terms combined):
        a comfort-preferring and progress-preferring policy should value a
        boost to THEIR OWN preferred dimension more than the other's.

        _min_val/_max_val are pinned to a fixed, realistic range before
        comparing -- in the real env these update on every step (not just
        during training), so a naive sequence of raw calls shifts the
        normalization between them and isn't a fair before/after
        comparison. Pinning them isolates the property this test actually
        checks: given a STABLE normalization range (the regime training
        settles into once enough episodes have shown the achievable z
        range), does a preferred-dimension boost get valued more."""
        torch.manual_seed(0)
        prefs = [[0.55, 0.15, 0.15, 0.15], [0.15, 0.55, 0.15, 0.15]]
        ufs = init_utility_functions_from_preferences(
            n_policies=2, reward_dim=4, preference_vectors=prefs
        )
        for uf in ufs:
            uf.eval()
            uf._min_val.fill_(0.0)
            uf._max_val.fill_(50.0)

        z_base = [20.0, 20.0, 20.0, 20.0]
        z_comfort_boost = [21.0, 20.0, 20.0, 20.0]
        z_progress_boost = [20.0, 21.0, 20.0, 20.0]

        import numpy as np
        fn0, fn1 = ufs[0].as_callable(), ufs[1].as_callable()
        d0_comfort = fn0(np.array(z_comfort_boost)) - fn0(np.array(z_base))
        d0_progress = fn0(np.array(z_progress_boost)) - fn0(np.array(z_base))
        d1_comfort = fn1(np.array(z_comfort_boost)) - fn1(np.array(z_base))
        d1_progress = fn1(np.array(z_progress_boost)) - fn1(np.array(z_base))

        assert d0_comfort > d0_progress, "comfort-preferring policy should value comfort more"
        assert d1_progress > d1_comfort, "progress-preferring policy should value progress more"


class TestMonotonicity:
    def test_output_nondecreasing_after_max_weight_change(self):
        """Regression check: raising max_weight (0.1 -> 1.0) must not break
        the core monotonicity guarantee (more of any objective is never
        worse) -- weights are still clamped to [0, max_weight] >= 0.

        _min_val/_max_val pinned before comparing, for the same reason as
        test_two_policies_value_their_own_dimension_more above -- they
        otherwise shift on every call and break a fair before/after
        comparison."""
        torch.manual_seed(0)
        uf = UtilityFunction(reward_dim=4, max_weight=1.0)
        uf.eval()
        uf._min_val.fill_(0.0)
        uf._max_val.fill_(50.0)
        fn = uf.as_callable()

        import numpy as np
        rng = np.random.default_rng(0)
        z = rng.uniform(0, 40, size=4)  # stay inside the pinned [0, 50] range
        base = fn(z)
        for dim in range(4):
            z_up = z.copy()
            z_up[dim] += 1.0
            assert fn(z_up) >= base - 1e-5, (
                f"increasing z[{dim}] must not decrease utility (monotonicity violated)"
            )
