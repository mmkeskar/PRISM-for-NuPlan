"""
Unit tests for StochasticActionHead's log_std clamp.

Pure PyTorch, no nuPlan/gym dependency. Run with:
    cd PRISM && python -m pytest tests/test_action_heads.py -v
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from prism.models.common.action_heads import (
    StochasticActionHead,
    _LOG_STD_MAX,
    _LOG_STD_MIN,
)


class TestLogStdClamp:
    def test_runaway_high_log_std_is_clamped(self):
        """A log_std that drifted far above _LOG_STD_MAX (the exact failure
        mode observed in a real run -- entropy climbing unboundedly) must
        not translate into unboundedly large action noise."""
        head = StochasticActionHead(action_dim=2, init_log_std=-0.5)
        with torch.no_grad():
            head.log_std.fill_(100.0)  # simulate unbounded drift

        mean = torch.zeros(4, 2)
        _, _, entropy = head(mean, actions=torch.zeros(4, 2))

        expected_std = torch.tensor(_LOG_STD_MAX).exp()
        # entropy of a diagonal Gaussian, per dim: 0.5*log(2*pi*e*std^2)
        expected_entropy_per_dim = 0.5 * torch.log(2 * torch.pi * torch.e * expected_std ** 2)
        expected_entropy = 2 * expected_entropy_per_dim  # summed over action_dim=2

        assert torch.allclose(entropy, expected_entropy.expand_as(entropy), atol=1e-4), (
            f"entropy should reflect the CLAMPED std ({expected_std:.3f}), not the raw "
            f"unclamped log_std=100 value; got {entropy}"
        )

    def test_runaway_low_log_std_is_clamped(self):
        """A log_std driven far below _LOG_STD_MIN must not collapse std to
        ~0 (which would produce NaN/Inf log_prob for any off-mean action)."""
        head = StochasticActionHead(action_dim=2, init_log_std=-0.5)
        with torch.no_grad():
            head.log_std.fill_(-100.0)

        mean = torch.zeros(4, 2)
        actions = torch.full((4, 2), 0.1)  # off-mean, would blow up log_prob if std~0
        _, log_prob, _ = head(mean, actions=actions)

        assert torch.isfinite(log_prob).all(), (
            "log_prob should stay finite even when the raw log_std parameter "
            "drifts far below the clamp floor"
        )

    def test_normal_range_log_std_passes_through_unchanged(self):
        """Sanity check the clamp doesn't distort behavior for log_std values
        already inside [_LOG_STD_MIN, _LOG_STD_MAX] (the whole normal
        operating range, including the -0.5 default init)."""
        head = StochasticActionHead(action_dim=2, init_log_std=-0.5)
        mean = torch.zeros(4, 2)
        actions = torch.zeros(4, 2)
        _, log_prob, entropy = head(mean, actions=actions)

        expected_std = torch.tensor(-0.5).exp()
        expected_entropy_per_dim = 0.5 * torch.log(2 * torch.pi * torch.e * expected_std ** 2)
        expected_entropy = 2 * expected_entropy_per_dim
        assert torch.allclose(entropy, expected_entropy.expand_as(entropy), atol=1e-4)

    def test_clamp_bounds_are_sane(self):
        assert _LOG_STD_MIN < -0.5 < _LOG_STD_MAX, (
            "the default init_log_std=-0.5 must fall strictly inside the clamp range"
        )
