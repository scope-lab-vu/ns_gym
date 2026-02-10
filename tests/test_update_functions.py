import pytest
import numpy as np
import inspect

import ns_gym
import ns_gym.base as base
import ns_gym.update_functions as update_mod
from ns_gym.schedulers import ContinuousScheduler, PeriodicScheduler
from ns_gym.update_functions.single_param import (
    DeterministicTrend,
    RandomWalkWithDriftAndTrend,
    RandomWalk,
    RandomWalkWithDrift,
    IncrementUpdate,
    DecrementUpdate,
    StepWiseUpdate,
    NoUpdate,
    OscillatingUpdate,
    ExponentialDecay,
    GeometricProgression,
    OrnsteinUhlenbeck,
    SigmoidTransition,
    CyclicUpdate,
    BoundedRandomWalk,
    PolynomialTrend,
    LinearInterpolation,
)
from ns_gym.update_functions.distribution import (
    RandomCategorical,
    DistributionIncrementUpdate,
    DistributionDecrementUpdate,
    DistributionStepWiseUpdate,
    DistributionNoUpdate,
    UniformDrift,
    TargetReversion,
    DistributionLinearInterpolation,
    DistributionCyclicUpdate,
)


# --- Discovery-based tests ---

update_classes = []

for name, obj in inspect.getmembers(ns_gym.update_functions.single_param):
    if inspect.isclass(obj) and obj.__module__ == ns_gym.update_functions.single_param.__name__:
        update_classes.append(obj)

for name, obj in inspect.getmembers(ns_gym.update_functions.distribution):
    if inspect.isclass(obj) and obj.__module__ == ns_gym.update_functions.distribution.__name__:
        update_classes.append(obj)


@pytest.mark.parametrize("UpdateFn", update_classes)
def test_inheritance(UpdateFn):
    """Tests that all update functions inherit from UpdateFn."""
    assert issubclass(UpdateFn, ns_gym.base.UpdateFn)


@pytest.mark.parametrize("UpdateFn", update_classes)
def test_has_update_method(UpdateFn):
    """Tests that all update functions implement the '_update' method."""
    assert hasattr(UpdateFn, "_update"), f"{UpdateFn.__name__} does not implement '_update' method"


# --- Helpers ---

def _always_scheduler():
    return ContinuousScheduler()


def _never_scheduler():
    """A scheduler that never fires (start in the future)."""
    return ContinuousScheduler(start=999999, end=999999)


def _periodic_scheduler(period=2):
    return PeriodicScheduler(period=period)


# ============================================================
# Single Parameter Update Functions
# ============================================================

# --- NoUpdate ---

class TestNoUpdate:
    def test_returns_same_param(self):
        fn = NoUpdate(_always_scheduler())
        param, changed, delta = fn(5.0, 1)
        assert param == 5.0
        assert changed == 1
        assert delta == 0.0

    def test_with_different_types(self):
        fn = NoUpdate(_always_scheduler())
        param, changed, _ = fn(10, 0)
        assert param == 10

        fn2 = NoUpdate(_always_scheduler())
        param, changed, _ = fn2(3.14, 5)
        assert param == 3.14


# --- IncrementUpdate ---

class TestIncrementUpdate:
    def test_increments_correctly(self):
        fn = IncrementUpdate(_always_scheduler(), k=2.0)
        param, changed, delta = fn(10.0, 0)
        assert param == 12.0
        assert changed == 1
        assert delta == 2.0

    def test_negative_increment(self):
        fn = IncrementUpdate(_always_scheduler(), k=-3.0)
        param, changed, delta = fn(10.0, 0)
        assert param == 7.0
        assert changed == 1

    def test_no_update_when_scheduler_false(self):
        fn = IncrementUpdate(_never_scheduler(), k=5.0)
        param, changed, delta = fn(10.0, 0)
        assert param == 10.0
        assert changed == 0
        assert delta == 0.0

    def test_multiple_calls_accumulate(self):
        fn = IncrementUpdate(_always_scheduler(), k=1.0)
        param = 0.0
        for t in range(5):
            param, _, _ = fn(param, t)
        assert param == 5.0


# --- DecrementUpdate ---

class TestDecrementUpdate:
    def test_decrements_correctly(self):
        fn = DecrementUpdate(_always_scheduler(), k=2.0)
        param, changed, delta = fn(10.0, 0)
        assert param == 8.0
        assert changed == 1

    def test_no_update_when_scheduler_false(self):
        fn = DecrementUpdate(_never_scheduler(), k=5.0)
        param, changed, delta = fn(10.0, 0)
        assert param == 10.0
        assert changed == 0


# --- DeterministicTrend ---

class TestDeterministicTrend:
    def test_trend_at_t1(self):
        fn = DeterministicTrend(_always_scheduler(), slope=2.0)
        param, changed, delta = fn(10.0, 1)
        assert param == 12.0  # 10 + 2*1
        assert changed == 1

    def test_trend_at_t0(self):
        fn = DeterministicTrend(_always_scheduler(), slope=5.0)
        param, changed, delta = fn(10.0, 0)
        assert param == 10.0  # 10 + 5*0

    def test_negative_slope(self):
        fn = DeterministicTrend(_always_scheduler(), slope=-1.0)
        param, changed, delta = fn(10.0, 3)
        assert param == 7.0  # 10 + (-1)*3

    def test_no_update_when_scheduler_false(self):
        fn = DeterministicTrend(_never_scheduler(), slope=5.0)
        param, changed, delta = fn(10.0, 1)
        assert param == 10.0
        assert changed == 0


# --- RandomWalk ---

class TestRandomWalk:
    def test_returns_numeric(self):
        fn = RandomWalk(_always_scheduler(), mu=0, sigma=1, seed=42)
        param, changed, delta = fn(10.0, 0)
        assert isinstance(param, (int, float, np.floating))
        assert changed == 1

    def test_seed_reproducibility(self):
        fn1 = RandomWalk(_always_scheduler(), mu=0, sigma=1, seed=42)
        fn2 = RandomWalk(_always_scheduler(), mu=0, sigma=1, seed=42)
        p1, _, _ = fn1(10.0, 0)
        p2, _, _ = fn2(10.0, 0)
        assert np.isclose(p1, p2)

    def test_zero_sigma_no_change(self):
        fn = RandomWalk(_always_scheduler(), mu=0, sigma=0, seed=42)
        param, _, _ = fn(10.0, 0)
        assert np.isclose(param, 10.0)

    def test_no_update_when_scheduler_false(self):
        fn = RandomWalk(_never_scheduler(), mu=0, sigma=1, seed=42)
        param, changed, _ = fn(10.0, 0)
        assert param == 10.0
        assert changed == 0


# --- RandomWalkWithDrift ---

class TestRandomWalkWithDrift:
    def test_returns_numeric(self):
        fn = RandomWalkWithDrift(_always_scheduler(), alpha=1.0, mu=0, sigma=1, seed=42)
        param, changed, delta = fn(10.0, 0)
        assert changed == 1

    def test_drift_component(self):
        fn = RandomWalkWithDrift(_always_scheduler(), alpha=2.0, mu=0, sigma=0, seed=42)
        param, _, _ = fn(10.0, 0)
        assert np.isclose(param, 12.0, atol=1e-6)  # 2.0 + 10 + 0

    def test_seed_reproducibility(self):
        fn1 = RandomWalkWithDrift(_always_scheduler(), alpha=1.0, mu=0, sigma=1, seed=42)
        fn2 = RandomWalkWithDrift(_always_scheduler(), alpha=1.0, mu=0, sigma=1, seed=42)
        p1, _, _ = fn1(10.0, 0)
        p2, _, _ = fn2(10.0, 0)
        assert np.isclose(p1, p2)


# --- RandomWalkWithDriftAndTrend ---

class TestRandomWalkWithDriftAndTrend:
    def test_returns_numeric(self):
        fn = RandomWalkWithDriftAndTrend(
            _always_scheduler(), alpha=1.0, mu=0, sigma=1, slope=0.5, seed=42
        )
        param, changed, delta = fn(10.0, 1)
        assert changed == 1

    def test_components_with_zero_noise(self):
        fn = RandomWalkWithDriftAndTrend(
            _always_scheduler(), alpha=1.0, mu=0, sigma=0, slope=2.0, seed=42
        )
        param, _, _ = fn(10.0, 3)
        # alpha + param + noise(0) + slope*t = 1 + 10 + 0 + 2*3 = 17
        assert np.isclose(param, 17.0, atol=1e-6)


# --- StepWiseUpdate ---

class TestStepWiseUpdate:
    def test_steps_through_values(self):
        fn = StepWiseUpdate(_always_scheduler(), param_list=[20.0, 30.0, 40.0])
        param, changed, _ = fn(10.0, 0)
        assert param == 20.0

        param, changed, _ = fn(param, 1)
        assert param == 30.0

        param, changed, _ = fn(param, 2)
        assert param == 40.0

    def test_empty_list_keeps_param(self):
        fn = StepWiseUpdate(_always_scheduler(), param_list=[])
        param, changed, _ = fn(10.0, 0)
        assert param == 10.0

    def test_no_update_when_scheduler_false(self):
        fn = StepWiseUpdate(_never_scheduler(), param_list=[99.0])
        param, changed, _ = fn(10.0, 0)
        assert param == 10.0
        assert changed == 0


# --- OscillatingUpdate ---

class TestOscillatingUpdate:
    def test_at_t_zero(self):
        fn = OscillatingUpdate(_always_scheduler(), delta=1.0)
        param, changed, _ = fn(10.0, 0)
        assert np.isclose(param, 10.0)  # sin(0) = 0

    def test_at_t_pi_half(self):
        fn = OscillatingUpdate(_always_scheduler(), delta=2.0)
        param, _, _ = fn(10.0, np.pi / 2)
        assert np.isclose(param, 12.0, atol=1e-6)  # 10 + 2*sin(pi/2) = 12

    def test_oscillation_pattern(self):
        fn = OscillatingUpdate(_always_scheduler(), delta=1.0)
        p1, _, _ = fn(0.0, 0)
        p2, _, _ = fn(0.0, np.pi)
        assert np.isclose(p1, 0.0, atol=1e-6)
        assert np.isclose(p2, 0.0, atol=1e-6)  # sin(pi) ≈ 0


# --- ExponentialDecay ---

class TestExponentialDecay:
    def test_decay_at_t_zero(self):
        fn = ExponentialDecay(_always_scheduler(), decay_rate=0.5)
        param, _, _ = fn(10.0, 0)
        assert np.isclose(param, 10.0)  # 10 * exp(0) = 10

    def test_decay_at_t_one(self):
        fn = ExponentialDecay(_always_scheduler(), decay_rate=1.0)
        param, _, _ = fn(10.0, 1)
        assert np.isclose(param, 10.0 * np.exp(-1.0))

    def test_decay_decreases_value(self):
        fn = ExponentialDecay(_always_scheduler(), decay_rate=0.1)
        param, _, _ = fn(100.0, 5)
        assert param < 100.0


# --- GeometricProgression ---

class TestGeometricProgression:
    def test_ratio_multiply(self):
        fn = GeometricProgression(_always_scheduler(), r=2.0)
        param, changed, _ = fn(5.0, 0)
        assert param == 10.0
        assert changed == 1

    def test_ratio_less_than_one(self):
        fn = GeometricProgression(_always_scheduler(), r=0.5)
        param, _, _ = fn(10.0, 0)
        assert param == 5.0

    def test_multiple_calls(self):
        fn = GeometricProgression(_always_scheduler(), r=3.0)
        param = 1.0
        for t in range(3):
            param, _, _ = fn(param, t)
        assert param == 27.0  # 1 * 3 * 3 * 3


# --- OrnsteinUhlenbeck ---

class TestOrnsteinUhlenbeck:
    def test_mean_reversion_direction(self):
        """When param > mu, the update should pull it back toward mu."""
        fn = OrnsteinUhlenbeck(
            _always_scheduler(), theta=0.5, mu=10.0, sigma=0, seed=42
        )
        param, changed, _ = fn(20.0, 0)
        assert changed == 1
        # theta * (mu - param) = 0.5 * (10 - 20) = -5, so 20 + (-5) = 15
        assert np.isclose(param, 15.0)

    def test_mean_reversion_from_below(self):
        """When param < mu, the update should push it toward mu."""
        fn = OrnsteinUhlenbeck(
            _always_scheduler(), theta=0.5, mu=10.0, sigma=0, seed=42
        )
        param, _, _ = fn(0.0, 0)
        # theta * (mu - param) = 0.5 * (10 - 0) = 5, so 0 + 5 = 5
        assert np.isclose(param, 5.0)

    def test_at_equilibrium_no_drift(self):
        """When param == mu and sigma=0, param stays at mu."""
        fn = OrnsteinUhlenbeck(
            _always_scheduler(), theta=0.5, mu=10.0, sigma=0, seed=42
        )
        param, _, _ = fn(10.0, 0)
        assert np.isclose(param, 10.0)

    def test_with_noise(self):
        """With sigma > 0, the output should differ from the deterministic case."""
        fn = OrnsteinUhlenbeck(
            _always_scheduler(), theta=0.5, mu=10.0, sigma=1.0, seed=42
        )
        param, _, _ = fn(10.0, 0)
        # At equilibrium + noise, should not be exactly 10.0
        assert not np.isclose(param, 10.0)

    def test_seed_reproducibility(self):
        fn1 = OrnsteinUhlenbeck(
            _always_scheduler(), theta=0.5, mu=10.0, sigma=1.0, seed=42
        )
        fn2 = OrnsteinUhlenbeck(
            _always_scheduler(), theta=0.5, mu=10.0, sigma=1.0, seed=42
        )
        p1, _, _ = fn1(5.0, 0)
        p2, _, _ = fn2(5.0, 0)
        assert np.isclose(p1, p2)

    def test_no_update_when_scheduler_false(self):
        fn = OrnsteinUhlenbeck(
            _never_scheduler(), theta=0.5, mu=10.0, sigma=1.0, seed=42
        )
        param, changed, _ = fn(5.0, 0)
        assert param == 5.0
        assert changed == 0


# --- SigmoidTransition ---

class TestSigmoidTransition:
    def test_starts_near_a(self):
        """At t << t0, the output should be close to a."""
        fn = SigmoidTransition(
            _always_scheduler(), a=1.0, b=10.0, k=1.0, t0=50
        )
        param, changed, _ = fn(0.0, 0)  # t=0, far before t0=50
        assert changed == 1
        assert np.isclose(param, 1.0, atol=0.1)

    def test_ends_near_b(self):
        """At t >> t0, the output should be close to b."""
        fn = SigmoidTransition(
            _always_scheduler(), a=1.0, b=10.0, k=1.0, t0=50
        )
        param, _, _ = fn(0.0, 100)  # t=100, far after t0=50
        assert np.isclose(param, 10.0, atol=0.1)

    def test_midpoint(self):
        """At t == t0, the output should be (a+b)/2."""
        fn = SigmoidTransition(
            _always_scheduler(), a=0.0, b=10.0, k=1.0, t0=50
        )
        param, _, _ = fn(0.0, 50)
        assert np.isclose(param, 5.0)

    def test_steepness_controls_sharpness(self):
        """Higher k should make the transition sharper."""
        fn_sharp = SigmoidTransition(
            _always_scheduler(), a=0.0, b=10.0, k=10.0, t0=50
        )
        fn_gradual = SigmoidTransition(
            _always_scheduler(), a=0.0, b=10.0, k=0.1, t0=50
        )
        # At t=51 (just past midpoint), sharp should be much closer to b
        p_sharp, _, _ = fn_sharp(0.0, 51)
        p_gradual, _, _ = fn_gradual(0.0, 51)
        assert p_sharp > p_gradual

    def test_no_update_when_scheduler_false(self):
        fn = SigmoidTransition(
            _never_scheduler(), a=0.0, b=10.0, k=1.0, t0=50
        )
        param, changed, _ = fn(5.0, 0)
        assert param == 5.0
        assert changed == 0


# --- CyclicUpdate ---

class TestCyclicUpdate:
    def test_cycles_through_values(self):
        fn = CyclicUpdate(_always_scheduler(), value_list=[10.0, 20.0, 30.0])
        p1, changed, _ = fn(0.0, 0)
        assert p1 == 10.0
        assert changed == 1

        p2, _, _ = fn(p1, 1)
        assert p2 == 20.0

        p3, _, _ = fn(p2, 2)
        assert p3 == 30.0

    def test_wraps_around(self):
        """After exhausting the list, it should cycle back to the start."""
        fn = CyclicUpdate(_always_scheduler(), value_list=[1.0, 2.0])
        results = []
        param = 0.0
        for t in range(6):
            param, _, _ = fn(param, t)
            results.append(param)
        assert results == [1.0, 2.0, 1.0, 2.0, 1.0, 2.0]

    def test_single_value_repeats(self):
        fn = CyclicUpdate(_always_scheduler(), value_list=[42.0])
        for t in range(5):
            param, _, _ = fn(0.0, t)
            assert param == 42.0

    def test_no_update_when_scheduler_false(self):
        fn = CyclicUpdate(_never_scheduler(), value_list=[99.0])
        param, changed, _ = fn(5.0, 0)
        assert param == 5.0
        assert changed == 0


# --- BoundedRandomWalk ---

class TestBoundedRandomWalk:
    def test_stays_within_bounds(self):
        fn = BoundedRandomWalk(
            _always_scheduler(), mu=0, sigma=10.0, lo=0.0, hi=20.0, seed=42
        )
        param = 10.0
        for t in range(100):
            param, _, _ = fn(param, t)
            assert 0.0 <= param <= 20.0, (
                f"BoundedRandomWalk produced {param} outside [0, 20] at step {t}"
            )

    def test_clamps_high(self):
        """Large positive noise should be clamped to hi."""
        fn = BoundedRandomWalk(
            _always_scheduler(), mu=100.0, sigma=0, lo=0.0, hi=15.0, seed=42
        )
        param, _, _ = fn(10.0, 0)
        assert param == 15.0

    def test_clamps_low(self):
        """Large negative noise should be clamped to lo."""
        fn = BoundedRandomWalk(
            _always_scheduler(), mu=-100.0, sigma=0, lo=5.0, hi=20.0, seed=42
        )
        param, _, _ = fn(10.0, 0)
        assert param == 5.0

    def test_seed_reproducibility(self):
        fn1 = BoundedRandomWalk(
            _always_scheduler(), mu=0, sigma=1, lo=-10, hi=10, seed=42
        )
        fn2 = BoundedRandomWalk(
            _always_scheduler(), mu=0, sigma=1, lo=-10, hi=10, seed=42
        )
        p1, _, _ = fn1(5.0, 0)
        p2, _, _ = fn2(5.0, 0)
        assert np.isclose(p1, p2)

    def test_no_update_when_scheduler_false(self):
        fn = BoundedRandomWalk(
            _never_scheduler(), mu=0, sigma=1, lo=0, hi=10, seed=42
        )
        param, changed, _ = fn(5.0, 0)
        assert param == 5.0
        assert changed == 0


# --- PolynomialTrend ---

class TestPolynomialTrend:
    def test_linear_via_coefficients(self):
        """coeffs=[2.0] should behave like DeterministicTrend with slope=2."""
        fn = PolynomialTrend(_always_scheduler(), coeffs=[2.0])
        param, changed, _ = fn(10.0, 3)
        # 10 + 2.0 * 3 = 16
        assert np.isclose(param, 16.0)
        assert changed == 1

    def test_quadratic(self):
        """coeffs=[0, 1.0] gives Y_t = Y_{t-1} + 1.0 * t^2."""
        fn = PolynomialTrend(_always_scheduler(), coeffs=[0, 1.0])
        param, _, _ = fn(10.0, 3)
        # 10 + 0*3 + 1.0*9 = 19
        assert np.isclose(param, 19.0)

    def test_cubic(self):
        """coeffs=[0, 0, 0.5] gives Y_t = Y_{t-1} + 0.5 * t^3."""
        fn = PolynomialTrend(_always_scheduler(), coeffs=[0, 0, 0.5])
        param, _, _ = fn(0.0, 2)
        # 0 + 0 + 0 + 0.5*8 = 4
        assert np.isclose(param, 4.0)

    def test_mixed_coefficients(self):
        """coeffs=[1, -0.5] gives Y_t = Y_{t-1} + 1*t + (-0.5)*t^2."""
        fn = PolynomialTrend(_always_scheduler(), coeffs=[1.0, -0.5])
        param, _, _ = fn(10.0, 4)
        # 10 + 1*4 + (-0.5)*16 = 10 + 4 - 8 = 6
        assert np.isclose(param, 6.0)

    def test_at_t_zero(self):
        fn = PolynomialTrend(_always_scheduler(), coeffs=[5.0, 3.0])
        param, _, _ = fn(10.0, 0)
        # all terms are 0 at t=0
        assert np.isclose(param, 10.0)

    def test_no_update_when_scheduler_false(self):
        fn = PolynomialTrend(_never_scheduler(), coeffs=[1.0])
        param, changed, _ = fn(10.0, 5)
        assert param == 10.0
        assert changed == 0


# --- LinearInterpolation ---

class TestLinearInterpolation:
    def test_at_start(self):
        fn = LinearInterpolation(
            _always_scheduler(), start_val=0.0, end_val=10.0, T=100
        )
        param, changed, _ = fn(0.0, 0)
        assert np.isclose(param, 0.0)
        assert changed == 1

    def test_at_end(self):
        fn = LinearInterpolation(
            _always_scheduler(), start_val=0.0, end_val=10.0, T=100
        )
        param, _, _ = fn(0.0, 100)
        assert np.isclose(param, 10.0)

    def test_midpoint(self):
        fn = LinearInterpolation(
            _always_scheduler(), start_val=0.0, end_val=10.0, T=100
        )
        param, _, _ = fn(0.0, 50)
        assert np.isclose(param, 5.0)

    def test_clamps_beyond_T(self):
        """Past t=T, the value should stay at end_val."""
        fn = LinearInterpolation(
            _always_scheduler(), start_val=0.0, end_val=10.0, T=100
        )
        param, _, _ = fn(0.0, 200)
        assert np.isclose(param, 10.0)

    def test_decreasing_interpolation(self):
        fn = LinearInterpolation(
            _always_scheduler(), start_val=10.0, end_val=0.0, T=100
        )
        param, _, _ = fn(0.0, 50)
        assert np.isclose(param, 5.0)

    def test_no_update_when_scheduler_false(self):
        fn = LinearInterpolation(
            _never_scheduler(), start_val=0.0, end_val=10.0, T=100
        )
        param, changed, _ = fn(5.0, 50)
        assert param == 5.0
        assert changed == 0


# ============================================================
# Distribution Update Functions
# ============================================================

# --- DistributionNoUpdate ---

class TestDistributionNoUpdate:
    def test_returns_same_distribution(self):
        fn = DistributionNoUpdate(_always_scheduler())
        param, changed, delta = fn([0.5, 0.3, 0.2], 0)
        assert param == [0.5, 0.3, 0.2]
        assert changed == 1

    def test_requires_list_input(self):
        fn = DistributionNoUpdate(_always_scheduler())
        with pytest.raises(AssertionError):
            fn(0.5, 0)


# --- RandomCategorical ---

class TestRandomCategorical:
    def test_returns_valid_distribution(self):
        fn = RandomCategorical(_always_scheduler(), seed=42)
        param, changed, _ = fn([0.25, 0.25, 0.25, 0.25], 0)
        assert changed == 1
        assert len(param) == 4
        assert np.isclose(sum(param), 1.0)
        assert all(0 <= p <= 1 for p in param)

    def test_seed_reproducibility(self):
        fn1 = RandomCategorical(_always_scheduler(), seed=42)
        fn2 = RandomCategorical(_always_scheduler(), seed=42)
        p1, _, _ = fn1([0.5, 0.5], 0)
        p2, _, _ = fn2([0.5, 0.5], 0)
        assert np.allclose(p1, p2)

    def test_preserves_length(self):
        fn = RandomCategorical(_always_scheduler(), seed=42)
        for n in [2, 3, 5, 10]:
            param, _, _ = fn([1.0 / n] * n, 0)
            assert len(param) == n


# --- DistributionIncrementUpdate ---

class TestDistributionIncrementUpdate:
    def test_increments_first_element(self):
        fn = DistributionIncrementUpdate(_always_scheduler(), k=0.1)
        dist = [0.5, 0.25, 0.25]
        param, changed, _ = fn(dist, 0)
        assert changed == 1
        assert np.isclose(param[0], 0.6)
        assert np.isclose(sum(param), 1.0)

    def test_capped_at_one(self):
        fn = DistributionIncrementUpdate(_always_scheduler(), k=0.9)
        dist = [0.5, 0.25, 0.25]
        param, _, _ = fn(dist, 0)
        assert param[0] <= 1.0
        assert np.isclose(sum(param), 1.0)

    def test_redistributes_remaining(self):
        fn = DistributionIncrementUpdate(_always_scheduler(), k=0.2)
        dist = [0.4, 0.3, 0.3]
        param, _, _ = fn(dist, 0)
        assert np.isclose(param[0], 0.6)
        remaining = (1 - 0.6) / 2
        assert np.isclose(param[1], remaining)
        assert np.isclose(param[2], remaining)


# --- DistributionDecrementUpdate ---

class TestDistributionDecrementUpdate:
    def test_decrements_first_element(self):
        fn = DistributionDecrementUpdate(_always_scheduler(), k=0.1)
        dist = [0.5, 0.25, 0.25]
        param, changed, _ = fn(dist, 0)
        assert changed == 1
        assert np.isclose(param[0], 0.4)
        assert np.isclose(sum(param), 1.0)

    def test_floored_at_zero(self):
        fn = DistributionDecrementUpdate(_always_scheduler(), k=0.9)
        dist = [0.5, 0.25, 0.25]
        param, _, _ = fn(dist, 0)
        assert param[0] >= 0.0
        assert np.isclose(sum(param), 1.0)

    def test_redistributes_remaining(self):
        fn = DistributionDecrementUpdate(_always_scheduler(), k=0.2)
        dist = [0.6, 0.2, 0.2]
        param, _, _ = fn(dist, 0)
        assert np.isclose(param[0], 0.4)
        remaining = (1 - 0.4) / 2
        assert np.isclose(param[1], remaining)
        assert np.isclose(param[2], remaining)


# --- DistributionStepWiseUpdate ---

class TestDistributionStepWiseUpdate:
    def test_steps_through_distributions(self):
        values = [[0.6, 0.2, 0.2], [0.3, 0.4, 0.3]]
        fn = DistributionStepWiseUpdate(_always_scheduler(), update_values=values)

        param, changed, _ = fn([0.5, 0.25, 0.25], 0)
        assert param == [0.6, 0.2, 0.2]
        assert changed == 1

        param, changed, _ = fn(param, 1)
        assert param == [0.3, 0.4, 0.3]

    def test_empty_list_keeps_param(self):
        fn = DistributionStepWiseUpdate(_always_scheduler(), update_values=[])
        dist = [0.5, 0.25, 0.25]
        param, changed, _ = fn(dist, 0)
        assert param == [0.5, 0.25, 0.25]


# --- UniformDrift ---

class TestUniformDrift:
    def test_moves_toward_uniform(self):
        """With rate=1.0, should become exactly uniform."""
        fn = UniformDrift(_always_scheduler(), rate=1.0)
        dist = [0.8, 0.1, 0.1]
        param, changed, _ = fn(dist, 0)
        assert changed == 1
        expected = [1 / 3] * 3
        assert np.allclose(param, expected)

    def test_rate_zero_no_change(self):
        """With rate=0, the distribution should not change."""
        fn = UniformDrift(_always_scheduler(), rate=0.0)
        dist = [0.7, 0.2, 0.1]
        param, changed, _ = fn(dist, 0)
        assert np.allclose(param, [0.7, 0.2, 0.1])

    def test_partial_drift(self):
        """rate=0.5 should be halfway between current and uniform."""
        fn = UniformDrift(_always_scheduler(), rate=0.5)
        dist = [1.0, 0.0, 0.0]
        param, _, _ = fn(dist, 0)
        # (1-0.5)*[1,0,0] + 0.5*[1/3,1/3,1/3] = [0.5+1/6, 1/6, 1/6] = [2/3, 1/6, 1/6]
        expected = [2 / 3, 1 / 6, 1 / 6]
        assert np.allclose(param, expected)

    def test_output_is_valid_distribution(self):
        fn = UniformDrift(_always_scheduler(), rate=0.3)
        dist = [0.6, 0.3, 0.1]
        param, _, _ = fn(dist, 0)
        assert np.isclose(sum(param), 1.0)
        assert all(p >= 0 for p in param)

    def test_preserves_length(self):
        fn = UniformDrift(_always_scheduler(), rate=0.5)
        for n in [2, 4, 5]:
            dist = [1.0 / n] * n
            param, _, _ = fn(dist, 0)
            assert len(param) == n

    def test_no_update_when_scheduler_false(self):
        fn = UniformDrift(_never_scheduler(), rate=0.5)
        dist = [0.8, 0.1, 0.1]
        param, changed, _ = fn(dist, 0)
        assert param == [0.8, 0.1, 0.1]
        assert changed == 0


# --- TargetReversion ---

class TestTargetReversion:
    def test_full_reversion(self):
        """theta=1.0 should jump directly to the target."""
        target = [0.2, 0.3, 0.5]
        fn = TargetReversion(_always_scheduler(), target=target, theta=1.0)
        dist = [0.8, 0.1, 0.1]
        param, changed, _ = fn(dist, 0)
        assert changed == 1
        assert np.allclose(param, target)

    def test_no_reversion(self):
        """theta=0.0 should not change the distribution."""
        target = [0.2, 0.3, 0.5]
        fn = TargetReversion(_always_scheduler(), target=target, theta=0.0)
        dist = [0.8, 0.1, 0.1]
        param, _, _ = fn(dist, 0)
        assert np.allclose(param, [0.8, 0.1, 0.1])

    def test_partial_reversion(self):
        """theta=0.5 should be halfway between current and target."""
        target = [0.0, 0.5, 0.5]
        fn = TargetReversion(_always_scheduler(), target=target, theta=0.5)
        dist = [1.0, 0.0, 0.0]
        param, _, _ = fn(dist, 0)
        # [1,0,0] + 0.5*([0,0.5,0.5] - [1,0,0]) = [1,0,0] + [-0.5,0.25,0.25] = [0.5,0.25,0.25]
        assert np.allclose(param, [0.5, 0.25, 0.25])

    def test_output_is_valid_distribution(self):
        target = [0.1, 0.4, 0.5]
        fn = TargetReversion(_always_scheduler(), target=target, theta=0.7)
        dist = [0.6, 0.3, 0.1]
        param, _, _ = fn(dist, 0)
        assert np.isclose(sum(param), 1.0)
        assert all(p >= -1e-10 for p in param)

    def test_at_target_stays_at_target(self):
        target = [0.25, 0.25, 0.25, 0.25]
        fn = TargetReversion(_always_scheduler(), target=target, theta=0.5)
        param, _, _ = fn(list(target), 0)
        assert np.allclose(param, target)

    def test_no_update_when_scheduler_false(self):
        target = [0.5, 0.5]
        fn = TargetReversion(_never_scheduler(), target=target, theta=0.5)
        dist = [0.8, 0.2]
        param, changed, _ = fn(dist, 0)
        assert param == [0.8, 0.2]
        assert changed == 0


# --- DistributionLinearInterpolation ---

class TestDistributionLinearInterpolation:
    def test_at_start(self):
        start = [0.8, 0.1, 0.1]
        end = [0.2, 0.4, 0.4]
        fn = DistributionLinearInterpolation(
            _always_scheduler(), start_dist=start, end_dist=end, T=100
        )
        param, changed, _ = fn([0.0, 0.0, 1.0], 0)
        assert changed == 1
        assert np.allclose(param, start)

    def test_at_end(self):
        start = [0.8, 0.1, 0.1]
        end = [0.2, 0.4, 0.4]
        fn = DistributionLinearInterpolation(
            _always_scheduler(), start_dist=start, end_dist=end, T=100
        )
        param, _, _ = fn([0.0, 0.0, 1.0], 100)
        assert np.allclose(param, end)

    def test_midpoint(self):
        start = [1.0, 0.0]
        end = [0.0, 1.0]
        fn = DistributionLinearInterpolation(
            _always_scheduler(), start_dist=start, end_dist=end, T=100
        )
        param, _, _ = fn([0.0, 1.0], 50)
        assert np.allclose(param, [0.5, 0.5])

    def test_clamps_beyond_T(self):
        start = [0.8, 0.2]
        end = [0.3, 0.7]
        fn = DistributionLinearInterpolation(
            _always_scheduler(), start_dist=start, end_dist=end, T=100
        )
        param, _, _ = fn([0.5, 0.5], 200)
        assert np.allclose(param, end)

    def test_output_is_valid_distribution(self):
        start = [0.7, 0.2, 0.1]
        end = [0.1, 0.3, 0.6]
        fn = DistributionLinearInterpolation(
            _always_scheduler(), start_dist=start, end_dist=end, T=100
        )
        for t in [0, 25, 50, 75, 100]:
            param, _, _ = fn([0.0, 0.0, 1.0], t)
            assert np.isclose(sum(param), 1.0), f"sum != 1 at t={t}"
            assert all(p >= -1e-10 for p in param), f"negative prob at t={t}"

    def test_no_update_when_scheduler_false(self):
        start = [0.8, 0.2]
        end = [0.3, 0.7]
        fn = DistributionLinearInterpolation(
            _never_scheduler(), start_dist=start, end_dist=end, T=100
        )
        dist = [0.5, 0.5]
        param, changed, _ = fn(dist, 50)
        assert param == [0.5, 0.5]
        assert changed == 0


# --- DistributionCyclicUpdate ---

class TestDistributionCyclicUpdate:
    def test_cycles_through_distributions(self):
        dists = [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]
        fn = DistributionCyclicUpdate(_always_scheduler(), dist_list=dists)

        p1, changed, _ = fn([0.33, 0.33, 0.34], 0)
        assert changed == 1
        assert p1 == [0.8, 0.1, 0.1]

        p2, _, _ = fn(p1, 1)
        assert p2 == [0.1, 0.8, 0.1]

        p3, _, _ = fn(p2, 2)
        assert p3 == [0.1, 0.1, 0.8]

    def test_wraps_around(self):
        dists = [[0.9, 0.1], [0.1, 0.9]]
        fn = DistributionCyclicUpdate(_always_scheduler(), dist_list=dists)
        results = []
        param = [0.5, 0.5]
        for t in range(6):
            param, _, _ = fn(param, t)
            results.append(list(param))
        assert results == [
            [0.9, 0.1], [0.1, 0.9],
            [0.9, 0.1], [0.1, 0.9],
            [0.9, 0.1], [0.1, 0.9],
        ]

    def test_single_distribution_repeats(self):
        dists = [[0.5, 0.5]]
        fn = DistributionCyclicUpdate(_always_scheduler(), dist_list=dists)
        for t in range(5):
            param, _, _ = fn([0.0, 1.0], t)
            assert param == [0.5, 0.5]

    def test_no_update_when_scheduler_false(self):
        dists = [[0.9, 0.1]]
        fn = DistributionCyclicUpdate(_never_scheduler(), dist_list=dists)
        dist = [0.5, 0.5]
        param, changed, _ = fn(dist, 0)
        assert param == [0.5, 0.5]
        assert changed == 0


# ============================================================
# __call__ interface tests
# ============================================================

class TestCallInterface:
    """Tests the (param, changed_flag, delta_change) return interface."""

    def test_update_returns_three_tuple_when_fired(self):
        fn = IncrementUpdate(_always_scheduler(), k=1.0)
        result = fn(10.0, 0)
        assert len(result) == 3
        param, changed, delta = result
        assert changed == 1

    def test_update_returns_three_tuple_when_not_fired(self):
        fn = IncrementUpdate(_never_scheduler(), k=1.0)
        result = fn(10.0, 0)
        assert len(result) == 3
        param, changed, delta = result
        assert changed == 0
        assert delta == 0.0

    def test_delta_change_reflects_update(self):
        fn = IncrementUpdate(_always_scheduler(), k=5.0)
        _, _, delta = fn(10.0, 0)
        assert delta == 5.0

    def test_prev_param_tracked(self):
        fn = IncrementUpdate(_always_scheduler(), k=1.0)
        fn(10.0, 0)
        assert fn.prev_param == 10.0
        assert fn.prev_time == 0

    def test_requires_scheduler_instance(self):
        with pytest.raises(AssertionError):
            IncrementUpdate("not_a_scheduler", k=1.0)

    def test_t_must_be_numeric(self):
        fn = IncrementUpdate(_always_scheduler(), k=1.0)
        with pytest.raises(AssertionError):
            fn(10.0, "not_a_number")

    def test_distribution_requires_list(self):
        fn = DistributionNoUpdate(_always_scheduler())
        with pytest.raises(AssertionError):
            fn(5.0, 0)

    def test_periodic_scheduler_integration(self):
        fn = IncrementUpdate(_periodic_scheduler(period=2), k=1.0)
        results = []
        param = 0.0
        for t in range(6):
            param, changed, _ = fn(param, t)
            results.append(changed)
        # Period 2: fires at t=0,2,4 -> changed flags: [1,0,1,0,1,0]
        assert results == [1, 0, 1, 0, 1, 0]


# ============================================================
# Regression: RandomWalk* must return scalar, not numpy array
# https://github.com/scope-lab-vu/ns_gym/issues/XX
# rng.normal(mu, sigma, 1) returns a 1-element array which
# causes ValueError in downstream dynamics (e.g. Acrobot).
# ============================================================

class TestRandomWalkReturnsScalar:
    """All RandomWalk* update functions must return a plain Python scalar,
    not a numpy array, so they can be used directly in environment dynamics."""

    def test_random_walk_returns_scalar(self):
        fn = RandomWalk(_always_scheduler(), mu=0, sigma=1, seed=42)
        param, _, _ = fn(10.0, 0)
        assert np.isscalar(param), (
            f"RandomWalk returned {type(param)}, expected scalar"
        )

    def test_random_walk_with_drift_returns_scalar(self):
        fn = RandomWalkWithDrift(
            _always_scheduler(), alpha=1.0, mu=0, sigma=1, seed=42
        )
        param, _, _ = fn(10.0, 0)
        assert np.isscalar(param), (
            f"RandomWalkWithDrift returned {type(param)}, expected scalar"
        )

    def test_random_walk_with_drift_and_trend_returns_scalar(self):
        fn = RandomWalkWithDriftAndTrend(
            _always_scheduler(), alpha=1.0, mu=0, sigma=1, slope=0.5, seed=42
        )
        param, _, _ = fn(10.0, 1)
        assert np.isscalar(param), (
            f"RandomWalkWithDriftAndTrend returned {type(param)}, expected scalar"
        )

    def test_random_walk_scalar_survives_multiple_steps(self):
        """Ensure the scalar type is preserved across multiple update steps."""
        fn = RandomWalk(_always_scheduler(), mu=0, sigma=1, seed=42)
        param = 10.0
        for t in range(10):
            param, _, _ = fn(param, t)
            assert np.isscalar(param), (
                f"RandomWalk returned {type(param)} at step {t}, expected scalar"
            )

    def test_random_walk_with_drift_scalar_survives_multiple_steps(self):
        fn = RandomWalkWithDrift(
            _always_scheduler(), alpha=0.5, mu=0, sigma=1, seed=42
        )
        param = 10.0
        for t in range(10):
            param, _, _ = fn(param, t)
            assert np.isscalar(param), (
                f"RandomWalkWithDrift returned {type(param)} at step {t}, expected scalar"
            )

    def test_random_walk_with_drift_and_trend_scalar_survives_multiple_steps(self):
        fn = RandomWalkWithDriftAndTrend(
            _always_scheduler(), alpha=0.5, mu=0, sigma=1, slope=0.1, seed=42
        )
        param = 10.0
        for t in range(10):
            param, _, _ = fn(param, t)
            assert np.isscalar(param), (
                f"RandomWalkWithDriftAndTrend returned {type(param)} at step {t}, expected scalar"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
