import pytest
import numpy as np
import inspect

import ns_gym
import ns_gym.schedulers as sched_mod
import ns_gym.base as base
from ns_gym.schedulers import (
    RandomScheduler,
    CustomScheduler,
    ContinuousScheduler,
    DiscreteScheduler,
    PeriodicScheduler,
    MemorylessScheduler,
    BurstScheduler,
    DecayingProbabilityScheduler,
    WindowScheduler,
)


# --- Discovery-based tests ---

scheduler_classes = []
for name, obj in inspect.getmembers(ns_gym.schedulers):
    if inspect.isclass(obj) and obj.__module__ == ns_gym.schedulers.__name__:
        scheduler_classes.append(obj)


@pytest.mark.parametrize("Scheduler", scheduler_classes)
def test_inheritance(Scheduler):
    """Tests that all schedulers inherit from Scheduler."""
    assert issubclass(Scheduler, ns_gym.base.Scheduler)


@pytest.mark.parametrize("Scheduler", scheduler_classes)
def test_has_call_method(Scheduler):
    """Tests that all schedulers implement __call__."""
    assert callable(Scheduler), f"{Scheduler.__name__} is not callable"


# --- ContinuousScheduler ---

class TestContinuousScheduler:
    def test_always_true_in_range(self):
        sched = ContinuousScheduler(start=0, end=10)
        for t in range(11):
            assert sched(t) is True

    def test_false_outside_range(self):
        sched = ContinuousScheduler(start=5, end=10)
        assert sched(4) is False
        assert sched(11) is False

    def test_default_infinite_end(self):
        sched = ContinuousScheduler()
        assert sched(0) is True
        assert sched(1000000) is True

    def test_boundary_values(self):
        sched = ContinuousScheduler(start=3, end=7)
        assert sched(3) is True
        assert sched(7) is True
        assert sched(2) is False
        assert sched(8) is False

    def test_single_point_range(self):
        """start == end should fire only at that exact time step."""
        sched = ContinuousScheduler(start=5, end=5)
        assert sched(4) is False
        assert sched(5) is True
        assert sched(6) is False

    def test_delayed_start(self):
        sched = ContinuousScheduler(start=100)
        assert sched(0) is False
        assert sched(99) is False
        assert sched(100) is True
        assert sched(101) is True

    def test_returns_bool_type(self):
        sched = ContinuousScheduler(start=0, end=10)
        assert isinstance(sched(5), bool)
        assert isinstance(sched(20), bool)


# --- DiscreteScheduler ---

class TestDiscreteScheduler:
    def test_fires_at_event_times(self):
        events = {2, 5, 8}
        sched = DiscreteScheduler(event_list=events, start=0, end=10)
        for t in range(11):
            if t in events:
                assert sched(t) is True
            else:
                assert not sched(t)

    def test_does_not_fire_outside_events(self):
        events = {3, 6}
        sched = DiscreteScheduler(event_list=events, start=0, end=10)
        assert not sched(0)
        assert not sched(1)
        assert not sched(4)
        assert not sched(10)

    def test_assertion_start_after_first_event(self):
        with pytest.raises(AssertionError):
            DiscreteScheduler(event_list={1, 5}, start=3, end=10)

    def test_assertion_end_before_last_event(self):
        with pytest.raises(AssertionError):
            DiscreteScheduler(event_list={1, 5, 12}, start=0, end=10)

    def test_single_event(self):
        sched = DiscreteScheduler(event_list={5}, start=0, end=10)
        assert sched(5) is True
        assert not sched(4)
        assert not sched(6)

    def test_returns_false_not_none_outside_range(self):
        """__call__ should return False (not None) when t is outside [start, end]."""
        events = {5}
        sched = DiscreteScheduler(event_list=events, start=3, end=7)
        result = sched(1)  # outside range
        assert result is False

    def test_many_events(self):
        events = set(range(0, 100, 7))
        sched = DiscreteScheduler(event_list=events, start=0, end=100)
        for t in range(101):
            if t in events:
                assert sched(t) is True
            else:
                assert not sched(t)

    def test_events_at_boundaries(self):
        """Events at exactly start and end should fire."""
        sched = DiscreteScheduler(event_list={0, 10}, start=0, end=10)
        assert sched(0) is True
        assert sched(10) is True


# --- PeriodicScheduler ---

class TestPeriodicScheduler:
    def test_fires_at_period_multiples(self):
        sched = PeriodicScheduler(period=3)
        assert sched(0) is True
        assert sched(3) is True
        assert sched(6) is True
        assert sched(9) is True

    def test_does_not_fire_off_period(self):
        sched = PeriodicScheduler(period=3)
        assert sched(1) is False
        assert sched(2) is False
        assert sched(4) is False

    def test_period_one(self):
        sched = PeriodicScheduler(period=1)
        for t in range(10):
            assert sched(t) is True

    def test_respects_start_end(self):
        """Should not fire outside [start, end]."""
        sched = PeriodicScheduler(period=3, start=5, end=15)
        assert sched(0) is False   # before start, but 0 % 3 == 0
        assert sched(3) is False   # before start, but 3 % 3 == 0
        assert sched(6) is True    # in range and on period
        assert sched(18) is False  # after end, but 18 % 3 == 0

    def test_large_period(self):
        sched = PeriodicScheduler(period=100)
        assert sched(0) is True
        assert sched(50) is False
        assert sched(100) is True
        assert sched(99) is False

    def test_returns_bool_type(self):
        sched = PeriodicScheduler(period=2)
        assert isinstance(sched(0), bool)
        assert isinstance(sched(1), bool)


# --- RandomScheduler ---

class TestRandomScheduler:
    def test_returns_bool(self):
        sched = RandomScheduler(probability=0.5, seed=42)
        result = sched(0)
        assert isinstance(result, (bool, np.bool_))

    def test_always_fires_with_probability_one(self):
        sched = RandomScheduler(probability=1.0, seed=42)
        for t in range(100):
            assert sched(t) is True

    def test_never_fires_with_probability_zero(self):
        sched = RandomScheduler(probability=0.0, seed=42)
        for t in range(100):
            assert sched(t) is False

    def test_outside_range_returns_false(self):
        sched = RandomScheduler(probability=1.0, start=5, end=10, seed=42)
        assert sched(3) is False
        assert sched(12) is False
        assert sched(5) is True

    def test_seed_reproducibility(self):
        results1 = [RandomScheduler(probability=0.5, seed=123)(t) for t in range(20)]
        results2 = [RandomScheduler(probability=0.5, seed=123)(t) for t in range(20)]
        assert results1 == results2

    def test_different_seeds_diverge(self):
        sched1 = RandomScheduler(probability=0.5, seed=1)
        sched2 = RandomScheduler(probability=0.5, seed=999)
        results1 = [sched1(t) for t in range(50)]
        results2 = [sched2(t) for t in range(50)]
        assert results1 != results2

    def test_statistical_frequency(self):
        """Over many calls, the firing rate should approximate the probability."""
        sched = RandomScheduler(probability=0.3, seed=42)
        n = 10000
        fires = sum(1 for t in range(n) if sched(t))
        observed_rate = fires / n
        assert abs(observed_rate - 0.3) < 0.03, (
            f"Expected ~0.3 fire rate, got {observed_rate}"
        )

    def test_boundary_at_start(self):
        sched = RandomScheduler(probability=1.0, start=5, end=10, seed=42)
        assert sched(5) is True
        assert sched(10) is True

    def test_single_call_with_same_seed_is_reproducible(self):
        """Two fresh instances with same seed should agree on first call."""
        s1 = RandomScheduler(probability=0.5, seed=42)
        s2 = RandomScheduler(probability=0.5, seed=42)
        assert s1(0) == s2(0)


# --- CustomScheduler ---

class TestCustomScheduler:
    def test_custom_function(self):
        sched = CustomScheduler(event_function=lambda t: t % 2 == 0)
        assert sched(0) is True
        assert sched(1) is False
        assert sched(2) is True

    def test_outside_range(self):
        sched = CustomScheduler(event_function=lambda t: True, start=5, end=10)
        assert sched(3) is False
        assert sched(11) is False
        assert sched(7) is True

    def test_custom_always_false(self):
        sched = CustomScheduler(event_function=lambda t: False)
        for t in range(10):
            assert sched(t) is False

    def test_custom_always_true(self):
        sched = CustomScheduler(event_function=lambda t: True)
        for t in range(10):
            assert sched(t) is True

    def test_time_dependent_function(self):
        """Event function that fires only at prime-ish times."""
        primes = {2, 3, 5, 7, 11, 13}
        sched = CustomScheduler(event_function=lambda t: t in primes)
        assert sched(2) is True
        assert sched(4) is False
        assert sched(7) is True

    def test_stateful_function(self):
        """Event function that tracks its own state (fires every other call)."""
        state = {"count": 0}
        def every_other(t):
            state["count"] += 1
            return state["count"] % 2 == 0
        sched = CustomScheduler(event_function=every_other)
        results = [sched(t) for t in range(6)]
        assert results == [False, True, False, True, False, True]

    def test_range_overrides_function(self):
        """Even if function returns True, outside range should return False."""
        sched = CustomScheduler(event_function=lambda t: True, start=10, end=20)
        assert sched(5) is False
        assert sched(15) is True
        assert sched(25) is False


# --- MemorylessScheduler ---

class TestMemorylessScheduler:
    def test_returns_bool(self):
        sched = MemorylessScheduler(p=0.5, seed=42)
        result = sched(0)
        assert isinstance(result, (bool, np.bool_))

    def test_eventually_fires(self):
        sched = MemorylessScheduler(p=0.5, seed=42)
        fired = False
        for t in range(1000):
            if sched(t):
                fired = True
                break
        assert fired, "MemorylessScheduler never fired in 1000 steps"

    def test_resamples_after_fire(self):
        sched = MemorylessScheduler(p=1.0, seed=42)
        # With p=1.0, geometric distribution always returns 1
        # So it should fire at t=1, then schedule next at t=2, etc.
        initial_transition = sched.transition_time.copy()
        # Find the transition time and trigger it
        t = int(initial_transition[0])
        result = sched(t)
        assert result is True
        # After firing, transition_time should be updated
        assert sched.transition_time != initial_transition

    def test_seed_creates_deterministic_schedule(self):
        sched1 = MemorylessScheduler(p=0.3, seed=99)
        sched2 = MemorylessScheduler(p=0.3, seed=99)
        assert np.array_equal(sched1.transition_time, sched2.transition_time)

    def test_respects_start_end(self):
        """Should not fire before start or after end."""
        sched = MemorylessScheduler(p=1.0, start=10, end=20, seed=42)
        # With p=1.0, geometric always returns 1, so first transition_time = 1
        # Should not fire before start=10
        assert sched(1) is False

    def test_fires_multiple_times(self):
        """With p=1.0, should fire at every step (geometric(1.0) always returns 1)."""
        sched = MemorylessScheduler(p=1.0, seed=42)
        fire_times = []
        for t in range(10):
            if sched(t):
                fire_times.append(t)
        assert len(fire_times) > 1, "Should fire more than once over 10 steps"

    def test_inter_event_times_are_geometric(self):
        """Collect inter-event times and verify they follow geometric(p)."""
        sched = MemorylessScheduler(p=0.5, seed=42)
        fire_times = []
        for t in range(5000):
            if sched(t):
                fire_times.append(t)
        assert len(fire_times) >= 10, "Need enough fires for statistical test"
        # Inter-event times
        gaps = [fire_times[i+1] - fire_times[i] for i in range(len(fire_times) - 1)]
        mean_gap = np.mean(gaps)
        # Expected mean of geometric(0.5) = 1/p = 2
        assert abs(mean_gap - 2.0) < 0.5, (
            f"Expected mean inter-event time ~2.0, got {mean_gap}"
        )

    def test_high_p_fires_frequently(self):
        sched = MemorylessScheduler(p=0.9, seed=42)
        fires = sum(1 for t in range(100) if sched(t))
        # geometric(0.9) has mean 1/0.9 ≈ 1.11, so nearly every step
        assert fires > 50

    def test_low_p_fires_rarely(self):
        sched = MemorylessScheduler(p=0.01, seed=42)
        fires = sum(1 for t in range(100) if sched(t))
        # geometric(0.01) has mean 100, so very few fires in 100 steps
        assert fires < 10

    def test_different_seeds_give_different_schedules(self):
        sched1 = MemorylessScheduler(p=0.3, seed=1)
        sched2 = MemorylessScheduler(p=0.3, seed=999)
        assert not np.array_equal(sched1.transition_time, sched2.transition_time)


# --- BurstScheduler ---

class TestBurstScheduler:
    def test_fires_during_on_phase(self):
        """Should fire for the first on_duration steps of each cycle."""
        sched = BurstScheduler(on_duration=3, off_duration=2)
        # Cycle length = 5: on at 0,1,2 off at 3,4 on at 5,6,7 off at 8,9
        assert sched(0) is True
        assert sched(1) is True
        assert sched(2) is True

    def test_silent_during_off_phase(self):
        sched = BurstScheduler(on_duration=3, off_duration=2)
        assert sched(3) is False
        assert sched(4) is False

    def test_full_cycle_pattern(self):
        """Verify two full cycles of on/off behavior."""
        sched = BurstScheduler(on_duration=2, off_duration=3)
        # cycle=5: on at 0,1 off at 2,3,4 on at 5,6 off at 7,8,9
        expected = [True, True, False, False, False,
                    True, True, False, False, False]
        results = [sched(t) for t in range(10)]
        assert results == expected

    def test_respects_start_end(self):
        sched = BurstScheduler(on_duration=2, off_duration=2, start=5, end=15)
        assert sched(0) is False   # before start
        assert sched(4) is False   # before start
        assert sched(5) is True    # in range, on phase of cycle
        assert sched(16) is False  # after end

    def test_on_one_off_one(self):
        """Alternating on/off like a toggle."""
        sched = BurstScheduler(on_duration=1, off_duration=1)
        expected = [True, False, True, False, True, False]
        results = [sched(t) for t in range(6)]
        assert results == expected

    def test_long_on_short_off(self):
        sched = BurstScheduler(on_duration=5, off_duration=1)
        # cycle=6: on 0-4, off 5, on 6-10, off 11
        for t in range(5):
            assert sched(t) is True
        assert sched(5) is False
        assert sched(6) is True

    def test_returns_bool_type(self):
        sched = BurstScheduler(on_duration=3, off_duration=3)
        assert isinstance(sched(0), bool)
        assert isinstance(sched(3), bool)

    def test_large_time_values(self):
        """Pattern should repeat correctly at large t."""
        sched = BurstScheduler(on_duration=2, off_duration=3)
        # cycle=5, t=1000: 1000 % 5 = 0 -> on
        assert sched(1000) is True
        # t=1002: 1002 % 5 = 2 -> off
        assert sched(1002) is False


# --- DecayingProbabilityScheduler ---

class TestDecayingProbabilityScheduler:
    def test_returns_bool(self):
        sched = DecayingProbabilityScheduler(
            initial_probability=0.5, decay_rate=0.01, seed=42
        )
        assert isinstance(sched(0), (bool, np.bool_))

    def test_high_initial_probability_fires_early(self):
        """With p=1.0 and slow decay, should fire at t=0."""
        sched = DecayingProbabilityScheduler(
            initial_probability=1.0, decay_rate=0.001, seed=42
        )
        assert sched(0) is True

    def test_decayed_to_zero_never_fires(self):
        """With very high decay rate, probability drops to ~0 quickly."""
        sched = DecayingProbabilityScheduler(
            initial_probability=0.5, decay_rate=10.0, seed=42
        )
        # At t=100, p = 0.5 * exp(-10*100) ≈ 0
        fires = sum(1 for t in range(100, 200) if sched(t))
        assert fires == 0

    def test_fires_more_early_than_late(self):
        """Early time steps should fire more often than later ones."""
        sched = DecayingProbabilityScheduler(
            initial_probability=0.8, decay_rate=0.02, seed=42
        )
        early_fires = sum(1 for t in range(500) if sched(t))

        sched2 = DecayingProbabilityScheduler(
            initial_probability=0.8, decay_rate=0.02, seed=42
        )
        # Fast-forward through early steps (consuming RNG state)
        for t in range(500):
            sched2(t)
        late_fires = sum(1 for t in range(500, 1000) if sched2(t))

        assert early_fires > late_fires

    def test_seed_reproducibility(self):
        results1 = [
            DecayingProbabilityScheduler(
                initial_probability=0.5, decay_rate=0.01, seed=123
            )(t) for t in range(50)
        ]
        results2 = [
            DecayingProbabilityScheduler(
                initial_probability=0.5, decay_rate=0.01, seed=123
            )(t) for t in range(50)
        ]
        assert results1 == results2

    def test_respects_start_end(self):
        sched = DecayingProbabilityScheduler(
            initial_probability=1.0, decay_rate=0.0, start=5, end=10, seed=42
        )
        assert sched(3) is False
        assert sched(12) is False
        assert sched(5) is True

    def test_zero_decay_rate_is_constant(self):
        """With decay_rate=0, should behave like RandomScheduler."""
        sched = DecayingProbabilityScheduler(
            initial_probability=1.0, decay_rate=0.0, seed=42
        )
        for t in range(100):
            assert sched(t) is True

    def test_statistical_decay(self):
        """Firing rate in first 100 steps should be higher than in steps 400-500."""
        sched = DecayingProbabilityScheduler(
            initial_probability=0.9, decay_rate=0.01, seed=42
        )
        all_results = [sched(t) for t in range(500)]
        early_rate = sum(all_results[:100]) / 100
        late_rate = sum(all_results[400:]) / 100
        assert early_rate > late_rate


# --- WindowScheduler ---

class TestWindowScheduler:
    def test_fires_within_window(self):
        sched = WindowScheduler(windows=[(5, 10)])
        for t in range(5, 11):
            assert sched(t) is True

    def test_silent_outside_window(self):
        sched = WindowScheduler(windows=[(5, 10)])
        assert sched(0) is False
        assert sched(4) is False
        assert sched(11) is False

    def test_multiple_windows(self):
        sched = WindowScheduler(windows=[(0, 3), (10, 13), (20, 23)])
        assert sched(2) is True
        assert sched(5) is False
        assert sched(11) is True
        assert sched(15) is False
        assert sched(21) is True

    def test_single_point_window(self):
        """A window where start == end should fire at exactly that time."""
        sched = WindowScheduler(windows=[(5, 5)])
        assert sched(4) is False
        assert sched(5) is True
        assert sched(6) is False

    def test_adjacent_windows(self):
        """Two windows that share a boundary should both fire at the boundary."""
        sched = WindowScheduler(windows=[(0, 5), (5, 10)])
        assert sched(5) is True
        assert sched(3) is True
        assert sched(7) is True

    def test_respects_start_end(self):
        """Global start/end should override windows."""
        sched = WindowScheduler(windows=[(0, 100)], start=10, end=20)
        assert sched(5) is False    # in window but before global start
        assert sched(15) is True    # in window and in global range
        assert sched(25) is False   # in window but after global end

    def test_empty_windows_never_fires(self):
        sched = WindowScheduler(windows=[])
        for t in range(20):
            assert sched(t) is False

    def test_returns_bool_type(self):
        sched = WindowScheduler(windows=[(0, 10)])
        assert isinstance(sched(0), bool)
        assert isinstance(sched(15), bool)

    def test_large_gap_between_windows(self):
        sched = WindowScheduler(windows=[(0, 2), (1000, 1002)])
        assert sched(1) is True
        assert sched(500) is False
        assert sched(1001) is True

    def test_many_windows(self):
        """Build windows from periodic pattern and verify."""
        windows = [(i * 10, i * 10 + 3) for i in range(10)]
        sched = WindowScheduler(windows=windows)
        assert sched(0) is True
        assert sched(3) is True
        assert sched(5) is False
        assert sched(10) is True
        assert sched(14) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
