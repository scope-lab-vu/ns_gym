import pytest
import ns_gym
import inspect
import ns_gym.schedulers as sched_mod



scheduler_classes = []
for name, obj in inspect.getmembers(ns_gym.schedulers):
    if inspect.isclass(obj) and obj.__module__ == ns_gym.schedulers.__name__:
        scheduler_classes.append(obj)


@pytest.mark.parametrize("Scheduler", scheduler_classes)
def test_inheritance(Scheduler):
    """Tests that all schedulers inherit from Scheduler."""
    assert issubclass(Scheduler, ns_gym.base.Scheduler)

if __name__ == "__main__":
    pass