import pytest
import ns_gym
import inspect
import ns_gym.update_functions as dist_mod



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
    """Tests that all update functions implement the 'update' method."""
    assert hasattr(UpdateFn, "_update"), f"{UpdateFn.__name__} does not implement '_update' method"




if __name__ == "__main__":
    pass


        