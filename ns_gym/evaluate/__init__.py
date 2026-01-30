import importlib as _importlib

from .metrics import EnsembleMetric, PAMCTS_Bound

# Lazy-loaded attributes from run_experiment (pulls in torch, pandas)
_LAZY_IMPORTS = {
    "run_experiment": (".run_experiment", "run_experiment"),
    "run_episode": (".run_experiment", "run_episode"),
    "read_experiment_results": (".run_experiment", "read_experiment_results"),
    "action_type_checker": (".run_experiment", "action_type_checker"),
    "array_to_list_if_array": (".run_experiment", "array_to_list_if_array"),
    "write_results_to_file": (".run_experiment", "write_results_to_file"),
}

__all__ = ["EnsembleMetric", "PAMCTS_Bound"]


def __getattr__(name):
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        module = _importlib.import_module(module_path, __name__)
        attr = getattr(module, attr_name)
        globals()[name] = attr
        return attr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
