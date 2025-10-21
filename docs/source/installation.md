# Installation

There are two options to install NS-Gym, `pip` install from PyPI, or build directly from source.


```bash
pip install ns_gym
```

To install the latest development version from GitHub, use:

```bash
pip install git+https://github.com/scope-lab-vu/ns_gym
```

NS-Gym welcomes contributions! If you would like to contribute, please consider installing from source as well as the development dependencies (see below).

```bash
git clone https://github.com/scope-lab-vu/ns_gym
cd ns_gym
pip install -e .[dev]
```
This will install NS-Gym in "editable" mode, meaning that any changes you make to the source code will be reflected immediately without needing to reinstall. The `[dev]` option installs additional packages useful for development, such as testing and documentation tools.

