# Installation

You can install NS-Gym using pip. 

```bash
pip install ns-gym
```

For nightly builds, you can install directly from the GitHub repository:

```bash
pip install git+https://github.com/scope-lab-vu/ns_gym
```

## Development and testing

We welcome any contributions to this NS-Gym project! If you find a bug or want to add a new feature, please feel free to open an issue or submit a pull request.

Fork then clone the repository, install the required dependencies in editable mode, and run the tests to ensure everything is working correctly. We use UV for package management. 

```bash
git clone https://github.com/scope-lab-vu/ns_gym.git
cd ns_gym
uv pip install -e ".[all]" 
```

To run all test in the project run:

```bash
pytest tests/
```

