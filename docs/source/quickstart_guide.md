# Quickstart Guide

## Installation

Create a virtual environment (optional but recommended). Here we use `uv` for development, but you can also use `venv` or `conda`.

```bash
uv venv
source venv/bin/activate 
```

To install NS-Gym, you can use pip. For now we recommend pip installing from our GitHub repository to ensure you have the latest version:

```bash
uv pip install git+https://github.com/scope-lab-vu/ns_gym
```

## Building a non-stationary environment

We can build a non-stationary environment in six lines of code. First, we import the necessary modules from NS-Gym and Gymnasium:

```python
import ns_gym
import gymnasium as gym
```

We will then import the necessary wrappers to create a non-stationary environment. 

Then as you would in Gymnasium, we create an environment.

```python
env = gym.make("CartPole-v1")
```
