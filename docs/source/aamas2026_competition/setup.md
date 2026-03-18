# Environment Setup

Welcome to the NS-Gym competition setup guide!

This tutorial series will walk you through the full submission workflow:

1. This tutorial: create your repository, configure Python, and build Docker images.
2. [Build Your Agent](build_agents.md): implement a model-based or model-free agent and register it.
3. [Create a Custom Environment](make_env.md): define non-stationarity with schedulers and update functions.
4. [Evaluation Criteria](evaluation.md): understand how submissions are scored and ranked.
5. [Submit Your Agent](submission.md): run final checks and send your repository for evaluation.

Let's get started!




> “A journey of a thousand miles begins with a single step.” 
— Lao Tzu



## 1. Create your repository

1. Open the template repository: `https://github.com/scope-lab-vu/ns-gym-comp-template`.
2. Click **Use this template**.
3. Create a new repository called `ns-gym-comp-submission`.

## 2. Clone and configure remotes

Then, clone your new repository and add this template as an upstream remote so you can pull future updates (new environments, examples, evaluation changes, etc.):
```bash
git clone https://github.com/<your-username>/ns-gym-comp-submission.git
cd ns-gym-comp-submission
git remote add upstream https://github.com/scope-lab-vu/ns-gym-comp-template.git
git remote -v
```

You should see something like:

- `origin` -> your repository
- `upstream` -> `scope-lab-vu/ns-gym-comp-template`

## 3. Set up local Python with `uv`

We use `uv` for development. If needed, install `uv` first:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then, we use `uv` to create and activate the virtual environment, and download necessary packages:

```bash
uv venv --python 3.13
source .venv/bin/activate
uv pip install -e .
```

To verify the install:

```bash
uv pip list
```

## 4. Build the Docker images

Additionally, we will use Docker to create a standarized environment to evaluate all competitors. The docker consists of two images:

- `docker/base.Dockerfile`: base image with Python, uv, and project dependencies. Do not modify this file!
- `docker/eval.Dockerfile`: evaluation image that builds on the base. Add non-Python dependencies  (system packages, etc.) your agent needs.

To build both images, the easiest way is to [install Docker Desktop](https://www.docker.com/products/docker-desktop/), and run:

```bash
docker compose build
```

Note that all submitted agents will be evaluated using this Docker image with 32 GB RAM and a single NVIDIA GPU. GPU access is available inside the container — agents can use CUDA via PyTorch. Ensure your agent runs correctly within this environment.

## Outcome

For now, you should have:

- your own `ns-gym-comp-submission` repository,
- a local Python environment managed by `uv`,
- Docker images that match the evaluation environment,
- an `upstream` remote to pull template updates.


Need help? Join in our [Office Hour sessions](../aamas2026_competition.md#office-hours)!


In the next tutorial, we will cover the basics of building an agent using the Ns-Gym competition template. 
