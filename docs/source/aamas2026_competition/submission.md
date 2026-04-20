# Submit Your Agent


This tutorial series will walk you through the full submission workflow:

- [Environment Setup](setup.md): create your repository, configure Python, and build Docker images.
- [Build Your Agent](build_agents.md): implement a model-based or model-free agent and register it.
- [Create a Custom Environment](make_env.md): define non-stationarity with schedulers and update functions.
- [Evaluation Criteria](evaluation.md): understand how submissions are scored and ranked.
- This tutorial: run final checks and send your repository for evaluation.

Let's get started!


> “One who travels a hundred miles sees ninety as only half” – The Strategies of the Warring States, Volume 5

## 1. Final local checks

Run both checks before submission:

```bash
uv run python evaluator.py
docker compose run --rm test-submission
```

## 2. Verify repository contents

Confirm the following are present and up to date:

- `submission.py` returns your agent from `get_agent(env_id)`.
- model files are in `models/`.
- Python dependencies are in `pyproject.toml`.
- non-Python dependencies are in `docker/eval.Dockerfile`.

## 3. Grant organizer access

If your repository is private, add these GitHub users as collaborators:

- `nkepling`
- `ayanmukhopadhyay`

## 4. Notify the organizers

When ready, either:

- open an issue in `scope-lab-vu/ns-gym-comp-template` with your repository link, or
- email:
  - `nathaniel.s.keplinger [at] vanderbilt.edu`
  - `amukhopadhyay [at] wm.edu`
  - `yli113 [at] wm.edu`

## What happens next

We will pull your repository, run standardized evaluation, and report results. Congrats!
