# Evaluating Your Agent

Welcome! This is the fourth tutorial in a series for the full submission workflow:

1. [Environment Setup](setup.md): create your repository, configure Python, and build Docker images.
2. [Build Your Agent](build_agents.md): implement a model-based or model-free agent and register it.
3. [Create a Custom Environment](make_env.md): define non-stationarity with schedulers and update functions.
4. This tutorial: understand how submissions are scored and ranked.
5. [Submit Your Agent](submission.md): run final checks and send your repository for evaluation.

For each submission, we will evaluate and rank according to four criteria. When you run the evaluator through Docker, you should see your scoring metrics on your machine. We will officiate your result after submission.

We mainly evaluate based on:

1. **Adaptability**: a measure of how fast an agent could adapt to change. At an unknown timestep, we will make a change to the non-stationary environment parameter. Your algorithm needs to learn to recover from initial failures, and adapt to find a good solution.

   To evaluate, we will have an oracle solution to compare against your submission, starting from the time of change until some timestep of `max(N, terminates)`.

   *Unnotify category only.*

2. **Performance**: this is the average undiscounted episodic reward achieved under non-stationary conditions. The environment we use to evaluate this will be more non-stationary.

   *Unnotify and partial-notify categories.*

3. **Resilience**: a good algorithm needs a robust policy that still thrives with slight perturbations. We want to measure the agent's performance immediately after the change, before the agent has time to adapt. To get ranked on this leaderboard, agents have to pass a specific performance threshold.

   *Fully-notify only.*

4. **Efficiency**: this is measured in two aspects: fewer timesteps and less wall-clock (real-life) time the agent consumes in finding a solution. We rank submissions based on the ratio between the two. To get ranked on this leaderboard, agents have to pass a specific performance threshold.

   *Unnotify and partial-notify categories.*

## Next step

Ready to send your code? Go to [Submit Your Agent](submission.md) for the final checklist and submission steps.
