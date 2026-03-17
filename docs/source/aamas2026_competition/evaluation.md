# Evaluating Your Agent

For each submission, we will evaluate and rank according to four criteria. When you run the evaluator through docker, you should see your scoring metrics produced in your machine. We will offciate your result after the submission.

We mainly evaluate based on:

1. **Adaptability**: a measure of how fast an agent could adapt to change. At an unknown timestep, we will make change to the non-stationarity environment parameter. Your algorithm need to learn to recover from initial failures, and adapt to find a good solution. 

To evaluate, we will have an oracle solution to compare against your submission, starting from the time of change, until some timestep of max(N, terminates).

Unnotify category only.

2. **Performance**: this is the average undiscounted episodic reward achieved under non-stationary conditions. The environment we use to evaluate this will be more non-stationary. 

Unnotify and partial-notify categories.

3. **Resilience**: a good algorithm needs a robust policy that still thrives with slight pertubations. We want to measure the agent's performance immediately after the change, before the agent has time to adapt. To get ranked on this leaderboard, agents have to pass a specific performance threshold.

Fully-notify only.

4. **Efficiency**: this is measured in two aspects - less timesteps and less wall-clock (real-life) time the agent consumes in finding a solution. We rank submission based on the ratio between the two. To get ranked on this leaderboard, agents have to pass a specific performance threshold.

Unnotify, and partial-notify categories.