import gymnasium as gym
from ns_gym.wrappers import (
    NSClassicControlWrapper,
    NSBridgeWrapper,
    NSCliffWalkingWrapper,
    NSFrozenLakeWrapper,
)
from ns_gym.schedulers import ContinuousScheduler
from ns_gym.update_functions import StepWiseUpdate, DistributionStepWiseUpdate
import numpy as np
from stable_baselines3 import PPO

import matplotlib.pyplot as plt
import argparse
import os
from datetime import datetime


def make_env_with_context(
    env_name, context_value, context_parameter_name="masscart", seed=None
):
    """Creates a ns_gym environment with a specified context parameter value.

    Args:
        env_name (str): Gymnasium environement name.
        context_value (float): The value for the context parameter.
        context_parameter_name (str): The name of the parameter to tune.
        seed (Optional[int]): Seed for the environment.

    Returns:
        gym.Env: The configured environment.

    Caution:
        Not implemented for all environments. Currently supports:
            - Classic Control: CartPole-v1, Acrobot-v1, MountainCarContinuous-v0, MountainCar-v0, Pendulum-v1
            - Gridworlds: CliffWalking-v0, FrozenLake-v1, ns_gym/Bridge-v0
    """

    if env_name in [
        "CartPole-v1",
        "Acrobot-v1",
        "MountainCarContinuous-v0",
        "MountainCar-v0",
        "Pendulum-v1",
    ]:
        env = gym.make(env_name)

        scheduler = ContinuousScheduler(start=0, end=0)
        update_fn = StepWiseUpdate(
            scheduler, [context_value]
        )  # Provides the context value

        tunable_params = {context_parameter_name: update_fn}

        ns_env = NSClassicControlWrapper(
            env=env,
            tunable_params=tunable_params,
            change_notification=True,
            delta_change_notification=True,
        )

    elif env_name in ["CliffWalking-v0", "FrozenLake-v1", "ns_gym/Bridge-v0"]:
        env = gym.make(env_name)

        scheduler = ContinuousScheduler(start=0, end=0)
        update_fn = DistributionStepWiseUpdate(
            scheduler, [context_value]
        )  # Provides the context value

        tunable_params = {context_parameter_name: update_fn}

        if env_name == "FrozenLake-v1":
            ns_env = NSFrozenLakeWrapper(
                env,
                tunable_params,
                change_notification=True,
                delta_change_notification=True,
                initial_prob_dist=[
                    context_value,
                    (1 - context_value) / 2,
                    (1 - context_value) / 2,
                ],
            )
        elif env_name == "CliffWalking-v0":
            ns_env = NSCliffWalkingWrapper(
                env,
                tunable_params,
                change_notification=True,
                delta_change_notification=True,
                initial_prob_dist=[
                    context_value,
                    (1 - context_value) / 3,
                    (1 - context_value) / 3,
                    (1 - context_value) / 3,
                ],
            )
        elif env_name == "ns_gym/Bridge-v0":
            ns_env = NSBridgeWrapper(
                env,
                tunable_params,
                change_notification=True,
                delta_change_notification=True,
                initial_prob_dist=[
                    context_value,
                    (1 - context_value) / 2,
                    (1 - context_value) / 2,
                ],
            )

    else:
        raise ValueError("Invalid environment")

    ns_env = gym.wrappers.TransformObservation(ns_env, lambda obs: obs["state"], ns_env.unwrapped.observation_space)
    ns_env = gym.wrappers.TransformReward(ns_env, lambda rew: rew.reward)

    if seed is not None:
        ns_env.reset(seed=seed)
    return ns_env


def run_context_episode(agent, ns_env_instance, num_episodes):
    """Runs an StableBaselines3 policy in a given ns_gym environment for a number of episodes.

    Args:
        agent (StableBaselines3 Policy): The trained agent/policy to evaluate.
        ns_env_instance (gym.Env): The ns_gym environment instance.
        num_episodes (int): Number of episodes to run.
    """

    reward_list = []
    for ep in range(num_episodes):
        ep_reward = 0.0
        done = False
        truncated = False
        obs, info = ns_env_instance.reset()

        if not isinstance(obs, np.ndarray) and hasattr(obs, "state"):
            obs = np.array(obs.state, dtype=np.float32)
        elif not isinstance(obs, np.ndarray):
            obs = np.array(obs, dtype=np.float32)

        while not (done or truncated):
            action, _states = agent.predict(obs, deterministic=True)
            obs, reward, done, truncated, current_info = ns_env_instance.step(action)

            if not isinstance(obs, np.ndarray) and hasattr(obs, "state"):
                obs = np.array(obs.state, dtype=np.float32)
            elif not isinstance(obs, np.ndarray):
                obs = np.array(obs, dtype=np.float32)

            if not isinstance(reward, (float, int)) and hasattr(reward, "reward"):
                reward = float(reward.reward)
            elif not isinstance(reward, (float, int)):
                reward = float(reward)

            ep_reward += reward
        reward_list.append(ep_reward)
    return np.mean(reward_list), np.std(reward_list)


def eval_target_contexts(
    policy, make_env_func_partial, num_episodes_per_context, target_context_range
):
    """Evaluates a given policy across a range of target contexts.
    Args:
        policy: The trained StableBaselines3 agent/policy to evaluate. Should be compatible with StableBaselines3.
        make_env_func_partial: A partial function of make_env_with_context (with context_parameter_name fixed).
        num_episodes_per_context: How many episodes to run for each target context.
        target_context_range: Array of target context values to evaluate on.
    Returns:
        Array of mean rewards for each target context, Array of std deviations.
    """
    mean_rewards = np.zeros(len(target_context_range))
    std_rewards = np.zeros(len(target_context_range))

    for i, target_ctx_val in enumerate(target_context_range):
        eval_env = make_env_func_partial(
            context_value=target_ctx_val
        )  # Pass only context_value
        mean_rewards[i], std_rewards[i] = run_context_episode(
            policy, eval_env, num_episodes_per_context
        )
        eval_env.close()
    return mean_rewards, std_rewards


def normalize_rewards_matrix(U_matrix_raw):
    """Normalizes a reward matrix using min-max scaling (0-1 range)."""
    min_val = np.min(U_matrix_raw)
    max_val = np.max(U_matrix_raw)

    if max_val == min_val:
        U_matrix_normalized = np.full_like(U_matrix_raw, 0.5 if min_val != 0 else 0.0)
        return U_matrix_normalized, min_val, max_val

    U_matrix_normalized = (U_matrix_raw - min_val) / (max_val - min_val)
    return U_matrix_normalized, min_val, max_val


def calculate_sem(data_array):
    """Calculates the standard error of the mean for a 1D array."""
    if len(data_array) < 2:
        return 0.0
    return np.std(data_array, ddof=1) / np.sqrt(len(data_array))


def calculate_generalized_performance(
    agent_list,
    trained_agent_source_contexts,
    make_env_func_partial,
    num_episodes_per_eval_context,
    evaluation_target_context_range,
    normalize=True,
):
    """
    Calculates the generalized performance matrix U (raw and normalized),
    related metrics, and their standard errors.
    """
    num_agents = len(agent_list)
    num_target_contexts = len(evaluation_target_context_range)
    U_matrix_raw = np.zeros((num_agents, num_target_contexts))

    print("\n--- Evaluating Generalized Performance (Raw Rewards) ---")
    for i, agent in enumerate(agent_list):
        source_ctx = trained_agent_source_contexts[i]
        print(
            f"  Evaluating Agent {i + 1}/{num_agents} (trained on source context: {source_ctx:.2f})..."
        )

        mean_rewards_for_agent, _ = eval_target_contexts(
            agent,
            make_env_func_partial,
            num_episodes_per_eval_context,
            evaluation_target_context_range,
        )
        U_matrix_raw[i, :] = mean_rewards_for_agent

    peak_perf_per_policy_raw = np.max(U_matrix_raw, axis=1)
    U_envelope_raw = np.max(U_matrix_raw, axis=0)
    overall_system_perf_paper_raw = np.mean(U_envelope_raw)
    sem_overall_system_perf_paper_raw = calculate_sem(U_envelope_raw)

    U_matrix_normalized = None
    peak_perf_per_policy_normalized = None
    overall_system_perf_paper_normalized = None
    sem_overall_system_perf_paper_normalized = None
    normalization_params = {}

    if normalize:
        print("\n--- Normalizing Rewards (Min-Max Scaling) ---")
        U_matrix_normalized, min_r, max_r = normalize_rewards_matrix(U_matrix_raw)
        normalization_params = {
            "min_reward_observed": min_r,
            "max_reward_observed": max_r,
        }

        peak_perf_per_policy_normalized = np.max(U_matrix_normalized, axis=1)
        U_envelope_normalized = np.max(U_matrix_normalized, axis=0)
        overall_system_perf_paper_normalized = np.mean(U_envelope_normalized)
        sem_overall_system_perf_paper_normalized = calculate_sem(U_envelope_normalized)
        print(
            f"Normalization complete. Min observed: {min_r:.2f}, Max observed: {max_r:.2f}"
        )

    print("--- Evaluation Complete ---")
    return (
        U_matrix_raw,
        peak_perf_per_policy_raw,
        overall_system_perf_paper_raw,
        sem_overall_system_perf_paper_raw,
        U_matrix_normalized,
        peak_perf_per_policy_normalized,
        overall_system_perf_paper_normalized,
        sem_overall_system_perf_paper_normalized,
        normalization_params,
    )


def plot_performance_curves(
    U_matrix_to_plot,
    eval_context_range,
    train_source_contexts,
    overall_perf_to_display,
    context_param_name,
    is_normalized,
    output_filename=None,
):
    """
    Plots performance curves (either raw or normalized) and saves the plot.
    """
    plt.figure(figsize=(12, 8))
    for i in range(U_matrix_to_plot.shape[0]):
        plt.plot(
            eval_context_range,
            U_matrix_to_plot[i, :],
            label=f"Trained on Ctx: {train_source_contexts[i]:.2f}",
            alpha=0.7,
        )

    U_envelope_to_plot = np.max(U_matrix_to_plot, axis=0)
    plt.plot(
        eval_context_range,
        U_envelope_to_plot,
        color="k",
        linestyle="--",
        linewidth=2,
        label=f"Upper Envelope (Overall Perf: {overall_perf_to_display:.2f})",
    )

    y_label = "Mean Reward during Evaluation"
    plot_title = f"Zero-Shot Transfer Performance on CartPole ({context_param_name})"
    if is_normalized:
        y_label = "Normalized " + y_label
        plot_title = "Normalized " + plot_title
        plt.ylim(-0.05, 1.05)

    plt.xlabel(f"Target Context ({context_param_name})")
    plt.ylabel(y_label)
    plt.title(plot_title)
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()

    if output_filename:
        try:
            plt.savefig(output_filename)
            print(f"Plot saved to {output_filename}")
        except Exception as e:
            print(f"Error saving plot to {output_filename}: {e}")
    plt.show()


def save_metrics_to_file(
    filename,
    U_matrix_raw,
    peak_performances_raw,
    overall_system_performance_raw,
    sem_overall_raw,
    U_matrix_normalized,
    peak_performances_normalized,
    overall_system_performance_normalized,
    sem_overall_norm,
    normalization_params,
    source_contexts,
    context_range,
    args_dict,
):
    """
    Saves both raw and normalized metrics, including SEM, to a text file.
    """
    try:
        with open(filename, "w") as f:
            f.write("--- Experiment Configuration ---\n")
            for key, value in args_dict.items():
                f.write(f"{key}: {value}\n")
            f.write("\n--- Source Contexts for Training ---\n")
            f.write(f"{source_contexts.tolist()}\n")
            f.write("\n--- Target Contexts Evaluated ---\n")
            f.write(f"{np.round(context_range, 3).tolist()}\n")

            # --- Raw Metrics ---
            f.write("\n\n--- RAW METRICS ---\n")
            f.write("Overall System Generalized Performance (Raw):\n")
            f.write(
                f"Mean: {overall_system_performance_raw:.4f}, SEM: {sem_overall_raw:.4f}\n"
            )
            f.write("\nPeak Performance for each Policy (Raw):\n")
            for i, peak_p in enumerate(peak_performances_raw):
                f.write(
                    f"  Agent trained on Ctx {source_contexts[i]:.2f}: Max Reward = {peak_p:.4f}\n"
                )
            f.write("\nPerformance Matrix U (Raw):\n")
            for i in range(U_matrix_raw.shape[0]):
                row_str = ", ".join([f"{val:.4f}" for val in U_matrix_raw[i, :]])
                f.write(f"Agent {i} (Ctx {source_contexts[i]:.2f}): [{row_str}]\n")

            # --- Normalized Metrics ---
            if (
                U_matrix_normalized is not None
                and overall_system_performance_normalized is not None
                and sem_overall_norm is not None
            ):
                f.write("\n\n--- NORMALIZED METRICS ---\n")
                f.write(
                    f"Normalization Parameters: Min Observed = {normalization_params.get('min_reward_observed', 'N/A'):.4f}, Max Observed = {normalization_params.get('max_reward_observed', 'N/A'):.4f}\n"
                )
                f.write("Overall System Generalized Performance (Normalized):\n")
                f.write(
                    f"Mean: {overall_system_performance_normalized:.4f}, SEM: {sem_overall_norm:.4f}\n"
                )
                f.write("\nPeak Performance for each Policy (Normalized):\n")
                for i, peak_p in enumerate(peak_performances_normalized):  # type: ignore
                    f.write(
                        f"  Agent trained on Ctx {source_contexts[i]:.2f}: Max Reward = {peak_p:.4f}\n"
                    )
                f.write("\nPerformance Matrix U (Normalized):\n")
                for i in range(U_matrix_normalized.shape[0]):
                    row_str = ", ".join(
                        [f"{val:.4f}" for val in U_matrix_normalized[i, :]]
                    )
                    f.write(f"Agent {i} (Ctx {source_contexts[i]:.2f}): [{row_str}]\n")

        print(f"Metrics saved to {filename}")
    except Exception as e:
        print(f"Error saving metrics to {filename}: {e}")


if __name__ == "__main__":
    # Example test context switching experiment code.
    parser = argparse.ArgumentParser(
        description="Run Model-Based Transfer Learning Evaluation for CartPole."
    )
    parser.add_argument(
        "--timesteps_train",
        type=int,
        default=30000,
        help="Total timesteps to train each agent.",
    )
    parser.add_argument(
        "--episodes_eval",
        type=int,
        default=20,
        help="Number of episodes for evaluation on each target context.",
    )
    parser.add_argument(
        "--context_param",
        type=str,
        default="masscart",
        help="Environment parameter to modify (e.g., 'masscart', 'length').",
    )
    parser.add_argument(
        "--num_target_contexts",
        type=int,
        default=100,
        help="Number of points in the target context range for evaluation.",
    )
    parser.add_argument(
        "--target_context_min",
        type=float,
        default=0.1,
        help="Minimum value for the target context range.",
    )
    parser.add_argument(
        "--target_context_max",
        type=float,
        default=10.0,
        help="Maximum value for the target context range.",
    )
    parser.add_argument("--env_name", type=str, help="Environment name")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_output_dir = f"experiment_results_{timestamp}"

    parser.add_argument(
        "--output_dir",
        type=str,
        default=default_output_dir,
        help="Directory to save plot and metrics files.",
    )
    parser.add_argument(
        "--plot_file",
        type=str,
        default="performance_plot.png",
        help="Filename for the saved plot (relative to output_dir).",
    )
    parser.add_argument(
        "--metrics_file",
        type=str,
        default="performance_metrics.txt",
        help="Filename for the saved metrics (relative to output_dir).",
    )
    parser.add_argument(
        "--normalize_rewards",
        type=bool,
        default=True,
        help="normalize generalized performance",
    )

    args = parser.parse_args()

    from stable_baselines3.common.env_util import make_vec_env

    SOURCE_CONTEXTS = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])

    # SOURCE_CONTEXTS = np.linspace(0.0025 - 0.001, 0.05,9)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory: {args.output_dir}")

    plot_filepath = os.path.join(args.output_dir, args.plot_file)
    metrics_filepath = os.path.join(args.output_dir, args.metrics_file)

    # Generate context_range based on args
    current_context_range = np.linspace(
        args.target_context_min, args.target_context_max, args.num_target_contexts
    )

    context_size = len(current_context_range)

    random_integers = np.random.choice(context_size, size=9, replace=False)
    SOURCE_CONTEXTS = current_context_range[random_integers]

    print("--- Starting Experiment: Model-Based Transfer Learning Evaluation ---")
    print("Using arguments:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")

    trained_agents = []

    # Use functools.partial to pass the fixed context_parameter_name to make_env_with_context
    from functools import partial

    make_env_partial_fn = partial(
        make_env_with_context,
        env_name=args.env_name,
        context_parameter_name=args.context_param,
    )

    print(f"\n--- Training {len(SOURCE_CONTEXTS)} Agents ---")
    for i, s_ctx in enumerate(SOURCE_CONTEXTS):
        print(
            f"Training Agent {i + 1}/{len(SOURCE_CONTEXTS)} for source context ({args.context_param}): {s_ctx:.2f}"
        )

        # Pass the partial function to make_vec_env
        # The lambda now only needs to provide context_value and seed
        train_env = make_vec_env(
            lambda: make_env_partial_fn(context_value=s_ctx, seed=i), n_envs=1
        )

        model = PPO(
            "MlpPolicy",
            train_env,
            verbose=0,
            tensorboard_log=None,
            device="auto",
            seed=i,
        )

        # model = DQN(
        #     "MlpPolicy",
        #     train_env,
        #     verbose=0,
        #     tensorboard_log=None,
        #     device="auto",
        #     seed=i
        # )

        model.learn(total_timesteps=args.timesteps_train, progress_bar=True)
        trained_agents.append(model)
        train_env.close()

    print("--- All Agents Trained ---")

    (
        U_raw,
        peaks_raw,
        overall_raw,
        sem_raw,
        U_norm,
        peaks_norm,
        overall_norm,
        sem_norm,
        norm_params,
    ) = calculate_generalized_performance(
        trained_agents,
        SOURCE_CONTEXTS,
        make_env_partial_fn,
        args.episodes_eval,
        current_context_range,
        normalize=args.normalize_rewards,
    )

    print("\n--- Results Summary ---")
    print(f"Shape of Performance Matrix U (Raw): {U_raw.shape}")
    if U_norm is not None:
        print(f"Shape of Performance Matrix U (Normalized): {U_norm.shape}")

    print("\nPeak Performance for each Policy (Raw):")
    for i, peak_p in enumerate(peaks_raw):
        print(
            f"  Agent trained on Ctx {SOURCE_CONTEXTS[i]:.2f}: Max Reward = {peak_p:.2f}"
        )
    if peaks_norm is not None:
        print("\nPeak Performance for each Policy (Normalized):")
        for i, peak_p in enumerate(peaks_norm):
            print(
                f"  Agent trained on Ctx {SOURCE_CONTEXTS[i]:.2f}: Max Reward = {peak_p:.2f}"
            )

    print(
        f"\nOverall System Generalized Performance (Paper's V-metric, Raw): Mean = {overall_raw:.2f}, SEM = {sem_raw:.4f}"
    )
    if overall_norm is not None and sem_norm is not None:
        print(
            f"Overall System Generalized Performance (Paper's V-metric, Normalized): Mean = {overall_norm:.2f}, SEM = {sem_norm:.4f}"
        )

    print("\nSaving metrics and plotting performance curves...")
    save_metrics_to_file(
        metrics_filepath,
        U_raw,
        peaks_raw,
        overall_raw,
        sem_raw,
        U_norm,
        peaks_norm,
        overall_norm,
        sem_norm,
        norm_params,
        SOURCE_CONTEXTS,
        current_context_range,
        vars(args),
    )

    # Decide which matrix to plot based on normalization flag
    if args.normalize_rewards and U_norm is not None:
        plot_performance_curves(
            U_norm,
            current_context_range,
            SOURCE_CONTEXTS,
            overall_norm,
            args.context_param,
            True,
            plot_filepath,
        )
    else:
        plot_performance_curves(
            U_raw,
            current_context_range,
            SOURCE_CONTEXTS,
            overall_raw,
            args.context_param,
            False,
            plot_filepath,
        )

    print("\n--- Experiment Finished ---")
