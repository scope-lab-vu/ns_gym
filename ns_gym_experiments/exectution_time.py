import gymnasium as gym
import time
import numpy as np
import ns_gym
from ns_gym.wrappers import NSClassicControlWrapper,NSCliffWalkingWrapper,NSFrozenLakeWrapper,NSBridgeWrapper


def make_no_update_env(env_name,context_parameter_name):

    if env_name in ["CartPole-v1","Acrobot-v1","MountainCarContinuous-v0","MountainCar-v0","Pendulum-v1"]:
            
        env = gym.make(env_name)

        scheduler = ns_gym.schedulers.ContinuousScheduler()
        update_fn = ns_gym.update_functions.NoUpdate(scheduler)

        tunable_params = {context_parameter_name: update_fn}

        ns_env = NSClassicControlWrapper(
            env=env,
            tunable_params=tunable_params,
            change_notification=True,
            delta_change_notification=True
        ) 

    elif env_name in ["CliffWalking-v0","FrozenLake-v1", "ns_gym/Bridge-v0"]:
        env = gym.make(env_name)
        
        scheduler = ns_gym.schedulers.ContinuousScheduler()
        update_fn =  ns_gym.update_functions.DistributionNoUpdate(scheduler)# Provides the context value

        tunable_params = {context_parameter_name: update_fn}

        if env_name == "FrozenLake-v1":
            ns_env = NSFrozenLakeWrapper(env,
                                         tunable_params,
                                         change_notification=True,
                                         delta_change_notification=True)
        elif env_name == "CliffWalking-v0":
            ns_env = NSCliffWalkingWrapper(env,
                                           tunable_params,
                                           change_notification=True,
                                           delta_change_notification=True)
        elif env_name == "ns_gym/Bridge-v0":
            ns_env = NSBridgeWrapper(env,tunable_params,change_notification=True,delta_change_notification=True)
    
    else:
        raise ValueError("Invalid environment")




    # ns_env = gym.wrappers.TransformObservation(ns_env, lambda obs: obs.state,None)
    # ns_env = gym.wrappers.TransformReward(ns_env, lambda rew: rew.reward)

    return ns_env



# List of environments to test
env_names = [
    "FrozenLake-v1",
    "CliffWalking-v0",
    "Acrobot-v1",
    "MountainCar-v0",
    "CartPole-v1",
    "Pendulum-v1",
]

context_parameter_names = ["P","P","LINK_MASS_1","gravity","masscart","m"]

env_name_dict = dict(zip(env_names,context_parameter_names))

# Number of steps to collect data for timing
NUM_STEPS = 10000 # Using a sufficiently large number for meaningful SEM

# Dictionary to store results
# Each entry will be: env_name -> {"mean": mean_us, "sem": sem_us, "count": count}
results = {}

print(f"Measuring average step time and standard error over {NUM_STEPS} calls...\n")

# Iterate over each environment
for env_name,context_parameter_name in env_name_dict.items():
    env = None # Initialize env to None for the finally block
    step_durations_sec = [] # List to store individual step durations in seconds

    try:
        print(f"--- Testing NS Environment: {env_name} ---")

        # Create the environment
        # env = gym.make(env_name)
        env = make_no_update_env(env_name,context_parameter_name)
        observation, info = env.reset()

        overall_start_time = time.perf_counter()

        for i in range(NUM_STEPS):
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)

            step_start_time = time.perf_counter()
            planning_env = env.get_planning_env()
            step_end_time = time.perf_counter()

            step_duration_sec = step_end_time - step_start_time
            step_durations_sec.append(step_duration_sec)

            if terminated or truncated:
                observation, info = env.reset()

        overall_end_time = time.perf_counter()
        total_loop_duration_sec = overall_end_time - overall_start_time
        
        num_actual_steps = len(step_durations_sec)

        if num_actual_steps > 0:
            mean_time_sec = np.mean(step_durations_sec)
            
            if num_actual_steps > 1:
                # Use ddof=1 for sample standard deviation
                std_dev_time_sec = np.std(step_durations_sec, ddof=1)
                sem_time_sec = std_dev_time_sec / np.sqrt(num_actual_steps)
            else: # For n=1, std_dev and SEM are typically 0 or undefined.
                std_dev_time_sec = 0.0
                sem_time_sec = 0.0

            mean_time_us = mean_time_sec * 1_000_000  # Convert to microseconds
            sem_time_us = sem_time_sec * 1_000_000    # Convert to microseconds
            
            results[env_name] = {
                "mean": mean_time_us,
                "sem": sem_time_us,
                "count": num_actual_steps
            }
            
            print(f"Environment: {env_name}")
            print(f"Total wall-clock time for {num_actual_steps} steps: {total_loop_duration_sec:.4f} seconds")
            print(f"Average step() time: {mean_time_us:.4f} \u00B1 {sem_time_us:.4f} \u00B5s (n={num_actual_steps})")
            # \u00B1 is ±, \u00B5 is µ
        else:
            print(f"Environment: {env_name} - No steps were executed.")
            results[env_name] = None

    except gym.error.DependencyNotInstalled as e:
        print(f"Error creating environment {env_name}: {e}")
        print("You might need to install additional dependencies.")
        print("Try: pip install gymnasium[classic_control,toy_text]")
        results[env_name] = {"error": "Dependency Missing"}
    except Exception as e:
        print(f"An unexpected error occurred with environment {env_name}: {e}")
        results[env_name] = {"error": f"{type(e).__name__}"}
    finally:
        if env is not None:
            env.close()
        print("-" * (len(env_name) + 25))
        print()


# Print summary of results
print("\n--- Summary (Mean \u00B1 SEM in \u00B5s) ---")
for env_name, res_data in results.items():
    if res_data and "mean" in res_data:
        print(f"- {env_name:<15}: {res_data['mean']:.4f} \u00B1 {res_data['sem']:.4f} \u00B5s (n={res_data['count']})")
    elif res_data and "error" in res_data:
        print(f"- {env_name:<15}: Error: {res_data['error']}")
    else:
        print(f"- {env_name:<15}: No data or error.")