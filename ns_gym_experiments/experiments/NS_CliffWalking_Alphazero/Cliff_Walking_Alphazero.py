from build_model_tf import build_model_tf
import math
from multiprocessing import Pool
import numpy as np
import os
import pandas as pd
import time
from utils import convert_to_one_hot_encoding


ENV_WIDTH = 12
ENV_HEIGHT = 4
START_STATE = 36
GOAL_STATE = 47
HOLES = [37, 38, 39, 40, 41, 42, 43, 44, 45, 46]

MOVE_LEFT = 3
MOVE_RIGHT = 1
MOVE_DOWN = 0
MOVE_UP = 2
MAX_DEPTH = 500
STEP_LIMIT = 200


def calculate_probabilities(step_count):
    base_prob = max(1.0 - 0.02 * step_count, 0.8)
    other_prob = (1.0 - base_prob) / 3
    return [base_prob, other_prob, other_prob, other_prob]


def transition(state, action, transition_type, step_count):
    x = int(state / ENV_WIDTH)
    y = state % ENV_WIDTH
    shape = [ENV_HEIGHT, ENV_WIDTH]

    # ; Calculate dynamic probabilities
    if transition_type[0] == "single":
        probs = [transition_type[1], (1.0 - transition_type[1]) / 3, (1.0 - transition_type[1]) / 3, (1.0 - transition_type[1]) / 3]
    else:
        probs = calculate_probabilities(step_count)

    # Decide the direction based on the dynamic probabilities
    chosen_action = np.random.choice([action, (action + 1) % 4, (action + 2) % 4, (action + 3) % 4], p=probs)

    if chosen_action == MOVE_UP:  # UP
        x = max(x - 1, 0)
    elif chosen_action == MOVE_RIGHT:  # RIGHT
        y = min(y + 1, shape[1] - 1)
    elif chosen_action == MOVE_DOWN:  # DOWN
        x = min(x + 1, shape[0] - 1)
    elif chosen_action == MOVE_LEFT:  # LEFT
        y = max(y - 1, 0)

    state = x * ENV_WIDTH + y
    state = int(state)

    # Check if agent is in the cliff region
    if state in HOLES or step_count > STEP_LIMIT:
        return state, -100, True

    # Check if agent reached the goal
    if state == GOAL_STATE:
        return state, -step_count, True

    return state, 0, False


def select_action(C, state, n, v, p):
    state_n = n[state]
    state_v = v[state]
    state_p = p[state]
    N = sum(state_n)
    for i in range(3):
        if state_n[i] <= 1:
            return i
    return np.argmax(state_v / state_n + C * state_p * np.sqrt(math.log(N) / state_n))


def MCTS(root_state, network, transition_type, iterations, C, gamma):
    n = np.ones((ENV_WIDTH * ENV_HEIGHT, 3))
    v = np.zeros((ENV_WIDTH * ENV_HEIGHT, 3))
    p = np.zeros((ENV_WIDTH * ENV_HEIGHT, 3))
    r = np.zeros((ENV_WIDTH * ENV_HEIGHT, 3))
    for state in range(ENV_WIDTH * ENV_HEIGHT):
        # ^ tensorflow
        state_vec = convert_to_one_hot_encoding(state)
        outputs_combo = network.predict(x=np.array(state_vec).reshape((1, ENV_WIDTH * ENV_HEIGHT)), batch_size=1, verbose=0)
        prob_priors = outputs_combo[0][0]
        value = outputs_combo[1][0]
        for action in range(3):
            p[state, action] = prob_priors[action] 
            r[state, action] = value[action]
    for i in range(iterations):
        state = root_state
        done = False
        reward = None
        sa_trajectory = []
        depth = 0
        step_count = 0
        while not done:
            action = select_action(C, state, n, v, p)
            sa_trajectory.append((state, action))
            (next_state, reward, done) = transition(state, action, transition_type=transition_type, step_count=step_count)
            step_count += 1
            depth += 1
            if done or depth > MAX_DEPTH:
                break
            if n[state, action] == 1:
                reward = r[state, action]
                break
            state = next_state
        for (state, action) in sa_trajectory:
            n[state, action] += 1
            v[state, action] += reward
    return np.argmax(v[root_state] / n[root_state])


def simulate_episode(policy, network, iterations, C, transition_type, gamma):
    steps = 0
    state = START_STATE
    step_count = 0
    while True:
        action = policy(state, network, transition_type, iterations, C, gamma)

        (next_state, reward, done) = transition(state, action, transition_type=transition_type, step_count=step_count)
        step_count += 1
        print(f"State: {state}; Move: {action}; Next state: {next_state}; Reward: {reward}; Done: {done}")
        steps += 1
        if done or steps > MAX_DEPTH:
            return steps, reward
        state = next_state


def evaluate_policy(policy, verbose, network, iterations, C, transition_type, sample_id, gamma, file_name):
    start_time = time.time()
    lengths = np.zeros(1)
    outcomes = np.zeros(1)
    for episode in range(1):
        (length, outcome) = simulate_episode(policy, network, iterations, C, transition_type, gamma)
        lengths[episode] = length
        outcomes[episode] = outcome
        if verbose:
            print(f"Episode: {episode}; Length: {length}; Outcome: {outcome}.")
        results = [[outcomes[0], transition_type, iterations, sample_id, time.time() - start_time, lengths[0], C, gamma]]
        if not os.path.exists(file_name):
            with open(file_name, "w") as f:
                pd.DataFrame(results).to_csv(f, header=["cumulative_reward", "transition_type", "iterations",
                                                        "sample id", "computation time", "step_counter", "C", "gamma"], index=False)
        else:
            with open(file_name, "a") as f:
                pd.DataFrame(results).to_csv(f, header=False, index=False)


def run_simulations(args):
    transition_type = args["transition_type"]
    iterations = args["iterations"]
    C = args["C"]
    sample_id = args["sample_id"]
    gamma = args["gamma"]
    file_name = args["file_name"]
    weights_file = args["weights_file"]
    start_time = time.time()
    network = build_model_tf(num_hidden_layers=5, state_size=ENV_WIDTH * ENV_HEIGHT)
    network.load_weights(weights_file)
    print(f"Network loaded from {weights_file}")
    evaluate_policy(MCTS, verbose=True, network=network, iterations=iterations, C=C,
                    transition_type=transition_type, sample_id=sample_id, gamma=gamma, file_name=file_name)
    print(f"Sample {sample_id} Time taken: {time.time() - start_time} seconds.")


def calculate_stats(file_path):
    # Load the CSV file
    data = pd.read_csv(file_path)
    
    # Group by transition_type
    grouped = data.groupby('transition_type')['cumulative_reward']
    
    # Calculate mean and standard error
    stats = grouped.agg(['mean', 'sem']).reset_index()
    
    # Print the results
    for _, row in stats.iterrows():
        print(f"Transition Type: {row['transition_type']}")
        print(f"  Average Cumulative Reward: {row['mean']}")
        print(f"  Standard Error: {row['sem']}\n")


def main():
    num_cpus = 20
    file_name = "cliff_walking_alphazero.csv"
    if os.path.exists(file_name):
        os.remove(file_name)
    weights_file = "cliff_walking_alphazero_weights.h5f"
    begin_time = time.time()
    args = []
    for transition_type in [("continuous", None), ("single", 1.0), ("single", 0.8), ("single", 0.6), ("single", 0.4)]:
        for iterations in [1000]:
            for C in [1.414]:
                for gamma in [0.99]:
                    for sample_id in range(100):
                        args.append({"transition_type": transition_type, "iterations": iterations, "weights_file": weights_file,
                                    "sample_id": sample_id, "C": C, "gamma": gamma, "file_name": file_name})
    
    with Pool(processes=num_cpus) as pool:
        pool.map(run_simulations, args)

    print(f"experiments completed")
    with open("cliff_walking_alphazero_execution_time.txt", "w") as f:
        f.write(f"cliff walking alphazero experiments execution time: {time.time() - begin_time}")


if __name__ == "__main__":
    # calculate_stats("cliff_walking_alphazero.csv")
    main()
