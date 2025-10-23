import os
import logging
import datetime
from experiment_setup import *
from pathlib import Path
import ns_gym as nsb
import gymnasium as gym
from multiprocessing import Pool
import multiprocessing
import random
import itertools
import time

config_file_path = "/media/--/home/n--/ns_gym/experiments/NS_Bridge/configs/PAMCTS_continuous_config.yaml"

gym_config,wrapper_config,agent_config,exp_config = read_config_file(config_file_path)


script_path = Path(__file__)
script_dir = script_path.parent

current_datetime = datetime.datetime.now()
formatted_date = current_datetime.strftime('%Y-%m-%d')
experiment_name = exp_config["experiment_name"]+f"_{formatted_date}"

logsdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),"logs")
log_name = experiment_name + ".log"
os.makedirs(logsdir,exist_ok=True)
logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                datefmt='%Y-%m-%d:%H:%M:%S', level=logging.INFO,
                handlers=[logging.FileHandler(os.path.join(logsdir,log_name), mode='w')])
logger = logging.getLogger()

def run_all_experiments(sample_id,alpha,q,seed,agent_config,wrapper_config,gym_config):
    """Run all experiments with the given gridsearch over parameters.
    """
    try:
        logger.info(f"STARTING: Experiment with num_samples: {sample_id},seed: {seed}")

        ########## Setup the environment ##########
        realworld_env = make_contin_env(wrapper_config=wrapper_config)
        max_episode_steps = gym_config["max_episode_steps"]
        gamma = agent_config["gamma"]
        c = agent_config["c"]
        num_mcts_simulations = agent_config["num_mcts_simulations"]
        max_mcts_search_depth = agent_config["mcts_search_depth"]
        state_space = agent_config["state_size"]
        action_size = agent_config["action_size"]
        num_layers = agent_config["num_layers"]
        num_hidden_units = agent_config["num_hidden_units"]
        model_path = agent_config["model_path"]
        ############# Run the experiment #############
        episode_reward = 0
        start_time = time.time()
        done = False
        truncated = False
        obs,_= realworld_env.reset(seed=seed)
        count = 0

        model = ns_gym.benchmark_algorithms.DDQN.DQN(state_size=state_space,action_size=action_size,num_layers=num_layers,num_hidden_units=num_hidden_units,seed=seed)
        agent = ns_gym.benchmark_algorithms.PAMCTS(alpha=alpha,
                                                     mcts_iter=num_mcts_simulations,
                                                     mcts_search_depth=max_mcts_search_depth,
                                                     mcts_exploration_constant=c,
                                                     mcts_discount_factor=gamma,
                                                     state_space_size=state_space,
                                                     action_space_size=action_size,
                                                     DDQN_model=model,
                                                     DDQN_model_path=model_path,
                                                     seed=seed
                                                     )
        

        while not done and not truncated and count < max_episode_steps:
            planning_env = realworld_env.get_planning_env()
            action,_ = agent.act(obs,planning_env)
            obs,reward,done, truncated, info = realworld_env.step(action)
            episode_reward += reward.reward
            count += 1

        total_time = time.time() - start_time

        logger.info(f"FINISHED: Experiment with num_samples: {sample_id},seed: {seed}")
        result = [sample_id,episode_reward,experiment_name,num_mcts_simulations,total_time,seed,realworld_env.P,count,alpha]
        q.put(result)

    except Exception as e:
        logger.error(f"Error in running experiment: {e}", exc_info=True)
        print(f"Error in running experiment: {e}")
        return None
    
def main():
    manager = multiprocessing.Manager()
    queue = manager.Queue()

    sample_id= [x for x in range(exp_config["num_samples"])]
    alpha = agent_config["alpha"]

    num_experiments = len(sample_id) * len(alpha) 

    seeds = random.sample(range(100000), num_experiments)

    results_dir = os.path.join(script_dir, "results")

    os.makedirs(results_dir,exist_ok=True)
    outfile = os.path.join(results_dir,experiment_name + ".csv")

    with open(outfile, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["sample_id","reward","experiment_name","num_iter","total_time","seed","final_dist","num_steps","alpha"])

    writer_process = multiprocessing.Process(target=write_results_to_file, args=(queue,outfile,logger))
    writer_process.start()

    parameter_combinations = itertools.product(sample_id,alpha)
    input = [(*params,queue,seeds[i],agent_config,wrapper_config,gym_config) for i,params in enumerate(parameter_combinations)]

    with Pool(20) as p:
        p.starmap(run_all_experiments, input)

    print("number of experiments",num_experiments)
    queue.put("DONE")
    writer_process.join()


if __name__ == "__main__":
    print("Starting experiment")
    main()
    print("Experiment finished")



