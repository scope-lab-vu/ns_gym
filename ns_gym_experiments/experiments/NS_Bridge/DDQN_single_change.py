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

config_file_path = "/media/--/home/n--/ns_gym/experiments/NS_Bridge/configs/DDQN_single_config.yaml"

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

def run_all_experiments(sample_id,p,q,seed,agent_config,wrapper_config,gym_config):
    """Run all experiments with the given gridsearch over parameters.
    """
    try:
        logger.info(f"STARTING: Experiment with num_samples: {sample_id},seed: {seed}")

        ########## Setup the environment ##########
        realworld_env = make_contin_env(wrapper_config=wrapper_config)
        max_episode_steps = gym_config["max_episode_steps"]
        state_size = agent_config["state_size"]
        action_size = agent_config["action_size"]
        model_path = agent_config["model_path"]
        lr = agent_config["lr"]
        do_update = agent_config["do_update"]
        ############# Run the experiment #############
        episode_reward = 0
        start_time = time.time()
        done = False
        truncated = False
        obs,_= realworld_env.reset(seed=seed)
        count = 0
        

        agent = ns_gym.benchmark_algorithms.DDQN.DQNAgent(state_size=state_size,
                                                            action_size=action_size,
                                                            seed=seed,
                                                            model_path=model_path,
                                                            do_update=do_update,
                                                            lr=lr)

        while not done and not truncated and count < max_episode_steps:
            planning_env = realworld_env.get_planning_env()
            action,_= agent.act(state=obs)
            obs,reward,done, truncated, info = realworld_env.step(action)
            ns_gym.benchmark_algorithms.DDQN.do_gradient_updates(obs,planning_env,agent=agent,time_budget=0.17)
            episode_reward += reward.reward
            count += 1

        total_time = time.time() - start_time

        logger.info(f"FINISHED: Experiment with num_samples: {sample_id},seed: {seed}")
        result = [sample_id,episode_reward,experiment_name,p,total_time,seed,count]
        q.put(result)

    except Exception as e:
        logger.error(f"Error in running experiment: {e}", exc_info=True)
        print(f"Error in running experiment: {e}")
        return None
    
def main():
    manager = multiprocessing.Manager()
    queue = manager.Queue()

    sample_id= [x for x in range(exp_config["num_samples"])]
    p = exp_config["p"]
    num_experiments = len(sample_id) * len(p)


    seeds = random.sample(range(100000), num_experiments)

    results_dir = os.path.join(script_dir, "results")

    os.makedirs(results_dir,exist_ok=True)
    outfile = os.path.join(results_dir,experiment_name + ".csv")

    with open(outfile, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["sample_id","reward","experiment_name","p","total_time","seed","num_steps"])

    writer_process = multiprocessing.Process(target=write_results_to_file, args=(queue,outfile,logger))
    writer_process.start()

    parameter_combinations = itertools.product(sample_id,p)
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



