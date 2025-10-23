import json 
import os
import sys
import argparse
import numpy as np
import gymnasium as gym
import ns_gym as nsb 
import time
import logging
from multiprocessing import Pool
import multiprocessing

"""
This script is used to test the NS-FrozenLake environment where the probabilty distibution happends at the first step.

This experiment set up follows the experiments conducted in the PAMCTs paper.
"""

def make_env(prob_dist,change_notification,delta_change_notification,in_sim_change):
    env = gym.make('FrozenLake-v0',is_slippery = False)
    fl_scheduler = nsb.shedulers.ContinuousScheduler()
    fl_updateFn = nsb.update_functions.DistributionNoUpdate(fl_scheduler)
    param = {"P":fl_updateFn}
    env = nsb.wrappers.NSFrozenLake(env,
                                    param, 
                                    change_notification = change_notification, 
                                    delta_change_notification = delta_change_notification, 
                                    in_sim_change = in_sim_change,
                                    initial_prob_dist = prob_dist)

    return env

def run_experiment(ags):
    pass

if __name__ == "__main__":
    # Probdists
    # MCTS iteration depth
    change_notification = False
    delta_change_notification = False
    in_sim_change = False
    prob_dists = [[1,0,0],[0.8,0.1,0.1],[0.7,0.15,0.15],[0.6,0.2,0.2],[0.5,0.25,0.25],[0.4,0.3,0.3],[1/3,1/3,1/3]]
    mcts_iters = [25,50,100,200,500,1000,2000,5000,10000,12000,15000]
    args = []
    num_samples = 50

    for p in prob_dists:
        for m in mcts_iters:
            for sample_id in range(num_samples):
                args.append({"prob_dist":p, "mcts_iters":m, "sample_id":sample_id})

    num_cpus = multiprocessing.cpu_count() - 10 
    with Pool(processes = num_cpus) as pool:
        pool.map(run_experiment,args)

    # 30,000 iterations .... 




