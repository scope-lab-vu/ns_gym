import csv
from multiprocessing import Pool

import code.agents.asynchronous_dp as asyndp
import rats


def csv_write(row, path, mode):
    with open(path, mode) as csvfile:
        w = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        w.writerow(row)


def run(agent, env, tmax, verbose=False):
    """
    Run single episode
    Return: (undiscounted_return, total_time, discounted_return)
    """
    done = False
    undiscounted_return, total_time, discounted_return = 0.0, 0, 0.0
    if verbose:
        env.render()
    for t in range(tmax):
        action = agent.act(env,done)
        _, r, done, _ = env.step(action)
        undiscounted_return += r
        discounted_return += (agent.gamma**t) * r
        if verbose:
            env.render()
        if (t+1 == tmax) or done:
            total_time = t+1
            break
    return undiscounted_return, total_time, discounted_return


def multithread_run(
        env_name,
        _env,
        n_env,
        env,
        agt_name,
        _agt,
        n_agt,
        agt,
        _prm,
        n_prm,
        prm,
        tmax,
        n_epi,
        _thr,
        save,
        path,
        verbose,
        save_period
):
    saving_pool = []
    for _epi in range(n_epi):
        if verbose:
            print('Environment', env_name, _env+1, '/', n_env, 'agent', agt_name, _prm+1, '/', n_prm,
                  'running episode', _epi+1, '/', n_epi, '(thread nb', _thr, ')')
        env.reset()
        undiscounted_return, total_time, discounted_return = run(agt, env, tmax)
        if save:
            saving_pool.append([env_name, _env, agt_name, _prm] + prm + [_thr, undiscounted_return, total_time,
                                                                         discounted_return])
            if len(saving_pool) == save_period:
                for row in saving_pool:
                    csv_write(row, path, 'a')
                saving_pool = []
    if save:
        for row in saving_pool:
            csv_write(row, path, 'a')


def multithread_benchmark(
        env,
        env_name,
        epsilon,
        agent_name_pool,
        agent_pool,
        param_pool,
        param_names_pool,
        n_epi,
        tmax,
        save,
        paths_pool,
        n_thread,
        verbose=True,
        save_period=1
):
    """
    Benchmark multiple agents within a single environment.
    Multithread method.
    env_name         : name of the generated environment
    agent_name_pool  : list containing the names of the agents for saving purpose
    agent_pool       : list containing the agent objects
    param_pool       : list containing lists of parameters for each agent object
    param_names_pool : list containing the parameters names
    n_epi            : number of episodes per generated environment
    tmax             : timeout for each episode
    save             : save the results or not
    paths_pool       : list containing the saving path for each agent
    n_thread         : number of threads
    verbose          : if true, display informations during benchmark
    """
    assert len(agent_name_pool) == len(agent_pool) == len(param_pool)
    pool = Pool(processes=n_thread)
    n_agt = len(param_pool)
    n_epi = int(n_epi / n_thread)
    if save:
        assert len(paths_pool) == n_agt
        for _agt in range(n_agt): # Init save files for each agent
            csv_write(['env_name', 'env_number', 'agent_name', 'param_number'] +
                      param_names_pool[_agt] +
                      ['thread_number', 'undiscounted_return', 'total_time', 'discounted_return'],
                      paths_pool[_agt], 'w')
    env.set_epsilon(epsilon)
    for _agt in range(n_agt):
        agt_name = agent_name_pool[_agt]
        agt = agent_pool[_agt]
        n_prm = len(param_pool[_agt])
        for _prm in range(n_prm):
            prm = param_pool[_agt][_prm]
            agt.reset(prm)
            if verbose:
                print('Created agent', _agt+1, '/', n_agt,'with parameters', _prm+1, '/', n_prm)
                agt.display()
            results_pool = []
            for _thr in range(n_thread):
                results_pool.append(pool.apply_async(multithread_run, [env_name, 1, 1, env, agt_name, _agt, n_agt, agt,
                                                                       _prm, n_prm, prm, tmax, n_epi, _thr+1, save,
                                                                       paths_pool[_agt], verbose, save_period]))
            for result in results_pool:
                result.get()


def main():
    # Parameters
    n_epi = 24
    timeout = 10
    n_thread = 4
    depth = 6
    env_name = 'NSBridge-v0'
    saving_path = 'data/'
    agent_name_pool = ['RATS', 'AsynDP-snapshot', 'AsynDP-NSMDP']
    param_names_pool = [
        ['action_space', 'gamma', 'max_depth'],
        ['action_space', 'gamma', 'max_depth', 'is_model_dynamic'],
        ['action_space', 'gamma', 'max_depth', 'is_model_dynamic']
    ]

    epsilons = [0.0, 0.25, 0.5, 0.75, 1.0]

    for epsilon in epsilons:
        paths_pool = []
        for agent_name in agent_name_pool:
            paths_pool.append(saving_path + env_name + '-' + str(epsilon) + '-' + agent_name + '.csv')

        env = nsg.NSBridgeV0()
        env.set_epsilon(epsilon)
        A = env.action_space
        agent_pool = [rats.RATS(A), asyndp.AsynDP(A), asyndp.AsynDP(A)]
        param_pool = [
            [[A, 0.9, depth]],
            [[A, 0.9, depth, False]],
            [[A, 0.9, depth, True]]
        ]

        multithread_benchmark(
            env=env,
            env_name=env_name,
            epsilon=epsilon,
            agent_name_pool=agent_name_pool,
            agent_pool=agent_pool,
            param_pool=param_pool,
            param_names_pool=param_names_pool,
            n_epi=n_epi,
            tmax=timeout,
            save=True,
            paths_pool=paths_pool,
            n_thread=n_thread,
            verbose=True,
            save_period=1
        )


if __name__ == "__main__":
    main()
