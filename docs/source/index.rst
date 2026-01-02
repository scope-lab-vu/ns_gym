NS-Gym
======

NS-Gym (Non-Stationary Gym) is a flexible framework providing a standardized abstraction for both modeling non-stationary Markov Decision processes (NS-MDPs) and the key problem types that a decision-making entity may encounter in such environments. NS-Gym is built on top of the popular `Gymnasium <https://gymnasium.farama.org/>`_ library and provides a set of wrappers to for several existing environments, making it easy to incorporate non-stationary dynamics and manage the nature of agent-environment interaction specific to NS-MDPs. A key feature of NS-Gym is emulating the key problem types of decision-making in a non-stationary settings; these problem types concern not only the ability to adapt to changes in the environment but also the ability to detect and characterize these changes. To get started with NS-Gym, check out our :doc:`installation` instructions and :doc:`quickstart_guide`. For a deep dive into the core concepts behind NS-Gym, visit our :doc:`core_concepts` page or take a look at our paper on `ArXiv <https://arxiv.org/abs/2501.09646>`_ published at NeurIPS 2025 Dataset and Benchmarks track. 


Paper
-------------------------------------------

Please take a look at our paper accepted to the NeurIPS 2025 Dataset and Benchmarks Track for more details on NS-Gym:

`NS-Gym: Open-Source Simulation Environments and Benchmarks for Non-Stationary Markov Decision Processes <https://openreview.net/pdf?id=YOXZuRy40U>`_.

.. code-block::

     @inproceedings{
          keplinger2025nsgym,
          title={{NS}-Gym: A Comprehensive and Open-Source Simulation Framework for Non-Stationary Markov Decision Processes},
          author={Nathaniel S Keplinger and Baiting Luo and Yunuo Zhang and Kyle Hollins Wray and Aron Laszka and Abhishek Dubey and Ayan Mukhopadhyay},
          booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
          year={2025},
          url={https://openreview.net/forum?id=YOXZuRy40U}
     }


Installation
-----------------
To install NS-Gym, you can use pip:

.. code-block::
     pip install ns_gym

Alternatively, you can install the latest development version directly from the GitHub repository:
.. code-block::

    pip install git+https://github.com/scope-lab-vu/ns_gym



Decision Making Algorithm Support
-------------------------------------------------------------

NS-Gym is designed to be compatible with existing reinforcement learning libraries such as `Stable Baselines3 <https://stable-baselines3.readthedocs.io/en/master/>`_. Additionally, NS-Gym provides baseline algorithms designed explicitly for non-stationary environments, as well as a leaderboard to compare the performance of different algorithms on various non-stationary tasks. 



NS-Gym in Action
-------------------------------------------------------------

Here are three examples of non-stationary environments created using NS-Gym. Each demonstrates a transition from an initial MDP :math:`\mathcal{MDP}_0` to a modified MDP :math:`\mathcal{MDP}_1` by changing environment parameters :math:`\theta_0 \rightsquigarrow \theta_1`. We show examples from the classic control suite (CartPole), stochastic gridworlds (FrozenLake), and the MuJoCo suite (Ant).

Note that this type of parameter shift is just one example of how an NS-MDP can be implemented. The policy controlling the CartPole and FrozenLake agents is the NS-Gym implementation of :py:class:`Monte Carlo Tree Search <ns_gym.benchmark_algorithms.MCTS>`, while the Ant environment is controlled by a Stable-Baselines3 PPO policy.


.. list-table::
   :header-rows: 1
   :widths: 40 20 40
   :class: no-border-table

   * - .. container:: text-center

          Stationary MDP

     - .. container:: text-center

          :math:`\large \theta_0 \rightsquigarrow \theta_1`

     - .. container:: text-center

          Non-Stationary MDP

   * - .. image:: /_static/images/cartpole_stationary.gif
          :width: 300px
          :alt: CartPole Stationary

     - .. rst-class:: table-vcenter

       **At timestep** :math:`t` **gravity massively increases according to a user defined step function** 

     - .. image:: /_static/images/cartpole_nonstationary.gif
          :width: 300px
          :alt: CartPole Non-Stationary

   * - .. image:: /_static/images/frozen_lake_stationary.gif
          :width: 300px
          :alt: FrozenLake Stationary   

     - .. rst-class:: table-vcenter

       **Probability of moving in the intended direction goes to 0 just before reaching the goal**

     - .. image:: /_static/images/frozenlake_nonstationary.gif
          :width: 300px
          :alt: FrozenLake Non-Stationary

   * - .. image:: /_static/images/ant_stationary.gif
          :width: 300px
          :alt: Ant Stationary

     - .. rst-class:: table-vcenter

       **Magnitude of gravity gradually decreases at each timestep following a geometric progression**

     - .. image:: /_static/images/ant_non_stationary.gif
          :width: 300px
          :alt: Ant Non-Stationary


.. toctree::
  :caption: Contents
  :maxdepth: 1

  installation.md
  quickstart_guide.md
  core_concepts.rst
  tutorials.md
  environments.md
  algorithms.md
  leaderboard.md
  aamas2026_competition.md
  reference.md