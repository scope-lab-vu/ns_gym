.. NS-Gym documentation master file, created by
   sphinx-quickstart on Tue Oct  8 13:25:32 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

NS-Gym documentation
====================

Welcome to the NS-Gym documentation! NS-Gym (non-stationary Gym) is a simulation tool kit toolkit for non-stationary Markov decsion processes (NS-MDPs) that provides a tailored, standardized,
and principled set of interfaces for non-stationary environments. It is built on top of OpenAI Gymnasium as general framwork to construct 
and test NS-MDPs givne the breadth of treatment of non-stationary stochastic control porcesses in the literature.

In many real-world applications, agents must make sequential decisions in environments where conditions are subject to change due to various exogenous factors. These
nonstationary environments pose significant challenges to traditional decision-making models, which typically assume stationary dynamics. Non-stationary Markov decision processes
(NS-MDPs) offer a framework to model and solve decision problems under such changing conditions. However, the lack of standardized benchmarks and simulation tools has hindered systematic evaluation and advance in this field. We
present NS-Gym, the first simulation toolkit designed explicitly for NS-MDPs, integrated within the popular Gymnasium framework. In NS-Gym, we segregate the evolution of the
environmental parameters that characterize non-stationarity from the agent’s decision-making module, allowing for modular and flexible adaptations to dynamic environments.

In its current version, NS-Gym provides a set of wrappers to augment the classic control suite of Gymnasium environments and three gridworld environments. We refer to these Gymnasium environments (i.e., the stationary counterparts of the non-stationary
environments we develop) as base environments. At a high level, each wrapper introduces non-stationarity by modifying some parameters that the base environment exposes. The
modification potentially occurs at each decision epoch or through specific functions over decision epochs configured by the user. For example, in a deterministic environment
such as the “CartPole” (we provide a detailed description of the environment in the technical appendix), an example change is varying the value of the gravity, thereby altering the dynamics of the cart. In stochastic environments, the
probability distribution over possible next states, given the current state action pair, changes. For example, in the classic Frozen Lake environment, this change might increase (or decrease) the coefficient of friction, making the movement of the agent more (or less) uncertain.


Quickstart
-----------------




.. toctree::
   :maxdepth: 2
   :caption: Contents: 

   modules

