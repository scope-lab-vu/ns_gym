import numpy as np


ENV_WIDTH = 12
ENV_HEIGHT = 4


def convert_to_one_hot_encoding(state):
    state_vec = np.zeros((ENV_WIDTH * ENV_HEIGHT,))
    state_vec[state] = 1
    state_vec = np.reshape(state_vec, (1, ENV_WIDTH * ENV_HEIGHT))
    return state_vec
