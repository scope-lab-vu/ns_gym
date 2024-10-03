from nsfrozenlake.nsfrozenlake_v0 import NSFrozenLakeV0 as model

import numpy as np
import pickle
import random

task = model()
task.compute_distance(0.7,0.2)