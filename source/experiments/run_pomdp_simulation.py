from ctbn.generative_ctbn import GenerativeCTBN
import constants
import matplotlib.pyplot as plt
from ctbn.simulation_pomdp import *
import logging
import time
import os

if __name__ == "__main__":
    t = int(time.time())
    folder = f'../data/simulation_pomdp/{t}/'
    os.makedirs(folder, exist_ok=True)

    cfg = {
        constants.GRAPH_STRUCT: {'1': [],
                                 '2': [],
                                 '3': []},
        constants.T_MAX: 50,
        constants.N_STATES: 2,  # for every node
        # constants.N_ACTIONS: 3,
        constants.N_Q: 4  # size of the Q3 set
    }

    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.basicConfig(filename=os.path.join(folder + f'debug.log'), level=logging.DEBUG)

    pomdp_sim = POMDPSimulation(cfg, save_folder=folder, save_time=t)

    df_traj = pomdp_sim.sample_trajectory()
