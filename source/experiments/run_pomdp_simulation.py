from ctbn.generative_ctbn import GenerativeCTBN
import constants
import matplotlib.pyplot as plt
from ctbn.simulation_pomdp import *
import logging
import time
import os


if __name__ == "__main__":
    folder = '../data/simulation_pomdp/'
    os.makedirs(folder, exist_ok=True)
    t = time.time()

    cfg = {
        constants.PARENTS: {'1': [],
                            '2': [],
                            '3': []},
        constants.T_MAX: 10,
        constants.N_VALUES: 2,
        constants.N_ACTIONS: 3,
        constants.N_Q: 5
    }

    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.basicConfig(filename=folder+f'{t}_debug.log', level=logging.DEBUG)

    pomdp_sim = POMDPSimulation(cfg, save_folder=folder, save_time=t)

    df_traj = pomdp_sim.sample_trajectory()
