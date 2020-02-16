from pomdp.simulation_pomdp import *
import logging
import time
import os
from constants import *

if __name__ == "__main__":
    t = int(time.time())
    folder = f'../data/simulation_pomdp/'
    # os.makedirs(folder, exist_ok=True)

    cfg = {
        GRAPH_STRUCT: {'1': [],
                       '2': [],
                       # '3': []
                       },
        T_MAX: 10,
        STATES: [0, 1],  # for every node
        INITIAL_PROB: [0.5, 0.5],
        N_Q: 4,  # size of the Q3 set
        HOW_TO_PRED_STATE: 'expectation',  # 'maximum_likely' or 'expectation'
        TIME_GRAIN: 0.001
    }

    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.basicConfig(filename=os.path.join(folder + f'debug.log'), level=logging.DEBUG)

    pomdp_sim = POMDPSimulation(cfg, save_folder=folder, save_time=t)

    df_traj = pomdp_sim.sample_trajectory()
