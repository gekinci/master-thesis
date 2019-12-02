from ctbn.generative_ctbn import GenerativeCTBN
import constants
import matplotlib.pyplot as plt
from ctbn.simulation_pomdp import *


if __name__ == "__main__":
    cfg = {
        constants.PARENTS: {'1': [],
                            '2': [],
                            '3': []},
        constants.T_MAX: 20,
        constants.N_VALUES: 2,
        constants.N_ACTIONS: 3,
        constants.N_Q: 5
    }

    # logging.basicConfig(filename='example.log', level=logging.DEBUG)
    pomdp_sim = POMDPSimulation(cfg)

    df_traj = pomdp_sim.sample_trajectory()
