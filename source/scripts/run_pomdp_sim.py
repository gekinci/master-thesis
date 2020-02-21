from simulations.pomdp import POMDPSimulation
from utils.constants import *
from utils.helpers import *

import logging
import yaml
import os


if __name__ == "__main__":

    folder = create_folder_for_experiment(folder_name='../_data/pomdp_simulation/')

    with open('../configs/pomdp_sim.yaml') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.basicConfig(filename=os.path.join(folder, 'debug.log'), level=logging.DEBUG)

    pomdp_sim = POMDPSimulation(cfg, save_folder=folder)

    df_traj = pomdp_sim.sample_trajectory()
