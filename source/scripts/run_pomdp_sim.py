from simulations.pomdp import POMDPSimulation
from utils.constants import *
from utils.helpers import *

import yaml
import os


if __name__ == "__main__":

    folder = create_folder_for_experiment(folder_name='../_data/pomdp_simulation/')

    with open('../configs/pomdp_sim.yaml', 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    with open(os.path.join(folder, 'cfg.yaml'), 'w') as f:
        yaml.dump(cfg, f)

    np.random.seed(cfg[SEED])
    pomdp_sim = POMDPSimulation(cfg, save_folder=folder, import_data='1582930774')

    df_traj = pomdp_sim.sample_trajectory()
    df_traj.to_csv(os.path.join(folder, 'df_traj.csv'))
    pomdp_sim.df_b.to_csv(os.path.join(folder, 'df_belief.csv'))
    pomdp_sim.policy.to_csv(os.path.join(folder, 'df_policy.csv'))
    pomdp_sim.df_Qz.to_csv(os.path.join(folder, 'df_Qz.csv'))
