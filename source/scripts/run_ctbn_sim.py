from simulations.ctbn import CTBNSimulation
from utils.constants import *
from utils.helpers import *
from utils.visualization import *

import matplotlib.pyplot as plt
import yaml


if __name__ == "__main__":
    folder = create_folder_for_experiment(folder_name='../_data/ctbn_simulation/')

    n_traj = 1

    with open('../configs/ctbn_sim.yaml') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    ctbn = CTBNSimulation(cfg, save_folder=folder)

    df_traj_hist = ctbn.sample_and_save_trajectories(n_traj=n_traj)

    # Selecting a random experiment to plot
    exp_to_plot = np.random.randint(1, n_traj + 1)
    df_traj_to_plot = df_traj_hist[df_traj_hist[TRAJ_ID] == exp_to_plot-1]

    visualize_trajectories(df_traj_to_plot, node_list=ctbn.node_list, path_to_save=folder)
