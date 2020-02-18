import os
from ctbn.generative_ctbn import GenerativeCTBN
from ctbn.config import graph_config
import numpy as np
import matplotlib.pyplot as plt
from utils.constants import *
from utils.helper import *
import time

if __name__ == "__main__":
    folder = create_folder_for_experiment(folder_name='../data/generative_ctbn/')

    n_traj = 20

    ctbn = GenerativeCTBN(graph_config, save_folder=folder)

    df_traj_hist = ctbn.sample_and_save_trajectories(n_traj=n_traj)

    # Selecting a random experiment to plot
    exp_to_plot = np.random.randint(1, n_traj + 1)
    df_traj_to_plot = df_traj_hist[df_traj_hist[TRAJ_ID] == exp_to_plot]

    # Saving and plotting the trajectories
    fig, ax = plt.subplots(ctbn.num_nodes)
    for i, var in enumerate(ctbn.node_list):
        ax[i].step(df_traj_to_plot[TIME], df_traj_to_plot[var])
        ax[i].set_ylim([-.5, 1.5])
        ax[i].set_ylabel(var)
        ax[i].set_xlabel('time')

    fig.savefig(os.path.join(folder, 'trajectory_plot.png'))
