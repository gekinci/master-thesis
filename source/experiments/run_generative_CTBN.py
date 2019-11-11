import os
from ctbn.generative_CTBN.utils import *
import ctbn.config as cfg
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    os.makedirs('../data', exist_ok=True)

    run_generative_ctbn()

    df_traj_hist = pd.read_csv('../data/trajectory_hist.csv')

    # Selecting a random experiment to plot
    exp_to_plot = np.random.randint(1, cfg.n_experiments + 1)
    df_traj_to_plot = df_traj_hist[df_traj_hist.experiment == exp_to_plot]

    # Saving and plotting the trajectories
    fig, ax = plt.subplots(cfg.num_nodes)
    for i, var in enumerate(cfg.node_list):
        ax[i].step(df_traj_to_plot.time, df_traj_to_plot[var])
        ax[i].set_ylim([-.5, 1.5])
        ax[i].set_ylabel(var)
        ax[i].set_xlabel('time')

    fig.savefig(f'../data/trajectory_plot.png')
