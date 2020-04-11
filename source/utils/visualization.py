from utils.constants import *

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import pandas as pd
import os


def visualize_optimal_policy_map(df, path_to_save='../_data/'):
    step_cols = df.columns[df.columns.str.startswith('step')]

    for s in step_cols:
        fig = plt.figure()
        ax = Axes3D(fig)
        groupedbyAction = df.groupby(by=s)

        for a, df_a in list(groupedbyAction):
            ax.scatter(df_a['b1'], df_a['b2'], df_a['b3'], label=a)
            ax.set_xlabel('b1')
            ax.set_ylabel('b2')
            ax.set_zlabel('b3')
            ax.legend()
        fig.savefig(path_to_save + f'OPM_{s.split("_")[-1]}.png')
        plt.show()
    return


def plot_trajectories(df, node_list=None, path_to_save=None):
    if node_list is None:
        node_list = ['X', 'Y', 'o', 'Z']

    # Saving and plotting the trajectories
    fig, ax = plt.subplots(len(node_list))
    for i, node in enumerate(node_list):
        ax[i].step(df[TIME], df[node])
        ax[i].set_ylim([-.5, 1.5])
        ax[i].set_ylabel(node)
        ax[i].set_xlabel(TIME)

    if path_to_save:
        fig.savefig(os.path.join(path_to_save, 'trajectory_plot.png'))


def visualize_pomdp_simulation(df_traj, df_b, df_Q, node_list=None, path_to_save='../_data/'):
    if node_list is None:
        node_list = ['X', 'Y', 'o', 'Z']

    plot_trajectories(df_traj, node_list=node_list, path_to_save=path_to_save)

    fig, ax = plt.subplots(2, 1)
    df_b.plot(ax=ax[0])
    df_Q.plot(ax=ax[1])

    fig.savefig(os.path.join(path_to_save, 'b_Q_plot.png'))
    plt.close()


if __name__ == '__main__':
    S = ['00', '01', '10', '11']
    folder = '../_data/pomdp_simulation/1582983024/'
    df_belief = pd.read_csv(folder + 'df_belief.csv')
    df_traj = pd.read_csv(folder + 'env_traj.csv')
    df_Q = pd.read_csv(folder + 'df_Qz.csv')

    visualize_pomdp_simulation(df_traj, df_belief[S], df_Q[S], node_list=['X', 'Y', 'o'], path_to_save=folder)
