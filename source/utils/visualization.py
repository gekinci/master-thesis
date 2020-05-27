from utils.constants import *
from utils.helpers import *

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
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


def plot_trajectories(df, node_list=None, path_to_save=None, tag=''):
    if node_list is None:
        node_list = ['X1', 'X2', 'y', 'X3']

    # Saving and plotting the trajectories
    fig, ax = plt.subplots(len(node_list), sharex=True)
    for i, node in enumerate(node_list):
        ax[i].step(df[TIME], df[node], where='post')
        if node == 'y':
            ax[i].set_ylim([-.5, 2.5])
        else:
            ax[i].set_ylim([-.5, 1.5])
        ax[i].set_ylabel(node + '(t)')
    ax[len(node_list) - 1].set_xlabel('t')

    if path_to_save:
        fig.savefig(os.path.join(path_to_save, 'trajectory_plot_' + tag + '.png'))
    plt.close('all')


def visualize_pomdp_simulation(df_traj, df_b, df_Q, node_list=None, path_to_save='../_data/', tag='',
                               belief_method=EXACT):
    if node_list is None:
        node_list = [r'$X_{1}$', r'$X_{2}$', 'y', r'$X_{3}$']

    plot_trajectories(df_traj, node_list=node_list, path_to_save=path_to_save, tag=tag)

    t_max = df_traj[TIME].values[-1]
    df_b = df_b.truncate(before=to_decimal(0), after=to_decimal(t_max))
    df_Q = df_Q.truncate(before=to_decimal(0), after=to_decimal(t_max))

    fig, ax = plt.subplots(3, 1, sharex=True)

    ax[0].step(df_traj[TIME], df_traj['y'], where='post')
    ax[0].set_ylim([-.5, 2.5])
    ax[0].set_ylabel('y(t)')

    for col in df_b.columns:
        ax[1].plot(df_b.index, df_b[col]) if belief_method == EXACT else ax[1].step(df_b.index, df_b[col], where='post')
    ax[1].set_ylabel(r'$b(x_{1},x_{2};t)$')
    ax[1].legend(['00', '01', '10', '11'],
                 bbox_to_anchor=(1.02, 1.0), loc='upper left')

    for col in df_Q.columns:
        ax[2].plot(df_Q.index, df_Q[col]) if belief_method == EXACT else ax[2].step(df_Q.index, df_Q[col], where='post')
    ax[2].set_ylabel(r'$Q_{3}(t)$')
    ax[2].set_xlabel('t')
    ax[2].legend([r'$q_{0}$', r'$q_{1}$'])

    plt.tight_layout()
    fig.savefig(os.path.join(path_to_save, 'b_Q_plot_' + tag + '.png'))
    plt.close('all')


if __name__ == '__main__':
    S = ['00', '01', '10', '11']
    folder = '../_data/pomdp_simulation/1582983024/'
    df_belief = pd.read_csv(folder + 'df_belief.csv')
    df_traj = pd.read_csv(folder + 'env_traj.csv')
    df_Q = pd.read_csv(folder + 'df_Qz.csv')

    visualize_pomdp_simulation(df_traj, df_belief[S], df_Q[S], node_list=['X1', 'X2', 'y'], path_to_save=folder)
