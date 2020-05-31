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
                               belief_method=None):
    belief_method = [EXACT] if belief_method is None else belief_method
    node_list = [r'$X_{1}$', r'$X_{2}$', 'y', r'$X_{3}$'] if node_list is None else node_list
    df_Q_cols = ['01', '10']
    df_b_cols = ['00', '01', '10', '11']
    b_axis = 4 if len(df_b) > 1 else 1
    q_axis = 2 if len(df_Q) > 1 else 1

    plot_trajectories(df_traj, node_list=node_list, path_to_save=path_to_save, tag=tag)

    t_max = df_traj[TIME].values[-1]
    df_b = [df_.truncate(before=to_decimal(0), after=to_decimal(t_max)) for df_ in df_b]
    df_Q = [df_.truncate(before=to_decimal(0), after=to_decimal(t_max)) for df_ in df_Q]

    fig, ax = plt.subplots(b_axis + q_axis + 1, 1, sharex=True, figsize=(15, 12))

    ax[0].step(df_traj[TIME], df_traj['y'], where='post')
    ax[0].set_ylim([-.5, 2.5])
    ax[0].set_ylabel('y(t)')

    count = 1
    if len(belief_method) == 1:
        for col in df_b_cols:
            ax[count].plot(df_b[0].index, df_b[0][col]) if EXACT in belief_method else \
                ax[count].step(df_b[0].index, df_b[0][col], where='post')
        ax[count].set_ylabel(r'$b(x_{1},x_{2};t)$')
        ax[count].legend(['00', '01', '10', '11'], bbox_to_anchor=(1.02, 1.0), loc='upper left')
        count += 1
    else:
        for col in df_b_cols:
            for i, m in enumerate(belief_method):
                ax[count].plot(df_b[1].index, df_b[1][col]) if m == EXACT else \
                    ax[count].step(df_b[i].index, df_b[i][col], where='post')
            ax[count].set_ylabel(r'$b(x_{1},x_{2}$' + f' = {col};t)')
            count += 1
        ax[count - 4].legend(belief_method, bbox_to_anchor=(1.02, 1.0), loc='upper left')

    if len(df_Q) == 1:
        for col in df_Q_cols:
            ax[count].step(df_Q[0].index, df_Q[0][col], where='post')
        ax[count].set_ylabel(r'$Q_{3}(t)$')
        ax[count].set_title('from particle filter')
        ax[count].set_xlabel('t')
        ax[count].legend([r'$q_{0}$', r'$q_{1}$'], bbox_to_anchor=(1.02, 1.0), loc='upper left')
        count += 1
    else:
        for col in df_Q_cols:
            for i, m in enumerate(belief_method):
                ax[count].step(df_Q[i].index, df_Q[i][col], where='post')
            ax[count].set_ylabel(r'$q_{0}(t)$') if col == '01' else ax[count].set_ylabel(r'$q_{1}(t)$')
            count += 1
        ax[count - 2].legend(belief_method, bbox_to_anchor=(1.02, 1.0),
                             loc='upper left')

    plt.tight_layout()
    fig.savefig(os.path.join(path_to_save, 'b_Q_plot_' + tag + '.png'))
    plt.close('all')
