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
        fig.savefig(path_to_save + f'OPM_{s.split("_")[-1]}.pdf')
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
        fig.savefig(os.path.join(path_to_save, 'trajectory_plot_' + tag + '.pdf'))
    plt.close('all')


def visualize_pomdp_simulation(df_traj, dict_b, dict_Q, node_list=None, path_to_save='../_data/', tag=''):
    node_list = [r'$X_{1}$', r'$X_{2}$', 'y', r'$X_{3}$'] if node_list is None else node_list
    df_Q_cols = ['01', '10']
    df_b_cols = ['00', '01', '10', '11']
    b_axis = 4 if len(dict_b) > 1 else 1
    q_axis = 2 if len(dict_Q) > 1 else 1

    plot_trajectories(df_traj, node_list=node_list, path_to_save=path_to_save, tag=tag)

    t_max = df_traj[TIME].values[-1]
    dict_b = {m: df_.truncate(before=to_decimal(0), after=to_decimal(t_max)) for m, df_ in dict_b.items()}
    dict_Q = {m: df_.truncate(before=to_decimal(0), after=to_decimal(t_max)) for m, df_ in dict_Q.items()}

    fig, ax = plt.subplots(b_axis + q_axis + 1, 1, sharex=True, figsize=(15, 12))

    ax[0].step(df_traj[TIME], df_traj['y'], where='post')
    ax[0].set_ylim([-.5, 2.5])
    ax[0].set_ylabel('y(t)')

    count = 1
    if len(dict_b) == 1:
        df_ = dict_b[list(dict_b.keys())[0]]
        for col in df_b_cols:
            ax[count].plot(df_.index, df_[col]) if EXACT in list(dict_b.keys()) else ax[count].step(df_.index, df_[col],
                                                                                                    where='post')
        ax[count].set_ylabel(r'$b(x_{1},x_{2};t)$')
        ax[count].legend(['00', '01', '10', '11'], bbox_to_anchor=(1.02, 1.0), loc='upper left')
        ax[count].set_title(list(dict_b.keys())[0])
        count += 1
    else:
        for col in df_b_cols:
            for m, df in dict_b.items():
                ax[count].plot(df.index, df[col]) if m == EXACT else ax[count].step(df.index, df[col], where='post')
            ax[count].set_ylabel(r'$b(x_{1},x_{2}$' + f' = {col};t)')
            count += 1
        ax[count - 4].legend(list(dict_b.keys()), bbox_to_anchor=(1.02, 1.0), loc='upper left')

    if len(dict_Q) == 1:
        df_ = dict_Q[list(dict_Q.keys())[0]]
        for col in df_Q_cols:
            ax[count].step(df_.index, df_[col], where='post')
        ax[count].set_ylabel(r'$Q_{3}(t)$')
        ax[count].set_title(list(dict_Q.keys())[0])
        ax[count].set_xlabel('t')
        ax[count].legend([r'$q_{0}$', r'$q_{1}$'], bbox_to_anchor=(1.02, 1.0), loc='upper left')
        count += 1
    else:
        for col in df_Q_cols:
            for m, df in dict_Q.items():
                ax[count].step(df.index, df[col], where='post')
            ax[count].set_ylabel(r'$q_{0}(t)$') if col == '01' else ax[count].set_ylabel(r'$q_{1}(t)$')
            count += 1
        ax[count - 2].legend(list(dict_b.keys()), bbox_to_anchor=(1.02, 1.0), loc='upper left')

    plt.tight_layout()
    fig.savefig(os.path.join(path_to_save, 'b_Q_plot_' + tag + '.pdf'))
    plt.close('all')


def visualize_llh(dict_L, n_train, path_to_save):
    for m, df_L in dict_L.items():
        df_L_norm = df_L.cumsum().div((df_L.index + 1), axis=0)
        plt.figure()
        df_L_norm.head(n_train).plot()
        plt.xlabel('Number of trajectories')
        plt.ylabel('Average log-likelihood')
        plt.tight_layout()
        plt.savefig(os.path.join(path_to_save, f'llh_{m}.pdf'))
        plt.close()
