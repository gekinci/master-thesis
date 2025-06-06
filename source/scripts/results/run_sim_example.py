import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from utils.constants import *
from utils.helpers import *
from scripts.run_inference_mp import visualize_llh


def get_state(row):
    if (row[parent_list_] == [0, 0]).all():
        state = 0
    elif (row[parent_list_] == [0, 1]).all():
        state = 1
    elif (row[parent_list_] == [1, 0]).all():
        state = 2
    else:
        state = 3
    return state


if __name__ == "__main__":
    path_to_data = '/home/gizem/DATA'
    folder_name = "sim_example"
    path_to_data = os.path.join(path_to_data, folder_name)
    path_to_thesis = '/home/gizem/master_thesis/docs/thesis/figures/sim_example'

    tick_font = 14
    label_font = 18
    legend_font = 14
    df_traj = pd.read_csv(path_to_data + '/df_traj.csv', index_col=0)
    df_traj[r'$X_{P}$'] = ''
    df_traj.loc[:, r'$X_{P}$'] = df_traj.apply(get_state, axis=1)

    node_list = [r'$X_{1}$', r'$X_{2}$', r'$X_{P}$', r'y']
    fig, ax = plt.subplots(len(node_list), figsize=(12, 7), sharex=True)
    for i, node in enumerate(node_list):
        ax[i].step(df_traj[TIME], df_traj[node], where='post')
        if node == 'y':
            ax[i].set_ylim([-.5, 2.5])
            ax[i].set_yticks([0, 1, 2])
            ax[i].set_yticklabels([0, 1, 2], fontsize=tick_font)
        elif node == r'$X_{P}$':
            ax[i].set_ylim([-.5, 3.5])
            ax[i].set_yticks([0, 1, 2, 3])
            ax[i].set_yticklabels(['00', '01', '10', '11'], fontsize=tick_font)
        else:
            ax[i].set_ylim([-.5, 1.5])
            ax[i].set_yticks([0, 1])
            ax[i].set_yticklabels(['0', '1'], fontsize=tick_font)
        ax[i].set_ylabel(node + '(t)', fontsize=label_font)
        if i == 0:
            sec = ax[i].secondary_yaxis(location='right')
            sec.set_ylabel('  (a)', fontsize=label_font, rotation='horizontal', ha='left')
            sec.set_yticks([])
        elif i == 1:
            sec = ax[i].secondary_yaxis(location='right')
            sec.set_ylabel('  (b)', fontsize=label_font, rotation='horizontal', ha='left')
            sec.set_yticks([])
        elif i == 2:
            sec = ax[i].secondary_yaxis(location='right')
            sec.set_ylabel('  (c)', fontsize=label_font, rotation='horizontal', ha='left')
            sec.set_yticks([])
        else:
            sec = ax[i].secondary_yaxis(location='right')
            sec.set_ylabel('  (d)', fontsize=label_font, rotation='horizontal', ha='left')
            sec.set_yticks([])
    ax[len(node_list) - 1].set_xlabel('t / s', fontsize=label_font)
    ax[len(node_list) - 1].set_xticks([0, 1, 2, 3, 4, 5])
    ax[len(node_list) - 1].set_xticklabels([0, 1, 2, 3, 4, 5], fontsize=tick_font)
    plt.tight_layout()
    fig.savefig(os.path.join(path_to_data, 'parent_traj.pdf'))
    fig.savefig(os.path.join(path_to_thesis, 'parent_traj.pdf'))

    ##############################################################################################
    # tick_font = 14
    # label_font = 16
    df_b_cols = ['00', '01', '10', '11']
    b_axis = 4
    t_max = 5
    df_belief_exact = pd.read_csv(path_to_data + '/belief_exactUpdate_.csv', index_col=0)

    df_belief_part = pd.read_csv(path_to_data + '/belief_particleFilter_.csv', index_col=0)
    df_b_list = []
    for i in range(9):
        df_b_list += [pd.read_csv(path_to_data + f'/belief_particleFilter{i}_.csv', index_col=0)]

    ind_merged = df_belief_part.index.append([df.index for df in df_b_list]).sort_values().unique()

    df_belief_part = df_belief_part.reindex(df_belief_part.index.union(ind_merged)).fillna(method='ffill')
    for i in range(9):
        df = df_b_list[i]
        df = df.reindex(df.index.union(ind_merged)).fillna(method='ffill')
        df_b_list[i] = df

    dict_b = {'exactUpdate': df_belief_exact.truncate(before=to_decimal(0), after=to_decimal(t_max)),
              'particleFilter': df_belief_part.truncate(before=to_decimal(0), after=to_decimal(t_max))}

    fig, ax = plt.subplots(b_axis, 1, sharex=True, figsize=(12, 8))

    count = 0
    for col in df_b_cols:
        # for m, df in dict_b.items():
        #     ax[count].plot(df.index, df[col]) if m == EXACT else ax[count].step(df.index, df[col], where='post')
        ax[count].plot(dict_b[EXACT].index, dict_b[EXACT][col])
        df_col = pd.DataFrame(index=ind_merged)
        for i in range(9):
            df_col[i] = df_b_list[i][col]
        df_col = df_col.quantile([0.25, 0.5, 0.75], axis=1)
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        ax[count].plot(df_col.columns.astype(float), df_col.loc[0.5], color=colors[1], label='marginal particle filter')
        ax[count].fill_between(df_col.columns.astype(float), df_col.loc[0.25], df_col.loc[0.75], color=colors[1],
                               alpha=0.3)
        ax[count].set_ylabel(r'$b(x_{P}$' + f'={col};t)', fontsize=label_font)
        ax[count].set_ylim([-0.1, 1.1])
        ax[count].set_yticks([0, 0.5, 1])
        ax[count].set_yticklabels([0, 0.5, 1], fontsize=tick_font)
        sec = ax[count].secondary_yaxis(location='right')
        if count == 0:
            sec.set_ylabel('  (a)', fontsize=label_font, rotation='horizontal', ha='left')
        elif count == 1:
            sec.set_ylabel('  (b)', fontsize=label_font, rotation='horizontal', ha='left')
        elif count == 2:
            sec.set_ylabel('  (c)', fontsize=label_font, rotation='horizontal', ha='left')
        else:
            sec.set_ylabel('  (d)', fontsize=label_font, rotation='horizontal', ha='left')
        sec.set_yticks([])
        count += 1
    ax[1].legend(['exact update', 'marginal particle filter'], loc='best', fontsize=legend_font)
    ax[count - 1].set_xlabel('t / s', fontsize=label_font)
    ax[count - 1].set_xticks([0, 1, 2, 3, 4, 5])
    ax[count - 1].set_xticklabels([0, 1, 2, 3, 4, 5], fontsize=tick_font)
    plt.tight_layout()
    fig.savefig(os.path.join(path_to_thesis, 'belief_traj.pdf'))

    ################################################################################
    # tick_font = 14
    # label_font = 16
    df_q_cols = ['01', '10']
    q_axis = 3
    t_max = 5
    df_q = pd.read_csv(path_to_data + '/Q_agent_particleFilter_.csv', index_col=0).truncate(before=to_decimal(0),
                                                                                            after=to_decimal(t_max))

    fig, ax = plt.subplots(q_axis, 1, sharex=True, figsize=(12, 6))

    for col in df_b_cols:
        ax[0].step(df_belief_part.index, df_belief_part[col], where='post')
    ax[0].set_ylim([-0.1, 1.1])
    ax[0].set_yticks([0, 0.5, 1])
    ax[0].set_yticklabels([0, 0.5, 1], fontsize=tick_font)
    ax[0].set_ylabel(r'$b(x_{P}$' + f';t)', fontsize=label_font)
    ax[0].legend([r'$x_{P}$=00', r'$x_{P}$=01', r'$x_{P}$=10', r'$x_{P}$=11'], fontsize=legend_font)
    sec = ax[0].secondary_yaxis(location='right')
    sec.set_ylabel('  (a)', fontsize=label_font, rotation='horizontal', ha='left')
    sec.set_yticks([])

    for col in df_q_cols:
        ax[1].step(df_q.index, df_q[col], where='post')
    ax[1].set_ylim([-.2, 3.2])
    ax[1].set_yticks([0, 1, 2, 3])
    ax[1].set_yticklabels([0, 1, 2, 3], fontsize=tick_font)
    ax[1].set_ylabel(r'$Q_{3}(t)$', fontsize=label_font)
    ax[1].legend([r'$q_{0}$', r'$q_{1}$'], fontsize=legend_font)
    sec = ax[1].secondary_yaxis(location='right')
    sec.set_ylabel('  (b)', fontsize=label_font, rotation='horizontal', ha='left')
    sec.set_yticks([])

    ax[2].step(df_traj[TIME], df_traj[r'$X_{3}$'], where='post')
    ax[2].set_ylim([-.5, 1.5])
    ax[2].set_yticks([0, 1])
    ax[2].set_yticklabels([0, 1], fontsize=tick_font)
    ax[2].set_ylabel(r'$X_{3}$(t)', fontsize=label_font)
    sec = ax[2].secondary_yaxis(location='right')
    sec.set_ylabel('  (c)', fontsize=label_font, rotation='horizontal', ha='left')
    sec.set_yticks([])
    ax[2].set_xlabel('t / s', fontsize=label_font)
    ax[2].set_xticks([0, 1, 2, 3, 4, 5])
    ax[2].set_xticklabels([0, 1, 2, 3, 4, 5], fontsize=tick_font)

    plt.tight_layout()
    fig.savefig(os.path.join(path_to_thesis, 'q_traj.pdf'))
