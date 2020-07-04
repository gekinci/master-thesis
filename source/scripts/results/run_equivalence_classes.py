import numpy as np
import pandas as pd
import os
# import seaborn as sns;
#
# sns.set()
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


def run_all_data():
    path_to_data = '/home/gizem/DATA/equivalence_classes/'
    folder_name = "ROC_5MODEL_particleFilter_200samples_class0"
    obs_model = 'psi_3'
    path_to_exp = os.path.join(os.path.join(path_to_data, folder_name), obs_model)
    path_to_thesis = '../../docs/thesis/figures/equivalence_classes'
    print(obs_model)

    phi_set = np.load('../configs/psi_set_81.npy')

    df_L = pd.read_csv(path_to_exp + '/llh_particleFilter.csv', index_col=0)
    # df_L = pd.read_csv(path_to_exp + '/llh_exactUpdate.csv', index_col=0)
    dict_L = {PART_FILT: df_L}
    visualize_llh(dict_L, 200, path_to_save=path_to_thesis)

    # for s in df_L.sum(axis=0).unique():
    #     print(s)
    #     class_list = df_L.columns[df_L.sum(axis=0) == s]
    #     # print(class_list)
    #     # phi_list = []
    #     for c in class_list:
    #         ind = int(c.split('_')[-1].split('$')[0])
    #         # print(ind, phi_set[ind])
    #         # if s == -16464.560956165187:
    #         #     phi_list += [phi_set[ind]]
    # phi_list = [phi_set[0], phi_set[25], phi_set[29], phi_set[51], phi_set[55]]
    # print(phi_list)
    # np.save('../configs/psi_set_same_class.npy', phi_list)


def run_ap_traj():
    path_to_data = '/home/gizem/DATA/equivalence_classes'
    folder_name = "same_behaviour"
    # obs_model = 'psi_0'
    path_to_exp = os.path.join(path_to_data, folder_name)
    path_to_thesis = '../../docs/thesis/figures/equivalence_classes/' + folder_name

    tick_font = 14
    label_font = 18
    df_traj = pd.read_csv(path_to_exp + '/traj.csv', index_col=0)
    df_traj[r'$X_{P}$'] = ''
    df_traj.loc[:, r'$X_{P}$'] = df_traj.apply(get_state, axis=1)

    node_list = [r'$X_{1}$', r'$X_{2}$', r'$X_{P}$']
    fig, ax = plt.subplots(len(node_list), figsize=(12, 6), sharex=True)
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


def get_observation(row):
    phi_set = np.load('../configs/psi_set_81.npy')
    PSI = phi_set[16]
    S = cartesian_products(len(parent_list_))
    O = [0, 1, 2]
    state = ''.join(map(str, row[parent_list_].values.astype(int)))
    state_index = S.index(state)
    obs = np.random.choice(O, p=PSI[state_index])
    return obs


def run_ap_beliefs():
    path_to_data = '/home/gizem/DATA/equivalence_classes'
    folder_name = "same_behaviour"
    obs_model = 'psi_16'
    path_to_exp = os.path.join(os.path.join(path_to_data, folder_name), obs_model)
    path_to_thesis = '../../docs/thesis/figures/equivalence_classes/' + folder_name

    tick_font = 14
    label_font = 18
    legend_font = 14
    df_b_cols = ['00', '01', '10', '11']
    df_q_cols = ['01', '10']
    df_traj = pd.read_csv(os.path.join(path_to_data, folder_name) + '/traj.csv', index_col=0)
    df_traj['y'] = df_traj.apply(get_observation, axis=1)

    df_belief = pd.read_csv(path_to_exp + '/belief_exactUpdate_2.csv', index_col=0)
    df_q = pd.read_csv(path_to_exp + '/Q_agent_exactUpdate_2.csv', index_col=0)

    n_axis = 3
    fig, ax = plt.subplots(n_axis, 1, sharex=True, figsize=(12, 6))

    ax[0].step(df_traj[TIME], df_traj['y'], where='post')
    ax[0].set_ylim([-.5, 2.5])
    ax[0].set_ylabel('y(t)', fontsize=label_font)
    sec = ax[0].secondary_yaxis(location='right')
    sec.set_ylabel('  (a)', fontsize=label_font, rotation='horizontal', ha='left')
    sec.set_yticks([])
    ax[0].set_ylim([-.5, 2.5])
    ax[0].set_yticks([0, 1, 2])
    ax[0].set_yticklabels([0, 1, 2], fontsize=tick_font)

    for col in df_b_cols:
        ax[1].step(df_belief.index, df_belief[col], where='post')
    ax[1].set_ylim([-0.1, 1.1])
    ax[1].set_yticks([0, 0.5, 1])
    ax[1].set_yticklabels([0, 0.5, 1], fontsize=tick_font)
    ax[1].set_ylabel(r'$b(x_{p}$' + f';t)', fontsize=label_font)
    ax[1].legend([r'$x_{p}$=00', r'$x_{p}$=01', r'$x_{p}$=10', r'$x_{p}$=11'], fontsize=legend_font)
    sec = ax[1].secondary_yaxis(location='right')
    sec.set_ylabel('  (b)', fontsize=label_font, rotation='horizontal', ha='left')
    sec.set_yticks([])

    for col in df_q_cols:
        ax[2].step(df_q.index, df_q[col], where='post')
    ax[2].set_ylim([-.2, 3.2])
    ax[2].set_yticks([0, 1, 2, 3])
    ax[2].set_yticklabels([0, 1, 2, 3], fontsize=tick_font)
    ax[2].set_ylabel(r'$Q_{3}(t)$', fontsize=label_font)
    ax[2].legend([r'$q_{0}$', r'$q_{1}$'], fontsize=legend_font)
    sec = ax[2].secondary_yaxis(location='right')
    sec.set_ylabel('  (b)', fontsize=label_font, rotation='horizontal', ha='left')
    sec.set_yticks([])

    ax[n_axis - 1].set_xlabel('t / s', fontsize=label_font)
    ax[n_axis - 1].set_xticks([0, 1, 2, 3, 4, 5])
    ax[n_axis - 1].set_xticklabels([0, 1, 2, 3, 4, 5], fontsize=tick_font)
    plt.tight_layout()
    fig.savefig(os.path.join(os.path.join(path_to_data, folder_name), obs_model + '.pdf'))
    fig.savefig(os.path.join(path_to_thesis, obs_model + '.pdf'))


if __name__ == "__main__":
    run_all_data()
