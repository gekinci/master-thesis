import numpy as np
import pandas as pd
import os
# import seaborn as sns;
#
# sns.set()
import matplotlib.pyplot as plt
from utils.constants import *
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
    folder_name = "INFER_81MODEL_exactUpdate"
    obs_model = 'psi_0'
    path_to_exp = os.path.join(os.path.join(path_to_data, folder_name), obs_model)
    path_to_thesis = '../../docs/thesis/figures/equivalence_classes'
    print(obs_model)

    phi_set = np.load('../configs/psi_set_81.npy')

    # df_L = pd.read_csv(path_to_exp + '/llh_particleFilter.csv', index_col=0)
    # df_L = df_L[[r'$\psi_3$', r'$\psi_4$']]
    df_L = pd.read_csv(path_to_exp + '/llh_exactUpdate.csv', index_col=0)
    dict_L = {PART_FILT: df_L}
    visualize_llh(dict_L, 200, path_to_save=path_to_thesis)

    for s in df_L.sum(axis=0).unique():
        print(s)
        class_list = df_L.columns[df_L.sum(axis=0) == s]
        # print(class_list)
        # phi_list = []
        for c in class_list:
            ind = int(c.split('_')[-1].split('$')[0])
            # print(ind, phi_set[ind])
            # if s == -16464.560956165187:
            #     phi_list += [phi_set[ind]]
    phi_list = [phi_set[0], phi_set[25], phi_set[29], phi_set[51], phi_set[55]]
    print(phi_list)
    np.save('../configs/psi_set_same_class.npy', phi_list)


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


if __name__ == "__main__":
    run_all_data()
