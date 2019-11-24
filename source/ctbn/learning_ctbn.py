from ctbn.utils import *
import numpy as np
import pandas as pd
import ctbn.config as cfg
from sklearn.metrics import mean_squared_error
import constants


def get_time_of_stay_in_state(df_traj, node, df_all=None):
    df = df_traj.copy()
    df.loc[:, 'block'] = (df[node].shift(1) != df[node]).astype(int).cumsum()
    tmp = df.reset_index().groupby([node, 'block'])['index'].apply(np.array)
    T = np.zeros(2)
    for val in [0, 1]:
        if val in tmp.index:
            for arr in tmp.loc[val].values:
                start = arr[0]
                end = arr[-1] + 1 if arr[-1] != df.index.values[-1] else arr[-1]
                if df_all is None:
                    T[val] += df.loc[end, constants.TIME] - df.loc[start, constants.TIME]
                else:
                    T[val] += df_all.loc[end, constants.TIME] - df_all.loc[start, constants.TIME]
    return T


def get_number_of_transitions(df_traj, node):
    change0to1 = (df_traj[node].diff() == 1)
    change1to0 = (df_traj[node].diff() == -1)
    M0 = change0to1.sum()  # also M0
    M1 = change1to0.sum()  # also M1
    return M0, M1


def get_sufficient_statistics(df_all):
    stats = {}

    for traj in df_all.trajectory_id.unique():
        df_traj = df_all[df_all[constants.TRAJ_ID] == traj]

        stats[traj] = {}

        for node in cfg.node_list:

            stats[traj][node] = {}

            parents_list = cfg.graph_dict[node]
            n_parents = len(parents_list)
            if n_parents == 0:
                stats[traj][node]['M0'], stats[traj][node]['M1'] = get_number_of_transitions(df_traj, node)
                stats[traj][node]['T0'], stats[traj][node]['T1'] = get_time_of_stay_in_state(df_traj, node)
            else:

                parent_cart_prod = cartesian_products(n_parents)
                for parent_code in parent_cart_prod:

                    stats[traj][node][parent_code] = {}

                    parent_values = [int(i) for i in parent_code]
                    df_traj_par = df_traj.loc[(df_traj[parents_list] == parent_values).all(axis=1)]
                    if df_traj_par.empty:
                        stats[traj][node][parent_code]['M0'], stats[traj][node][parent_code]['M1'], \
                        stats[traj][node][parent_code]['T0'], stats[traj][node][parent_code]['T1'] = 0, 0, 0, 0
                    else:
                        stats[traj][node][parent_code]['M0'], stats[traj][node][parent_code][
                            'M1'] = get_number_of_transitions(df_traj_par, node)
                        stats[traj][node][parent_code]['T0'], stats[traj][node][parent_code][
                            'T1'] = get_time_of_stay_in_state(df_traj_par, node, df_all=df_traj)

    return stats


def learn_ctbn_parameters(stats):
    Q_pred = {}

    for node in cfg.node_list:
        parents_list = cfg.graph_dict[node]
        n_parents = len(parents_list)

        if n_parents == 0:
            M0_list = [sublist[node]['M0'] for sublist in stats]
            T0_list = [sublist[node]['T0'] for sublist in stats]
            M1_list = [sublist[node]['M1'] for sublist in stats]
            T1_list = [sublist[node]['T1'] for sublist in stats]

            Q_pred[node] = [[0.0, zero_div(np.sum(M0_list), np.sum(T0_list))],
                            [zero_div(np.sum(M1_list), np.sum(T1_list)), 0.0]]

        else:
            Q_pred[node] = {}
            parent_cart_prod = cartesian_products(n_parents)
            for parent_code in parent_cart_prod:
                M0_list = [sublist[node][parent_code]['M0'] for sublist in stats]
                T0_list = [sublist[node][parent_code]['T0'] for sublist in stats]
                M1_list = [sublist[node][parent_code]['M1'] for sublist in stats]
                T1_list = [sublist[node][parent_code]['T1'] for sublist in stats]
                Q_pred[node][parent_code] = [[0.0, zero_div(np.sum(M0_list), np.sum(T0_list))],
                                             [zero_div(np.sum(M1_list), np.sum(T1_list)), 0.0]]

    return Q_pred


def calculate_log_likelihood(df_all, Q):
    stats = get_sufficient_statistics(df_all)

    L_list = []
    for traj in stats.keys():
        L = 0
        stats_traj = stats[traj]
        for node in cfg.node_list:
            parents_list = cfg.graph_dict[node]
            n_parents = len(parents_list)
            if n_parents == 0:
                L += (stats_traj[node]['M0'] * np.log(Q[node][0][1]) - stats_traj[node]['T0'] * Q[node][0][1] +
                      stats_traj[node]['M1'] * np.log(Q[node][1][0]) - stats_traj[node]['T1'] * Q[node][1][0])
            else:
                parent_cart_prod = cartesian_products(n_parents)
                for parent_code in parent_cart_prod:
                    L += (stats_traj[node][parent_code]['M0'] * np.log(Q[node][parent_code][0][1]) -
                          stats_traj[node][parent_code]['T0'] * Q[node][parent_code][0][1] +
                          stats_traj[node][parent_code]['M1'] * np.log(Q[node][parent_code][1][0]) -
                          stats_traj[node][parent_code]['T1'] * Q[node][parent_code][1][0])
        L_list += [L]
    return L_list


def calculate_mse_for_Q(Q, Q_pred):
    mse_list = []
    count = 0
    for key in Q:
        if type(Q[key]) == dict:
            for subkey in Q[key]:
                mse_list += [mean_squared_error(Q[key][subkey], Q_pred[key][subkey])]
                count += 1
        else:
            mse_list += [mean_squared_error(Q[key], Q_pred[key])]
            count += 1

    return mse_list


def train_and_evaluate(ctbn, df_train, df_test):
    n_train = df_train[constants.TRAJ_ID].max()
    n_test = df_test[constants.TRAJ_ID].max()

    suff_stats = get_sufficient_statistics(df_train)

    # L_true_model = calculate_log_likelihood(df_test, ctbn.Q)
    # L_true_model_avg = np.average(L_true_model)
    # L_true_model_std = np.std(L_true_model)
    # df_L = pd.DataFrame()

    df_mse = pd.DataFrame()

    for traj in range(n_train):
        stats_trained = [v for k, v in suff_stats.items() if k <= traj]

        Q_pred = learn_ctbn_parameters(stats_trained)

        mse_iter = calculate_mse_for_Q(ctbn.Q, Q_pred)

        mse = {'number_of_trajectories': np.tile(traj + 1, len(mse_iter)),
               'mean_squared_error': mse_iter}
        df_mse = df_mse.append(pd.DataFrame.from_dict(mse))

        # likelihood = {'number_of_trajectories': np.tile(traj + 1, n_test),
        #               'test_log_likelihood': calculate_log_likelihood(df_test, Q_pred)}
        # df_L = df_L.append(pd.DataFrame.from_dict(likelihood))

    return df_mse
