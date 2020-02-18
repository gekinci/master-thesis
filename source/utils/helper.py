import itertools
import numpy as np
import pandas as pd
import time
import os


def create_folder_for_experiment(folder_name='../data'):
    t = int(time.time())
    folder_exp = os.path.join(folder_name, str(t))
    os.makedirs(folder_exp, exist_ok=True)
    return folder_exp


def cartesian_products(n_par, n_states=2):
    # it is adapted to non-binary states
    state_str = '' + "".join(map(str, range(n_states)))
    return ["".join(seq) for seq in itertools.product(state_str, repeat=n_par)]


def zero_div(x, y):
    return x / y if y != 0 else 0


# def merge_independent_ctbn_trajectories(df1, df2):
#     df_joined = pd.concat([df1, df2], ignore_index=True, sort=False).sort_values(
#         by=constants.TIME).ffill().drop_duplicates(subset=constants.TIME, keep='last')
#     return df_joined


def random_q_matrix(n_values):
    tmp = np.random.rand(n_values, n_values)
    np.fill_diagonal(tmp, 0)
    np.fill_diagonal(tmp, -tmp.sum(axis=1))
    Q = tmp.tolist()
    return Q


def get_amalgamated_trans_matrix(Q1, Q2):
    x1 = Q1[0][1]
    y1 = Q1[1][0]
    x2 = Q2[0][1]
    y2 = Q2[1][0]
    T = np.array([[0, x2, x1, .0],
                  [y2, 0, .0, x1],
                  [y1, .0, 0, x2],
                  [.0, y1, y2, 0]])
    np.fill_diagonal(T, -T.sum(axis=1))
    return T


def generate_belief_grid(step, path_to_save='./'):
    cols = ['b1', 'b2', 'b3', 'b4']
    b = []

    for b1 in np.arange(0, 1 + 0.001, step):
        b234 = 1 - b1
        for b2 in np.arange(0, b234 + 0.001, step):
            b34 = b234 - b2
            for b3 in np.arange(0, b34 + 0.001, step):
                b4 = b34 - b3
                b += [[b1.round(3), b2.round(3), b3.round(3), abs(b4).round(3)]]
    df = pd.DataFrame(b, columns=cols)
    df.to_csv(path_to_save + f'b_grid_{str(step).split(".")[0]}{str(step).split(".")[-1]}.csv')
    return df
