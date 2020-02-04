import itertools
import constants
import pandas as pd
import numpy as np


def cartesian_products(n_par, n_states=2):
    # it is adapted to non-binary states
    state_str = '' + "".join(map(str, range(n_states)))
    return ["".join(seq) for seq in itertools.product(state_str, repeat=n_par)]


def zero_div(x, y):
    return x / y if y != 0 else 0


#
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


def amalgamation_independent_cim(Q1, Q2):
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
