import itertools
import constants
import pandas as pd
import numpy as np


def cartesian_products(n):
    return ["".join(seq) for seq in itertools.product("01", repeat=n)]


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
    Q = tmp.tolist()
    return Q
