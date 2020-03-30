import numpy as np
import itertools
from utils.constants import *
from utils.helpers import *


def sample_obs_model(n_states, n_obs):
    z = np.zeros((n_states, n_obs))
    z[np.arange(n_states), np.random.randint(0, n_obs, size=(n_states))] = 1
    return z


def obs_model_set(n_states, n_obs):
    return np.array(list(itertools.product(np.eye(n_obs), repeat=n_states)))


def get_downsampled_obs_set(n_sample, orig_phi, n_states=2, n_obs=3):
    phi_set = obs_model_set(n_states ** 2, n_obs)
    phi_subset = phi_set[np.random.choice(range(len(phi_set)), size=n_sample, replace=False)]
    if not (np.all(orig_phi == phi_subset, axis=(1, 2))).any():
        phi_subset = np.append(phi_subset[:-1], [orig_phi], axis=0)
    return phi_subset[::-1, :, :]


def log_likelihood_inhomogeneous_ctmc(df_traj, df_Q, node='Z'):
    L = 0
    df_trans = df_traj.loc[df_traj[node].diff() != 0]
    trans_tuples = list(
        zip(df_trans[node].astype(int), df_trans[node][1:].astype(int), df_trans[TIME], df_trans[TIME][1:]))
    for i in trans_tuples:
        trans_tag = ''.join(map(str, i[0:2]))
        stay_tag = ''.join((str(i[0]), str(i[0])))
        times = i[2:4]
        q_trans = df_Q.truncate(after=to_decimal(times[1])).iloc[-1].loc[trans_tag]
        df_Q_ = df_Q.truncate(before=to_decimal(times[0]), after=to_decimal(times[1]))
        prob_stay_integral = (df_Q_[stay_tag].multiply(df_Q_[T_DELTA], axis="index")).cumsum().loc[to_decimal(times[1])]
        L += np.log(q_trans) + prob_stay_integral
    return L
