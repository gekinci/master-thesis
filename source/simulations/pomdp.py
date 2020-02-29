from utils.constants import *
from utils.helpers import *
from utils.visualization import *
from simulations.ctbn import CTBNSimulation

import matplotlib.pyplot as plt
import time
import logging
import scipy
from timer import Timer


class POMDPSimulation:
    def __init__(self, cfg, save_folder='../_data/pomdp_simulation', import_data=None):

        self.FOLDER = save_folder
        self.IMPORT = import_data

        self.parent_ctbn = CTBNSimulation(cfg, save_folder=self.FOLDER)
        self.t_max = cfg[T_MAX] if cfg[T_MAX] else 20

        self.parent_list = cfg[PARENT_LIST] if cfg[PARENT_LIST] else ['X', 'Y']
        self.n_parents = len(self.parent_list)

        self.states = cfg[STATES] if cfg[STATES] else [0, 1]
        self.n_states = len(self.states)

        self.HOW_TO_PRED_STATE = cfg[HOW_TO_PRED_STATE]
        self.time_grain = cfg[TIME_GRAIN]

        self.S = cartesian_products(self.n_parents, states=self.states)
        # [list(seq) for seq in itertools.product(self.states, repeat=self.n_parents)]
        self.O = cfg[OBS_SPACE]
        self.A = [str(i) for i in cfg[ACT_SPACE]]

        self.Qz = {k: random_q_matrix(self.n_states) for k in self.A}

        self.policy = self.generate_random_stoch_policy()

        # T(a, s, s')
        self.T = get_amalgamated_trans_matrix(self.parent_ctbn.Q[self.parent_list[0]],
                                              self.parent_ctbn.Q[self.parent_list[1]])

        # Z(a, s', o)
        self.Z = np.array([[1, 0, 0],
                           [0, 1, 0],
                           [0, 1, 0],
                           [0, 0, 1]])

        self.belief_state = np.tile(1 / len(self.S), len(self.S))

        self.df_b = pd.DataFrame(columns=self.S + ['t_delta'])
        self.df_Qz = pd.DataFrame(columns=self.S + ['t_delta'])

    def generate_random_stoch_policy(self):
        policy = generate_belief_grid(step=0.01, cols=self.S)
        for action in self.A:
            policy[str(action)] = np.random.random(len(policy))
        policy[self.A] = policy[self.A].div(policy[self.A].sum(axis=1), axis=0)
        return policy

    def get_observation(self, df):  # Stochastic observation
        state = ''.join(map(str, df[self.parent_list].values.astype(int)))
        state_index = self.S.index(state)
        obs = np.random.choice(self.O, p=self.Z[state_index])
        return obs

    def update_belief_state(self, obs, t):
        self.belief_state = self.Z[:, obs] * (self.belief_state @ scipy.linalg.expm(self.T * self.time_grain))
        self.belief_state = self.belief_state / self.belief_state.sum()
        self.df_b.loc[custom_round(t, decimals=3), self.S] = self.belief_state
        self.df_b.loc[custom_round(t, decimals=3), 't_delta'] = 0

    def continuous_belief_state(self, t, t_next):

        t = custom_round(t, decimals=3)
        t_next = custom_round(t_next, decimals=3)

        def helper(row):
            row[self.S] = row[self.S].values @ scipy.linalg.expm(self.T * row['t_delta'])
            return row

        tmp = self.df_b.loc[(self.df_b.index >= t) & (self.df_b.index < t_next)].copy()
        tmp.t_delta = tmp.t_delta.ffill() + tmp.groupby(
            tmp.t_delta.notnull().cumsum()).cumcount() * self.time_grain
        tmp.ffill(inplace=True)
        self.df_b.loc[(self.df_b.index >= t) & (self.df_b.index < t_next), :] = tmp.apply(helper, axis=1)
        self.belief_state = self.df_b.loc[(self.df_b.index < t_next), :].iloc[-1][self.S].values

    def get_prob_action(self, belief=None):
        if belief is None:
            belief = self.belief_state
        return self.policy.loc[np.argmin(abs(self.policy[self.S].values - belief.astype(float)).sum(axis=1)), self.A]

    def get_Qz(self, p_act):
        Q = np.sum([np.array(self.Qz[i]) * p_act[i] for i in self.A], axis=0)
        return Q

    def get_belief_traj(self, df_traj):
        t_end = df_traj[TIME].values[-1]
        time_grid = custom_decimal_range(0, t_end, self.time_grain)
        self.df_b = self.df_b.reindex(time_grid)

        for i, row in df_traj.iterrows():
            self.update_belief_state(int(row[OBS]), row[TIME])

            if i == df_traj.index[-1]:
                break
            else:
                t = row[TIME]
                t_next = df_traj[TIME].values[i + 1]
                self.continuous_belief_state(t, t_next)

    def get_Qz_traj(self):
        def helper(v):
            p_a = self.get_prob_action(belief=v.values)
            Qz = self.get_Qz(p_a)
            v[self.S] = Qz.flatten()
            return v

        self.df_Qz[self.S] = self.df_b[self.S].apply(helper, axis=1)

    def sample_parent_trajectory(self):
        return self.parent_ctbn.sample_trajectory()

    def sample_trajectory(self):

        if self.IMPORT is None:
            df_traj = self.sample_parent_trajectory()
            df_traj.loc[:, OBS] = df_traj.apply(self.get_observation, axis=1)

            self.get_belief_traj(df_traj)
            self.get_Qz_traj()

        else:
            df_traj = pd.read_csv(f'../_data/pomdp_simulation/{self.IMPORT}/env_traj.csv', index_col=0)
            self.df_b = pd.read_csv(f'../_data/pomdp_simulation/{self.IMPORT}/df_belief.csv', index_col=0)
            self.df_b.index = self.df_b.index.map(custom_round)
            self.policy = pd.read_csv(f'../_data/pomdp_simulation/{self.IMPORT}/df_policy.csv', index_col=0)
            self.df_Qz = pd.read_csv(f'../_data/pomdp_simulation/{self.IMPORT}/df_Qz.csv', index_col=0)

        df_traj.to_csv(os.path.join(self.FOLDER, 'env_traj.csv'))
        self.df_b.to_csv(os.path.join(self.FOLDER, 'df_belief.csv'))
        self.policy.to_csv(os.path.join(self.FOLDER, 'df_policy.csv'))
        self.df_Qz.to_csv(os.path.join(self.FOLDER, 'df_Qz.csv'))

        # t = 0
        #
        # while t < self.t_max:
        #     t = self.do_step(t)
        #
        # plot_trajectories(df_traj, node_list=['X', 'Y', 'o'], path_to_save=self.FOLDER)

        # self.df_b[self.S].plot()

        return df_traj

    # def Qintegral(self):
    #     return
    #
    # def do_step(self, t):
    #     t_rounded = custom_round(t)
    #     F = 1 - np.exp(-(self.df_Qz.truncate(before=t_rounded, after=self.df_Qz.index[-1])[
    #                          ['01', '10']] * self.time_grain).cumsum())  # TODO hardcoded
    #     rnd = np.random.uniform()
    #     ind = np.min([F.loc[F[col] < rnd].index[-1] for col in F.columns])
    #     coln = F.columns[np.argmin([F.loc[F[col] < rnd].index[-1] for col in F.columns])]
    #
    #     t_next = np.float(ind) - np.log(1 - (rnd - F.loc[ind, coln])) / self.df_Qz.loc[ind:, coln].values[1]
    #
    #
    #     return t_next
