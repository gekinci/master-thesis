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
        self.initial_probs = cfg[INITIAL_PROB] if cfg[INITIAL_PROB] else np.ones(len(self.states)) / len(self.states)

        self.HOW_TO_PRED_STATE = cfg[HOW_TO_PRED_STATE]
        self.time_grain = cfg[TIME_GRAIN]

        # self.initial_states = {**self.parent_ctbn.initial_states, **self.initialize_agent(cfg)}

        self.S = cartesian_products(self.n_parents, states=self.states)
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

        time_grid = custom_decimal_range(0, self.t_max + .00000001, self.time_grain)
        self.df_b = pd.DataFrame(columns=self.S + [T_DELTA], index=time_grid)
        self.df_Qz = pd.DataFrame(columns=self.S + [T_DELTA], index=time_grid)

    def initialize_agent(self):
        return {'Z': np.random.choice(self.states, p=self.initial_probs)}

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
        if t > 0:
            self.belief_state = self.df_b.truncate(before=self.df_b.index[0], after=to_decimal(t)).iloc[-1][
                self.S].values
        self.belief_state = self.Z[:, obs] * (self.belief_state @ scipy.linalg.expm(self.T * self.time_grain))
        self.belief_state = self.belief_state / self.belief_state.sum()
        self.df_b.loc[to_decimal(t), self.S] = self.belief_state
        self.df_b.loc[to_decimal(t), T_DELTA] = 0
        self.df_b.sort_index(inplace=True)
        self.df_Qz.loc[to_decimal(t), :] = np.nan
        self.df_Qz.sort_index(inplace=True)

    def get_cont_belief_state(self, t, t_next):
        def helper(row):
            row[self.S] = row[self.S].values @ scipy.linalg.expm(self.T * row[T_DELTA])
            return row

        ind = (self.df_b.index >= to_decimal(t)) & (self.df_b.index < to_decimal(t_next))
        tmp = self.df_b.loc[ind].copy()
        tmp.t_delta = (tmp.index - tmp.index[0]).values.astype(float)
        tmp.ffill(inplace=True)
        self.df_b.loc[ind, :] = tmp.apply(helper, axis=1)

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
                self.get_cont_belief_state(t, t_next)

    def get_cont_Q(self, t=0, t_next=None):
        def helper(v):
            p_a = self.get_prob_action(belief=v.values)
            Qz = self.get_Qz(p_a)
            v[self.S] = Qz.flatten()
            return v

        t_next = self.t_max if t_next is None else t_next

        ind = (self.df_b.index >= to_decimal(t)) & (self.df_b.index < to_decimal(t_next))
        self.df_Qz.loc[ind, self.S] = self.df_b.loc[ind, self.S].apply(helper, axis=1)
        self.df_Qz.loc[ind, T_DELTA] = self.time_grain

    def sample_parent_trajectory(self):
        return self.parent_ctbn.sample_trajectory()

    def draw_time_contQ(self, t_start, t_end, df_Q):
        col_transition = ['01', '10']  # TODO hardcoded

        df_Q_ = df_Q.truncate(before=to_decimal(t_start), after=to_decimal(t_end))

        F = 1 - np.exp(-(df_Q_[col_transition].multiply(df_Q_[T_DELTA], axis="index")).cumsum().astype(float))
        rnd = np.random.uniform()

        tmp = {col: F.loc[F[col] > rnd].index[0] for col in F.columns if not F.loc[F[col] > rnd].empty}
        if tmp:
            ind = min(tmp.values())
            coln = min(tmp, key=tmp.get)

            t_diff = np.log((1 - rnd) / (1 - F.loc[ind, coln])) / df_Q.loc[:ind, coln].values[-2]
            t_next = np.float(ind) - t_diff

            self.df_Qz.loc[to_decimal(t_next), T_DELTA] = t_diff
            self.df_Qz.loc[to_decimal(t_next), self.S] = \
                self.df_Qz.loc[self.df_Qz.index < to_decimal(t_next)][self.S].iloc[-1]
            self.df_Qz.sort_index(inplace=True)
            self.df_b.loc[to_decimal(t_next), :] = np.nan
            self.df_b.sort_index(inplace=True)
        else:
            t_next = np.nan
        return t_next

    def do_step(self, prev_step):
        t = prev_step[TIME].values[0]

        tmp = self.parent_ctbn.do_step(prev_step)
        t_par_change = tmp[TIME].values[0]

        self.get_cont_belief_state(t, t_par_change)
        self.get_cont_Q(t=t, t_next=t_par_change)

        t_agent_change = self.draw_time_contQ(t, t_par_change, self.df_Qz)

        if t_agent_change < t_par_change:
            next_step = prev_step.copy()
            next_step.loc[:, TIME] = t_agent_change
            next_step.loc[:, 'Z'] = int(1 - next_step['Z'])  # TODO only for binary

            self.df_b.loc[self.df_b.index > to_decimal(t_agent_change)] = np.nan
            self.df_Qz.loc[self.df_Qz.index > to_decimal(t_agent_change)] = np.nan

            self.get_cont_belief_state(t_agent_change - self.time_grain, t_agent_change + .0000001)
            self.get_cont_Q(t=t_agent_change - self.time_grain, t_next=t_agent_change + .0000001)

            NEW_OBS = False
        else:
            next_step = tmp.copy()
            next_step.loc[:, OBS] = next_step.apply(self.get_observation, axis=1)
            NEW_OBS = next_step.iloc[0][OBS] != prev_step.iloc[0][OBS]

        return next_step, NEW_OBS

    def sample_trajectory(self):
        t = 0
        initial_states = {**self.parent_ctbn.initialize_nodes(), **self.initialize_agent(), **{TIME: t}}

        df_traj = pd.DataFrame().append(initial_states, ignore_index=True)
        df_traj.loc[:, OBS] = df_traj.apply(self.get_observation, axis=1)
        prev_step = pd.DataFrame(df_traj[-1:].values, columns=df_traj.columns)
        NEW_OBS = True

        while t < self.t_max:
            if NEW_OBS:
                self.update_belief_state(int(prev_step[OBS].values[0]), prev_step[TIME].values[0])
            new_step, NEW_OBS = self.do_step(prev_step)
            t = new_step[TIME].values[0]
            if t > self.t_max:
                prev_step.loc[:, TIME] = self.t_max
                df_traj = df_traj.append(prev_step, ignore_index=True)
                break
            df_traj = df_traj.append(new_step, ignore_index=True)
            prev_step = new_step.copy()

        df_traj.to_csv(os.path.join(self.FOLDER, 'df_traj.csv'))
        self.df_b.to_csv(os.path.join(self.FOLDER, 'df_belief.csv'))
        self.policy.to_csv(os.path.join(self.FOLDER, 'df_policy.csv'))
        self.df_Qz.to_csv(os.path.join(self.FOLDER, 'df_Qz.csv'))

        visualize_pomdp_simulation(df_traj, self.df_b[self.S], self.df_Qz[['01', '10']], path_to_save=self.FOLDER)

        return df_traj
