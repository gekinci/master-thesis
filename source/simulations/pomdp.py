from utils.constants import *
from utils.helpers import *
from simulations.ctbn import CTBNSimulation

import matplotlib.pyplot as plt
import time
import logging
import scipy


class POMDPSimulation:
    def __init__(self, cfg, save_folder='../_data/pomdp_simulation'):
        self.FOLDER = save_folder

        self.parent_ctbn = CTBNSimulation(cfg, save_folder=self.FOLDER)
        self.t_max = cfg[T_MAX] if cfg[T_MAX] else 20

        self.parent_list = cfg[PARENT_LIST] if cfg[PARENT_LIST] else ['X', 'Y']
        self.n_parents = len(self.parent_list)

        self.states = cfg[STATES] if cfg[STATES] else [0, 1]
        self.n_states = len(self.states)

        self.HOW_TO_PRED_STATE = cfg[HOW_TO_PRED_STATE]
        self.time_grain = cfg[TIME_GRAIN]

        self.S = [list(seq) for seq in itertools.product(self.states, repeat=self.n_parents)]  # state space
        self.O = cfg[OBS_SPACE]
        self.A = cfg[ACT_SPACE]

        self.Qz = {k: random_q_matrix(self.n_states) for k in self.A}

        self.policy = self.generate_random_stoch_policy()

        # T(a, s, s')
        self.T = get_amalgamated_trans_matrix(self.ctbn.Q[self.parent_list[0]],
                                              self.ctbn.Q[self.parent_list[1]])

        # Z(a, s', o)
        self.Z = np.array([[1, 0, 0],
                           [0, 1, 0],
                           [0, 1, 0],
                           [0, 0, 1]])

        self.belief_state = np.tile(1 / len(self.S), len(self.S))

        time_grid = np.around(np.arange(0, self.t_max + self.time_grain, self.time_grain), decimals=3)
        self.df_b = pd.DataFrame(columns=self.S + ['t_delta'], index=time_grid)

    def generate_random_stoch_policy(self):
        policy = generate_belief_grid(step=0.01)
        for action in self.A:
            policy[action] = np.random.random(len(policy))
        policy[self.A] = policy[self.A].div(policy[self.A].sum(axis=1), axis=0)
        return policy

    def get_observation(self, df):  # Stochastic observation
        state = ''.join(map(str, df[self.parent_list].values.astype(int)))  # parent values
        state_index = self.S.index(state)
        obs = np.random.choice(self.O, p=self.Z[state_index])
        logging.debug(f'Observation: {obs}')
        return obs

    def update_belief_state(self, obs, t):
        self.belief_state = self.Z[:, obs] * (self.belief_state @ scipy.linalg.expm(self.T * t))
        self.belief_state = self.belief_state / self.belief_state.sum()
        self.df_b.loc[t, self.S] = self.belief_state
        self.df_b.loc[t, 't_delta'] = 0

    def continuous_belief_state(self):

        def helper(row):
            row[self.S] = row[self.S].values @ scipy.linalg.expm(self.T * row['t_delta'])
            return row

        self.df_b.t_delta = self.df_b.t_delta.ffill() + self.df_b.groupby(
            self.df_b.t_delta.notnull().cumsum()).cumcount() * self.time_grain
        self.df_b.ffill(inplace=True)
        self.df_b.apply(helper, axis=1)

    def get_Qz(self, p_act):
        Q = np.sum([self.Qz[i]*p_act[i].values for i in self.A], axis=0)
        logging.debug(f"Q3 = \n{Q}")
        return Q

    def get_prob_action(self):
        b = self.belief_state.round(2)
        p = self.policy
        p_a = p[(p['b1'] == b[0]) & (p['b2'] == b[1]) & (p['b3'] == b[2]) & (p['b4'] == b[3])][self.A]
        return p_a

    def do_step(self, prev_step, t, prev_obs):
        obs = prev_step[OBS].values[0]
        if prev_obs is None or prev_obs != obs:  # if it is the very first step or obs has changed
            self.update_belief_state(obs, t)

            p_action = self.get_prob_action()

            Qz = self.get_Qz(p_action)
            self.ctbn.Q['3'] = Qz

        new_step = self.ctbn.do_step(prev_step)  # updates only for the nodes of ctbn!
        t = new_step[TIME].values[0].round(3)

        if (prev_step[self.parent_list].values == new_step[self.parent_list].values).all():
            new_step.loc[:, OBS] = prev_step.loc[0, OBS]
        else:
            new_step.loc[:, OBS] = new_step[self.parent_list].apply(self.get_observation, axis=1)

        return new_step, t, obs

    def sample_parent_trajectory(self):
        return self.parent_ctbn.sample_trajectory()

    def sample_trajectory(self):
        t = 0

        # Randomly initializing first states
        initial_states = {var: [np.random.randint(0, 2)] for var in self.ctbn.node_list}
        initial_states[TIME] = 0
        logging.debug(f'Initial states of the nodes: {initial_states}')

        df_traj = pd.DataFrame.from_dict(initial_states)

        # add first observation
        df_traj.loc[:, OBS] = df_traj[self.parent_list].apply(self.get_observation, axis=1)
        prev_step = pd.DataFrame(df_traj[-1:].values, columns=df_traj.columns)
        prev_obs = None

        while t < self.t_max:
            new_step, t, prev_obs = self.do_step(prev_step, t, prev_obs)
            df_traj = df_traj.append(new_step, ignore_index=True)
            prev_step = new_step.copy()

        df_traj.to_csv(self.FOLDER + f'/traj.csv')

        self.continuous_belief_state()
        return df_traj
