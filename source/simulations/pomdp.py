from utils.constants import *
from utils.helpers import *
from utils.visualization import *
from simulations.ctbn import CTBNSimulation
from simulations.particle_filter import ParticleFilter

import matplotlib.pyplot as plt
# import time
import logging
import scipy


class POMDPSimulation:
    def __init__(self, cfg, save_folder='../_data/pomdp_simulation', import_data=None):

        self.cfg = cfg
        self.FOLDER = save_folder
        self.IMPORT = import_data
        self.policy_type = cfg[POLICY]
        self.belief_update = cfg[B_UPDATE]

        self.parent_ctbn = CTBNSimulation(cfg, save_folder=self.FOLDER)
        self.t_max = cfg[T_MAX] if cfg[T_MAX] else 20
        self.parent_list = cfg[PARENT_LIST] if cfg[PARENT_LIST] else parent_list_
        self.n_parents = len(self.parent_list)
        self.states = cfg[STATES] if cfg[STATES] else [0, 1]
        self.n_states = len(self.states)
        self.initial_probs = cfg[INITIAL_PROB] if cfg[INITIAL_PROB] else np.ones(len(self.states)) / len(self.states)

        self.S = cartesian_products(self.n_parents, states=self.states)
        self.O = cfg[OBS_SPACE]
        self.A = [str(i) for i in cfg[ACT_SPACE]]
        self.Qz = self.set_Q_agent()

        self.policy = self.generate_policy()

        self.T = get_amalgamated_trans_matrix(self.parent_ctbn.Q[self.parent_list[0]],
                                              self.parent_ctbn.Q[self.parent_list[1]])
        self.Z = np.array(cfg[OBS_MODEL])
        self.init_belief_state = np.tile(1 / len(self.S), len(self.S))
        self.initial_states = self.initialize_nodes()

        self.belief_particle_filter = ParticleFilter(cfg, cfg[N_PARTICLE], self.Z, self.S, self.O)

        self.time_grain = cfg[TIME_GRAIN]
        time_grid = custom_decimal_range(0, self.t_max + .00000001, self.time_grain)
        self.df_b = pd.DataFrame(columns=self.S + [T_DELTA], index=time_grid)
        self.df_b.loc[0] = np.append(self.init_belief_state, 0)
        self.df_Qz = pd.DataFrame(columns=self.S + [T_DELTA], index=time_grid)

    def reset(self):
        self.initial_states = self.initialize_nodes()
        time_grid = custom_decimal_range(0, self.t_max + .00000001, self.time_grain)
        self.df_b = pd.DataFrame(columns=self.S + [T_DELTA], index=time_grid)
        self.df_b.loc[0] = np.append(self.init_belief_state, 0)
        self.df_Qz = pd.DataFrame(columns=self.S + [T_DELTA], index=time_grid)

    def reset_obs_model(self, new):
        self.Z = new
        self.belief_particle_filter.reset_obs_model(new)

    def set_Q_agent(self):
        Q_agent = self.cfg[Q3] if self.cfg[Q3] else {k: random_q_matrix(self.n_states) for k in self.A}
        return Q_agent

    def initialize_nodes(self):
        return {**self.parent_ctbn.initialize_nodes(), **{agent_: np.random.choice(self.states, p=self.initial_probs)}}

    def generate_policy(self):
        if self.policy_type == 'function':
            return np.random.random(len(self.S))
        else:
            policy = generate_belief_grid(step=0.01, cols=self.S)
            for action in self.A:
                policy[str(action)] = np.random.random(len(policy))
            policy[self.A] = policy[self.A].div(policy[self.A].sum(axis=1), axis=0)
            if self.policy_type == 'deterministic':
                policy[self.A] = policy[self.A].round()
                policy.loc[policy['00'] >= 0.5, '0'] = 0
                policy.loc[policy['00'] >= 0.5, '1'] = 1
                policy.loc[policy['01'] >= 0.5, '0'] = 0
                policy.loc[policy['01'] >= 0.5, '1'] = 1
                policy.loc[policy['10'] >= 0.5, '0'] = 0
                policy.loc[policy['10'] >= 0.5, '1'] = 1
                policy.loc[policy['11'] >= 0.5, '1'] = 0
                policy.loc[policy['11'] >= 0.5, '0'] = 1
                return policy
            elif self.policy_type == 'stochastic':
                return policy

    def get_observation(self, df):  # Stochastic observation
        state = ''.join(map(str, df[self.parent_list].values.astype(int)))
        state_index = self.S.index(state)
        obs = np.random.choice(self.O, p=self.Z[state_index])
        return obs

    def append_event(self, t, event_b=np.nan, event_q=np.nan):
        self.df_b.loc[to_decimal(t)] = event_b
        self.df_b.sort_index(inplace=True)
        self.df_Qz.loc[to_decimal(t)] = event_q
        self.df_Qz.sort_index(inplace=True)

    def get_belief_exact(self, obs, t):
        new_b = self.Z[:, obs] * self.df_b.loc[to_decimal(t), self.S]
        new_b /= new_b.sum()
        return new_b

    def update_belief_jump(self, b, t):
        self.df_b.loc[to_decimal(t)] = np.append(b, 0)

    def update_belief_cont(self, t, t_next):
        def helper(row):
            row[self.S] = row[self.S].values @ scipy.linalg.expm(self.T * row[T_DELTA])
            return row

        ind = (self.df_b.index >= to_decimal(t)) & (self.df_b.index <= to_decimal(t_next))
        tmp = self.df_b.loc[ind].copy()
        if not tmp.empty:
            tmp.t_delta = (tmp.index - tmp.index[0]).values.astype(float)
            tmp.ffill(inplace=True)
            self.df_b.loc[ind, :] = tmp.apply(helper, axis=1)
        else:
            pass

    def get_prob_action(self, belief=None):
        if self.policy_type == 'function':
            p_0 = np.round(np.sum(belief * self.policy))
            return pd.Series([p_0, 1 - p_0], index=[self.A])
        else:
            return self.policy.loc[
                np.argmin(abs(self.policy[self.S].values - belief.astype(float)).sum(axis=1)), self.A]

    def get_Qz(self, p_act):
        Q = np.sum([np.array(self.Qz[i]) * p_act[int(i)] for i in self.A], axis=0)
        return Q

    def get_belief_traj(self, df_traj):
        self.reset()
        self.belief_particle_filter.reset()
        prev = df_traj.iloc[0]
        for i, row in df_traj.iterrows():
            if row[TIME] == 0. or (row[OBS] != prev[OBS]):
                if self.belief_update == 'particle_filter':
                    new_b, new_Q = self.belief_particle_filter.update(int(row[OBS]), row[TIME])
                    self.T = get_amalgamated_trans_matrix(new_Q[self.parent_list[0]], new_Q[self.parent_list[1]])
                else:
                    new_b = self.get_belief_exact(int(row[OBS]), row[TIME])
                self.update_belief_jump(new_b, row[TIME])

            prev = row.copy()
            if i == df_traj.index[-1]:
                break
            else:
                t = row[TIME]
                t_next = df_traj[TIME].values[i + 1]
                self.append_event(t_next)
                self.update_belief_cont(t, t_next)

    def update_cont_Q(self, t=0, t_next=None):
        def helper(v):
            p_a = self.get_prob_action(belief=v.values)
            Qz = self.get_Qz(p_a)
            v[self.S] = Qz.flatten()
            return v

        t_next = self.t_max if t_next is None else t_next

        ind = (self.df_b.index >= to_decimal(t)) & (self.df_b.index <= to_decimal(t_next))
        self.df_Qz.loc[ind, self.S] = self.df_b.loc[ind, self.S].apply(helper, axis=1)
        t_diff = np.diff(self.df_Qz.loc[self.df_b.index <= to_decimal(t_next)].index)
        self.df_Qz.loc[(self.df_b.index <= to_decimal(t_next)), T_DELTA] = np.append(t_diff,
                                                                                     to_decimal(self.time_grain) -
                                                                                     t_diff[-1]).astype(float)

    def sample_parent_trajectory(self):
        df_par_traj = self.parent_ctbn.sample_trajectory()
        df_par_traj.loc[:, OBS] = df_par_traj.apply(self.get_observation, axis=1)
        return df_par_traj

    def draw_time_contQ(self, state, t_start, t_end):
        if state == 0:
            col_transition = '01'
        elif state == 1:
            col_transition = '10'  # TODO hardcoded
        else:
            raise Exception(f'State of cell unknown: {state}')

        df_Q_ = self.df_Qz.truncate(before=to_decimal(t_start), after=to_decimal(t_end))

        F = 1 - np.exp(-(df_Q_[col_transition].multiply(df_Q_[T_DELTA], axis="index")).cumsum().astype(float))
        rnd = np.random.uniform()

        if len(F.index[F < rnd]):
            ind = F.index[F < rnd][-1]
            t_diff = np.log((1 - rnd) / (1 - F.loc[ind])) / -self.df_Qz.loc[:ind, col_transition].values[-1]
            t_next = np.float(ind) + t_diff
        else:
            t_next = np.nan

        if t_next <= t_start:
            t_next = np.nan

        return t_next

    def do_step(self, prev_step, NEW_OBS, t_last_agent_change):
        t = prev_step[TIME].values[0]

        if NEW_OBS:
            if self.belief_update == 'particle_filter':
                new_b, new_Q = self.belief_particle_filter.update(prev_step[OBS].values[0], t)
                self.T = get_amalgamated_trans_matrix(new_Q[self.parent_list[0]], new_Q[self.parent_list[1]])
            else:
                new_b = self.get_belief_exact(int(prev_step[OBS].values[0]), t)
            self.update_belief_jump(new_b, t)

        tmp = self.parent_ctbn.do_step(prev_step)
        t_par_change = tmp[TIME].values[0]

        self.append_event(t_par_change)

        self.update_belief_cont(t, t_par_change)
        self.update_cont_Q(t=t, t_next=t_par_change)

        t_agent_change = self.draw_time_contQ(prev_step.iloc[0][agent_], t, t_par_change)

        if t_agent_change < t_par_change:
            next_step = prev_step.copy()
            next_step.loc[:, TIME] = t_agent_change
            next_step.loc[:, agent_] = int(1 - next_step[agent_])  # TODO only for binary

            self.append_event(t_agent_change, event_q=np.append(
                self.df_Qz.loc[self.df_Qz.index < to_decimal(t_agent_change)][self.S].iloc[-1], 0))

            self.df_b.loc[self.df_b.index > to_decimal(t)] = np.nan
            self.df_Qz.loc[self.df_Qz.index > to_decimal(t)] = np.nan

            self.update_belief_cont(t, t_agent_change)
            self.update_cont_Q(t=t, t_next=t_agent_change)

            NEW_OBS = False
            t_last_agent_change = t_agent_change
        else:
            next_step = tmp.copy()
            next_step.loc[:, OBS] = next_step.apply(self.get_observation, axis=1)
            NEW_OBS = next_step.iloc[0][OBS] != prev_step.iloc[0][OBS]

        return next_step, NEW_OBS, t_last_agent_change

    def sample_trajectory(self):
        self.reset()
        self.belief_particle_filter.reset()
        t = 0
        initial_states = {**self.initial_states, **{TIME: t}}

        df_traj = pd.DataFrame().append(initial_states, ignore_index=True)
        df_traj.loc[:, OBS] = df_traj.apply(self.get_observation, axis=1)
        prev_step = pd.DataFrame(df_traj[-1:].values, columns=df_traj.columns)
        NEW_OBS = True
        t_last_agent_change = 0

        while t < self.t_max:
            new_step, NEW_OBS, t_last_agent_change = self.do_step(prev_step, NEW_OBS, t_last_agent_change)
            t = new_step[TIME].values[0]
            if t > self.t_max:
                prev_step.loc[:, TIME] = self.t_max
                df_traj = df_traj.append(prev_step, ignore_index=True)
                break
            df_traj = df_traj.append(new_step, ignore_index=True)

            prev_step = new_step.copy()

        return df_traj
