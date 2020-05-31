from utils.constants import *
from utils.helpers import *
from utils.visualization import *
from simulations.ctbn import CTBNSimulation
from simulations.belief_updaters import *

import matplotlib.pyplot as plt
# import time
import logging
import scipy


class POMDPSimulation:
    def __init__(self, cfg):
        self.config = cfg
        self.POLICY_TYPE = cfg[POLICY_TYPE]
        self.BELIEF_UPDATE_METHOD = cfg[B_UPDATE_METHOD]

        self.parent_ctbn = CTBNSimulation(cfg)
        self.t_max = cfg[T_MAX] if cfg[T_MAX] else 20
        self.states = cfg[STATES] if cfg[STATES] else [0, 1]
        self.initial_probs = cfg[INITIAL_PROB] if cfg[INITIAL_PROB] else np.ones(len(self.states)) / len(self.states)

        self.S = cartesian_products(len(parent_list_), states=self.states)
        self.O = cfg[OBS_SPACE]
        self.A = [str(i) for i in cfg[ACT_SPACE]]
        self.Qset = self.set_Qset_agent()
        self.policy = self.generate_policy()
        self.PSI = np.array(cfg[OBS_MODEL])

        self.belief_updater = []
        self.df_Q_agent = []
        self.df_belief = []
        self.initial_states = {}

        self.reset()

    def reset(self):
        self.belief_updater = []
        self.df_Q_agent = []
        self.df_belief = []
        self.initial_states = self.initialize_nodes()
        for method in self.BELIEF_UPDATE_METHOD:
            if method == PART_FILT:
                self.belief_updater += [
                    ParticleFilterUpdate(self.config, self.config[GAMMA_PARAMS], self.config[N_PARTICLE], self.PSI,
                                         self.S, self.O)]
                self.df_Q_agent += [pd.DataFrame(columns=self.S + [T_DELTA])]
            if method == EXACT:
                T = get_amalgamated_trans_matrix(self.parent_ctbn.Q[parent_list_[0]],
                                                 self.parent_ctbn.Q[parent_list_[1]])
                self.belief_updater += [ExactUpdate(self.config, T, self.PSI, self.S, self.O)]
                self.df_Q_agent += [pd.DataFrame(columns=self.S + [T_DELTA])]
            if method == VANILLA_PART_FILT:
                Q_params = {}
                for key, val in self.config[GAMMA_PARAMS].items():
                    Q_params[key] = {}
                    for k, v in val.items():
                        Q_params[key][k] = [i * 1000000 for i in v]
                self.belief_updater += [
                    ParticleFilterUpdate(self.config, Q_params, self.config[N_PARTICLE], self.PSI, self.S, self.O)]
                self.df_Q_agent += [pd.DataFrame(columns=self.S + [T_DELTA])]
        print(self.BELIEF_UPDATE_METHOD)
        print(self.belief_updater)

    def reset_obs_model(self, new):
        self.PSI = new
        for upd in self.belief_updater:
            upd.reset_obs_model(new)

    def set_Qset_agent(self):
        Q_agent = self.config[Q3] if self.config[Q3] else {k: random_q_matrix(len(self.states)) for k in self.A}
        return Q_agent

    def initialize_nodes(self):
        return {**self.parent_ctbn.initialize_nodes(), **{agent_: np.random.choice(self.states, p=self.initial_probs)}}

    def generate_policy(self):
        if self.POLICY_TYPE == DET_FUNC:
            return np.random.random(len(self.S))
        else:
            policy = generate_belief_grid(step=0.01, cols=self.S)
            for action in self.A:
                policy[str(action)] = np.random.random(len(policy))
            policy[self.A] = policy[self.A].div(policy[self.A].sum(axis=1), axis=0)
            if self.POLICY_TYPE == DET_DF:
                policy[self.A] = policy[self.A].round()
                return policy
            elif self.POLICY_TYPE == STOC_DF:
                return policy

    def get_observation(self, df):  # Stochastic observation
        state = ''.join(map(str, df[parent_list_].values.astype(int)))
        state_index = self.S.index(state)
        obs = np.random.choice(self.O, p=self.PSI[state_index])
        return obs

    def get_prob_action(self, belief=None):
        if self.POLICY_TYPE == DET_FUNC:
            p_0 = np.round(np.sum(belief * self.policy))
            return pd.Series([p_0, 1 - p_0], index=[self.A])
        else:
            return self.policy.loc[
                np.argmin(abs(self.policy[self.S].values - belief.astype(float)).sum(axis=1)), self.A]

    def get_Qz(self, p_act):
        Q = np.sum([np.array(self.Qset[i]) * p_act[int(i)] for i in self.A], axis=0)
        return Q

    def update_cont_Q(self, t=None, t_next=None):
        def helper(v):
            p_a = self.get_prob_action(belief=v.values)
            Qz = self.get_Qz(p_a)
            v[self.S] = Qz.flatten()
            return v

        t = 0 if t is None else t
        t_next = self.t_max if t_next is None else t_next
        for i, upd in enumerate(self.belief_updater):
            t_ = t[i] if type(t) == list else t
            t_next_ = t_next[i] if type(t_next) == list else t_next
            df_b = upd.df_belief.copy()
            ind = (df_b.index >= to_decimal(t_)) & (df_b.index <= to_decimal(t_next_))
            df_Q = self.df_Q_agent[i].copy()
            if self.BELIEF_UPDATE_METHOD[i] == EXACT:
                df_Q = df_Q.combine_first(df_b.loc[ind, self.S].apply(helper, axis=1))
                t_diff = np.diff(df_Q.index)
                df_Q.loc[:, T_DELTA] = np.append(t_diff, to_decimal(self.config[TIME_INCREMENT]) - t_diff[-1]).astype(
                    float)
            elif self.BELIEF_UPDATE_METHOD[i] == PART_FILT or self.BELIEF_UPDATE_METHOD[i] == VANILLA_PART_FILT:
                df_Q = df_Q.combine_first(df_b.loc[ind, self.S].apply(helper, axis=1))
                df_Q.index = [to_decimal(i) for i in df_Q.index]
                df_Q.loc[:, T_DELTA] = np.append(np.diff(df_Q.index), 0).astype(float)
            self.df_Q_agent[i] = df_Q.copy()

    def update_belief_object(self, obs, t):
        for upd in self.belief_updater:
            upd.update(obs, t)

    def get_belief_for_inference(self, df_traj):
        self.reset()
        prev = df_traj.iloc[0]
        t_to_append = []
        for i, row in df_traj.iterrows():
            if row[TIME] == 0. or (row[OBS] != prev[OBS]) or ((i == len(df_traj) - 1) and len(t_to_append)):
                self.update_belief_object(int(row[OBS]), row[TIME])
                for t_ in t_to_append:
                    for upd in self.belief_updater:
                        upd.append_event(t_, t_ - self.config[TIME_INCREMENT])
                t_to_append = []
            else:
                t_to_append += [row[TIME]]
            prev = row.copy()
            if i == len(df_traj) - 1:
                self.update_belief_object(None, self.t_max)

        self.df_belief += [upd.df_belief for upd in self.belief_updater]

    def draw_time_contQ(self, state, t_start, t_end):
        change_list = []
        T = t_start
        df_b = self.belief_updater[0].df_belief.copy()
        while T < t_end:
            upper_bound = np.max([self.Qset[k][int(state)][int(1 - state)] for k in self.Qset.keys()])
            u = np.random.uniform()
            tao = -np.log(u) / upper_bound
            T += tao
            s = np.random.uniform()
            T_belief = df_b.loc[df_b.index < T, self.S].iloc[-1].values
            T_q = self.get_Qz(self.get_prob_action(belief=T_belief))[int(state)][int(1 - state)]
            if s <= T_q / upper_bound:
                change_list += [T]
                state = 1 - state
            else:
                continue
        return [x for x in change_list if x < t_end]

    def do_step(self, prev_step):
        t_now = prev_step[TIME].values[0]
        if t_now == 0.:
            self.update_belief_object(prev_step[OBS].values[0], t_now)

        parent_event = self.parent_ctbn.do_step(prev_step)
        new_obs = parent_event.apply(self.get_observation, axis=1).values[0]
        t_parent_event = parent_event[TIME].values[0]

        self.update_belief_object(new_obs, t_parent_event)
        self.update_cont_Q(t=t_now, t_next=t_parent_event)

        t_agent_change_list = self.draw_time_contQ(prev_step[agent_].values[-1], t_now, t_parent_event)

        agent_events = pd.DataFrame()
        if t_agent_change_list:
            new = prev_step.copy()
            for t_agent_change in t_agent_change_list:
                new.loc[:, TIME] = t_agent_change
                new.loc[:, agent_] = int(1 - new[agent_])
                agent_events = pd.concat([agent_events, new])
                for i, upd in enumerate(self.belief_updater):
                    upd.append_event(t_agent_change, self.df_Q_agent[i].loc[
                        self.df_Q_agent[i].index < to_decimal(t_agent_change)].index[-1])
                self.update_cont_Q(
                    t=[self.df_Q_agent[i].loc[self.df_Q_agent[i].index < to_decimal(t_agent_change)].index[-1] for i in
                       range(len(self.df_Q_agent))],
                    t_next=t_agent_change)
            parent_event[agent_] = agent_events.iloc[-1][agent_]
        next_step = pd.concat([agent_events, parent_event])
        next_step.loc[:, OBS] = next_step.apply(self.get_observation, axis=1)
        return next_step

    def sample_trajectory(self):
        self.reset()
        t = 0
        initial_states = {**self.initial_states, **{TIME: t}}

        df_traj = pd.DataFrame().append(initial_states, ignore_index=True)
        df_traj.loc[:, OBS] = df_traj.apply(self.get_observation, axis=1)
        prev_step = df_traj.copy()

        while t < self.t_max:
            new_step = self.do_step(prev_step)
            t = new_step[TIME].values[-1]
            df_traj = df_traj.append(new_step, ignore_index=True)
            if t > self.t_max:
                df_traj = df_traj[df_traj[TIME] < self.t_max]
                df_traj = df_traj.append(df_traj.iloc[-1], ignore_index=True)
                df_traj.loc[df_traj.index[-1], TIME] = self.t_max
                self.df_Q_agent = [df.truncate(after=to_decimal(self.t_max)) for df in self.df_Q_agent]
                self.df_belief = [upd.df_belief.truncate(after=to_decimal(self.t_max)) for upd in self.belief_updater]
                break

            prev_step = pd.DataFrame(new_step[-1:].values, columns=new_step.columns)

        return df_traj
