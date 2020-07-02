from utils.constants import *
from utils.helpers import *
from utils.visualization import *
from simulations.ctbn import CTBNSimulation
from ctbn.parameter_learning import *

from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import logging
import scipy


class ParticleFilterUpdate:
    def __init__(self, cfg, Q_params, n_particle, obs_llh, S, O):
        self.n_particle = n_particle
        self.obs_llh = obs_llh
        self.S = S
        self.O = O
        self.prev_obs = None

        self.Q_gamma_params = Q_params
        self.sampling_ctbn = CTBNSimulation(cfg)
        self.reset()

    def reset(self):
        self.sampling_ctbn.Q = {node: (np.array([[-1, 1], [1, -1]]) * (
                np.array(self.Q_gamma_params[node]['alpha']) / np.array(self.Q_gamma_params[node]['beta']))).T for
                                node in self.sampling_ctbn.node_list}
        self.particles = self.initialize_particles()
        self.weights = np.tile(1 / self.n_particle, self.n_particle)
        self.T = {key: np.zeros((self.n_particle, 2)) for key in parent_list_}
        self.M = {key: np.zeros((self.n_particle, 2)) for key in parent_list_}
        self.df_belief = pd.DataFrame(columns=self.S + [T_DELTA])

    def reset_obs_model(self, new_model):
        self.obs_llh = new_model

    def initialize_particles(self):
        def init_p(s):
            init_states = s.sampling_ctbn.initialize_nodes()
            state = ''.join(map(str, init_states.values()))
            p_init = {**init_states, **{'state': state}, **{TIME: 0}}
            df_p = pd.DataFrame().append(p_init, ignore_index=True)
            return df_p

        particle_list = []
        particle_list += [init_p(self) for i in range(self.n_particle)]
        return particle_list

    def get_state(self, row):
        state = ''.join(map(str, row[self.sampling_ctbn.node_list].values.astype(int)))
        return state

    def reestimate_Q(self, i, p):
        Q_dict = {}
        for node in self.sampling_ctbn.node_list:
            alpha = self.Q_gamma_params[node]['alpha']
            beta = self.Q_gamma_params[node]['beta']
            self.T[node][i] += get_time_of_stay_in_state(p, node=node)
            self.M[node][i] += get_number_of_transitions(p, node=node)
            Q_dict[node] = (np.array([[-1, 1], [1, -1]]) * ((np.array(alpha) + np.sum(self.M[node], axis=0)) / (
                    np.array(beta) + np.sum(self.T[node], axis=0)))).T
        return Q_dict

    def propagate_particles(self, t):
        new_p = []
        for i, p in enumerate(self.particles):
            while p.iloc[-1][TIME] <= t:
                new_step = self.sampling_ctbn.do_step(pd.DataFrame(p[-1:].values, columns=p.columns))
                new_step.loc[:, 'state'] = self.get_state(new_step.iloc[-1])
                p = p.append(new_step, ignore_index=True)
            p = p.iloc[:-1]
            if p.iloc[-1][TIME] < t:
                p = p.append(p.iloc[-1], ignore_index=True)
                p.loc[p.index[-1], TIME] = t
            new_p += [p]
            self.sampling_ctbn.Q = self.reestimate_Q(i, p)
        return new_p

    def update_weights(self, obs, new_p):
        obs_ind = self.O.index(obs)
        state_ind = [self.S.index(p.iloc[-1]['state']) for p in new_p]
        new_w = np.array([self.obs_llh[ind][obs_ind] for ind in state_ind])
        if new_w.sum() == 0:
            new_w = np.tile(1 / self.n_particle, self.n_particle)
        else:
            new_w /= new_w.sum()
        return new_w

    def resample_particles(self):
        resampled_ind = np.random.choice(self.n_particle, size=self.n_particle, p=self.weights)
        self.particles = [self.particles[i] for i in resampled_ind]

    def get_belief_df(self):
        master_df = pd.DataFrame()
        master_df = master_df.join(
            [p.set_index(TIME)[['state']].rename(columns={'state': f'{i}'}) for i, p in enumerate(self.particles)],
            how='outer')
        master_df.ffill(inplace=True)

        new_b = master_df.apply(pd.value_counts, axis=1)
        for col in self.S:
            if col not in new_b.columns:
                new_b[col] = 0.
        new_b.fillna(0, inplace=True)
        new_b = new_b.div(new_b.sum(axis=1), axis=0)
        new_b.index = [to_decimal(i) for i in new_b.index]
        return new_b

    def append_event(self, t, trivial_t):
        self.df_belief.loc[t] = self.df_belief.loc[self.df_belief.index < t].iloc[-1]
        self.df_belief.sort_index(inplace=True)

    def update(self, obs, t):
        new_particles = self.propagate_particles(t)
        new_weights = self.update_weights(obs, new_particles) if obs is not None else np.zeros(self.n_particle)

        self.particles = new_particles
        self.weights = new_weights

        new_b = self.get_belief_df()
        self.df_belief = self.df_belief.combine_first(new_b)

        if obs is not None:
            self.resample_particles()
            self.particles = [p.tail(1).reset_index(drop=True) for p in self.particles]
            self.prev_obs = obs


class ExactUpdate:
    def __init__(self, cfg, T, obs_llh, S, O):
        self.obs_llh = obs_llh
        self.S = S
        self.O = O
        self.T = T

        self.time_increment = cfg[TIME_INCREMENT]
        self.t_max = cfg[T_MAX]
        self.init_belief_state = np.tile(1 / len(self.S), len(self.S))
        self.reset()

    def reset(self):
        time_grid = custom_decimal_range(0, self.t_max + .00000001, self.time_increment)
        self.df_belief = pd.DataFrame(columns=self.S + [T_DELTA], index=time_grid)
        self.df_belief.loc[0] = np.append(self.init_belief_state, 0)

    def reset_obs_model(self, new_model):
        self.obs_llh = new_model

    def append_event(self, t, t_prev):
        self.df_belief.loc[to_decimal(t)] = np.nan
        self.df_belief.sort_index(inplace=True)
        self.update_belief_cont(t=t_prev, t_next=t)

    def update_jump(self, obs, t):
        new_b = self.obs_llh[:, obs] * self.df_belief.loc[to_decimal(t), self.S]
        new_b /= new_b.sum()
        self.df_belief.loc[to_decimal(t)] = np.append(new_b, 0)

    def update_belief_cont(self, t, t_next):
        def helper(row):
            row[self.S] = row[self.S].values @ scipy.linalg.expm(self.T * row[T_DELTA])
            return row
        ind = (self.df_belief.index >= to_decimal(t)) & (self.df_belief.index <= to_decimal(t_next))
        tmp = self.df_belief.loc[ind].copy()
        if not tmp.empty:
            tmp.t_delta = (tmp.index - tmp.index[0]).values.astype(float)
            tmp.ffill(inplace=True)
            self.df_belief.loc[ind, :] = tmp.apply(helper, axis=1)
        else:
            pass

    def update(self, obs, t):
        if t == 0.0:
            self.update_jump(obs, t)
        elif obs is not None:
            t_prev = self.df_belief.dropna().index[-1]
            self.append_event(t, t_prev)
            self.update_jump(obs, t)
        else:
            t_prev = self.df_belief.dropna().index[-1]
            self.append_event(t, t_prev)
