from utils.constants import *
from utils.helpers import *
from utils.visualization import *
from simulations.ctbn import CTBNSimulation
from ctbn.parameter_learning import *

from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import logging
import scipy


class ParticleFilter:
    def __init__(self, cfg, n_particle, obs_llh, S, O):
        self.n_particle = n_particle
        self.obs_llh = obs_llh
        self.S = S
        self.O = O

        self.Q_gamma_params = cfg[GAMMA_PARAMS]
        self.sampling_ctbn = CTBNSimulation(cfg)
        self.sampling_ctbn.Q = {node: (np.array([[-1, 1], [1, -1]]) * (
                np.array(self.Q_gamma_params[node]['alpha']) / np.array(self.Q_gamma_params[node]['beta']))).T for
                                node in self.sampling_ctbn.node_list}
        self.particles = self.initialize_particles()
        self.weights = np.tile(1 / self.n_particle, self.n_particle)

    def reset(self):
        self.sampling_ctbn.Q = {node: (np.array([[-1, 1], [1, -1]]) * (
                np.array(self.Q_gamma_params[node]['alpha']) / np.array(self.Q_gamma_params[node]['beta']))).T for
                                node in self.sampling_ctbn.node_list}
        self.particles = self.initialize_particles()
        self.weights = np.tile(1 / self.n_particle, self.n_particle)

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

    def reestimate_Q(self):
        Q_dict = {}
        for node in self.sampling_ctbn.node_list:
            alpha = self.Q_gamma_params[node]['alpha']
            beta = self.Q_gamma_params[node]['beta']
            T = [get_time_of_stay_in_state(p, node=node) for p in self.particles]
            M = [get_number_of_transitions(p, node=node) for p in self.particles]
            Q_dict[node] = (np.array([[-1, 1], [1, -1]]) * (
                        (np.array(alpha) + np.sum(M, axis=0)) / (np.array(beta) + np.sum(T, axis=0)))).T
        return Q_dict

    def propagate_particles(self, t):
        def prop_p(p, t):
            while p.iloc[-1][TIME] <= t:
                new_step = self.sampling_ctbn.do_step(pd.DataFrame(p[-1:].values, columns=p.columns))
                new_step.loc[:, 'state'] = self.get_state(new_step.iloc[-1])
                p = p.append(new_step, ignore_index=True)
            p = p.iloc[:-1]
            return p

        new_p = [prop_p(p, t) for p in self.particles]
        return new_p

    def update_weights(self, obs, new_p):
        obs_ind = self.O.index(obs)
        state_ind = [self.S.index(p.iloc[-1]['state']) for p in new_p]
        new_w = np.array([self.obs_llh[ind][obs_ind] for ind in state_ind])
        new_w /= new_w.sum()
        return new_w

    def resample_particles(self):
        resampled_ind = np.random.choice(self.n_particle, size=self.n_particle, p=self.weights)
        self.particles = [self.particles[i] for i in resampled_ind]

    def update(self, obs, t):
        UPDATE = True
        while UPDATE:
            new_particles = self.propagate_particles(t)
            new_weights = self.update_weights(obs, new_particles)
            if np.isnan(np.sum(new_weights)):
                print('ALL REJECTED!! NO PARTICLE LEFT!!')
                UPDATE = True
            else:
                UPDATE = False

        self.particles = new_particles
        self.weights = new_weights
        self.resample_particles()

        tmp = [p.iloc[-1]['state'] for p in self.particles]
        belief = [tmp.count(s) for s in self.S]
        belief /= np.sum(belief)

        Q_new = self.reestimate_Q()
        self.sampling_ctbn.Q = Q_new
        print(Q_new)
        return belief, Q_new
