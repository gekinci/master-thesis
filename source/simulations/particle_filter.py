from utils.constants import *
from utils.helpers import *
from utils.visualization import *
from simulations.ctbn import CTBNSimulation
from ctbn.parameter_learning import *

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
        self.particles = self.initialize_particles()
        self.weights = np.tile(1/n_particle, n_particle)

    def reset(self):
        self.particles = self.initialize_particles()
        self.weights = np.tile(1 / self.n_particle, self.n_particle)

    def initialize_particles(self):
        particle_list = []
        for i in range(self.n_particle):
            p_init = {**self.sampling_ctbn.initialize_nodes(), **{TIME: 0}}
            df_p = pd.DataFrame().append(p_init, ignore_index=True)
            df_p.loc[:, 'state'] = self.get_state(df_p.iloc[-1])
            particle_list += [df_p]

        return particle_list

    def get_state(self, row):
        state = ''.join(map(str, row[self.sampling_ctbn.node_list].values.astype(int)))
        return state

    def reestimate_Q_particle(self, p):
        Q_dict = {}
        for node in self.sampling_ctbn.node_list:
            alpha = self.Q_gamma_params[node]['alpha']
            beta = self.Q_gamma_params[node]['beta']
            T = get_time_of_stay_in_state(p, node=node)
            M = get_number_of_transitions(p, node=node)
            Q_dict[node] = (np.array([[-1, 1], [1, -1]])*((alpha+M)/(beta+T))).T
        return Q_dict

    def propagate_particle(self, p, t):
        while p.iloc[-1][TIME] <= t:
            new_step = self.sampling_ctbn.do_step(pd.DataFrame(p[-1:].values, columns=p.columns))
            new_step.loc[:, 'state'] = self.get_state(new_step.iloc[-1])
            p = p.append(new_step, ignore_index=True)
        p = p.iloc[:-1]
        return p

    def update_weights(self, obs, t):
        obs_ind = self.O.index(obs)
        for i, p in enumerate(self.particles):
            Q_ = self.reestimate_Q_particle(p)
            self.sampling_ctbn.Q = Q_
            p_prop = self.propagate_particle(p, t)  # propagating the particle in the system
            self.particles[i] = p_prop  # that is the new particle now
            state_ind = self.S.index(p_prop.iloc[-1]['state'])
            self.weights[i] = self.obs_llh[state_ind][obs_ind]
        self.weights /= self.weights.sum()

    def resample_particles(self):
        resampled_ind = np.random.choice(self.n_particle, size=self.n_particle, p=self.weights)
        self.particles = [self.particles[i] for i in resampled_ind]

    def update(self, obs, t):
        self.update_weights(obs, t)
        if np.isnan(np.sum(self.weights)):
            x = 5
        self.resample_particles()
        tmp = [p.iloc[-1]['state'] for p in self.particles]
        belief = [tmp.count(s) for s in self.S]
        belief /= np.sum(belief)
        return belief
