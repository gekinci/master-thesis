import constants
import pandas as pd
from ctbn.utils import *
from ctbn.generative_ctbn import GenerativeCTBN
import matplotlib.pyplot as plt
import time
import logging


# define policy: which Q to go with on which state
# transition probabilities
# observation probabilities
# def update_belief_state
# stochastic observation
# def run: observe --> update belief --> predict state --> select action --> select Q --> gillespie
# get transition prob from ctbn amalgamation
# TODO consider action for transition and observation function ( T, Z )

class POMDPSimulation:
    def __init__(self, cfg, save_folder='../data/', save_time=time.time()):
        self.FOLDER = save_folder
        self.TIME = save_time

        logging.debug('initializing the POMDP object...')

        self.ctbn = GenerativeCTBN(cfg, save_folder=save_folder, save_time=save_time)
        self.parent_list = ['1', '2']
        self.n_parents = len(self.parent_list)
        self.t_max = cfg[constants.T_MAX]

        self.A = ['A' + str(k) for k in range(1, cfg[constants.N_ACTIONS] + 1)]  # action space
        self.S = cartesian_products(self.n_parents)  # state space
        self.O = [0, 1]  # observation space

        self.q_list = ['Q' + str(k) for k in range(1, cfg[constants.N_Q] + 1)]
        self.Q = {k: random_q_matrix(cfg[constants.N_VALUES]) for k in self.q_list}

        self.policy = pd.DataFrame(index=self.S, columns=self.A,
                                   data=np.random.rand(len(self.S), len(self.A)))
        self.policy = self.policy.div(self.policy.sum(axis=1), axis=0)  # normalize to have probabilities

        self.behaviour = pd.DataFrame(index=self.A, columns=self.q_list,
                                      data=np.random.rand(len(self.A), len(self.q_list)))
        self.behaviour = self.behaviour.div(self.behaviour.sum(axis=1), axis=0)  # normalize to have probabilities
        # np.random.choice(self.node_list, p=q_list)

        # T(a, s, s')
        # self.T = np.repeat(np.random.rand(len(self.S), len(self.S))[np.newaxis, :, :], 3, axis=0)  #random
        self.T = np.repeat(np.repeat(np.tile(1 / len(self.S), len(self.S))[:, np.newaxis], 4, axis=1)
                           [np.newaxis, :, :], 3, axis=0)  # equiprobable
        # self.T = amalgamation_independent_cim(self.ctbn.Q['1'], self.ctbn.Q['2'])  # amalgamation from ctbn

        [np.fill_diagonal(self.T[i, ::-1, :], 0) for i in range(self.T.shape[0])]  # vars cannot change at the same time
        self.T /= self.T.sum(axis=2)[:, :, np.newaxis]

        # Z(a, s', o)
        self.Z = np.repeat(np.array([[.9, .1], [.1, .9], [.1, .9], [.9, .1]])[np.newaxis, :, :], 3, axis=0)

        self.belief_state = np.tile(1 / len(self.S), len(self.S))

        logging.debug(f'Action space: {self.A}')
        logging.debug(f'State space: {self.S}')
        logging.debug(f'Observation space: {self.O}')
        # logging.debug(f'Conditional intensity matrix (Q) space for node 3: \n')
        # for key in self.Q.keys():
        #     logging.debug(f'{self.Q[key]}')
        logging.debug(f'Stochastic policy Pr(a|s): \n{self.policy}\n')
        logging.debug(f'Stochastic behavior Pr(Q|a): \n{self.behavior}\n')
        logging.debug(f'State-transition function: \n{self.T}\n')
        logging.debug(f'Observation function: \n{self.Z}\n')
        logging.debug(f'Initial belief state: {self.belief_state}')
        logging.debug('POMDP simulator object initialized!\n')

    @staticmethod
    def segmentation_function(df):  # Deterministic observation
        return df.sum() % 2

    def get_observation(self, df):  # Stochastic observation
        state = ''.join(map(str, df[self.parent_list].values.astype(int)))  # parent values
        state_index = self.S.index(state)
        obs = np.random.choice(self.O, p=self.Z[0, state_index])
        logging.debug(f'Observation: {obs}')
        return obs

    def update_belief_state(self, obs):
        logging.debug(f"Old belief state= {self.belief_state}")
        logging.debug(f"Pr(o|s',a) = \n{self.Z[0][:, obs]}")
        logging.debug(f"sum over s(Pr(s'|s,a)*b(s) = \n{self.belief_state @ self.T[0]}")
        self.belief_state = self.Z[0][:, obs] * (self.belief_state @ self.T[0])
        self.belief_state = self.belief_state / self.belief_state.sum()
        logging.debug(f"New belief state: {self.belief_state}")

    def get_state(self):  # Deterministic state prediction, taking the maximum likely
        return self.S[self.belief_state.argmax()]

    def get_action(self, state, ax):  # Stochastic action
        ax[-2].clear()
        ax[-2].bar(self.A, self.policy.loc[state])
        ax[-2].set_ylabel('Pr(a|s)')
        # plt.pause(0.0001)
        logging.debug(f"P(a|s) = \n{self.policy.loc[state]}")
        return np.random.choice(self.A, p=self.policy.loc[state].values)

    def get_Q(self, action, ax):
        ax[-1].clear()
        ax[-1].bar(self.q_list, self.behaviour.loc[action])
        ax[-1].set_ylabel('Pr(Q|a)')
        # plt.pause(0.0001)
        Q_tag = np.random.choice(self.q_list, p=self.behaviour.loc[action].values)
        logging.debug(f"P(Q|a) = \n{self.behaviour.loc[action]}")
        return self.Q[Q_tag]

    def do_step(self, prev_step, t, prev_obs, ax):
        obs = prev_step[constants.OBS].values[0]
        if prev_obs is None or prev_obs != obs:  # if it is the very first step or obs has changed
            logging.debug(f'Prev obs was {prev_obs}, now observed {obs}!)\n'
                          'New observation! Updating the belief state! ...')
            self.update_belief_state(obs)
            ax[-3].clear()
            ax[-3].bar(self.S, self.belief_state)
            ax[-3].set_ylabel('belief_state')
            # plt.pause(0.0001)

            state_pred = self.get_state()
            logging.debug(f'Deterministic prediction of state: {state_pred}')

            action = self.get_action(state_pred, ax)
            logging.debug(f'Stochasticly selected decision: {action}')

            Q = self.get_Q(action, ax)
            self.ctbn.Q['3'] = Q
            logging.debug(f'Stochasticly selected Q: {Q}')

        logging.debug('Taking the new step at ctbn...')
        new_step, t = self.ctbn.do_step(prev_step, t)  # updates only for the nodes of ctbn!

        if (prev_step[self.parent_list].values == new_step[self.parent_list].values).all():
            new_step.loc[:, constants.OBS] = prev_step.loc[0, constants.OBS]
            logging.debug('Parents didnt change their states. No new observation! Continue with the Gillespie!')
        else:
            logging.debug('Parents changed their states...')
            new_step.loc[:, constants.OBS] = new_step[self.parent_list].apply(self.get_observation, axis=1)

        return new_step, t, obs

    def sample_trajectory(self):
        t = 0

        # Randomly initializing first states
        initial_states = {var: [np.random.randint(0, 2)] for var in self.ctbn.node_list}
        initial_states[constants.TIME] = 0
        logging.debug(f'Initial states of the nodes: {initial_states}')

        df_traj = pd.DataFrame.from_dict(initial_states)
        # add first observation
        df_traj.loc[:, constants.OBS] = df_traj[self.parent_list].apply(self.get_observation, axis=1)
        prev_step = pd.DataFrame(df_traj[-1:].values, columns=df_traj.columns)
        prev_obs = None

        plt.ion()
        fig, ax = plt.subplots(self.ctbn.num_nodes + 1 + 3)

        while t < self.t_max:
            logging.debug(f'Time is now {t} sec...')
            logging.debug(f'Next step in POMDP...')
            new_step, t, prev_obs = self.do_step(prev_step, t, prev_obs, ax)
            df_traj = df_traj.append(new_step, ignore_index=True)
            prev_step = new_step.copy()

            for i, var in enumerate(['1', '2', constants.OBS, '3']):
                ax[i].step(df_traj[constants.TIME], df_traj[var], 'b')
                ax[i].set_ylim([-.5, 1.5])
                ax[i].set_ylabel(var)
                ax[i].set_xlabel('time')
                plt.pause(0.0001)
            ax[1].xaxis.tick_top()
            time.sleep(2)
        plt.show(block=False)
        fig.savefig(self.FOLDER + f'{self.TIME}_traj.png')
        df_traj.to_csv(self.FOLDER + f'{self.TIME}_traj.csv')

        return df_traj
