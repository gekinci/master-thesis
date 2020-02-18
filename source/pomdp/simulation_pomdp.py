from utils.constants import *
from utils.helper import *
from ctbn.generative_ctbn import GenerativeCTBN

import matplotlib.pyplot as plt
import time
import logging
import scipy


class POMDPSimulation:
    def __init__(self, cfg, save_folder='../data/', save_time=time.time()):
        self.FOLDER = save_folder + str(int(save_time))

        logging.debug('initializing the POMDP object...')

        self.ctbn = GenerativeCTBN(cfg, save_folder=self.FOLDER, save_time=save_time)
        self.parent_list = ['1', '2']
        self.n_parents = len(self.parent_list)
        self.t_max = cfg[T_MAX]
        self.states = cfg[STATES]
        self.n_states = len(cfg[STATES])
        self.HOW_TO_PRED_STATE = cfg[HOW_TO_PRED_STATE]
        self.time_grain = cfg[TIME_GRAIN]

        self.S = cartesian_products(self.n_parents, self.n_states)  # state space
        self.O = [0, 1]  # observation space

        self.q_list = ['Q' + str(k) for k in range(1, cfg[N_Q] + 1)]
        self.Q = {k: random_q_matrix(self.n_states) for k in self.q_list}

        # policy is set to be deterministic!!!
        self.policy = pd.DataFrame(index=self.S, columns=self.q_list,
                                   data=np.eye(len(self.S), len(self.q_list)))
        self.policy = self.policy.div(self.policy.sum(axis=1), axis=0)  # normalize to have probabilities

        # T(a, s, s')
        self.T = get_amalgamated_trans_matrix(self.ctbn.Q['1'], self.ctbn.Q['2'])  # amalgamation from ctbn

        # Z(a, s', o)
        self.Z = np.array([[1, .0], [.0, 1], [.0, 1], [1, .0]])

        self.belief_state = np.tile(1 / len(self.S), len(self.S))
        time_grid = np.around(np.arange(0, self.t_max + self.time_grain, self.time_grain), decimals=3)
        self.df_b = pd.DataFrame(columns=self.S+['t_delta'], index=time_grid)

        logging.debug(f'State space: {self.S}')
        logging.debug(f'Observation space: {self.O}')
        logging.debug(f'Stochastic policy Pr(a|s): \n{self.policy}\n')
        logging.debug(f'State-transition function: \n{self.T}\n')
        logging.debug(f'Observation function: \n{self.Z}\n')
        logging.debug(f'Initial belief state: {self.belief_state}')
        logging.debug('POMDP simulator object initialized!\n')

    @staticmethod
    def segmentation_function(df):  # Deterministic observation
        return df.sum() % 2

    def app_func(self, row):
        row[self.S] = row[self.S].values @ scipy.linalg.expm(self.T * row['t_delta'])
        return row

    def get_observation(self, df):  # Stochastic observation
        state = ''.join(map(str, df[self.parent_list].values.astype(int)))  # parent values
        state_index = self.S.index(state)
        obs = np.random.choice(self.O, p=self.Z[state_index])
        logging.debug(f'Observation: {obs}')
        return obs

    def update_belief_state(self, obs, t):
        # logging.debug(f"Old belief state= {self.belief_state}")
        # logging.debug(f"Pr(o|s',a) = \n{self.Z[:, obs]}")
        # logging.debug(f"sum over s(Pr(s'|s,a)*b(s) = \n{self.belief_state @ self.T}")
        self.belief_state = self.Z[:, obs] * (self.belief_state @ scipy.linalg.expm(self.T*t))
        self.belief_state = self.belief_state / self.belief_state.sum()
        self.df_b.loc[t, self.S] = self.belief_state
        self.df_b.loc[t, 't_delta'] = 0
        # logging.debug(f"New belief state: {self.belief_state}")

    def continuous_belief_state(self):
        self.df_b.t_delta = self.df_b.t_delta.ffill() + self.df_b.groupby(
            self.df_b.t_delta.notnull().cumsum()).cumcount() * self.time_grain
        self.df_b.ffill(inplace=True)
        self.df_b.apply(self.app_func, axis=1)

    def get_state_pred_vect(self):
        # if parameter is 'maximum_likely', returns 1 for the state with the highest belief and others zero
        # if it is 'expected', returns belief
        pred_vect = np.zeros(len(self.S))
        if self.HOW_TO_PRED_STATE == 'maximum_likely':
            pred_vect[self.belief_state.argmax()] = 1
        elif self.HOW_TO_PRED_STATE == 'expectation':
            pred_vect = self.belief_state
        return pred_vect

    def get_Q(self, state_pred, ax):
        # ax[-1].clear()
        # ax[-1].bar(self.q_list, self.behavior.loc[action])
        # ax[-1].set_ylabel('Pr(Q|b)')
        Q = (np.array(list(self.Q.values()))*np.expand_dims(np.expand_dims(state_pred, axis=1), axis=1)).sum(axis=0)
        logging.debug(f"Q3 = \n{Q}")
        return Q

    def do_step(self, prev_step, t, prev_obs, ax):
        obs = prev_step[OBS].values[0]
        if prev_obs is None or prev_obs != obs:  # if it is the very first step or obs has changed
            logging.debug(f'Prev obs was {prev_obs}, now observed {obs}!)\n'
                          'New observation! Updating the belief state! ...')
            self.update_belief_state(obs, t)
            # ax[-3].clear()
            # ax[-3].bar(self.S, self.belief_state)
            # ax[-3].set_ylabel('b')

            state_pred_vect = self.get_state_pred_vect()
            logging.debug(f'Deterministic prediction of state: {state_pred_vect}')

            Q = self.get_Q(state_pred_vect, ax)
            self.ctbn.Q['3'] = Q
            logging.debug(f'Stochasticly selected Q: {Q}')

        logging.debug('Taking the new step at ctbn...')
        new_step = self.ctbn.do_step(prev_step)  # updates only for the nodes of ctbn!
        t = new_step[TIME].values[0].round(3)

        if (prev_step[self.parent_list].values == new_step[self.parent_list].values).all():
            new_step.loc[:, OBS] = prev_step.loc[0, OBS]
            logging.debug('Parents didnt change their states. No new observation! Continue with the Gillespie!')
        else:
            logging.debug('Parents changed their states...')
            new_step.loc[:, OBS] = new_step[self.parent_list].apply(self.get_observation, axis=1)

        return new_step, t, obs

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

        plt.ion()
        fig, ax = plt.subplots(self.ctbn.num_nodes + 2)

        while t < self.t_max:
            logging.debug(f'Time is now {t} sec...')
            logging.debug(f'Next step in POMDP...')
            new_step, t, prev_obs = self.do_step(prev_step, t, prev_obs, ax)
            df_traj = df_traj.append(new_step, ignore_index=True)
            prev_step = new_step.copy()

            for i, var in enumerate(['1', '2', OBS]):  # , '3']):
                ax[i].step(df_traj[TIME], df_traj[var], 'b')
                ax[i].set_ylim([-.5, 1.5])
                ax[i].set_ylabel(var)
                if i != 0:
                    ax[i].set_xticks([])
                plt.pause(0.0001)
            ax[0].xaxis.tick_top()
            ax[0].set_xlabel('time')
            ax[0].xaxis.set_label_position('top')
            time.sleep(1)
            fig.savefig(self.FOLDER + f'{t.round(3)}_step.png')
        plt.show(block=False)
        # fig.savefig(self.FOLDER + f'traj_plot.png')
        df_traj.to_csv(self.FOLDER + f'traj.csv')

        self.continuous_belief_state()
        ax[-1].clear()
        self.df_b[self.S].plot(ax=ax[-1])
        # plt.pause(0.0001)
        # time.sleep(1)
        fig.savefig(self.FOLDER + f'final.png')
        return df_traj
