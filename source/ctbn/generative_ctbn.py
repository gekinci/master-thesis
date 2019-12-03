import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ctbn.utils import *
import constants
import logging
import time


class GenerativeCTBN:
    def __init__(self, cfg, save_folder='../data/', save_time=time.time()):
        self.FOLDER = save_folder
        self.TIME = save_time
        logging.debug('initializing the CTBN object...')

        self.graph_dict = cfg[constants.GRAPH_STRUCT]
        self.t_max = cfg[constants.T_MAX]
        self.n_values = cfg[constants.N_VALUES]

        self.node_list = list(self.graph_dict.keys())
        self.num_nodes = len(self.node_list)
        self.net_struct = [[par, node] for node in self.node_list for par in self.graph_dict[node] if
                           len(self.graph_dict[node]) > 0]
        self.Q = self.initialize_generative_ctbn()
        for key in self.Q.keys():
            logging.debug(f'Q[{key}] = {self.Q[key]}')

        logging.debug('CTBN object initialized!')

    def initialize_generative_ctbn(self):
        self.create_and_save_graph()
        Q_dict = self.generate_conditional_intensity_matrices()
        return Q_dict

    def generate_conditional_intensity_matrices(self):
        Q = dict()

        # Randomly generated conditional intensity matrices
        for node in self.node_list:
            if len(self.graph_dict[node]) == 0:
                Q[node] = random_q_matrix(self.n_values)
            else:
                parent_list = self.graph_dict[node]
                n_parents = len(parent_list)
                parent_cart_prod = cartesian_products(n_parents)
                Q[node] = {}

                for prod in parent_cart_prod:
                    Q[node][prod] = random_q_matrix(self.n_values)
        return Q

    def create_and_save_graph(self):
        G = nx.DiGraph()
        G.add_edges_from(self.net_struct)
        pos = graphviz_layout(G, prog='dot')
        nx.draw_networkx(G, pos=pos, arrows=True)
        plt.savefig(self.FOLDER + f'{self.TIME}_graph.png')

    def get_parent_values(self, node, prev_step):
        parent_list = self.graph_dict[node]
        return ''.join(map(str, prev_step[parent_list].values[-1].astype(int)))

    def get_current_Q_for_node(self, node, prev_step):
        parent_list = self.graph_dict[node]
        if len(parent_list) == 0:
            node_Q = self.Q[node]
        else:
            par_values = self.get_parent_values(node, prev_step)
            node_Q = self.Q[node][par_values]
        return node_Q

    def draw_time(self, prev_step):
        random_draws = []
        for node in self.node_list:
            current_val = int(prev_step[node].values[-1])
            current_Q = self.get_current_Q_for_node(node, prev_step)
            q = current_Q[current_val][1 - current_val]
            random_draws += [np.random.exponential(1 / q)]

        return np.min(random_draws)

    def draw_variable(self, df_traj):
        q_list = []
        for node in self.node_list:
            current_val = int(df_traj[node].values[-1])
            current_Q = self.get_current_Q_for_node(node, df_traj)
            q_list += [current_Q[current_val][1 - current_val]]
        q_list /= np.sum(q_list)

        return np.random.choice(self.node_list, p=q_list)

    def draw_next_value(self):
        # TODO non-binary variables
        return

    def do_step(self, prev_step, t):
        tao = self.draw_time(prev_step)
        var = self.draw_variable(prev_step)
        logging.debug(f'Change is gonna happen in {tao} sec')
        logging.debug(f'{var} is gonna change from {int(prev_step[var])} to {int(1 - prev_step[var])}')

        new_step = prev_step.copy()
        # Adding new state change to the trajectories
        new_step.loc[:, constants.TIME] = t + tao
        new_step.loc[:, var] = int(1 - new_step[var])

        t += tao
        return new_step, t

    def sample_trajectory(self):
        t = 0

        # Randomly initializing first states
        initial_states = {var: [np.random.randint(0, 2)] for var in self.node_list}
        initial_states[constants.TIME] = 0
        df_traj = pd.DataFrame.from_dict(initial_states)
        prev_step = pd.DataFrame(df_traj[-1:].values, columns=df_traj.columns)

        while t < self.t_max:
            new_step, t = self.do_step(prev_step, t)
            df_traj = df_traj.append(new_step, ignore_index=True)
            prev_step = new_step.copy()

        return df_traj

    def sample_and_save_trajectories(self, n_traj, file_name='hist'):
        # Generates predefined number of trajectories for the same graph with the same Q!
        df_traj_hist = pd.DataFrame()

        for exp in range(n_traj):
            df_traj = self.sample_trajectory()

            df_traj.loc[:, constants.TRAJ_ID] = exp
            df_traj_hist = df_traj_hist.append(df_traj)

        # Save all the sampled trajectories
        df_traj_hist.to_csv(self.FOLDER + f'{self.TIME}_{file_name}_traj.csv')

        return df_traj_hist
