from utils.constants import *
from utils.helper import *

import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt
import logging
import time
import os


class GenerativeCTBN:
    def __init__(self, cfg, save_folder='../data/', save_time=time.time()):
        self.FOLDER = save_folder + str(int(save_time)) + '/'
        os.makedirs(self.FOLDER, exist_ok=True)
        logging.debug('initializing the CTBN object...')

        self.graph_dict = cfg[GRAPH_STRUCT]
        self.t_max = cfg[T_MAX]
        self.states = cfg[STATES]
        self.n_states = len(cfg[STATES])
        self.initial_probs = cfg[INITIAL_PROB]

        self.node_list = list(self.graph_dict.keys())
        self.num_nodes = len(self.node_list)
        self.net_struct = [[par, node] for node in self.node_list for par in self.graph_dict[node] if
                           len(self.graph_dict[node]) > 0]

        self.initial_states = self.initialize_nodes()

        self.Q = self.initialize_generative_ctbn(cfg)

        logging.debug(f'Q = {self.Q}')
        logging.debug('CTBN object initialized!')

    def initialize_generative_ctbn(self, cfg):
        # self.create_and_save_graph()
        if Q_DICT in cfg.keys():
            Q_dict = cfg['Q_dict']
        else:
            Q_dict = self.generate_conditional_intensity_matrices()
        return Q_dict

    def initialize_nodes(self):
        return {var: np.random.choice(self.states, p=self.initial_probs) for var in self.node_list}

    def generate_conditional_intensity_matrices(self):
        Q = dict()

        # Randomly generated conditional intensity matrices
        for node in self.node_list:
            if len(self.graph_dict[node]) == 0:
                Q[node] = random_q_matrix(self.n_states)
            else:
                parent_list = self.graph_dict[node]
                n_parents = len(parent_list)
                parent_cart_prod = cartesian_products(n_parents, n_states=self.n_states)
                Q[node] = {}

                for prod in parent_cart_prod:
                    Q[node][prod] = random_q_matrix(self.n_states)
        return Q

    def create_and_save_graph(self):
        G = nx.DiGraph()
        G.add_edges_from(self.net_struct)
        pos = graphviz_layout(G, prog='dot')
        nx.draw_networkx(G, pos=pos, arrows=True)
        plt.savefig(self.FOLDER + f'graph.png')

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

    def do_step(self, prev_step):
        t = prev_step[TIME].values[0]
        tao = self.draw_time(prev_step)
        var = self.draw_variable(prev_step)
        logging.debug(f'Change is gonna happen in {tao} sec')
        logging.debug(f'{var} is gonna change from {int(prev_step[var])} to {int(1 - prev_step[var])}')

        new_step = prev_step.copy()
        # Adding new state change to the trajectories
        new_step.loc[:, TIME] = t + tao
        new_step.loc[:, var] = int(1 - new_step[var])

        return new_step

    def sample_trajectory(self):
        t = 0

        # Randomly initializing first states
        initial_states = self.initial_states
        initial_states[TIME] = 0
        df_traj = pd.DataFrame().append(initial_states, ignore_index=True)
        prev_step = pd.DataFrame(df_traj[-1:].values, columns=df_traj.columns)

        while t < self.t_max:
            new_step= self.do_step(prev_step)
            t = new_step[TIME].values[0]
            df_traj = df_traj.append(new_step, ignore_index=True)
            prev_step = new_step.copy()

        return df_traj

    def sample_and_save_trajectories(self, n_traj, file_name='hist'):
        # Generates predefined number of trajectories for the same graph with the same Q!
        df_traj_hist = pd.DataFrame()

        for exp in range(n_traj):
            df_traj = self.sample_trajectory()

            df_traj.loc[:, TRAJ_ID] = exp
            df_traj_hist = df_traj_hist.append(df_traj)

        # Save all the sampled trajectories
        df_traj_hist.to_csv(self.FOLDER + f'{file_name}_traj.csv')

        return df_traj_hist
