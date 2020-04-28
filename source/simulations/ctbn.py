from utils.constants import *
from utils.helpers import *

import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt
import logging


class CTBNSimulation:
    def __init__(self, cfg, save_folder='../_data/generative_ctbn/'):
        self.FOLDER = save_folder
        self.graph_dict = cfg[GRAPH_STRUCT]
        self.t_max = cfg[T_MAX] if cfg[T_MAX] else 20
        self.states = cfg[STATES] if cfg[STATES] else [0, 1]
        self.n_states = len(self.states)
        self.initial_probs = cfg[INITIAL_PROB] if cfg[INITIAL_PROB] else np.ones(len(self.states)) / len(self.states)
        self.node_list = list(self.graph_dict.keys())
        self.num_nodes = len(self.node_list)
        # self.draw_and_save_graph()
        # self.initial_states = self.initialize_nodes()
        self.Q = self.initialize_intensity_matrices(cfg)

    def draw_and_save_graph(self):
        net_struct = [[par, node] for node in self.node_list for par in self.graph_dict[node] if
                      len(self.graph_dict[node]) > 0]
        G = nx.DiGraph()
        G.add_edges_from(net_struct)
        pos = graphviz_layout(G, prog='dot')
        nx.draw_networkx(G, pos=pos, arrows=True)
        plt.savefig(os.path.join(self.FOLDER, 'graph.png'))

    def initialize_intensity_matrices(self, cfg):
        if cfg[Q_DICT]:
            Q_dict = cfg[Q_DICT]
        elif cfg[GAMMA_PARAMS]:
            Q_dict = self.generate_conditional_intensity_matrices(param_dict=cfg[GAMMA_PARAMS])
        else:
            Q_dict = self.generate_conditional_intensity_matrices()
        return Q_dict

    def initialize_nodes(self):
        return {var: np.random.choice(self.states, p=self.initial_probs) for var in self.node_list}

    def generate_conditional_intensity_matrices(self, param_dict=None):
        Q = dict()

        # Randomly generated conditional intensity matrices
        for node in self.node_list:
            if len(self.graph_dict[node]) == 0:
                Q[node] = random_q_matrix(self.n_states, params=param_dict[node]) if param_dict else random_q_matrix(
                    self.n_states)
            else:
                parent_list = self.graph_dict[node]
                n_parents = len(parent_list)
                parent_cart_prod = cartesian_products(n_parents, states=self.states)
                Q[node] = {}

                for prod in parent_cart_prod:
                    Q[node][prod] = random_q_matrix(self.n_states,
                                                    params=param_dict[node]) if param_dict else random_q_matrix(
                        self.n_states)
        return Q

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
        q = 0
        for node in self.node_list:
            current_val = int(prev_step[node].values[-1])
            current_Q = self.get_current_Q_for_node(node, prev_step)
            q += abs(current_Q[current_val][current_val])

        tao = np.random.exponential(1 / q)

        return tao

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

        new_step = prev_step.copy()
        # Adding new state change to the trajectories
        new_step.loc[:, TIME] = t + tao
        new_step.loc[:, var] = int(1 - new_step[var])

        return new_step

    def sample_trajectory(self):
        t = 0

        initial_states = {**self.initialize_nodes(), **{TIME: 0}}
        df_traj = pd.DataFrame().append(initial_states, ignore_index=True)
        prev_step = pd.DataFrame(df_traj[-1:].values, columns=df_traj.columns)

        while t < self.t_max:
            new_step = self.do_step(prev_step)
            t = new_step[TIME].values[0]
            if t > self.t_max:
                prev_step.loc[:, TIME] = self.t_max
                df_traj = df_traj.append(prev_step, ignore_index=True)
                break
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
        df_traj_hist.to_csv(os.path.join(self.FOLDER, f'{file_name}_traj.csv'))

        return df_traj_hist
