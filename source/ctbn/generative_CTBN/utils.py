import networkx as nx
import ctbn.config as cfg
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def generate_random_conditional_intensity_matrices():
    Q = dict()

    # Randomly generated conditional intensity matrices
    for node in cfg.node_list:
        if len(cfg.node_dict[node]) == 0:
            tmp = np.random.rand(cfg.n_values, cfg.n_values)
            np.fill_diagonal(tmp, 0)
            Q[node] = tmp.tolist()
        else:
            parent = cfg.node_dict[node][0]
            Q[node] = {}
            Q[node][parent] = []
            tmp = np.random.rand(cfg.n_values, cfg.n_values)
            np.fill_diagonal(tmp, 0)
            Q[node][parent].append(tmp.tolist())
            tmp = np.random.rand(cfg.n_values, cfg.n_values)
            np.fill_diagonal(tmp, 0)
            Q[node][parent].append(tmp.tolist())
    return Q


def generate_and_save_graph():
    G = nx.DiGraph()
    G.add_edges_from(cfg.net_struct)
    pos = graphviz_layout(G, prog='dot')
    nx.draw_networkx(G, pos=pos, arrows=True)
    plt.savefig('../data/ctbn_graph.png')
    # A = nx.nx_agraph.to_agraph(G)
    # A.layout('dot', args='-Nfontsize=10 -Nwidth=".2" -Nheight=".2" -Nmargin=0 -Gfontsize=8')
    # A.draw('test.png')


def get_current_Q_for_node(node, df_traj, Q_dict):
    # TODO consider multiple parents
    parent = cfg.node_dict[node]
    if len(parent) == 0:
        Q = Q_dict[node]
    else:
        Q = Q_dict[node][parent[0]][int(df_traj[parent[0]].values[-1])]
    return Q


def draw_time(df_traj, Q_dict):
    random_draws = []
    for node in cfg.node_list:
        current_val = int(df_traj[node].values[-1])
        current_Q = get_current_Q_for_node(node, df_traj, Q_dict)
        q = current_Q[current_val][int(1-current_val)]
        random_draws += [np.random.exponential(1/q)]

    return np.min(random_draws)


def draw_variable(df_traj, Q_dict):
    q_list = []
    for node in cfg.node_list:
        current_val = int(df_traj[node].values[-1])
        current_Q = get_current_Q_for_node(node, df_traj, Q_dict)
        q_list += [current_Q[current_val][int(1-current_val)]]
    q_list /= np.sum(q_list)

    return np.random.choice(cfg.node_list, p=q_list)


def draw_next_value():
    # TODO non-binary variables
    return


def run_generative_ctbn():
    # Generates predefined number of trajectories for the same graph with the same Q!
    generate_and_save_graph()
    Q_dict = generate_random_conditional_intensity_matrices()

    df_traj_hist = pd.DataFrame()

    for exp in range(1, cfg.n_experiments + 1):
        t = 0

        # Randomly initializing first states
        initial_states = {var: [np.random.randint(0, 2)] for var in cfg.node_list}
        initial_states['time'] = 0
        df_traj = pd.DataFrame.from_dict(initial_states)

        while t < cfg.T:
            tao = draw_time(df_traj, Q_dict)
            var = draw_variable(df_traj, Q_dict)

            # Adding new state change to the trajectories
            new_data = pd.DataFrame(df_traj[-1:].values, columns=df_traj.columns)
            new_data['time'] = t + tao
            new_data[var] = int(1 - new_data[var])
            df_traj = df_traj.append(new_data, ignore_index=True)

            t += tao

        df_traj['experiment'] = exp
        df_traj_hist = df_traj_hist.append(df_traj)

    # Save all the sampled trajectories
    df_traj_hist.to_csv('../data/trajectory_hist.csv')
