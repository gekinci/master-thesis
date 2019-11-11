# Dictionary in the format of {node: its parents}
node_dict = {
    'X': [],
    'Y': ['X'],
    'Z': ['X'],
    'W': ['Y'],
    'Q': ['Z'],
    'T': ['Z']
}

n_experiments = 10
T = 50  # t_max to terminate sampling

n_values = 2  # binary {0, 1)

node_list = list(node_dict.keys())
num_nodes = len(node_list)
net_struct = [[par, node] for node in node_list for par in node_dict[node] if len(node_dict[node]) > 0]
