# Dictionary in the format of {node: [its parents]}
graph_dict = {
    'X': [],
    'Y': ['X'],
    # 'Z': ['Y'],
    # 'W': ['Z'],
    'Z': ['X'],
    'W': ['Y'],
    'Q': ['Z'],
    'T': ['Z', 'M'],
    'M': []
}

t_max = 20  # t_max to terminate sampling

n_values = 2  # binary {0, 1), if different sets for variables, could also be a dict

node_list = list(graph_dict.keys())
num_nodes = len(node_list)
net_struct = [[par, node] for node in node_list for par in graph_dict[node] if len(graph_dict[node]) > 0]
