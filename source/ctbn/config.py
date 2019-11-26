import constants

# Dictionary in the format of {node: [its parents]}
graph_config = {
    constants.PARENTS: {
        'X': [],
        'Y': ['X'],
        # 'Z': ['Y'],
        # 'W': ['Z'],
        'Z': ['X'],
        'W': ['Y'],
        'Q': ['Z'],
        'T': ['Z', 'M'],
        'M': []
    },
    constants.T_MAX: 20,  # t_max to terminate sampling
    constants.N_VALUES: 2  # binary {0, 1), if different sets for variables, could also be a dict
}
