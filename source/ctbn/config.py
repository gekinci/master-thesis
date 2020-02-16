from constants import *

# Dictionary in the format of {node: [its parents]}
graph_config = {
    GRAPH_STRUCT: {
        'X': [],
        'Y': ['X'],
        'Z': ['X'],
        'W': ['Y'],
        'Q': ['Z'],
        'T': ['Z', 'M'],
        'M': []
    },
    T_MAX: 20,  # t_max to terminate sampling
    STATES: [0, 1],
    INITIAL_PROB: [0.5, 0.5]
}
