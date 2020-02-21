import numpy as np
from utils.helpers import get_amalgamated_trans_matrix

############################
### Problem Parameters
############################

# states, 0: not-using, 1: using
parent_states = [0, 1]
env_states = [[0, 0], [0, 1], [1, 0], [1, 1]]

# Probabilities with which the parent states are initialized
p_0 = 0.5
prob_states = [p_0, 1 - p_0]

# The actions available to cell Z
actions = [0, 1]  # 0: not-using, 1: using

# The observations returned to the player after every actions
observations = list(set(np.sum(env_states, axis=1)))

# # The probability of getting the correct observation (and wrong observation)
# p_correct_obs = 0.9
# prob_obs = np.array([p_correct_obs, 1 - p_correct_obs])

Q_X = [[-10, 10], [0.5, -0.5]]

Q_Y = [[-2, 2], [1, -1]]

T = get_amalgamated_trans_matrix(Q_X, Q_Y)

Z = np.array([[1, 0, 0],
              [0, 1, 0],
              [0, 1, 0],
              [0, 0, 1]])


####################################
### Value Iteration Parameters
####################################

# The discount factor used in value iteration
GAMMA = 1

# each plan is a triple
# (action, map from observation to plans in old set, alpha vector)
# trivial plan
trivial_plan = (None, None, [0, 0, 0, 0])

# the trivial map from observations to plans in old set
trivial_map = {0: trivial_plan, 1: trivial_plan, 2: trivial_plan}

# set of height 1 plans
initial_set = [(0, trivial_map, []), (1, trivial_map, [])]
