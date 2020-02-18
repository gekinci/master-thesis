from pomdp.learn_pomdp.parameters import *
import scipy


def total_consumption(state, action):
    c = np.sum(state) + action
    return c


def transition(initial, final, time):
    """
    Function that returns the probability of reaching FINAL state from INITIAL state
    taking ACTION.
    """
    initial_ind = env_states.index(initial)
    final_ind = env_states.index(final)
    T_time = scipy.linalg.expm(T*time)  # TODO check
    return T_time[initial_ind, final_ind]


def sensor(observation, state):
    """
    Function that returns the probability of getting OBSERVATION after taking ACTION
    to land in STATE.
    """
    Z_normalized = Z / Z.sum(axis=0)
    obs_ind = observations.index(observation)
    s_ind = env_states.index(state)
    return Z[s_ind, obs_ind]


def getReward(state, action):
    """
    Function that returns the reward for taking ACTION in STATE.
    """
    r = - 1 * (3 == total_consumption(state, action)) \
        + 1 * (2 == total_consumption(state, action)) \
        - 1 * (1 == total_consumption(state, action)) \
        - 1 * (0 == total_consumption(state, action))

    return r


def getObservation(state):
    """
    Function that returns an observation and a new state to which the game is
    reinitialised after taking ACTION in STATE.
    """
    state_ind = env_states.index(state)
    observation = np.random.choice(observations, p=Z[state_ind, :])

    return observation
