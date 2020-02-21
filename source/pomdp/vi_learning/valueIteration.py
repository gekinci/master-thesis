from pomdp.vi_learning.model import *
from pomdp.vi_learning.parameters import *
from pomdp.vi_learning.pruning import *
import pandas as pd


def valueIteration(env_traj, next_set, df_b):
    """
    Returns the optimal action after evaluating policies using value iteration.
    """
    current_set = []

    step = env_traj.index[-1]
    while step:
        current_set = next_set
        instant = env_traj.loc[step, :]
        t_delta = instant['t_delta']

        for i in range(len(current_set)):
            plan = current_set[i]
            current_set[i] = get_value_of_plan(plan, t_delta)

        print(df_b.head())

        current_set, optimal_map, df_b = prune(current_set, df_b, step)

        next_set = []
        for action in actions:
            for plan0 in current_set:
                for plan1 in current_set:
                    for plan2 in current_set:
                        mapping = {}
                        mapping[0] = plan0
                        mapping[1] = plan1
                        mapping[2] = plan2
                        plan = (action, mapping, [])
                        next_set.append(plan)
        step -= 1
    return current_set, df_b


def pickBestAction(b_left, current_set):
    """
    Given a set of plans (with value functions), returns the best action to take
    given that the (left) belief state of the agent is b_left.
    """
    values = []
    for i in range(len(current_set)):
        alpha = current_set[i][2]
        value = evaluateBelief(b_left, alpha)
        values.append(value)
    opt_index = np.argmax(values)
    action = current_set[opt_index][0]
    return action


def get_value_of_plan(plan, time_nt=0):
    """
    Returns a new plan, with the same action and mapping as the old plan, but with
    the value of the plan for each state computed.
    """
    action = plan[0]
    map_to_old_plans = plan[1]

    values = []
    for initial_state in env_states:
        rwrd = getReward(initial_state, action)
        sum_over_states = 0
        for next_state in env_states:
            T = transition(initial_state, next_state, time_nt)
            sum_over_observations = 0
            for observation in observations:
                Z = sensor(observation, next_state)
                next_plan = map_to_old_plans[observation]
                value = next_plan[2][env_states.index(next_state)]
                sum_over_observations += (Z * value)
            sum_over_states += (T * sum_over_observations)
        value = rwrd + GAMMA * sum_over_states
        values.append(value)

    new_plan = (action, map_to_old_plans, values)
    return new_plan
