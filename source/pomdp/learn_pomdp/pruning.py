import numpy as np


def formAlphaSet(plan_set):
    """
    Given a set of conditional plans, returns a set of alpha vectors corresponding
    to the set of plans. Maintains the index of the conditional plans.
    """
    alpha_set = []
    for plan in plan_set:
        alpha_set.append(plan[2])
    return alpha_set


def evaluateBelief(belief, alpha):
    """
    Evaluates a plan over belief space.
    """
    alpha = np.array(alpha)
    return np.array(belief).dot(alpha)


def evaluateBeliefSpace(row, col, alpha_set, plan_set):
    belief = [row['b1'], row['b2'], row['b3'], row['b4']]
    values = []
    for i in range(len(alpha_set)):
        values.append(evaluateBelief(belief, alpha_set[i]))
    best = np.argmax(values)
    row['best_plan_ind'] = best
    row[col] = plan_set[best][0]
    return row


def prune(plan_set, df_b, step, b_jump=0.01):
    """
    Given a set of conditional plans, returns the set of conditional plans that are
    optimal over some interval in belief space and returns a dictionary mapping
    each conditional plan's index to the set of b_left over which it is optimal.
    b_jump is the size of each jump in belief space in the linear program.
    """
    parsimonius_set = []
    optimal_map = {}
    col = 'step_' + str(step)

    alpha_set = formAlphaSet(plan_set)

    df_b = df_b.apply(evaluateBeliefSpace, args=(col, alpha_set, plan_set), axis=1)

    for i in df_b['best_plan_ind'].unique():
        parsimonius_set.append(plan_set[int(i)])

    df_b.drop(columns=['best_plan_ind'], inplace=True)

    return parsimonius_set, optimal_map, df_b


def createOptimalActionMap(plan_set, optimal_map):
    """
    Given a set of conditional plans a map between the index of the plans and the
    region over belief space where it is optimal, returns a map that maps the
    initial action to regions over belief space where it is optimal.
    """
    action_map = []
    for i in list(optimal_map.keys()):
        plan = plan_set[i]
        action = plan[0]
        smallest = min(optimal_map[i])
        biggest = max(optimal_map[i])
        action_map.append((action, [str(round(smallest, 2)), str(round(biggest, 2))]))
    action_map = sorted(action_map, key=lambda x: x[1])
    return action_map
