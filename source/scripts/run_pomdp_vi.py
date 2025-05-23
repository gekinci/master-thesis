"""
Runs an instance of the tiger game. Command line input has the following format:

python play.py [player] [max_time]

player: The options are "Human" or "AI". If playing with a human player, the
user will be prompted with "What action would you like to make? " The options
are "left", "right" or "listen". The options are case sensitive. If playing with
an AI player, the player will play according to the predetermined strategy.

max_time: The number of time steps over which the game is played.
"""

from pomdp.vi_learning.environment import CellEnvironment
from pomdp.vi_learning.valueIteration import *
from pomdp.vi_learning.parameters import *
from utils.constants import *
from utils.helpers import *
from utils.visualization import *
import os


def get_env_trajectory(env, max_time=10):
    list_traj = [env.getState() + [env.observation, env.time]]

    while env.time < max_time:
        _, o, t = env.respond()
        list_traj.append(env.getState() + [o, t])

    df_traj = pd.DataFrame(list_traj, columns=['X1', 'X2', OBS, TIME])
    df_traj = df_traj.assign(t_delta=np.append(df_traj.time.diff().values[1:], .0))

    return df_traj


def train_given_trajectory(init_b, b_jump=0.01, max_time=5, import_env=None, folder='../_data/'):
    env = CellEnvironment(folder=folder)

    df_b_grid = generate_belief_grid(b_jump, cols=['b1', 'b2', 'b3', 'b4'], path_to_save='./')

    if import_env is None:
        env_traj = get_env_trajectory(env, max_time=max_time)
    else:
        env_traj = pd.read_csv(f'../_data/{import_env}/env_traj.csv', index_col=0)

    policy_tree, df_optimal_map = valueIteration(env_traj, initial_set, df_b_grid)

    df_optimal_map.to_csv(folder + 'df_belief.csv')
    env_traj.to_csv(folder + 'env_traj.csv')

    visualize_optimal_policy_map(df_optimal_map, path_to_save=folder)
    plot_trajectories(env_traj, node_list=['X1', 'X2'], path_to_save=folder)

    if not os.listdir(folder):
        os.rmdir(folder)

    # TODO get trajectory of Z, given the policy for parent trajectories
    # player = Agent(_b=init_b)
    # n_step = len(env_traj)
    # step = n_step

    # print("Initial belief state: " + str(init_b) + '\n')
    #
    # while step:
    #     print("Step " + str(step) + ":")
    #     env_instant = env_traj.loc[step - 1, :]
    #     s = [env_instant['X1'], env_instant['X2']]
    #     o = env_instant[OBS]
    #     t = env_instant[TIME]
    #     t_delta = env_instant['t_delta']
    #     player.step = n_step - step
    #     player.time = t
    #     player.time_nt = t_delta
    #     player.update_observation(o)
    #     current_set, optimal_map, next_set = valueIteration(current_set, t_delta)
    #     # a = player.pick_action()
    #     a = pickBestAction(player.b, current_set)
    #     print("State: " + str(s))
    #     print("Observation: " + str(o))
    #     print("Action_Z: " + str(a))
    #     reward = getReward(s, a)
    #     # reward, observation, time = env.respond(move)
    #     print("> Reward_Z " + str(reward) + "\n")
    #     player.update_reward(reward)
    #     # player.update_observation(observation)
    #     player.update_belief(time)
    #     print("> New b_left = " + str(player.b) + '\n')
    #     step -= 1
    # print("Game over! Total Reward: " + str(player.get_reward()) + "\n")


if __name__ == "__main__":

    folder = create_folder_for_experiment(folder_name='../_data/learning_pomdp/')
    initial_b = np.array([0.25, 0.25, 0.25, 0.25])
    max_time = 5
    b_jump = 0.03

    train_given_trajectory(initial_b, b_jump=b_jump,
                           max_time=max_time, folder=folder)
