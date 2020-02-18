from pomdp.learn_pomdp.valueIteration import *
import scipy


class Agent:
    """
    Cell Z
    """

    def __init__(self, _b=np.array([0.25, 0.25, 0.25, 0.25])):
        self._reward = 0
        self._observation = None
        self.b = _b
        self.time = 0
        self.step = 1
        self.time_nt = 0

    def act(self, game, action):
        reward, observation = game.respond(action)
        self._reward += reward
        self._observation = observation

    def update_reward(self, new_reward):
        self._reward += new_reward

    def update_observation(self, new_observation):
        self._observation = new_observation

    def get_reward(self):
        return self._reward

    def update_belief(self, t):
        obs = self._observation
        # TODO check
        self.b = Z[:, obs] * (self.b @ scipy.linalg.expm(T * t))
        self.b = self.b / self.b.sum()

    def pick_action(self):
        # TODO
        current_set = valueIteration(initial_set, self.time_nt, self.b)
        return pickBestAction(self.b, current_set)
