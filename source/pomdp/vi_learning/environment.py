from pomdp.vi_learning.model import *
from pomdp.vi_learning.parameters import *
from simulations.ctbn import CTBNSimulation
from utils.constants import *
import pandas as pd


class CellEnvironment:
    """
    The environment. Updates the time of the system, the states of the parents,
    and returns reward and observation.
    """

    def __init__(self, _state=None, folder='../_data/'):
        self._parents = Parents(Q_X, Q_Y, folder=folder)
        self._state = list(self._parents.init_state.values())
        self.time = 0
        self.observation = getObservation(self._state)

    def getState(self):
        return self._state

    def getParents(self):
        return self._parents

    def respond(self, action=None):
        """
        Responds to an agent's action with a reward and observation
        and updates the state of the parents.
        """
        reward = None
        if action:
            reward = getReward(self._state, action)
        self._state, self.time = self._parents.transition()
        obs = getObservation(self._state)
        return reward, obs, self.time


class Parents:
    """
    Cell X and Y. Initializes the ctbn and returns the transition of the parents
    to the CellEnvironment.
    """

    def __init__(self, Qx, Qy, folder='../_data/'):
        self.cfg = {GRAPH_STRUCT: {'X': [],
                                   'Y': []},
                    T_MAX: 50,
                    STATES: parent_states,
                    INITIAL_PROB: prob_states,
                    Q_DICT: {'X': Qx,
                             'Y': Qy}
                    }
        self.ctbn = CTBNSimulation(cfg=self.cfg, save_folder=folder)
        self.init_state = self.ctbn.initialize_nodes()
        self._prev_step = pd.DataFrame().append(self.init_state, ignore_index=True)
        self._prev_step[TIME] = 0

    def transition(self):
        self._prev_step = self.ctbn.do_step(self._prev_step)
        t = self._prev_step[TIME].values[0]
        state = [int(self._prev_step['X'].values[0]), int(self._prev_step['Y'].values[0])]
        return state, t
