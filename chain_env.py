""" 
The environment simulates the possibility of buying or selling a good. The agent can either have one unit or zero unit of that good. At each transaction with the market, the agent obtains a reward equivalent to the price of the good when selling it and the opposite when buying. In addition, a penalty of 0.5 (negative reward) is added for each transaction.
Two actions are possible for the agent:
- Action 0 corresponds to selling if the agent possesses one unit or idle if the agent possesses zero unit.
- Action 1 corresponds to buying if the agent possesses zero unit or idle if the agent already possesses one unit.
The state of the agent is made up of an history of two punctual observations:
- The price signal
- Either the agent possesses the good or not (1 or 0)
The price signal is build following the same rules for the training and the validation environment. That allows the agent to learn a strategy that exploits this successfully.

"""

import numpy as np
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import matplotlib.pyplot as plt

from deer.base_classes import Environment


class ChainEnv(Environment):
    def __init__(self, n_states=10):
        """Initialize environment."""
        # Defining the type of environment
        self.state = np.array(
            [0.0]
        )  # We start from state 0 and arrive at max state N-1 *0.1 (reward=1)

        self._counter = 1
        self.n_states = n_states

    def reset(self, mode):
        """Resets the environment for a new episode.

        Parameters
        -----------
        mode : int
            -1 is for the training phase, others are for validation/test.

        Returns
        -------
        list
            Initialization of the sequence of observations used for the pseudo-state; dimension must match self.inputDimensions().
            If only the current observation is used as a (pseudo-)state, then this list is equal to self.state.
        """

        self.state = np.array([0.0])

        self._counter = 1
        return self.state

    def act(self, action):
        """Performs one time-step within the environment and updates the current observation self._last_ponctual_observation

        Parameters
        -----------
        action : int
            Integer in [0, ..., N_A] where N_A is the number of actions given by self.nActions()
            0 means go back to the initial state, 1 means go to next

        Returns
        -------
        reward: float
        """
        reward = 0
        if action == 0:  # A
            self.state[0] += 0.1
            if self.state[0] > ((self.n_states - 1) * 0.1):
                reward = 1
                self.state[0] = (self.n_states - 1) * 0.1
        elif action == 1:  # B
            if self.state[0] == 0:
                reward = 0.2
            self.state[0] = 0
        return reward

    def summarizePerformance(self, test_data_set, *args, **kwargs):
        """
        This function is called at every PERIOD_BTW_SUMMARY_PERFS.
        Parameters
        -----------
            test_data_set
        """
        print(test_data_set)
        print("Summary was called but I don't wanna do it")

    def inputDimensions(self):
        return [(1,)]  # We consider an observation made up of an history of
        # - the last six for the first scalar element obtained
        # - the last one for the second scalar element

    def nActions(self):
        return 2  # The environment allows two different actions to be taken at each time step

    def inTerminalState(self):
        return False

    def observe(self):
        return np.array(self.state)


def main():
    # Can be used for debug purposes
    myenv = ChainEnv(10)
    for _ in range(20):
        print(myenv.act(0))
        print(myenv.observe())
    print(myenv.observe())


if __name__ == "__main__":
    main()
