import random
import numpy as np


# ------------------------------------------------------------------------------------------
# ------------------------------------- Base Policy ----------------------------------------
# ------------------------------------------------------------------------------------------

class BasePolicy:

	def __init__(self, state_space, action_space, qvalues=None, epislon=None):
		self.action_space = action_space
		self.state_space = state_space
		self.qvalues = qvalues
		self.epsilon = epislon

	def get_best_action(self, state):

		best_action = 0
		value = self.qvalues[state][0]
        
		for action in range(self.action_space):
			q_val = self.qvalues[state][action]
			if q_val > value:
				value = q_val
				best_action = action

		return best_action

# ------------------------------------------------------------------------------------------
# ------------------------------------- Specific Policies ----------------------------------
# ------------------------------------------------------------------------------------------

class RandomPolicy(BasePolicy):
	def step(self, state):
		return 	random.choice(range(self.action_space))

# ------------------------------------------------------------------------------------------

class GreedyPolicy(BasePolicy):
	def step(self, state):
		return self.get_best_action( state)

# ------------------------------------------------------------------------------------------

class EpsilonGreedyPolicy(BasePolicy):
	def step(self, state):
		if self.epsilon > np.random.uniform(0.0, 1.0):
			chosen_action = random.choice(range(self.action_space))
		else:
			chosen_action = self.get_best_action(state)

		return chosen_action

# ------------------------------------------------------------------------------------------


