import numpy as np
import random
import policy



# ------------------------------------------------------------------------------------------
# ------------------------------------- Base Agent -----------------------------------------
# ------------------------------------------------------------------------------------------

class BaseAgent:
	
	def __init__(self, alpha, epsilon, discount, environment):
 
		self.action_space = environment.action_space
		self.alpha = alpha
		self.epsilon = epsilon
		self.discount = discount
		self.qvalues = np.zeros((environment.state_space, environment.action_space), np.float32)
		self.policy = policy.RandomPolicy(environment.state_space, environment.action_space)
		self.explore_policy = self.policy
		
	def update(self, state, action, reward, next_state, done):

		# Q(s,a) = (1.0 - alpha) * Q(s,a) + alpha * (reward + discount * V(s'))

		if done==True:
			qval_dash = reward
		else:
			qval_dash = reward + self.discount * self.get_next_value(next_state)
			
		qval_old = self.qvalues[state][action]      
		qval = (1.0 - self.alpha)* qval_old + self.alpha * qval_dash

		self.qvalues[state][action] = qval
        
	def get_policy_action(self, state):
		return self.policy.step(state)

	def get_explore_action(self, state):
		return self.explore_policy.step(state)
        
	def get_next_value(self, next_state):		
		pass





class MCValueEstimator(BaseAgent):
	def __init__(self, alpha, epsilon, discount, env):
		super().__init__(alpha,	epsilon, discount, env)

		self.policy = policy.FixedRandomPolicy(env.state_space, env.action_space) 
		self.explore_policy = self.policy 
	
	def update(self, state, action, reward, next_state, done):
		qval_old = self.qvalues[state][action]      
		qval = reward + self.discount * ((1.0 - self.alpha)* qval_old + self.alpha * self.qvalues[next_state][action])
		self.qvalues[state][action] = qval


# ------------------------------------------------------------------------------------------
# ---------------------------------- First-Value MC Prediction -----------------------------
# ------------------------------------------------------------------------------------------

class FVMCPrediction(BaseAgent):

	def get_next_value(self, next_state):

		# estimate V(s) as maximum of Q(state,action) over possible actions
		value = self.qvalues[state][0]
       
		for action in range(self.action_space):
			q_val = self.qvalues[state][action]
			if q_val > value:
				value = q_val

		return value


# ------------------------------------------------------------------------------------------
# ---------------------------------- Q-Learning Agent --------------------------------------
# ------------------------------------------------------------------------------------------

class QLearningAgent(BaseAgent):

	def __init__(self, alpha, epsilon, discount, environment):
		super().__init__(alpha,	epsilon, discount, environment)

		self.policy = policy.GreedyPolicy(environment.state_space, environment.action_space, self.qvalues)
		self.explore_policy = policy.EpsilonGreedyPolicy(environment.state_space, environment.action_space, self.qvalues, self.epsilon)

	def get_next_value(self, next_state):
		return self.qvalues[next_state][self.get_policy_action(next_state)]

# ------------------------------------------------------------------------------------------
# ---------------------------------- SARSA Agent --------------------------------------
# ------------------------------------------------------------------------------------------

class SARSAAgent(BaseAgent):

	def __init__(self, alpha, epsilon, discount, environment):
		super().__init__(alpha,	epsilon, discount, environment)

		self.policy = policy.EpsilonGreedyPolicy(environment.state_space, environment.action_space, self.qvalues)
		self.explore_policy = self.policy

	def get_next_value(self, next_state):
		return self.qvalues[next_state][self.get_policy_action(next_state)]


# ------------------------------------------------------------------------------------------
# ------------------------------ Expected Value SARSA Agent --------------------------------
# ------------------------------------------------------------------------------------------
    
class EVSarsaAgent(BaseAgent):
    
	def get_next_value(self, next_state):
		
		# estimate V(s) as expected value of Q(state,action) over possible actions assuming epsilon-greedy policy
		# V(s) = sum [ p(a|s) * Q(s,a) ]
          
		best_action = 0
		max_val = self.qvalues[next_state][0]
		
		for action in range(self.action_space):
            
			q_val = self.qvalues[next_state][action]
			if q_val > max_val:
				max_val = q_val
				best_action = action
        
		state_value = 0.0
		n_actions = self.action_space
		
		for action in range(self.action_space):
            
			if action == best_action:
				trans_prob = 1.0 - self.epsilon + self.epsilon/n_actions
			else:
				trans_prob = self.epsilon/n_actions
                   
			state_value = state_value + trans_prob * self.qvalues[next_state][action]

		return state_value
