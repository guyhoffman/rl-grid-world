import environment
import agent
import numpy as np
import random

# Environment ---------
# gridH, gridW = 4, 8
# start_pos = (3, 0)
# end_positions = [(3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7)]
# end_rewards = [-100.0, -100.0, -100.0, -100.0, -100.0, -100.0, 100.0]
# blocked_positions = []
# default_reward= -1.0

# gridH, gridW = 9, 7
# start_pos = None
# end_positions = [(0, 3), (2, 4), (6, 2)]
# end_rewards = [20.0, -50.0, -50.0]
# blocked_positions = [(2, i) for i in range(3)] + [(6, i) for i in range(4, 7)]
# default_reward = -0.1

gridW, gridH = 4, 3
start_pos = (0, 0)
end_positions = [(3, 1), (3, 2)]
end_rewards = [-50.0, 10.0]
blocked_positions = [(1,1)]
default_reward= -1.0


scale=100
env = environment.Environment(gridW, gridH, end_positions, end_rewards, blocked_positions, start_pos, default_reward, scale)

# Agent -------------
alpha = 0.2
epsilon = 0.25
discount = 0.99
action_space = env.action_space
state_space = env.state_space

agent = agent.MCValueEstimator(alpha, epsilon, discount, env)

# Learning -----------
env.render(agent)
state = env.get_state()

while(True):

	input("Step ")

	action = agent.get_explore_action(state)
	next_state, reward, done = env.step(action)
	env.render(agent)

	agent.update(state, action, reward, next_state, done)
	state = next_state

	if done == True:	
		env.reset_state()
		env.render(agent)
		state = env.get_state()
		continue

