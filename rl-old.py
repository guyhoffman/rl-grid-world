import environment
import agent
import numpy as np
import random

#    { "unused":
#         "
#         # ------------------------ environment 2 -----------------------------
#         gridH, gridW = 8, 4
#         start_pos = (7, 0)
#         end_positions = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0)]
#         end_rewards = [10.0, -40.0, -40.0, -40.0, -40.0, -40.0, -40.0]
#         blocked_positions = []
#         default_reward= -0.2
#         # ------------------------ environment 3 -----------------------------
#         gridH, gridW = 8, 9
#         start_pos = None
#         end_positions = [(2, 2), (3, 5), (4, 5), (5, 5), (6, 5)]
#         end_rewards = [10.0, -30.0, -30.0, -30.0, -30.0]
#         blocked_positions = [(i, 1) for i in range(1, 7)] + \
#                             [(1, i) for i in range(1, 8)] + \
#                             [(i, 7) for i in range(1, 7)]
#         default_reward= -0.5
#         # ------------------------ environment 4 -----------------------------
#         gridH, gridW = 9, 7
#         start_pos = None
#         end_positions = [(0, 3), (2, 4), (6, 2)]
#         end_rewards = [20.0, -50.0, -50.0]
#         blocked_positions = [(2, i) for i in range(3)] + [(6, i) for i in range(4, 7)]
#         default_reward = -0.1
#                 "
#     }

# Environment ---------
gridH, gridW = 4, 4
start_pos = None
end_positions = [(0, 3), (1, 3)]
end_rewards = [10.0, -60.0]
blocked_positions = [(1, 1), (2, 1)]
default_reward= -0.2
scale=130

env = environment.Environment(gridH, gridW, end_positions, end_rewards, blocked_positions, start_pos, default_reward, scale)

print (env.state2idx)

# Agent -------------
alpha = 0.2
epsilon = 0.25
discount = 0.99
action_space = env.action_space
state_space = env.state_space

# agent = agent.EVSarsaAgent(alpha, epsilon, discount, action_space, state_space)
agent = agent.FVMCPrediction(alpha, epsilon, discount, action_space, state_space)

# Learning -----------
env.render(agent.qvalues, agent.get_policy_action)


state = env.get_state()

returns = []
for i in range(state_space):
	returns.append([])

episode_length = 26

cvalues = np.zeros((state_space, action_space), np.float32)

while(True):

	# == Single Episode == 
	print("Next Episode ")
	episode = []
	for i in range(episode_length):
		

		if epsilon > np.random.uniform(0.0, 1.0):
			action = random.choice(range(action_space))
		else:
			action = agent.get_policy_action(state,action_space)

		next_state, reward, done = env.step(action)
		env.render(agent.qvalues, agent.get_policy_action)

		episode.append((state, action, reward))
		state = next_state

		if done == True:	
			continue
	
	print(episode)

	# -- Calculate long-term reward for each state
	g = 0
	epi_states = list(list(zip(*episode))[0])
	print(epi_states, '===')
	for i, e in enumerate(reversed(episode[:-1])):
		g = g + e[2] # add reward
		if not e[0] in epi_states[0:-i-2]:
			returns[e[0]].append(g)

	for s in range(state_space):
		for a in range(action_space):
			if len(returns[s]) > 0:
				agent.qvalues[s,a] = np.mean(returns[s])

	# = improve policy
	for s in range(state_space):
		best_a = -9999999999
		for a in range(action_space):
			env.position = env.idx2state[s]
			next_s, r, d = env.step(a)
			# print (s, a, next_s,r)
			# print (agent.qvalues[next_s,0] + r , best_a)
			if agent.qvalues[next_s,0] + r > best_a :
				best_a = agent.qvalues[next_s,0] + r
				agent.policy[s] = a


	env.reset_state()
	env.render(agent.qvalues, agent.get_policy_action)
	state = env.get_state()




while(False):

	# == Single Episode == 
	print("Next Episode ")
	episode = []
	for i in range(episode_length):
		

		if epsilon > np.random.uniform(0.0, 1.0):
			action = random.choice(range(action_space))
		else:
			action = agent.get_policy_action(state,action_space)

		next_state, reward, done = env.step(action)
		env.render(agent.qvalues, agent.get_policy_action)

		episode.append((state, action, reward))
		state = next_state

		if done == True:	
			continue
	
	print(episode)

	# -- Calculate long-term reward for each state
	g = 0
	epi_states = list(list(zip(*episode))[0])
	print(epi_states, '===')
	for i, e in enumerate(reversed(episode[:-1])):
		g = g + e[2] # add reward
		if not e[0] in epi_states[0:-i-2]:
			returns[e[0]].append(g)

	for s in range(state_space):
		for a in range(action_space):
			if len(returns[s]) > 0:
				agent.qvalues[s,a] = np.mean(returns[s])

	# = improve policy
	for s in range(state_space):
		best_a = -9999999999
		for a in range(action_space):
			env.position = env.idx2state[s]
			next_s, r, d = env.step(a)
			# print (s, a, next_s,r)
			# print (agent.qvalues[next_s,0] + r , best_a)
			if agent.qvalues[next_s,0] + r > best_a :
				best_a = agent.qvalues[next_s,0] + r
				agent.policy[s] = a


	env.reset_state()
	env.render(agent.qvalues, agent.get_policy_action)
	state = env.get_state()



# - Original code below 
while(False):

	possible_actions = env.get_possible_actions()
	action = agent.get_action(state, possible_actions)
	next_state, reward, done = env.step(action)
	env.render(agent.qvalues, agent.get_policy_action)

	next_state_possible_actions = env.get_possible_actions()
	agent.update(state, action, reward, next_state, next_state_possible_actions, done)
	state = next_state

	input("Next")

	if done == True:	
		env.reset_state()
		env.render(agent.qvalues, agent.get_policy_action)
		state = env.get_state()
		continue

