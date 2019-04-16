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
start_pos = None # (0, 0)
end_positions = [(3, 1), (3, 2)]
end_rewards = [-50.0, 10.0]
blocked_positions = [(1,1)]
default_reward= -1.0

scale=150
env = environment.Environment(gridW, gridH, end_positions, end_rewards, blocked_positions, start_pos, default_reward, scale)

# Agent -------------
alpha = 0.2
epsilon = 0.2
discount = 0.99
action_space = env.action_space
state_space = env.state_space

agent = agent.SARSAAgent(alpha, epsilon, discount, env)

# Learning -----------
env.reset_state()
env.render(agent)

n = 3
while(True):
    input("------- Episode -------- ")

    states = np.full(n+1, np.nan)
    actions = np.full(n+1, np.nan)
    rewards = np.full(n+1, np.nan)
    state = env.get_state()
    states[0] = state
    print("initial state is", state)
    action = agent.get_explore_action(state)
    print("taking action", action)
    actions[0] = action
    T = np.infty

    t = 0
    tau = 0
    while (tau < T - 1):
        print("t=",t, ", tau=", tau, "T=", T)
        print("states=",states)
        print("actions=",actions)
        print("rewards=",rewards)
        input(" --- Step ")
        next_state, reward, done = env.step(action)
        print("ended up in state", next_state,"with reward", reward )
        rewards[(t + 1) % n] = reward
        states[(t + 1) % n] = next_state
        print("states=",states)
        print("actions=",actions)
        print("rewards=",rewards)
        if done == True:
            T = t + 1
        else:
            print("next state is", state)
            action = agent.get_explore_action(next_state)
            print("next action is", action)
            actions[(t + 1) % n] = action
        tau = t - n + 1
        if tau >= 0:
            print("updating Q values")
            G = 0
            for i in range(tau+1, min(tau+n, T)):
                G += discount**(i-tau-1) * rewards[i % n]
            if tau + n < T:
                G += discount**n * agent.qvalues[states[(tau + n) % n]][actions[ (tau + n) % n ]]
                agent.qvalues[states[tau % n]][actions[tau % n]] = (1 - alpha) * agent.qvalues[states[tau % n]][actions[tau % n]] + alpha * G  
        t = t + 1
        env.render(agent)

