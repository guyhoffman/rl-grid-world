import environment
import agent
import numpy as np
import random

SCALE=150

env = environment.Environment("fourbythree", scale=SCALE)

# Agent -------------
alpha = 0.2
discount = 0.99

agent = agent.ValueIterationAgent(alpha, discount, env)

# Set up environment  -----------

env.reset_state()

# Learning -----------
while (True):
    env.render(agent)
    input("------- Episode -------- ")
    while (True):
        print ("-- Step --")
        state = env.get_state()
        action = agent.get_action(state)
        next_state, reward, terminal = env.step(action)
        done = agent.update(state, action, reward, next_state, terminal)
        env.render(agent)
        if done:
            env.reset_state()
            break
