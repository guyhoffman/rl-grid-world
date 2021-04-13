import environment
import agent
import numpy as np
import random

# Visual scale
SCALE=100

# Envs are loaded from envs.json
env = environment.Environment("fourbythree", start_position=(0,0), scale=SCALE)

# Agent -------------
alpha = 0.2
discount = 0.99

agent = agent.FVMControl(alpha, discount, env)
# agent = agent.SARSAAgent(alpha, discount, env, epsilon=0.6)
# agent = agent.QLearningAgent(alpha, discount, env, epsilon=0.6)
# agent = agent.EVSarsaAgent(alpha, discount, env)

# Initialize environment state -----------
env.reset_state()

reward = 0
# Learning -----------
while (True):
    env.render(agent)
    input ("=== Episode === ") # Uncomment to inspect agent episode-by-episode

    while (True):
       # input ("== Step == ") # Uncomment to inspect agent step-by-step

        # Get current state
        state = env.get_state()
        # Choose action
        action = agent.get_action(state)
        # Try out the action
        next_state, reward, terminal = env.step(action)
        # Update the agent's internal variable
        done = agent.update(state, action, reward, next_state, terminal)

        env.render(agent)

        if done:
            env.reset_state()
            break
