import environment
import agent
import numpy as np
import random

# Visual scale
SCALE=100

# Envs are loaded from envs.json
env = environment.Environment("cliff", start_position=(0,0), scale=SCALE)

# Agent -------------
alpha = 0.2
discount = 0.99

agent = agent.QLearningAgent(alpha, discount, env, epsilon=0.1)
# agent = agent.SARSAAgent(alpha, discount, env, epsilon=0.1)

# Initialize environment state -----------
env.reset_state()

# Learning -----------
while (True):
    env.render(agent)

    while (True):
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
