import environment
import agents

# Visual scale
SCALE=100

# Envs are loaded from envs.json
env = environment.Environment("fourbythree", start_position=(0,0), scale=SCALE)

# Agent -------------
alpha = 0.2
discount = 0.9

# agent = agents.FVMCPrediction(alpha, discount, env)
# agent = agents.FVMCQPrediction(alpha, discount, env)
agent = agents.FVMCControl(alpha, discount, env)
# agent = agents.FVMCEpsiControl(alpha, discount, env, epsilon=0.2)
# agent = agents.OffPolicyMCControl(alpha, discount, env, epsilon=0.2)

# agent = agents.SARSAAgent(alpha, discount, env, epsilon=0.5)
# agent = agents.QLearningAgent(alpha, discount, env, epsilon=0.5)

# Initialize environment state -----------
env.reset_state()

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
        next_state, reward, terminal = env.step(action, 0.2)
        # Update the agent's internal variable
        done = agent.update(state, action, reward, next_state, terminal)

        env.render(agent)

        if done:
            env.reset_state()
            break
