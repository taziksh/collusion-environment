import numpy as np
import environs
import agent
import random
import math

np.set_printoptions(precision=2)

num_agents = 3

num_actions = 3

learning_rate = 0.9
discount_rate = 0.8
beta = 0.5 # for chosing like in the paper
decay_rate= 0.005
epsilon = 0.1

num_episodes = 100
max_steps = 30

env = environs.Cournot(num_agents)

agents = [agent.Agent(num_actions, learning_rate, beta, epsilon) for _ in range(num_agents)]

for episode in range(num_episodes):

    # reset the environment
    state, info = env.reset()
    done = False

    while not done:
        for s in range(max_steps):
            actions = [agent.select_action() for agent in agents]
            
            # take action and observe reward
            new_states, rewards, done, _ = env.step(actions, agents)

            # Q-learning algorithm
            for agent_idx, reward in enumerate(rewards):
                agents[agent_idx].update_qtable(actions[agent_idx], reward)
    
            if done:
                break
    
    # for agent in agents:
    #     agent.decrease_epsilon(episode)


print(f"Training completed over {num_episodes} episodes")
for i, agent in enumerate(agents):
    print(f"Q-values for agent {i}: {agent.qtable}")
