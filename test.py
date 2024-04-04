import numpy as np
import environs
import agent
import random
import math
import argparse

np.set_printoptions(precision=2)

num_agents = 2 # in paper: [2 , 3, 4 , 5 , 6]

num_actions = 40 # in paper: 40

demand_quantity = 40 # in paper it is denoted with u=40
demand_factor = 1 # in paper it is denoted with v=1

learning_rate = 0.5 # in paper: [ 0.05 , 0.25 , 0.5 , 1 ]
beta = 1000 # for chosing like in the paper, decay by factor 0.999 each time step

# with memory means we have states different states for each agent:
# - own previous production level
# - production level of others
# myopic says that the agents learn short term achieved reward gamma=0 to update q-table
# non-myopic means they learn by rewards and expected q-talbe long term values gamma=0.9

gamma = 0 # myopic firms=0 , non-myopic 0.9

epsilon = 0.1 # in paper: not used, but for less computational effort keep that approach

num_episodes = 1000
max_steps = 300

env = environs.Cournot(args.num_agents)

agents = [agent.Agent(args.num_actions, learning_rate, beta, epsilon) for _ in range(args.num_agents)]

for episode in range(args.num_episodes):

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

            beta = beta*0.99

            if done:
                break
    
    # for agent in agents:
    #     agent.decrease_epsilon(episode)

print('Beta is:' +str(beta))

print(f"Training completed over {args.num_episodes} episodes")
for i, agent in enumerate(agents):
    print(f"Q-values for agent {i}: {agent.qtable}")
