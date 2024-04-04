import numpy as np
import environs
import agent
import random
import math
import argparse

np.set_printoptions(precision=2)

parser = argparse.ArgumentParser()

parser.add_argument('--num_agents', type=int, default=3, help='number of agents')
parser.add_argument('--num_actions', type=int, default=3, help='number of actions')
parser.add_argument('--num_episodes', type=int, default=10000, help='number of episodes')

# num_agents = 3
# num_actions = 3
# num_episodes = 10000

args = parser.parse_args()

learning_rate = 0.9
discount_rate = 0.8
beta = 1000 # for chosing like in the paper
decay_rate= 0.005
epsilon = 0.1

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

            beta = beta*0.999

            if done:
                break
    
    # for agent in agents:
    #     agent.decrease_epsilon(episode)

print('Beta is:' +str(beta))

print(f"Training completed over {args.num_episodes} episodes")
for i, agent in enumerate(agents):
    print(f"Q-values for agent {i}: {agent.qtable}")
