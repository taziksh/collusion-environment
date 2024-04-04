import numpy as np
import environs
import random
import math

np.set_printoptions(precision=2)

num_agents = 2

num_actions = 5

learning_rate = 0.9
discount_rate = 0.8
beta = 0.5 # for chosing like in the paper
decay_rate= 0.005
#epsilon = 0.5 # for randomly chose

num_episodes = 100
max_steps = 30

env = environs.Cournot(num_agents)
#TODO: try np.zeros
qtables = [np.ones((1, num_actions)) for _ in range(num_agents)]

def softmax(q_values, beta):
    assert beta > 0
    q_exp = np.exp(q_values[0] / beta)
    probs = q_exp / np.sum(q_exp)
    return probs

for episode in range(num_episodes):

    # reset the environment
    state, info = env.reset()
    done = False

    for s in range(max_steps):
        actions = []

        for agent_idx in range(num_agents):            
            probs = softmax(qtables[agent_idx], beta)

            #randomly choose action
            action = np.random.choice(range(num_actions), p=probs)
            
            actions.append(action)

        #old code: explore v exploit
        '''if random.uniform(0,1) < epsilon:
            # explore
            action = env.action_space.sample()
        else:
            # exploit
            action = np.argmax(qtable[state,:])
        '''
        
        # take action and observe reward
        new_states, rewards, done, _ = env.step(actions)

        # Q-learning algorithm
        for agent_idx, reward in enumerate(rewards):
            qtable = qtables[agent_idx]
            action = actions[agent_idx]
            qtable[0,action] = (1-learning_rate)*qtable[0,action] + learning_rate * reward
        #print('{}, {}'.format(action,reward))
        #print(qtable)

        # Update to our new state
        # state = new_state

        # if done, finish episode
        if done:
            break

    # Decrease epsilon
    # epsilon = np.exp(-decay_rate*episode)

print(f"Training completed over {num_episodes} episodes")
for i, qtable in enumerate(qtables):
    print(f"Q-values for agent {i}: {qtable}")