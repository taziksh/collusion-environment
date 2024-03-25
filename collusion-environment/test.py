import numpy as np
import environs
import random
import math

env = environs.Cournot()

number_actions = 3

qtable = np.ones((1, number_actions))

learning_rate = 0.9
discount_rate = 0.8
#epsilon = 0.5 # for randomly chose
beta = 0.5 # for chosing like in the paper
decay_rate= 0.005

num_episodes = 100
max_steps = 30

for episode in range(num_episodes):

    # reset the environment
    state, info = env.reset()
    done = False

    for s in range(max_steps):

        # exploration-exploitation tradeoff
        
        logit = lambda a: math.exp(qtable[0,a]/beta) / sum([math.exp(qtable[0,a]/beta) for a in [0,1,2]])
        
        probs = [logit(action) for action in range(0,number_actions)]
        
        collect = 0
        p = []
        for prob in probs:
            collect = prob + collect
            p.append(collect)

        r = random.uniform(0,1)
        if r<p[0]:
            action = 0
        elif r<p[1]:
            action = 1
        else:
            action = 2

        '''if random.uniform(0,1) < epsilon:
            # explore
            action = env.action_space.sample()
        else:
            # exploit
            action = np.argmax(qtable[state,:])'''

        # take action and observe reward
        new_state, reward, done, info = env.step(action)

        # Q-learning algorithm
        qtable[state,action] = (1-learning_rate)*qtable[state,action] + learning_rate * reward
        #print('{}, {}'.format(action,reward))
        #print(qtable)

        # Update to our new state
        state = new_state

        # if done, finish episode
        if done:
            break

    # Decrease epsilon
    epsilon = np.exp(-decay_rate*episode)

print(f"Training completed over {num_episodes} episodes")

print(qtable)