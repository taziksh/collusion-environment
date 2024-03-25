import numpy as np
import environs
import random

env = environs.Cournot()

qtable = np.zeros((1, 3))

learning_rate = 0.9
discount_rate = 0.8
epsilon = 0.5
decay_rate= 0.005

num_episodes = 100
max_steps = 30

for episode in range(num_episodes):

    # reset the environment
    state, info = env.reset()
    done = False

    for s in range(max_steps):

        # exploration-exploitation tradeoff
        if random.uniform(0,1) < epsilon:
            # explore
            action = env.action_space.sample()
        else:
            # exploit
            action = np.argmax(qtable[state,:])

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