import numpy as np
import environs


env = environs.Cournot()

num_episodes = 100
max_steps = 30

agents = [environs.Agent() for i in range(3)]

for episode in range(num_episodes):

    # reset the environment
    state, info = env.reset()
    done = False

    for s in range(max_steps):

        # exploration-exploitation tradeoff
        actions = []

        for agent in agents:
            actions.append(agent.run())

        '''if random.uniform(0,1) < epsilon:
            # explore
            action = env.action_space.sample()
        else:
            # exploit
            action = np.argmax(qtable[state,:])'''

        # take action and observe reward
        new_state, rewards, done, info = env.step(actions,agents)

        for i, agent in zip(range(len(agents)),agents):
            agent.update(rewards[i],state,actions[i])
        
        # Update to our new state
        state = new_state

        # if done, finish episode
        if done:
            break

    # Decrease epsilon
    # epsilon = np.exp(-decay_rate*episode)

print(f"Training completed over {num_episodes} episodes")

for agent in agents:
    print(agent.qtable)