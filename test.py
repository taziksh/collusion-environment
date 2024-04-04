import numpy as np
import environs
import agent

np.set_printoptions(precision=2)

num_agents = 2 # in paper: [2 , 3, 4 , 5 , 6]

num_actions = 40 # in paper: 40

num_cumulative_action = 1 # with memory: num_agents*num_actions

demand_quantity = 40 # in paper it is denoted with u=40
demand_factor = 1 # in paper it is denoted with v=1

variable_cost = 4 # the variable cost w = 4
fixed_cost = 0 # in the paper we have f = 0

learning_rate = 0.5 # in paper: [ 0.05 , 0.25 , 0.5 , 1 ]
beta = 1000 # for chosing like in the paper
# In paper decay by factor 0.99999 each time step, but computationally 0.999 better

# with memory means we have states different states for each agent:
# - own previous production level
# - production level of others
# myopic says that the agents learn short term achieved reward gamma=0 to update q-table
# non-myopic means they learn by rewards and expected q-talbe long term values gamma=0.9

gamma = 0 # myopic firms=0 , non-myopic 0.9

epsilon = 0.1 # in paper: not used, but for less computational effort keep that approach

# unresolved: Read!
# TODO: Do they in the paper have several episodes so reset environment several times?
num_episodes = 100
max_steps = 300

env = environs.Cournot(num_agents,demand_quantity,demand_factor)

agents = [agent.Agent(num_actions, num_cumulative_action,
                      learning_rate, beta, epsilon, variable_cost, fixed_cost)
                      for _ in range(num_agents)]

for episode in range(num_episodes):

    # reset the environment
    state, info = env.reset()
    [agent.reset() for agent in agents]
    done = False
    sum_qs = 0

    for s in range(max_steps):
        actions = [agent.select_action(sum_qs) for agent in agents]
        
        # take action and observe reward
        sum_qs, rewards = env.step(actions, agents)

        # Q-learning algorithm
        for agent_idx, reward in enumerate(rewards):
            agents[agent_idx].update_qtable(actions[agent_idx], reward)

        beta = beta*0.99

    
    # for agent in agents:
    #     agent.decrease_epsilon(episode)

print('Beta is:' +str(beta))

print(f"Training completed over {num_episodes} episodes")
for i, agent in enumerate(agents):
    print(f"Q-values for agent {i}: {agent.qtable}")
