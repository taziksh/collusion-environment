import numpy as np
import environs
import agent
from tqdm import tqdm

np.set_printoptions(precision=2)

num_agents = 6 # in paper: [2 , 3, 4 , 5 , 6]

num_actions = 40 # in paper: 40

# cumulative actions: with memory: num_agents*num_actions, without memory: 1
num_cumulative_action = num_agents*num_actions # 1 or num_agents*num_actions
if num_cumulative_action>1:
    memory = True
else:
    memory = False

demand_quantity = 40 # in paper it is denoted with u=40
demand_factor = 1 # in paper it is denoted with v=1

variable_cost = 2 # the variable cost w = 4
fixed_cost = 0 # in the paper we have f = 0

learning_rate = 0.5 # in paper: [ 0.05 , 0.25 , 0.5 , 1 ]
beta = 1000 # for chosing like in the paper
beta_decay = 0.999
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
num_episodes = 900     #1000 in paper
max_steps = 10        #1000 in paper

env = environs.Cournot(num_agents,demand_quantity,demand_factor)

agents = [agent.Agent(num_actions, num_cumulative_action,learning_rate,
                      gamma, beta, epsilon, variable_cost, fixed_cost,memory)
                      for _ in range(num_agents)]

joint_profits = []
joint_quantities = []

for episode in tqdm(range(num_episodes)):

    # reset the environment
    state, info = env.reset()
    [agent.reset() for agent in agents]
    done = False
    sum_qs = 0

    for s in range(max_steps):
        actions = [agent.select_action(beta) for agent in agents]
        
        # take action and observe reward
        sum_qs, rewards = env.step(actions, agents)

        #results:
        joint_profits.append(sum(rewards))
        joint_quantities.append(sum_qs)

        # Q-learning algorithm
        for agent_idx, reward in enumerate(rewards):
            agents[agent_idx].update_qtable(reward,sum_qs)

        beta = beta*beta_decay
    
    # for agent in agents:
    #     agent.decrease_epsilon(episode)

print(f'beta after final episode {num_episodes} is: {beta}')

print(f"Training completed over {num_episodes} episodes")


for i, agent in enumerate(agents):
    '''print()
    print('Softmax values:')
    print(agent.softmax())
    print(f"Q-values for agent {i}:")
    #print("Mean actions for Q-values are given for states")
    print("actions are in this order [0,1,2,...,#actions]")

    # num_actions only if we have memory true. Otherwise 0+1=1 amount of states for actions
    for a in range((num_actions-1)*memory+1):
        print('-----------------------------------------------')
        print(f"state[0]: previous q_{i} = {a}:")
        print()

        for k in range(num_cumulative_action):
            #bullshit, because the weights are negative:
            #weights = [q_value / sum(agent.qtable[a][k]) for q_value in agent.qtable[a][k]]
            #mean_action = sum(weight*action for weight, action in zip(weights,range(num_actions)))
            #print(f"state[1]: mean for previous sum(q_i) = {k}: {mean_action}")
            print(f"state[1]: previous sum(q_i) = {k}: {agent.qtable[a][k]}")'''

print('---------------------------------------------------')
NEquantity = round((demand_quantity-variable_cost)*num_agents/(demand_factor*(num_agents+1)))
NEprofit = round((demand_quantity-variable_cost)**2*num_agents/(demand_factor*(num_agents+1)**2))
print(f'Nash equilibrium amount is: {NEquantity} with profit {NEprofit}')
COquantity = round((demand_quantity-variable_cost)/2/demand_factor)
COprofit = round((demand_quantity-variable_cost)**2/4/demand_factor)
print(f'Collusive equilibrium amount is: {COquantity} with profit {COprofit}')
print()
print('joint profits of the last 10 rounds:')
print(joint_profits[-10:])
print('joint quantities of the last 10 rounds:')
print(joint_quantities[-10:])

print(f"The mean joint profit of last 100 rounds: {np.mean(joint_profits[-10:])}")
print(f"The mean joint quantity of last 100 rounds: {np.mean(joint_quantities[-10:])}")