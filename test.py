import numpy as np
import simulation as sim

np.set_printoptions(precision=2)

'''
Parameter settings different from the paper:
demand:
u = 10 demandquantity
v = 1 demandfactor
maybe other demand function shape?

agents:
w = 1, 2 variable costs
f = {0,2} fixed costs
n = {2,3} agents
actions = 10

RL settings:
episodes 100
steps 1000
learning rate = 0.5
memory / no memory
gamma = {0,0.9}
beta minimum 0.5
'''
ws = [0,1]
fs = [0,2]
ns = [2,3]
gammemos = [[0,False],[0,True],[0.9,True]]

'Starting simulations...'

data = []

for w in ws:
    for f in fs:
        for n in ns:
            for gamma,memory in gammemos:

                agents,env,beta,num_episodes,max_steps, theory = sim.run(
                    num_agents=n,gamma=gamma,memory=memory,
                    fixed_cost=f,variable_cost=w)

                stamp = [n,f,w,gamma,memory]     

                overview = sim.print_overview(env,beta,num_episodes,max_steps,theory,stamp)

                sim.plot_traj(env,agents,theory,stamp,timesteps=50)

                data.append([env,agents,overview,theory,stamp])

sim.plot_tables(data,ws,fs,ns,gammemos)


'''

####################### Paper parameter settings #######################################

num_agents = 2 # in paper: [2 , 3, 4 , 5 , 6]

num_actions = 10 # in paper: 40

# cumulative actions: with memory: num_agents*num_actions, without memory: 1
memory = True
if memory:
    num_cumulative_action = num_agents*num_actions
else:
    num_cumulative_action = 1

demand_quantity = 10 # in paper it is denoted with u=40
demand_factor = 1 # in paper it is denoted with v=1

variable_cost = 1 # the variable cost w = 4
fixed_cost = 2 # in the paper we have f = 0

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


Not mentioned in the paper: How many episodes and how many steps?
num_episodes = 100     # in total 1,000,000 in paper
max_steps = 1000         '''