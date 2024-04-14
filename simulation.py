import numpy as np
import environs
import agent
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import pandas as pd

def run(num_agents,gamma,memory,fixed_cost,variable_cost):

    joint_profits = []
    joint_quantities = []

    learning_rate = 0.5 #perfect!

    num_actions = 10
    demand_quantity = 10
    demand_factor =1

    if memory:
        num_cumulative_action = num_agents*num_actions
    else:
        num_cumulative_action = 1

    num_episodes = 100     # perfect!
    max_steps = 1000      

    beta = 1000 
    beta_decay = 0.9999 # perfect!

    env = environs.Cournot(num_agents,demand_quantity,demand_factor)

    agents = [agent.Agent(num_actions, num_cumulative_action,learning_rate,
                      gamma, beta, variable_cost, fixed_cost,memory)
                      for _ in range(num_agents)]

    for episode in tqdm(range(num_episodes)):

        # reset the environment
        sum_qs, info = env.reset()
        [agent.reset() for agent in agents]
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

            beta = max(beta*beta_decay, 0.2)
    
    NEquantity = round((demand_quantity-variable_cost)*num_agents/(demand_factor*(num_agents+1)),2)
    NEprofit = round((demand_quantity-variable_cost)**2*num_agents/(demand_factor*(num_agents+1)**2),2)
    COquantity = round((demand_quantity-variable_cost)/2/demand_factor,2)
    COprofit = round((demand_quantity-variable_cost)**2/4/demand_factor,2)

    theory = [NEquantity, NEprofit, COquantity, COprofit]

    
    return agents,env,beta,num_episodes,max_steps, theory


def plot_traj(env,agents,theory,stamp,timesteps):
    
    total = env.sum_qs_traj[-timesteps:]
    individual_qs = []
    for agent in agents:
        individual_qs.append(agent.action_traj[-timesteps:])

    NEquantity, NEprofit, COquantity, COprofit = theory

    ls=['-.',':','--','-']

    plt.figure(figsize=(10,10))
    
    plt.plot([NEquantity]*timesteps, linestyle=(0, (1, 10)))
    plt.plot([COquantity]*timesteps, linestyle=(0, (5, 10)))
    plt.plot(total, linestyle=(0, (5, 1)))

    str_agents = []
    for i in range(len(agents)):
        plt.plot(individual_qs[i], linestyle=ls[i])
        str_agents.append('Firm '+str(i+1))

    plt.legend(['Nash equilibrium','Collusive equilibrium','Total']+str_agents)
    plt.ylabel('Quantities')
    plt.xlabel('timesteps t')

    n,f,w,gamma,memory = stamp

    plt.title(f'Quantites n={n}, f={f}, w={w}, memory={memory}, gamma={gamma}')
    if gamma>0:
        gamma = 'ON'
    else:
        gamma= 'OFF'
    plt.savefig(os.path.join(os.getcwd(),'data',f'n{n}_f{f}_w{w}_g{gamma}_m{memory}'))
    
    return 


def plot_tables(data,ws,fs,ns,gammemos):

    for gammemo in gammemos:
        
        tuples = [(f, w, c) for f in fs for w in ws for c in ['quantity','profit']]

        print(tuples)

        index = pd.MultiIndex.from_tuples(tuples, names=["fixed", "variable", "joint outcome"])

        empty_content = [['']*len(ns)]*(len(fs)*len(ws)*2)

        data_frame = pd.DataFrame( empty_content , index=index, columns = ns)

        for simu in data:
            env,agents,overview,theory,stamp = simu

            joint_quantities, joint_profits = overview

            NEquantity, NEprofit, COquantity, COprofit = theory

            n,f,w,gamma,memory = stamp

            if gammemo == [gamma,memory]:

                data_frame.loc[(f,w,'quantity'),n] = f'{joint_quantities} ({NEquantity})'
                data_frame.loc[(f,w,'profit'),n] = f'{joint_profits} ({NEprofit})'

        gamma, memory = gammemo

        if gamma>0:
            g = 'ON'
        else:
            g = 'OFF'

        data_frame.to_csv(os.path.join(os.getcwd(),'data',f'comparison_gamma{g}_memory{memory}'))

    return True



def print_overview(env,beta,num_episodes,max_steps, theory,stamp):

    n,f,w,gamma,memory = stamp

    print('SETTING')
    print(f'Cournot game: n={n}, f={f}, w={w}')
    print(f'RL-setting: gamma={gamma}, memory={memory}')

    print(f'beta after final episode {num_episodes} is: {beta}')

    print(f"Training completed over {num_episodes} episodes and {max_steps} steps")

    NEquantity, NEprofit, COquantity, COprofit = theory

    '''for i, agent in enumerate(agents):
        print()
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
    print(f'Nash equilibrium amount is: {NEquantity} with profit {NEprofit}')
    print(f'Collusive equilibrium amount is: {COquantity} with profit {COprofit}')
    print()

    joint_quantities = env.sum_qs_traj[-10:]
    joint_profits = [sum(rewards) for rewards in env.rewards_traj[-10:]]
    print('joint profits of the last 10 rounds:')
    print(joint_profits)
    print('joint quantities of the last 10 rounds:')
    print(joint_quantities)

    print(f"The mean joint profit of last 100 rounds: {np.mean(joint_profits)}")
    print(f"The mean joint quantity of last 100 rounds: {np.mean(joint_quantities)}")

    return np.mean(joint_quantities), np.mean(joint_profits)