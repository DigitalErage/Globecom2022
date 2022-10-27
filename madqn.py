import os
import time

import torch
import numpy as np
from copy import deepcopy

from dqn import DQNAgent
from environment import Scenario,ENV_dqn,ENV_ppo

import pickle

import argparse

def parse_args():
    parser = argparse.ArgumentParser("SF")
    parser.add_argument('--idx', metavar='N', type=int, default=0,
                    help='DRL training times/epoches')
    parser.add_argument('--step', metavar='N', type=int, default=10,
                    help='attack simulation times')
    parser.add_argument("--um", metavar='%', type=int, default=1)
    return parser.parse_args()

def train(batch_size=0,bplus=0,random_seed=0,initial_batch_list = [],last_evaluation_episodes=0,load=False,save=False,um=1):

    env_name = "SF"
    has_continuous_action_space = False
    
    sf = Scenario()
    
    nagents = 1
    
    length = 10
    
    if um == 0:
        env = ENV_dqn(length, sf, nagents, um = False)
    else:
        env = ENV_dqn(length, sf, nagents, um = True)

    # state space dimension
    state_dim = env.length

    # action space dimension
    if has_continuous_action_space:
        action_dim = len(env.action_space)
    else:
        action_dim = len(env.action_space)

    ###################### logging ######################

    max_ep_len = 1000                    # max timesteps in one episode
    max_training_timesteps = sum(initial_batch_list)+int(last_evaluation_episodes*state_dim-1)   # break training loop if timeteps > max_training_timesteps

    print_freq = max_ep_len     # print avg reward in the interval (in num timesteps)
    log_freq = max_ep_len * 2       # log avg reward in the interval (in num timesteps)
    save_model_freq = max_training_timesteps      # save model frequency (in num timesteps)

    action_std = None
    
    batch_list = initial_batch_list+[int(last_evaluation_episodes*state_dim)]
    
    batch_idx = 0
    current_batch = batch_list[batch_idx]

    eval_set1, eval_set2 = set(), set()

    random_seed = random_seed         # set random seed if required (0 = no random seed)

    run_num_pretrained = 0      #### change this to prevent overwriting weights in same env_name folder

    directory = "PPO_preTrained"
    if not os.path.exists(directory):
          os.makedirs(directory)

    directory = directory + '/' + env_name + '/'
    if not os.path.exists(directory):
          os.makedirs(directory)

    checkpoint_path = [directory + "PPO_{}_{}_{}_{}.pth".format(env_name, random_seed, batch_size,i) for i in range(nagents)]

    if random_seed:
        #print("--------------------------------------------------------------------------------------------")
        #print("setting random seed to ", random_seed)
        torch.manual_seed(random_seed)
        #env.seed(random_seed)
        np.random.seed(random_seed)
        
    # initialize a DQN agent
    agents = [DQNAgent(state_dim, action_dim, device = torch.device('cuda:0')).cuda() for i in range(nagents)]
    if load:
        for i in range(nagents):
            agents[i].load(checkpoint_path[i])
    # track total training time
    start_time = time.time()

    # printing variables
    print_running_reward = 0
    print_running_episodes = 0

    time_step = 0
    update_step = 0
    i_episode = 0
    
    reward_list = []
    eval_list = []
    
    total_metric_list = []
    

    # training loop
    while time_step <= max_training_timesteps:

        state = deepcopy(env.reset())
        current_ep_reward = 0
        metric_list = []
        for t in range(1, max_ep_len+1):

            # select action with policy
            actions = [agents[i].act(state,i) for i in range(nagents)]
            next_state, reward, done, eval_set1, eval_set2, metrics = env.step(actions, eval_set1 = eval_set1, eval_set2 = eval_set2)

            # saving reward and is_terminals
            for i in range(nagents):
                agents[i].remember(deepcopy(state[i]), deepcopy(actions[i]), reward, deepcopy(next_state[i]), done)
            state = deepcopy(next_state)
            time_step += 1
            update_step += 1
            current_ep_reward += reward
            metric_list += metrics
                
            # update DQN agent
            for i in range(nagents):
                if agents[i].buffer.size() > current_batch:
                    agents[i].replay(current_batch)

            if done:
                break
        
        total_metric_list.append(metric_list)
        reward_list.append(current_ep_reward)
        eval_list.append(len(eval_set1)+len(eval_set2))
        print_running_reward += current_ep_reward
        print_running_episodes += 1
        

        i_episode += 1
        if print_running_episodes % (current_batch/length) == 0:

            # print average reward till last episode
            end_time = time.time()
            print_avg_reward = print_running_reward / print_running_episodes
            print_avg_reward = round(print_avg_reward, 4)

            print("Episode : {} \t Timestep : {} \t Average Reward : {} \t Eval Times : {} \t Time Cost : {}".format(i_episode, time_step, print_avg_reward, len(eval_set1)+len(eval_set2), end_time - start_time))


            print_running_reward = 0
            print_running_episodes = 0
            current_batch = batch_list[batch_idx]
    if save:
        for i in range(nagents):
            agents[i].save(checkpoint_path[i])
    return reward_list, eval_list, total_metric_list

if __name__ == "__main__":
    arglist = parse_args()
    initial_batch_list = [500]*20#[440]*68
    rewards_list = []
    evals_list = []
    #metrics_list = []
    for i in range(arglist.idx*arglist.step,(arglist.idx+1)*arglist.step):
        reward_list, eval_list, metric_list = train(initial_batch_list=initial_batch_list,random_seed=i,um=arglist.um)
        rewards_list.append(reward_list)
        evals_list.append(eval_list)
        #metrics_list.append(metric_list)
        pickle.dump(rewards_list,open("results/r_madqn_%d_%d.pkl"%(arglist.um,arglist.idx),"wb"))
        pickle.dump(evals_list,open("results/e_madqn_%d_%d.pkl"%(arglist.um,arglist.idx),"wb"))
        pickle.dump(metric_list,open("results/m_madqn_%d_%d_%d.pkl"%(arglist.um,arglist.idx,i),"wb"))
