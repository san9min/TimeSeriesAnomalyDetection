import torch
import torch.nn as nn
import torch.optim
import numpy as np
import matplotlib.pyplot as plt

from config import get_parse
from datasets.build_data import load_data
from models.env import ENV
from models.agent import ICMagent
from util.ExperienceReplay import ReplayMemory
from test import main_test


def main(args):
    train, test = load_data(args)

    #init
    agent = ICMagent(args)
    replay = ReplayMemory(args)

    if args.pretrain :
        agent.load_state_dict(torch.load(args.pre_trained_weights))
        agent.train_mode()
        
    #loss and optim
    f_loss_func = nn.MSELoss().to(args.device)
    i_loss_func = nn.CrossEntropyLoss().to(args.device)
    q_loss_func = nn.MSELoss().to(args.device)
    loss_fns = (q_loss_func,f_loss_func,i_loss_func)
    optim = torch.optim.Adam(agent.get_params(), lr = args.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optim, lr_lambda=lambda epoch: 0.95 ** epoch)

    num_episodes = len(train)
    rewards_memory = []
    for e in range(num_episodes):
        _, state_set, label_set = train[e]
        env = ENV(state_set,label_set,args)
        print(f"{e+1} th episode starts")
        state= env.reset() #get initial state
        j = 0
        for i in range(args.epochs):
            optim.zero_grad()
            i -=  j
            action = agent.get_action(state)
            next_state, reward, done,info = env.step(i,action)
            if done:
                state = env.reset()
                j += i + 1
                continue
            i_reward = agent.get_intrinsic_reward(state,action,next_state)
            reward += i_reward
            replay.add_memory(state,action,reward,next_state,info)
            loss = agent.compute_loss(args,loss_fns,replay)
            loss.backward()
            optim.step()      
            
            env.render(reward)
            state = next_state  
            
            if (i+j+1) % 5 == 0:
                agent.targetnetwork.load_state_dict(agent.qnetwork.state_dict())
        scheduler.step()
        args.eps *= args.eps_decay

        print(f"Total Reward at {e+1}th episode : ",sum(env.reward_hist))        
        print(f"{e+1} th episode ends")
        print('*'*50)        
        rewards_memory.append(sum(env.reward_hist))
    
    torch.save(agent.state_dict(),'Check-point_'+ args.datasets) 

if __name__ == '__main__': 
    args = get_parse()
    main(args)
