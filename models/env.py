import torch
import numpy as np


class ENV:
    def __init__(self,state_set,label_set,args):
        self.state = state_set
        self.label = label_set
        self.len = len(self.label)

        self.window_size = args.window_size

        assert len(self.state) == len(self.label)

        self.tp = args.TP
        self.fp = args.FP
        self.fn = args.FN
        self.tn = args.TN

        self.reward_hist = []
        
        
    def reset(self): #initial state
        return self.state[0].transpose().reshape(1,self.window_size,2) #(1,-1,1)
    
    def step(self,idx,action): 
        a_pred = action
        a_real = self.label[idx]
        reward = self.get_reward(a_pred,a_real)

        if self.len == idx + 1:
            done = True
            next_state = None
        else:
            done = False
            next_state = self.state[idx+1].transpose().reshape(1,self.window_size,2)
        return next_state, reward, done, a_real

    def get_reward(self,action_pred,action_real):
        if action_real == 1: #anomaly
            if action_pred == 1: # TP
                return np.array([self.tp],dtype = np.float32)
            elif action_pred == 0: #FN
                return np.array([self.fn],dtype = np.float32)
        elif action_real == 0: #not anomaly
            if action_pred == 1: #FP
                return np.array([self.fp],dtype = np.float32)
            if action_pred == 0: #TN
                return np.array([self.tn],dtype = np.float32)
    
    def render(self,reward):
        self.reward_hist.append(reward)

