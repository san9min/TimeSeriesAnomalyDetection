from random import shuffle
import torch
import numpy as np

class ReplayMemory:
    def __init__(self, args):
        self.N = args.capacity
        self.batch_size = args.batch_size
        self.anomaly_ratio = args.anomaly_ratio

        self.memory_a = []
        self.memory_n = [] 
        self.counter_a = 0
        self.counter_n= 0
        

        self.device = args.device
        
    def add_memory(self, state1, action, reward, state2,a_real):
        if a_real == 1: #anomaly   
            self.counter_a +=1 
            if self.counter_a % 500 == 0:
                shuffle(self.memory_a)
            if len(self.memory_a) < self.N:
                self.memory_a.append((state1, action, reward, state2))
            else:
                rand_index = np.random.randint(0,self.N-1)
                self.memory_a[rand_index] = (state1, action, reward, state2)
        else: #not anomaly
            self.counter_n += 1
            if self.counter_n % 500 == 0:
                shuffle(self.memory_n)
            if len(self.memory_n) < self.N:
                self.memory_n.append((state1, action, reward, state2))
            else:
                rand_index = np.random.randint(0,self.N-1)
                self.memory_n[rand_index] = (state1, action, reward, state2)
        
    def get_batch(self): 
        if len(self.memory_a) > self.batch_size * self.anomaly_ratio and len(self.memory_n) > self.batch_size *(1-self.anomaly_ratio):
            idx_a = np.random.choice(np.arange(len(self.memory_a)),int(self.batch_size * self.anomaly_ratio))
            idx_n = np.random.choice(np.arange(len(self.memory_n)),int(self.batch_size *(1-self.anomaly_ratio)))
            batch = [self.memory_a[i] for i in idx_a] + [self.memory_n[i] for i in idx_n]        
        else:
            memory = self.memory_a + self.memory_n
            if len(memory) < self.batch_size:
                batch_size = len(memory)
            else:
                batch_size = self.batch_size            
            idx = np.random.choice(np.arange(len(memory)),batch_size)
            batch = [memory[i] for i in idx]

        state_batch = torch.stack([torch.from_numpy(x[0]).squeeze(dim = 0) for x in batch],dim=0).to(device = self.device,dtype = torch.float)
        action_batch = torch.Tensor([x[1] for x in batch]).long()
        reward_batch = torch.Tensor([x[2] for x in batch]).to(self.device)
        next_state_batch = torch.stack([torch.from_numpy(x[3]).squeeze(dim = 0) for x in batch],dim=0).to(device = self.device,dtype = torch.float)
        return state_batch, action_batch, reward_batch, next_state_batch
