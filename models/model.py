import torch
import torch.nn as nn
import torch.nn.functional as F

class ICMModel(nn.Module):
    def __init__(self, args):
        super(ICMModel, self).__init__()

        self.device = args.device

        self.encoder = nn.LSTM(2,64,3,batch_first = True).to(self.device)

        self.inverse_net = nn.Sequential(
            nn.Linear(64* 2,256),
            nn.ReLU(),
            nn.Linear(256,2)
        ).to(self.device)

        self.forward_net = nn.Sequential(
            nn.Linear(2 + 64 ,256),
            nn.LeakyReLU(),
            nn.Linear(256, 64)
        ).to(self.device)
        
        for p in self.modules():
            if isinstance(p, nn.Linear):
                nn.init.kaiming_uniform_(p.weight, a=1.0)
                p.bias.data.zero_()

            if isinstance(p,nn.LSTM):
                for name,param in p.named_parameters():
                    if 'bias' in name:
                        n = param.size(0)
                        start, end = n//4, n//2
                        param.data[start:end].fill_(1.0)
                        nn.init.ones_(param)
            
    def forward(self, inputs):
        state, action, next_state = inputs

        encode_state, _ = self.encoder(state)
        encode_next_state, _ = self.encoder(next_state)
        
        pred_action = torch.cat((encode_state[:,-1], encode_next_state[:,-1]), 1)
        pred_action = self.inverse_net(pred_action)

        # get pred next state
        pred_next_state = torch.cat((encode_state[:,-1].detach(), action), 1)
        pred_next_state = self.forward_net(pred_next_state)

        real_next_state = encode_next_state[:,-1]

        return real_next_state, pred_next_state, pred_action

class Qnetwork(nn.Module):
    def __init__(self,args):
        super(Qnetwork, self).__init__()

        self.rnn = nn.LSTM(2,64,3,batch_first= True).to(args.device)
        self.fc1 = nn.Linear(64,256).to(args.device)
        self.fc2 = nn.Linear(256,128).to(args.device)
        self.fc3 = nn.Linear(128,2).to(args.device)
        
        for p in self.modules():
            if isinstance(p, nn.Linear):
                nn.init.kaiming_uniform_(p.weight, a=1.0)
                p.bias.data.zero_()
            if isinstance(p,nn.LSTM):
                for name,param in p.named_parameters():
                    if 'bias' in name:
                        n = param.size(0)
                        start, end = n//4, n//2
                        param.data[start:end].fill_(1.0)
                        nn.init.ones_(param)

    def forward(self,x):
        outputs, _ = self.rnn(x)
        qvals = outputs[:,-1]
        qvals = F.relu(self.fc1(qvals))
        qvals = F.relu(self.fc2(qvals))
        qvals = self.fc3(qvals)
        return qvals #batch,2


def epsilon_greedy_policy(qvals,eps):
    batch = len(qvals)
    action = torch.zeros((batch,1))
    for b in range(batch):
        if torch.rand(1) < eps:
            action[b] = torch.randint(low=0,high=2,size=(1,)) # 0 : normal, 1 : anomaly
        else:
            action[b] = torch.argmax(qvals[b])
    return action