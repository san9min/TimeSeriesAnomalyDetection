import torch
from models.model import *
import torch.nn.functional as F
import numpy as np
from util.metric import fbeta_score

class ICMagent:

    def __init__(self,args):

        self.device = args.device

        self.icm = ICMModel(args)
        self.encoder = Encoder(args)
        self.qnetwork = Qnetwork(args)
        self.targetnetwork = Qnetwork(args)
        self.targetnetwork.eval()
        
        self.eps = args.eps
        self.eta = args.eta

    def get_action(self,state,greedy = False):

        state = torch.Tensor(state).to(self.device)
        state = state.float()
        state = self.encoder(state)

        qvals = self.qnetwork(state.detach())
        if greedy:
            action = torch.argmax(qvals,dim =1)    
        else:
            action = epsilon_greedy_policy(qvals,eps = self.eps)
        
        return action.data.cpu().numpy() # 0 or 1
    
    def get_intrinsic_reward(self,state,action,next_state):
        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action = self.action_one_hot_encoding(action.reshape(-1,1))

        state = self.encoder(state)
        next_state = self.encoder(next_state)

        real_next_state_feature, pred_next_state_feature, _ = self.icm([state, action, next_state])
        intrinsic_reward = self.eta * F.mse_loss(pred_next_state_feature,real_next_state_feature).unsqueeze(0) 
        return intrinsic_reward.data.cpu().numpy()
  
    def action_one_hot_encoding(self,action): #dim(action) = batch,1
        batch = len(action)
        action_oh = torch.zeros([batch,2])
        for i in range(batch):
            j = action[i] 
            action_oh[i][j] = 1        
        return action_oh.to(dtype=torch.float, device = self.device)

    def get_params(self):
        params = list(self.encoder.parameters()) + list(self.icm.forward_net.parameters()) + list(self.icm.inverse_net.parameters()) + list(self.qnetwork.parameters())
        return params

    def state_dict(self):
      param_group = {'qnetwork' : self.qnetwork.state_dict(), 'icm' : self.icm.state_dict(), 'encoder' : self.encoder.state_dict()}
      return param_group 

    def load_state_dict(self,params : dict):
        self.qnetwork.load_state_dict(params['qnetwork'])
        self.icm.load_state_dict(params['icm'])
        self.encoder.load_state_dict(params['encoder'])
        print('Weights loading success!')
    
    def train_mode(self):
        self.qnetwork.train()
        self.icm.train()
        self.encoder.train()

    def eval_mode(self):
        self.qnetwork.eval()
        self.icm.eval()
        self.encoder.eval()

    def total_loss(self, args, q_loss, forward_loss, inverse_loss):
        loss = args.lambda_ * q_loss + (1 - args.beta_) * inverse_loss + args.beta_ * forward_loss 
        return loss

    def compute_loss(self,args,loss_fns,replay):
    
        q_loss_func,f_loss_func,i_loss_func= loss_fns # loss functions unpack

        state1_batch, action_batch, reward_batch, state2_batch = replay.get_batch() 
        action_batch_value = action_batch.view(-1,1)
        reward_batch = reward_batch.view(-1,1).to(self.device)

        action_batch_one_hot = self.action_one_hot_encoding(action_batch_value)

        state1_batch = self.encoder(state1_batch)
        state2_batch = self.encoder(state2_batch)

        real_next_state, pred_next_state, pred_action = self.icm([state1_batch, action_batch_one_hot, state2_batch])
        #loss
        
        forward_pred_err = f_loss_func(pred_next_state,real_next_state.detach()) * args.fscale

        inverse_pred_err = i_loss_func(pred_action,action_batch_one_hot) * args.iscale
        
        #reward.
        reward = reward_batch 
        qvals = self.targetnetwork(state2_batch.detach())
        reward += args.gamma * torch.max(qvals,dim=1)[0].view(-1,1) 

        reward_pred = self.qnetwork(state1_batch.detach())
        reward_target = reward_pred.clone() #batch,2
        indices = torch.stack((torch.arange(action_batch_value.shape[0]),action_batch_value.squeeze(dim = 1)), dim=0)
        indices = indices.tolist()
        reward_target[indices] = reward.squeeze()

        q_loss = q_loss_func(reward_pred, reward_target.detach()) * args.qscale    

        loss = self.total_loss(args,q_loss, forward_pred_err, inverse_pred_err)
        return loss
