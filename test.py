import torch
import numpy as np


from config import get_parse
from datasets.Yahoo import build_yahoo
from models.agent import ICMagent


def test(args):
    print("Test start!")
    
    if args.datasets == 'Yahoo':
        _,test = build_yahoo(args)

    agent = ICMagent(args)

    if args.pretrain :
        agent.load_state_dict(torch.load(args.pre_trained_weights))
        agent.eval_mode()
    

    truepositive = 0
    tp_plus_fp = 0
    tp_plus_fn = 0
    num_episodes = len(test)
    for i in range(num_episodes):
        _,state_set,label_set = test[i]
        label_set = np.expand_dims(label_set,axis = 1)
        for j in range(len(state_set)):
            state = state_set[j].reshape(1,args.window_size,2)

            y_true = label_set[j]
            y_true = torch.FloatTensor(y_true).squeeze().float()

            y_pred = agent.get_action(state,1)
            y_pred = torch.FloatTensor(y_pred).float()   

            truepositive += (y_pred * y_true).sum()
            tp_plus_fp += y_pred.sum()
            tp_plus_fn += y_true.sum()
    precision = truepositive.div(tp_plus_fp + 1e-11)
    recall = truepositive.div(tp_plus_fn + 1e-11)



    f1_score = torch.mean((precision * recall * (2)).div(precision + recall)).item()

    print(f"Precision : {precision.item()}")
    print(f"Recall : {recall.item()}")
    print(f"F1-score : {f1_score}")    
    

if __name__ == '__main__': 
    args = get_parse()
    test(args)