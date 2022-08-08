import torch
import numpy as np

from config import get_parse
from datasets.build_data import load_data
from models.agent import ICMagent


def TEST_ALL(args):
    print("Test start!")

    agent = ICMagent(args)
    print('Hello, I was made by sangmin')
    if args.pretrain :
        agent.load_state_dict(torch.load(args.pre_trained_weights))
        agent.eval_mode()
    print('-'*50)
    for data_name in args.test_data_names :
        args.datasets = data_name
        test(args,agent)

def test(args,agent):

    print(f"Data : {args.datasets}")
    _, test = load_data(args)

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
    print('-'*50)
    

if __name__ == '__main__': 
    args = get_parse()
    TEST_ALL(args)